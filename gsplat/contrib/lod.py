# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Simple octree-based Level-of-Detail (LOD) for already-trained Gaussians.

This is a pure **inference-time / post-processing** technique: it never
touches the training loop or gradients. It takes a frozen (trained) set of
Gaussians and:

1. :func:`build_octree_lod` partitions the scene with an octree and, for
   every node, moment-matches all Gaussians underneath it into a single
   coarser "proxy" Gaussian (see :func:`_merge_gaussians`). This is a
   one-time, offline step run on a trained checkpoint.
2. :func:`select_lod` walks the tree once per frame given a camera position:
   nodes whose projected screen size is small are represented by their
   single proxy Gaussian; nodes that are still large on screen are expanded
   into their children, recursively, down to the original per-point data at
   the leaves.

The output of :func:`select_lod` is a plain dict of ``means`` / ``quats`` /
``scales`` / ``opacities`` / ``colors`` tensors that can be fed directly into
:func:`gsplat.rendering.rasterization`. ``colors`` may be plain RGB ``[N, 3]``
or spherical-harmonic coefficients ``[N, K, 3]``.

Limitations (kept intentionally simple):

- Appearance attributes are opacity-weighted when nodes are merged. This is a
  practical SH approximation, not a view-sampled fit of the merged Gaussian.
- Opacity is combined with a simple screen-door / alpha-compositing
  approximation (``1 - prod(1 - opacity_i)``), not a physically exact
  re-derivation of the blended appearance.
- The tree walk in :func:`select_lod` is a plain Python loop over nodes
  (not batched/vectorized), which is fine for interactive viewers but is not
  optimized for very large scenes or very high frame rates.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import Tensor

# NOTE: these two helpers are inlined (rather than imported from
# ``gsplat.cuda._math`` / ``gsplat.utils``) so this module stays self-contained.
# When gsplat is shipped as a PyInstaller-frozen package, unused internal
# submodules such as ``gsplat.cuda._math`` are pruned from the archive, which
# would break ``from ..cuda._math import _rotmat_to_quat``. Keeping local copies
# avoids depending on those internals.


def normalized_quat_to_rotmat(quat: Tensor) -> Tensor:
    """Convert a normalized (w, x, y, z) quaternion to a rotation matrix (..., 3, 3)."""
    assert quat.shape[-1] == 4, quat.shape
    w, x, y, z = torch.unbind(quat, dim=-1)
    mat = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )
    return mat.reshape(quat.shape[:-1] + (3, 3))


def _rotmat_to_quat(R: Tensor) -> Tensor:
    """Convert rotation matrix to (w, x, y, z) quaternion (port of GLM's quat_cast)."""
    B = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)

    fourXSquaredMinus1 = R_flat[:, 0, 0] - R_flat[:, 1, 1] - R_flat[:, 2, 2]
    fourYSquaredMinus1 = R_flat[:, 1, 1] - R_flat[:, 0, 0] - R_flat[:, 2, 2]
    fourZSquaredMinus1 = R_flat[:, 2, 2] - R_flat[:, 0, 0] - R_flat[:, 1, 1]
    fourWSquaredMinus1 = R_flat[:, 0, 0] + R_flat[:, 1, 1] + R_flat[:, 2, 2]

    fourBiggestSquaredMinus1 = torch.stack(
        [
            fourWSquaredMinus1,
            fourXSquaredMinus1,
            fourYSquaredMinus1,
            fourZSquaredMinus1,
        ],
        dim=1,
    )
    biggestIndex = torch.argmax(fourBiggestSquaredMinus1, dim=1)

    biggestVal = (
        torch.sqrt(
            fourBiggestSquaredMinus1.gather(1, biggestIndex.unsqueeze(1)).squeeze(1)
            + 1.0
        )
        * 0.5
    )
    mult = 0.25 / biggestVal

    quat = torch.zeros((R_flat.shape[0], 4), dtype=R.dtype, device=R.device)

    mask = biggestIndex == 0
    quat[mask, 0] = biggestVal[mask]
    quat[mask, 1] = (R_flat[mask, 2, 1] - R_flat[mask, 1, 2]) * mult[mask]
    quat[mask, 2] = (R_flat[mask, 0, 2] - R_flat[mask, 2, 0]) * mult[mask]
    quat[mask, 3] = (R_flat[mask, 1, 0] - R_flat[mask, 0, 1]) * mult[mask]

    mask = biggestIndex == 1
    quat[mask, 0] = (R_flat[mask, 2, 1] - R_flat[mask, 1, 2]) * mult[mask]
    quat[mask, 1] = biggestVal[mask]
    quat[mask, 2] = (R_flat[mask, 1, 0] + R_flat[mask, 0, 1]) * mult[mask]
    quat[mask, 3] = (R_flat[mask, 0, 2] + R_flat[mask, 2, 0]) * mult[mask]

    mask = biggestIndex == 2
    quat[mask, 0] = (R_flat[mask, 0, 2] - R_flat[mask, 2, 0]) * mult[mask]
    quat[mask, 1] = (R_flat[mask, 1, 0] + R_flat[mask, 0, 1]) * mult[mask]
    quat[mask, 2] = biggestVal[mask]
    quat[mask, 3] = (R_flat[mask, 2, 1] + R_flat[mask, 1, 2]) * mult[mask]

    mask = biggestIndex == 3
    quat[mask, 0] = (R_flat[mask, 1, 0] - R_flat[mask, 0, 1]) * mult[mask]
    quat[mask, 1] = (R_flat[mask, 0, 2] + R_flat[mask, 2, 0]) * mult[mask]
    quat[mask, 2] = (R_flat[mask, 2, 1] + R_flat[mask, 1, 2]) * mult[mask]
    quat[mask, 3] = biggestVal[mask]

    return quat.reshape(B + (4,))


__all__ = [
    "OctreeNode",
    "build_octree_lod",
    "build_bvh_lod",
    "select_lod",
    "save_lod_cache",
    "load_lod_cache",
]


@dataclass
class OctreeNode:
    """One node of the LOD octree.

    Attributes:
        center: ``[3]`` world-space center of this node's bounding cube.
        half_size: half the edge length of the bounding cube.
        depth: depth of this node in the tree (root is ``0``).
        num_points: number of original Gaussians underneath this node.
        proxy: single moment-matched Gaussian representing every point under
            this node (see :func:`_merge_gaussians`). Always populated, even
            for leaves, so callers that want a uniform coarse view of the
            whole tree can use it directly.
        leaf_data: exact per-point tensors for this node's Gaussians. Only
            populated when ``children is None`` (a leaf) since that is the
            finest level of detail available.
        children: up to 8 child nodes, or ``None`` for a leaf.
    """

    center: Tensor
    half_size: float
    depth: int
    num_points: int
    proxy: Dict[str, Tensor]
    leaf_data: Optional[Dict[str, Tensor]] = None
    children: Optional[List["OctreeNode"]] = None
    # Tight bounding-sphere radius of this node's contents, in world units.
    # Populated by :func:`build_bvh_lod` (whose nodes are tight AABBs rather
    # than fixed cubes); left ``None`` by :func:`build_octree_lod`, in which
    # case ``half_size`` (the cube half-edge) is used instead. When set, it
    # gives :func:`select_lod` an accurate on-screen size / frustum radius.
    bound_radius: Optional[float] = None

    @property
    def is_leaf(self) -> bool:
        return self.children is None


def _merge_gaussians(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    colors: Tensor,
    opacity_mode: str = "union",
) -> Dict[str, Tensor]:
    """Moment-match ``K`` Gaussians into a single representative Gaussian.

    Uses the standard Gaussian-mixture moment-matching identity (weighted by
    opacity): the merged mean and covariance reproduce the first and second
    moments of the union of the ``K`` input Gaussians, i.e.

    ``mu = sum_i w_i * mu_i``
    ``Sigma = sum_i w_i * (Sigma_i + (mu_i - mu) (mu_i - mu)^T)``

    Args:
        means: ``[K, 3]``.
        quats: ``[K, 4]`` unit quaternions, wxyz order.
        scales: ``[K, 3]`` positive scales.
        opacities: ``[K]`` opacities in ``[0, 1]``.
        colors: appearance tensors ``[K, ..., 3]``. This supports both plain
            RGB ``[K, 3]`` and SH coefficients ``[K, C, 3]``.

    Returns:
        A dict of un-batched tensors: ``means [3]``, ``quats [4]``,
        ``scales [3]``, ``opacities []``, and appearance ``colors [..., 3]``.
    """
    k = means.shape[0]
    total_opacity = opacities.sum()
    if float(total_opacity.item()) > 1e-8:
        weights = opacities / total_opacity
    else:
        weights = opacities.new_full((k,), 1.0 / k)

    merged_mean = (weights[:, None] * means).sum(dim=0)  # [3]

    rotmats = normalized_quat_to_rotmat(quats)  # [K, 3, 3]
    covars = rotmats @ torch.diag_embed(scales**2) @ rotmats.transpose(-1, -2)
    delta = means - merged_mean  # [K, 3]
    spread = delta[:, :, None] * delta[:, None, :]  # [K, 3, 3]
    merged_covar = (weights[:, None, None] * (covars + spread)).sum(dim=0)  # [3, 3]
    # Guard against floating-point drift before the eigendecomposition.
    merged_covar = 0.5 * (merged_covar + merged_covar.transpose(-1, -2))

    eigvals, eigvecs = torch.linalg.eigh(merged_covar)
    eigvals = eigvals.clamp_min(1e-12)
    merged_scale = eigvals.sqrt()  # [3]

    rot = eigvecs
    # `eigh` only guarantees an orthonormal basis, not a proper rotation
    # (det=+1); flip the last column if it came out as a reflection.
    if torch.det(rot) < 0:
        rot = rot.clone()
        rot[:, -1] = -rot[:, -1]
    merged_quat = _rotmat_to_quat(rot)  # [4]

    if opacity_mode == "mass":
        source_mass = (
            opacities.clamp(0.0, 1.0) * scales.prod(dim=-1)
        ).sum()
        merged_opacity = source_mass / merged_scale.prod().clamp_min(1e-12)
    elif opacity_mode == "union":
        merged_opacity = 1.0 - torch.prod(1.0 - opacities.clamp(0.0, 1.0))
    else:
        raise ValueError(f"unsupported opacity_mode: {opacity_mode!r}")
    merged_opacity = merged_opacity.clamp(0.0, 1.0)

    appearance_weights = weights.reshape((k,) + (1,) * (colors.ndim - 1))
    merged_color = (appearance_weights * colors).sum(dim=0)

    return {
        "means": merged_mean,
        "quats": merged_quat,
        "scales": merged_scale,
        "opacities": merged_opacity,
        "colors": merged_color,
    }


def build_octree_lod(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    colors: Tensor,
    max_depth: int = 6,
    min_points_per_node: int = 64,
    opacity_mode: str = "union",
) -> OctreeNode:
    """Build a simple octree LOD hierarchy from a frozen, trained scene.

    Pure post-processing: no gradients, no retraining. Call this once on a
    trained checkpoint (offline or at load time), then call
    :func:`select_lod` every frame with the current camera position.

    Args:
        means: ``[N, 3]`` world-space positions.
        quats: ``[N, 4]`` unit quaternions, wxyz order.
        scales: ``[N, 3]`` positive scales (already ``exp()``-activated).
        opacities: ``[N]`` opacities in ``[0, 1]`` (already
            ``sigmoid()``-activated).
        colors: appearance tensors ``[N, ..., 3]``.
        max_depth: maximum octree depth.
        min_points_per_node: stop splitting a node once it has this many
            points or fewer.

    Returns:
        The root :class:`OctreeNode`.

    Raises:
        ValueError: if input shapes are inconsistent or the scene is empty.
    """
    n = means.shape[0]
    if means.shape != (n, 3):
        raise ValueError(f"means must be [N, 3], got {tuple(means.shape)}")
    if quats.shape != (n, 4):
        raise ValueError(f"quats must be [N, 4], got {tuple(quats.shape)}")
    if scales.shape != (n, 3):
        raise ValueError(f"scales must be [N, 3], got {tuple(scales.shape)}")
    if opacities.shape != (n,):
        raise ValueError(f"opacities must be [N], got {tuple(opacities.shape)}")
    if colors.ndim < 2 or colors.shape[0] != n or colors.shape[-1] != 3:
        raise ValueError(
            f"colors must be [N, ..., 3], got {tuple(colors.shape)}"
        )
    if n == 0:
        raise ValueError("cannot build an octree LOD from an empty scene")
    if max_depth < 0:
        raise ValueError(f"max_depth must be >= 0, got {max_depth}")
    if min_points_per_node < 1:
        raise ValueError(
            f"min_points_per_node must be >= 1, got {min_points_per_node}"
        )
    if opacity_mode not in ("union", "mass"):
        raise ValueError(f"unsupported opacity_mode: {opacity_mode!r}")

    bbox_min = means.min(dim=0).values
    bbox_max = means.max(dim=0).values
    center = (bbox_min + bbox_max) / 2
    half_size = float((bbox_max - bbox_min).max().item()) / 2 + 1e-6

    indices = torch.arange(n, device=means.device)
    return _build_node(
        means,
        quats,
        scales,
        opacities,
        colors,
        indices,
        center,
        half_size,
        depth=0,
        max_depth=max_depth,
        min_points_per_node=min_points_per_node,
        opacity_mode=opacity_mode,
    )


def _build_node(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    colors: Tensor,
    indices: Tensor,
    center: Tensor,
    half_size: float,
    depth: int,
    max_depth: int,
    min_points_per_node: int,
    opacity_mode: str,
) -> OctreeNode:
    node_means = means[indices]
    node_quats = quats[indices]
    node_scales = scales[indices]
    node_opacities = opacities[indices]
    node_colors = colors[indices]
    proxy = _merge_gaussians(
        node_means,
        node_quats,
        node_scales,
        node_opacities,
        node_colors,
        opacity_mode=opacity_mode,
    )
    num_points = int(indices.numel())

    can_split = depth < max_depth and num_points > min_points_per_node
    if can_split:
        # Octant id in [0, 8): bit 2 = x > cx, bit 1 = y > cy, bit 0 = z > cz.
        octant_bits = torch.tensor([4, 2, 1], device=means.device, dtype=torch.long)
        octant = ((node_means > center).long() * octant_bits).sum(dim=-1)
        unique_octants = torch.unique(octant)
        if unique_octants.numel() > 1:
            child_half = half_size / 2.0
            children = []
            for oct_id in unique_octants.tolist():
                child_mask = octant == oct_id
                offset = torch.tensor(
                    [
                        1.0 if oct_id & 4 else -1.0,
                        1.0 if oct_id & 2 else -1.0,
                        1.0 if oct_id & 1 else -1.0,
                    ],
                    device=means.device,
                    dtype=means.dtype,
                ) * child_half
                children.append(
                    _build_node(
                        means,
                        quats,
                        scales,
                        opacities,
                        colors,
                        indices[child_mask],
                        center + offset,
                        child_half,
                        depth + 1,
                        max_depth,
                        min_points_per_node,
                        opacity_mode,
                    )
                )
            return OctreeNode(
                center=center,
                half_size=half_size,
                depth=depth,
                num_points=num_points,
                proxy=proxy,
                leaf_data=None,
                children=children,
            )
        # Degenerate split: every point landed in the same octant (e.g. a
        # tight cluster). Recursing further would just repeat this node, so
        # fall through and treat it as a leaf instead.

    leaf_data = {
        "means": node_means,
        "quats": node_quats,
        "scales": node_scales,
        "opacities": node_opacities,
        "colors": node_colors,
    }
    return OctreeNode(
        center=center,
        half_size=half_size,
        depth=depth,
        num_points=num_points,
        proxy=proxy,
        leaf_data=leaf_data,
        children=None,
    )


def build_bvh_lod(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    colors: Tensor,
    max_depth: int = 32,
    min_points_per_node: int = 64,
    opacity_mode: str = "union",
) -> OctreeNode:
    """Build a median-split BVH LOD hierarchy from a frozen, trained scene.

    This is a drop-in alternative to :func:`build_octree_lod` that partitions
    the *set of Gaussians* (not space): every internal node splits its points
    into two halves at the data median of its longest bounding-box axis. The
    returned tree uses the same :class:`OctreeNode` structure and is consumed
    by the same :func:`select_lod`, so callers only need to swap the builder.

    Compared to the octree builder, this is far more robust on scenes with
    non-uniform density or far-away outliers (e.g. driving logs): because it
    splits by data rather than by fixed spatial planes, it can never dump a
    huge fraction of the scene into a single giant leaf, and every node's
    bounding sphere is tight (giving :func:`select_lod` an accurate on-screen
    size estimate). See ``bound_radius`` on :class:`OctreeNode`.

    Args:
        means: ``[N, 3]`` world-space positions.
        quats: ``[N, 4]`` unit quaternions, wxyz order.
        scales: ``[N, 3]`` positive scales (already ``exp()``-activated).
        opacities: ``[N]`` opacities in ``[0, 1]`` (already
            ``sigmoid()``-activated).
        colors: appearance tensors ``[N, ..., 3]``.
        max_depth: maximum tree depth (a safety cap; a balanced median split
            reaches its leaves in ``~log2(N / min_points_per_node)`` levels).
        min_points_per_node: stop splitting a node once it has this many
            points or fewer.

    Returns:
        The root :class:`OctreeNode`.

    Raises:
        ValueError: if input shapes are inconsistent or the scene is empty.
    """
    n = means.shape[0]
    if means.shape != (n, 3):
        raise ValueError(f"means must be [N, 3], got {tuple(means.shape)}")
    if quats.shape != (n, 4):
        raise ValueError(f"quats must be [N, 4], got {tuple(quats.shape)}")
    if scales.shape != (n, 3):
        raise ValueError(f"scales must be [N, 3], got {tuple(scales.shape)}")
    if opacities.shape != (n,):
        raise ValueError(f"opacities must be [N], got {tuple(opacities.shape)}")
    if colors.ndim < 2 or colors.shape[0] != n or colors.shape[-1] != 3:
        raise ValueError(
            f"colors must be [N, ..., 3], got {tuple(colors.shape)}"
        )
    if n == 0:
        raise ValueError("cannot build an octree LOD from an empty scene")
    if max_depth < 0:
        raise ValueError(f"max_depth must be >= 0, got {max_depth}")
    if min_points_per_node < 1:
        raise ValueError(
            f"min_points_per_node must be >= 1, got {min_points_per_node}"
        )
    if opacity_mode not in ("union", "mass"):
        raise ValueError(f"unsupported opacity_mode: {opacity_mode!r}")

    device = means.device

    # Per-point world-space 3x3 covariance, computed once (vectorized on GPU),
    # so leaf proxies can be moment-matched without recomputing it per node.
    rotmats = normalized_quat_to_rotmat(quats)  # [N, 3, 3]
    covars = rotmats @ torch.diag_embed(scales ** 2) @ rotmats.transpose(-1, -2)

    means_np = means.detach().cpu().numpy().astype(np.float64)
    covars_np = covars.detach().cpu().numpy().astype(np.float64)
    scale_volume_np = (
        scales.detach().cpu().numpy().astype(np.float64).prod(axis=1)
    )
    op_np = opacities.detach().clamp(0.0, 1.0).cpu().numpy().astype(np.float64)
    color_np = colors.detach().cpu().numpy().astype(np.float64)

    # Node record arrays (structure-of-arrays), filled in post-order so a
    # parent always appears after its two children. All the heavy work
    # (median partitioning + composable moment-matching) happens in numpy on
    # the CPU; the only GPU work is a single batched eigendecomposition and a
    # handful of batched gathers at the very end.
    n_mu: List[np.ndarray] = []      # [3] opacity-weighted mean (proxy mean)
    n_sigma: List[np.ndarray] = []   # [3, 3] moment-matched covariance
    n_color: List[np.ndarray] = []   # [3] opacity-weighted color
    n_alpha: List[float] = []        # combined opacity 1 - prod(1 - a_i)
    n_mass: List[float] = []         # opacity-weighted Gaussian volume
    n_w: List[float] = []            # total weight = sum of opacities
    n_amin: List[np.ndarray] = []    # [3] AABB min
    n_amax: List[np.ndarray] = []    # [3] AABB max
    n_depth: List[int] = []
    n_num: List[int] = []
    n_left: List[int] = []           # child node index, or -1 for a leaf
    n_right: List[int] = []
    leaf_node_ids: List[int] = []    # node indices that are leaves (build order)
    leaf_idx_arrays: List[np.ndarray] = []  # their point-index arrays

    import sys as _sys
    _prev_reclimit = _sys.getrecursionlimit()
    _sys.setrecursionlimit(max(_prev_reclimit, 4 * (max_depth + 10)))

    def _build(idx: np.ndarray, depth: int) -> int:
        x = means_np[idx]
        amin = x.min(axis=0)
        amax = x.max(axis=0)
        num = idx.shape[0]

        if depth < max_depth and num > min_points_per_node:
            extent = amax - amin
            axis = int(np.argmax(extent))
            coords = x[:, axis]
            med = np.median(coords)
            left_mask = coords <= med
            nl = int(left_mask.sum())
            if nl == 0 or nl == num:
                # Degenerate median (many identical coords along this axis):
                # fall back to an even split of the sorted order so both
                # halves are non-empty and the recursion still makes progress.
                order = np.argsort(coords, kind="stable")
                half = num // 2
                left_idx = idx[order[:half]]
                right_idx = idx[order[half:]]
            else:
                left_idx = idx[left_mask]
                right_idx = idx[~left_mask]

            li = _build(left_idx, depth + 1)
            ri = _build(right_idx, depth + 1)

            # Compose the two child proxies into this node's proxy. Because
            # moment matching only needs each side's total weight, mean and
            # covariance, combining the aggregates reproduces exactly the
            # moment match of all underlying points (parallel-axis theorem).
            wa, wb = n_w[li], n_w[ri]
            w = wa + wb
            mua, mub = n_mu[li], n_mu[ri]
            if w > 1e-12:
                mu = (wa * mua + wb * mub) / w
                da = mua - mu
                db = mub - mu
                sigma = (
                    wa * (n_sigma[li] + np.outer(da, da))
                    + wb * (n_sigma[ri] + np.outer(db, db))
                ) / w
                color = (wa * n_color[li] + wb * n_color[ri]) / w
            else:
                mu = 0.5 * (mua + mub)
                sigma = 0.5 * (n_sigma[li] + n_sigma[ri])
                color = 0.5 * (n_color[li] + n_color[ri])
            alpha = 1.0 - (1.0 - n_alpha[li]) * (1.0 - n_alpha[ri])
            mass = n_mass[li] + n_mass[ri]
            amn = np.minimum(n_amin[li], n_amin[ri])
            amx = np.maximum(n_amax[li], n_amax[ri])

            ni = len(n_mu)
            n_mu.append(mu); n_sigma.append(sigma); n_color.append(color)
            n_alpha.append(alpha); n_w.append(w)
            n_mass.append(mass)
            n_amin.append(amn); n_amax.append(amx)
            n_depth.append(depth); n_num.append(num)
            n_left.append(li); n_right.append(ri)
            return ni

        # Leaf: moment-match this node's original points directly.
        a = op_np[idx]
        wsum = float(a.sum())
        wnorm = a / wsum if wsum > 1e-12 else np.full(num, 1.0 / num)
        mu = (wnorm[:, None] * x).sum(axis=0)
        d = x - mu
        sigma = np.einsum("k,kij->ij", wnorm, covars_np[idx] + d[:, :, None] * d[:, None, :])
        appearance_weights = wnorm.reshape((num,) + (1,) * (color_np.ndim - 1))
        color = (appearance_weights * color_np[idx]).sum(axis=0)
        alpha = 1.0 - float(np.prod(1.0 - a))
        mass = float(np.sum(a * scale_volume_np[idx]))

        ni = len(n_mu)
        n_mu.append(mu); n_sigma.append(sigma); n_color.append(color)
        n_alpha.append(alpha); n_w.append(wsum)
        n_mass.append(mass)
        n_amin.append(amin); n_amax.append(amax)
        n_depth.append(depth); n_num.append(num)
        n_left.append(-1); n_right.append(-1)
        leaf_node_ids.append(ni)
        leaf_idx_arrays.append(idx)
        return ni

    root_id = _build(np.arange(n), 0)
    _sys.setrecursionlimit(_prev_reclimit)

    m = len(n_mu)

    # Batched conversion of every node's covariance into (scales, quats).
    # The eigendecomposition is done in numpy on the CPU: the node
    # covariances are tiny (3x3) but there are hundreds of thousands of them,
    # and cuSOLVER's batched eigh workspace on the GPU can balloon to tens of
    # GiB. numpy's batched eigh handles this in a few hundred ms.
    sigma_np = np.stack(n_sigma)
    sigma_np = 0.5 * (sigma_np + np.transpose(sigma_np, (0, 2, 1)))
    eigvals_np, eigvecs_np = np.linalg.eigh(sigma_np)  # ascending eigvals
    proxy_scales_all = torch.tensor(
        np.sqrt(np.clip(eigvals_np, 1e-12, None)), dtype=torch.float32, device=device
    )  # [M, 3]
    # Ensure a proper rotation (det > 0) by flipping the last eigenvector
    # where the eigenbasis is a reflection.
    dets = np.linalg.det(eigvecs_np)
    eigvecs_np[dets < 0, :, -1] *= -1.0
    rot = torch.tensor(eigvecs_np, dtype=torch.float32, device=device)  # [M, 3, 3]
    proxy_quats_all = _rotmat_to_quat(rot)  # [M, 4]
    proxy_means_all = torch.tensor(np.stack(n_mu), dtype=torch.float32, device=device)
    proxy_colors_all = torch.tensor(np.stack(n_color), dtype=torch.float32, device=device)
    if opacity_mode == "mass":
        proxy_volume = proxy_scales_all.prod(dim=-1).clamp_min(1e-12)
        proxy_opac_all = torch.tensor(
            np.asarray(n_mass), dtype=torch.float32, device=device
        ) / proxy_volume
        proxy_opac_all = proxy_opac_all.clamp(0.0, 1.0)
    else:
        proxy_opac_all = torch.tensor(
            np.asarray(n_alpha), dtype=torch.float32, device=device
        )

    amin_np = np.stack(n_amin)
    amax_np = np.stack(n_amax)
    centers_all = torch.tensor((amin_np + amax_np) / 2, dtype=torch.float32, device=device)
    radii_np = 0.5 * np.linalg.norm(amax_np - amin_np, axis=1)

    # Gather all leaf point data in a few batched ops, then slice per leaf.
    all_leaf_idx = np.concatenate(leaf_idx_arrays) if leaf_idx_arrays else np.zeros((0,), dtype=np.int64)
    all_leaf_idx_t = torch.as_tensor(all_leaf_idx, dtype=torch.long, device=device)
    g_means = means[all_leaf_idx_t]
    g_quats = quats[all_leaf_idx_t]
    g_scales = scales[all_leaf_idx_t]
    g_opac = opacities[all_leaf_idx_t]
    g_colors = colors[all_leaf_idx_t]
    leaf_slice_of = {}
    off = 0
    for lid, arr in zip(leaf_node_ids, leaf_idx_arrays):
        k = arr.shape[0]
        leaf_slice_of[lid] = (off, off + k)
        off += k

    sqrt3 = 1.7320508075688772
    nodes: List[Optional[OctreeNode]] = [None] * m
    for i in range(m):
        proxy = {
            "means": proxy_means_all[i],
            "quats": proxy_quats_all[i],
            "scales": proxy_scales_all[i],
            "opacities": proxy_opac_all[i],
            "colors": proxy_colors_all[i],
        }
        leaf_data = None
        if n_left[i] == -1:
            s, e = leaf_slice_of[i]
            leaf_data = {
                "means": g_means[s:e],
                "quats": g_quats[s:e],
                "scales": g_scales[s:e],
                "opacities": g_opac[s:e],
                "colors": g_colors[s:e],
            }
        bound_radius = float(radii_np[i])
        nodes[i] = OctreeNode(
            center=centers_all[i],
            half_size=bound_radius / sqrt3 + 1e-6,
            depth=n_depth[i],
            num_points=n_num[i],
            proxy=proxy,
            leaf_data=leaf_data,
            children=None,
            bound_radius=bound_radius,
        )
    for i in range(m):
        if n_left[i] != -1:
            nodes[i].children = [nodes[n_left[i]], nodes[n_right[i]]]

    return nodes[root_id]


@dataclass
class _Frustum:
    """Camera-space view frustum, expressed as 5 half-spaces (left, right,
    top, bottom, near; the far plane is intentionally omitted -- LOD scenes
    are usually unbounded in depth and a missing far plane only costs a
    few extra, harmless node visits).

    Each plane is stored as ``(nx, ny, nz, d)`` such that a camera-space
    point ``p`` is inside the plane iff ``nx*px + ny*py + nz*pz + d >= 0``
    (planes are pre-normalized so this value is also a true signed
    distance, in world units).
    """

    planes: List[tuple]
    w2c: List[List[float]]  # 4x4 world-to-camera, row-major, plain Python floats

    def sphere_visible(self, center_world: tuple, radius: float) -> bool:
        w2c = self.w2c
        cx, cy, cz = center_world
        # world -> camera space (row-major 4x4, affine top-left 3x3 + translation)
        px = w2c[0][0] * cx + w2c[0][1] * cy + w2c[0][2] * cz + w2c[0][3]
        py = w2c[1][0] * cx + w2c[1][1] * cy + w2c[1][2] * cz + w2c[1][3]
        pz = w2c[2][0] * cx + w2c[2][1] * cy + w2c[2][2] * cz + w2c[2][3]
        for nx, ny, nz, d in self.planes:
            if nx * px + ny * py + nz * pz + d < -radius:
                return False
        return True


def _build_frustum(
    w2c: Tensor, fx: float, fy: float, cx: float, cy: float, width: float, height: float, near: float
) -> _Frustum:
    """Build a 5-plane camera-space frustum (see :class:`_Frustum`) from a
    world-to-camera matrix and pinhole intrinsics. Standard OpenCV/COLMAP
    convention: camera space is x-right, y-down, z-forward (points in view
    have ``z > 0``).
    """
    w2c_list = [[float(v) for v in row] for row in w2c.detach().cpu().tolist()]

    def _norm(nx, ny, nz):
        n = (nx * nx + ny * ny + nz * nz) ** 0.5
        return (nx / n, ny / n, nz / n)

    # Side-plane normals derived from the pinhole projection boundaries
    # (see module docstring / PR description for the derivation):
    #   left:   fx*x + cx*z   >= 0
    #   right: -fx*x + (w-cx)*z >= 0
    #   top:    fy*y + cy*z   >= 0
    #   bottom:-fy*y + (h-cy)*z >= 0
    left = _norm(fx, 0.0, cx)
    right = _norm(-fx, 0.0, width - cx)
    top = _norm(0.0, fy, cy)
    bottom = _norm(0.0, -fy, height - cy)
    planes = [
        (left[0], left[1], left[2], 0.0),
        (right[0], right[1], right[2], 0.0),
        (top[0], top[1], top[2], 0.0),
        (bottom[0], bottom[1], bottom[2], 0.0),
        (0.0, 0.0, 1.0, -near),  # near plane: z >= near
    ]
    return _Frustum(planes=planes, w2c=w2c_list)


# Flat attributes that are safe to keep in fp16. ``means`` are deliberately
# excluded: world coordinates can be hundreds of metres from the origin, where
# fp16's ~1-part-in-1024 resolution (>0.1 m error past 128 m) causes visible
# spatial jitter. Everything else here is bounded/normalized (quats in [-1, 1],
# opacities/colors in [0, 1], scales small and positive), so fp16 is visually
# lossless while halving both the VRAM footprint and, more importantly, the
# per-frame gather bandwidth in :func:`select_lod`.
_LOD_FP16_ATTRS = (
    "_flat_proxy_quats",
    "_flat_proxy_scales",
    "_flat_proxy_opacities",
    "_flat_proxy_colors",
    "_flat_leaf_quats",
    "_flat_leaf_scales",
    "_flat_leaf_opacities",
    "_flat_leaf_colors",
)


def _cast_flat_fp16(root) -> None:
    """Downcast the fp16-safe flat attributes (see :data:`_LOD_FP16_ATTRS`) in
    place. The gathered rows are widened back to fp32 in :func:`select_lod`
    before rasterization, so this is purely an internal storage/bandwidth
    optimization and does not change the rasterizer's fp32 inputs."""
    for attr in _LOD_FP16_ATTRS:
        t = getattr(root, attr, None)
        if t is not None and t.dtype != torch.float16:
            setattr(root, attr, t.to(torch.float16))


def _ensure_flat_cache(root: OctreeNode) -> None:
    """Flatten every node's ``proxy`` Gaussian (and every leaf's exact
    per-point data) into a handful of large contiguous tensors, once, the
    first time :func:`select_lod` is called on this tree.

    :func:`select_lod` visits thousands of nodes per frame; building the
    output purely by Python-appending each node's own small tensor and
    ``torch.cat``-ing the whole list back together every frame is
    expensive (measured ~9ms/frame for ~8k nodes) because ``torch.cat``
    over thousands of tiny fragments does not amortize well. Instead, once
    this flat cache exists, ``select_lod`` only needs to collect *integer
    indices* per frame and do a small, constant number of single gather +
    concat calls at the end.
    """
    if getattr(root, "_flat_proxy_means", None) is not None:
        return  # already flattened (cached across all future frames)

    all_nodes: List[OctreeNode] = []
    parents: List[int] = []
    stack = [(root, -1)]
    while stack:
        node, pidx = stack.pop()
        idx = len(all_nodes)
        all_nodes.append(node)
        parents.append(pidx)
        if node.children is not None:
            for ch in node.children:
                stack.append((ch, idx))

    proxy_means = []
    proxy_quats = []
    proxy_scales = []
    proxy_opacities = []
    proxy_colors = []
    leaf_means = []
    leaf_quats = []
    leaf_scales = []
    leaf_opacities = []
    leaf_colors = []

    leaf_cursor = 0
    for i, node in enumerate(all_nodes):
        node._proxy_idx = i
        proxy_means.append(node.proxy["means"])
        proxy_quats.append(node.proxy["quats"])
        proxy_scales.append(node.proxy["scales"])
        proxy_opacities.append(node.proxy["opacities"])
        proxy_colors.append(node.proxy["colors"])
        if node.is_leaf:
            data = node.leaf_data
            assert data is not None
            k = data["means"].shape[0]
            node._leaf_start = leaf_cursor
            node._leaf_end = leaf_cursor + k
            leaf_cursor += k
            leaf_means.append(data["means"])
            leaf_quats.append(data["quats"])
            leaf_scales.append(data["scales"])
            leaf_opacities.append(data["opacities"])
            leaf_colors.append(data["colors"])

    device = root.proxy["means"].device
    root._flat_proxy_means = torch.stack(proxy_means, dim=0)
    root._flat_proxy_quats = torch.stack(proxy_quats, dim=0)
    root._flat_proxy_scales = torch.stack(proxy_scales, dim=0)
    root._flat_proxy_opacities = torch.stack(proxy_opacities, dim=0)
    root._flat_proxy_colors = torch.stack(proxy_colors, dim=0)

    root._flat_leaf_means = (
        torch.cat(leaf_means, dim=0) if leaf_means else root._flat_proxy_means.new_zeros((0, 3))
    )
    root._flat_leaf_quats = (
        torch.cat(leaf_quats, dim=0) if leaf_quats else root._flat_proxy_quats.new_zeros((0, 4))
    )
    root._flat_leaf_scales = (
        torch.cat(leaf_scales, dim=0) if leaf_scales else root._flat_proxy_scales.new_zeros((0, 3))
    )
    root._flat_leaf_opacities = (
        torch.cat(leaf_opacities, dim=0) if leaf_opacities else root._flat_proxy_opacities.new_zeros((0,))
    )
    root._flat_leaf_colors = (
        torch.cat(leaf_colors, dim=0)
        if leaf_colors
        else root._flat_proxy_colors.new_zeros(
            (0,) + tuple(root._flat_proxy_colors.shape[1:])
        )
    )
    root._leaf_index_device = device

    # Per-node table (indexed by node position in ``all_nodes``, which equals
    # each node's ``_proxy_idx``) used by the fully vectorized selection in
    # :func:`select_lod`. Building these once turns the per-frame tree walk
    # into a handful of batched GPU tensor ops.
    sqrt3 = 1.7320508075688772
    m = len(all_nodes)
    centers = torch.stack([n.center for n in all_nodes], dim=0).to(device)
    size_list = [
        (n.bound_radius if n.bound_radius is not None else n.half_size) for n in all_nodes
    ]
    radius_list = [
        (n.bound_radius if n.bound_radius is not None else n.half_size * sqrt3)
        for n in all_nodes
    ]
    leaf_start_list = [getattr(n, "_leaf_start", 0) for n in all_nodes]
    leaf_len_list = [
        (getattr(n, "_leaf_end", 0) - getattr(n, "_leaf_start", 0)) if n.is_leaf else 0
        for n in all_nodes
    ]
    root._node_center = centers
    root._node_size = torch.tensor(size_list, dtype=torch.float32, device=device)
    root._node_radius = torch.tensor(radius_list, dtype=torch.float32, device=device)
    root._node_is_leaf = torch.tensor(
        [n.is_leaf for n in all_nodes], dtype=torch.bool, device=device
    )
    root._node_parent = torch.tensor(parents, dtype=torch.long, device=device)
    root._node_leaf_start = torch.tensor(leaf_start_list, dtype=torch.long, device=device)
    root._node_leaf_len = torch.tensor(leaf_len_list, dtype=torch.long, device=device)
    root._num_nodes = m
    _cast_flat_fp16(root)


def select_lod(
    root: OctreeNode,
    cam_pos: Tensor,
    focal_px: float,
    error_threshold_px: float = 2.0,
    w2c: Optional[Tensor] = None,
    K: Optional[Tensor] = None,
    width: Optional[float] = None,
    height: Optional[float] = None,
    near: float = 0.01,
) -> Dict[str, Tensor]:
    """Select which octree nodes to render, given a camera position.

    Walks the tree from ``root``. A node's angular size on screen is
    approximated as ``2 * half_size * focal_px / distance_to_camera``. Nodes
    at or below ``error_threshold_px`` are rendered using their single
    merged ``proxy`` Gaussian; otherwise the walk recurses into the node's
    children. Leaf nodes always contribute their exact ``leaf_data`` since
    there is nothing coarser to fall back to.

    If ``w2c``, ``K``, ``width`` and ``height`` are all provided, nodes
    whose bounding sphere lies entirely outside the camera's view frustum
    are skipped (neither their proxy nor any of their descendants are
    visited) -- this is a pure performance optimization for scenes much
    larger than the camera's field of view; the far plane is intentionally
    not culled against since most LOD scenes are open/unbounded in depth.

    Args:
        root: root of the octree built by :func:`build_octree_lod`.
        cam_pos: ``[3]`` camera position, in the same world frame the octree
            was built in.
        focal_px: representative focal length in pixels (e.g. ``K[0, 0]``)
            used for the screen-space size estimate.
        error_threshold_px: maximum allowed projected node size, in pixels,
            before recursing into finer children.
        w2c: optional ``[4, 4]`` world-to-camera matrix (OpenCV/COLMAP
            convention: x-right, y-down, z-forward), for frustum culling.
        K: optional ``[3, 3]`` pinhole intrinsics matrix, for frustum
            culling.
        width: optional image width in pixels, for frustum culling.
        height: optional image height in pixels, for frustum culling.
        near: near-plane distance (world units) used by frustum culling.

    Returns:
        A dict with ``means [M, 3]``, ``quats [M, 4]``, ``scales [M, 3]``,
        ``opacities [M]``, ``colors [M, 3]``, ready to pass to
        :func:`gsplat.rendering.rasterization`.
    """
    _ensure_flat_cache(root)

    # Fully vectorized LOD "cut" selection. Instead of a per-frame Python
    # walk over every node (which dominates cost on large trees with
    # hundreds of thousands of nodes), we evaluate the projected on-screen
    # size of *all* nodes at once on the GPU and pick, in a couple of batched
    # ops, exactly the set of nodes the recursive walk would have emitted:
    #
    #   * proxy node  : node small enough on screen AND whose parent is *not*
    #                   small (so the parent was expanded rather than
    #                   collapsed to its own proxy);
    #   * leaf points : a leaf that is still too big on screen (nothing
    #                   coarser than its exact points is available).
    #
    # This reproduces the recursive semantics because a node's bounding
    # sphere is (essentially) contained in its parent's, so "small on screen"
    # is monotone down the tree and "reached" reduces to "parent not small".
    device = root._leaf_index_device
    cam = cam_pos.to(device=device, dtype=torch.float32).reshape(3)

    centers = root._node_center                       # [M, 3]
    dist = (centers - cam).norm(dim=1).clamp_min(1e-6)  # [M]
    projected = 2.0 * root._node_size * float(focal_px) / dist
    small = projected <= float(error_threshold_px)    # [M] bool

    visible = torch.ones_like(small)
    if w2c is not None and K is not None and width is not None and height is not None:
        fx = float(K[0, 0]); fy = float(K[1, 1])
        cx = float(K[0, 2]); cy = float(K[1, 2])
        w = float(width); h = float(height)
        w2c_d = w2c.to(device=device, dtype=torch.float32)
        R = w2c_d[:3, :3]
        t = w2c_d[:3, 3]
        pc = centers @ R.transpose(0, 1) + t          # [M, 3] camera space
        px, py, pz = pc[:, 0], pc[:, 1], pc[:, 2]
        radius = root._node_radius                    # [M]

        def _n(a, b, c):
            n = (a * a + b * b + c * c) ** 0.5
            return a / n, b / n, c / n

        planes = [
            (*_n(fx, 0.0, cx), 0.0),
            (*_n(-fx, 0.0, w - cx), 0.0),
            (*_n(0.0, fy, cy), 0.0),
            (*_n(0.0, -fy, h - cy), 0.0),
            (0.0, 0.0, 1.0, -float(near)),
        ]
        for nx, ny, nz, d in planes:
            visible &= (nx * px + ny * py + nz * pz + d) >= -radius

    parent = root._node_parent                        # [M] long, root = -1
    small_parent = small[parent.clamp_min(0)]
    small_parent = small_parent & (parent >= 0)       # root has no parent
    is_leaf = root._node_is_leaf

    use_proxy = visible & small & (~small_parent)
    use_leaf = visible & is_leaf & (~small)

    # ---- gather selected leaf points + proxies straight into one buffer ----
    # Both the ragged leaf gather and the proxy gather write *directly* into a
    # single preallocated output tensor via ``index_select(out=...)``. This
    # fuses the per-attribute gather and the final ``torch.cat([leaf, proxy])``
    # into one pass: previously each of the millions of selected leaf rows was
    # materialized once by fancy-indexing and then copied a second time by the
    # concat. One preallocated write halves that traffic and is the dominant
    # cost of this function on large scenes.
    leaf_sel = use_leaf.nonzero(as_tuple=False).squeeze(1)
    proxy_idx = use_proxy.nonzero(as_tuple=False).squeeze(1)
    P = int(proxy_idx.numel())

    leaf_idx = None
    total = 0
    if leaf_sel.numel() > 0:
        ls = root._node_leaf_start[leaf_sel]
        ll = root._node_leaf_len[leaf_sel]
        total = int(ll.sum().item())  # single required device->host sync
        if total > 0:
            group_first = torch.repeat_interleave(torch.cumsum(ll, 0) - ll, ll)
            base = torch.repeat_interleave(ls, ll)
            ar = torch.arange(total, device=device)
            leaf_idx = base + (ar - group_first)

    def _combine(flat_leaf, flat_proxy, tail_shape):
        out = flat_leaf.new_empty((total + P,) + tail_shape)
        if total > 0:
            torch.index_select(flat_leaf, 0, leaf_idx, out=out[:total])
        if P > 0:
            torch.index_select(flat_proxy, 0, proxy_idx, out=out[total:])
        # fp16-stored attributes (everything but means) are widened back to
        # fp32 here so the rasterizer still receives float32 inputs.
        return out if out.dtype == torch.float32 else out.float()

    return {
        "means": _combine(root._flat_leaf_means, root._flat_proxy_means, (3,)),
        "quats": _combine(root._flat_leaf_quats, root._flat_proxy_quats, (4,)),
        "scales": _combine(root._flat_leaf_scales, root._flat_proxy_scales, (3,)),
        "opacities": _combine(root._flat_leaf_opacities, root._flat_proxy_opacities, ()),
        "colors": _combine(
            root._flat_leaf_colors,
            root._flat_proxy_colors,
            tuple(root._flat_proxy_colors.shape[1:]),
        ),
    }


# ---------------------------------------------------------------------------
# Persistent (on-disk) LOD cache
#
# Building the hierarchy is the expensive part (tens of seconds for millions
# of Gaussians). Everything :func:`select_lod` needs at run time, however, is
# a fixed set of flat tensors (the node table plus the per-node proxy and
# per-leaf point arrays). We can therefore serialize just those tensors and,
# on the next launch, reload them into a lightweight stand-in object instead
# of rebuilding the tree. The tree topology / error threshold do not affect
# these tensors, so a saved cache is valid for any ``error_threshold_px``.
# ---------------------------------------------------------------------------

_LOD_CACHE_VERSION = 2

# The exact attributes consumed by :func:`select_lod` (via
# :func:`_ensure_flat_cache`). Persisting these is sufficient to reload a
# fully functional LOD root without the original tree.
_LOD_FLAT_ATTRS = (
    "_node_center",
    "_node_size",
    "_node_radius",
    "_node_is_leaf",
    "_node_parent",
    "_node_leaf_start",
    "_node_leaf_len",
    "_flat_proxy_means",
    "_flat_proxy_quats",
    "_flat_proxy_scales",
    "_flat_proxy_opacities",
    "_flat_proxy_colors",
    "_flat_leaf_means",
    "_flat_leaf_quats",
    "_flat_leaf_scales",
    "_flat_leaf_opacities",
    "_flat_leaf_colors",
)


class _CachedLODRoot:
    """Lightweight stand-in for a built LOD hierarchy.

    Only carries the flat tensors that :func:`select_lod` reads, so a persisted
    cache can be reloaded without reconstructing the ``OctreeNode`` tree. It
    exposes ``_flat_proxy_means`` etc. so :func:`_ensure_flat_cache` treats it
    as already-built and returns immediately.
    """

    pass


def _windows_extended_path(path: str) -> str:
    """Use the Win32 extended-length form for deeply nested package paths."""
    absolute = os.path.abspath(path)
    if os.name != "nt" or absolute.startswith("\\\\?\\"):
        return absolute
    if absolute.startswith("\\\\"):
        return "\\\\?\\UNC\\" + absolute[2:]
    return "\\\\?\\" + absolute


def save_lod_cache(root: OctreeNode, path: str) -> None:
    """Serialize the flat run-time tensors of a built LOD ``root`` to ``path``.

    The (small) proxy/node tensors and the (larger) per-leaf point tensors are
    moved to CPU and written atomically. Call after building with
    :func:`build_bvh_lod` / :func:`build_octree_lod`.
    """
    _ensure_flat_cache(root)
    blob: Dict[str, object] = {
        "version": _LOD_CACHE_VERSION,
        "num_nodes": int(getattr(root, "_num_nodes", root._node_center.shape[0])),
    }
    for attr in _LOD_FLAT_ATTRS:
        blob[attr] = getattr(root, attr).detach().to("cpu").contiguous()

    path = _windows_extended_path(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    torch.save(blob, tmp)
    os.replace(tmp, path)  # atomic on the same filesystem


def load_lod_cache(path: str, device="cuda") -> "_CachedLODRoot":
    """Reload a LOD cache written by :func:`save_lod_cache` onto ``device``.

    Returns a stand-in root accepted by :func:`select_lod`. Raises
    ``ValueError`` on a version mismatch and ``KeyError`` if the file is
    missing an expected tensor (either should trigger a rebuild by the caller).
    """
    blob = torch.load(_windows_extended_path(path), map_location="cpu")
    version = blob.get("version") if isinstance(blob, dict) else None
    if version != _LOD_CACHE_VERSION:
        raise ValueError(
            f"LOD cache version mismatch: got {version!r}, "
            f"expected {_LOD_CACHE_VERSION}"
        )
    dev = torch.device(device)
    root = _CachedLODRoot()
    for attr in _LOD_FLAT_ATTRS:
        root.__dict__[attr] = blob[attr].to(dev)
    root._num_nodes = int(blob.get("num_nodes", root._node_center.shape[0]))
    root._leaf_index_device = dev
    # Older caches were saved entirely in fp32; downcast the fp16-safe
    # attributes so a reloaded tree matches a freshly built one.
    _cast_flat_fp16(root)
    return root
