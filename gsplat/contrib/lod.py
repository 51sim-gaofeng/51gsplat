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
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, List, Optional

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
    "CUDA_FLAT_BVH_VERSION",
    "OctreeNode",
    "build_octree_lod",
    "build_bvh_lod",
    "build_bvh_lod_batched",
    "select_lod",
    "save_lod_cache",
    "load_lod_cache",
]

CUDA_FLAT_BVH_VERSION = 2
LOD_BUILD_RESOURCE_VERSION = 3


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

    if means.is_cuda:
        return _build_bvh_lod_cuda_flat(
            means,
            quats,
            scales,
            opacities,
            colors,
            max_depth,
            min_points_per_node,
            opacity_mode,
        )

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


def _segment_reduce(data: Tensor, reduce: str, lengths: Tensor) -> Tensor:
    segment_ids = torch.repeat_interleave(
        torch.arange(lengths.numel(), device=data.device),
        lengths,
    )
    index = segment_ids.reshape(
        (-1,) + (1,) * (data.ndim - 1)
    ).expand_as(data)
    output_shape = (lengths.numel(),) + tuple(data.shape[1:])
    if reduce in ("sum", "mean"):
        output = data.new_zeros(output_shape)
        output.scatter_add_(0, index, data)
        if reduce == "mean":
            divisor = lengths.clamp_min(1).reshape(
                (-1,) + (1,) * (data.ndim - 1)
            )
            output = output / divisor
        return output
    if reduce == "min":
        output = data.new_full(output_shape, float("inf"))
        return output.scatter_reduce_(0, index, data, reduce="amin")
    if reduce == "max":
        output = data.new_full(output_shape, float("-inf"))
        return output.scatter_reduce_(0, index, data, reduce="amax")
    raise ValueError(f"unsupported segment reduction: {reduce}")


def _eigh_symmetric_3x3(covariances: Tensor) -> tuple[Tensor, Tensor]:
    """Use the workspace-light analytic CUDA solver when it is available."""
    try:
        from pytorch3d.common.workaround.symeig3x3 import symeig3x3
    except ImportError:
        covariance_np = covariances.detach().cpu().numpy()
        eigvals_np, eigvecs_np = np.linalg.eigh(covariance_np)
        return (
            torch.from_numpy(eigvals_np).to(covariances.device),
            torch.from_numpy(eigvecs_np).to(covariances.device),
        )
    return symeig3x3(covariances)


def _build_bvh_lod_flat(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    colors: Tensor,
    max_depth: int,
    min_points_per_node: int,
    opacity_mode: str,
    retain_root_stats: bool = False,
):
    """Build the median-split hierarchy on the input tensors' device."""
    device = means.device
    n = means.shape[0]
    point_order = torch.arange(n, dtype=torch.long, device=device)
    active_lengths = torch.tensor([n], dtype=torch.long, device=device)
    active_parents = torch.full((1,), -1, dtype=torch.long, device=device)

    parent_levels = []
    depth_levels = []
    count_levels = []
    amin_levels = []
    amax_levels = []
    left_levels = []
    right_levels = []
    leaf_node_parts = []
    leaf_point_parts = []
    leaf_length_parts = []
    node_cursor = 0

    for depth in range(max_depth + 1):
        segment_count = active_lengths.numel()
        node_ids = torch.arange(
            node_cursor,
            node_cursor + segment_count,
            dtype=torch.long,
            device=device,
        )
        node_cursor += segment_count
        points = means[point_order]
        amin = _segment_reduce(points, "min", active_lengths)
        amax = _segment_reduce(points, "max", active_lengths)
        is_leaf = (active_lengths <= min_points_per_node) | (depth == max_depth)

        parent_levels.append(active_parents)
        depth_levels.append(torch.full_like(active_lengths, depth))
        count_levels.append(active_lengths)
        amin_levels.append(amin)
        amax_levels.append(amax)

        point_segments = torch.repeat_interleave(
            torch.arange(segment_count, device=device),
            active_lengths,
        )
        leaf_points_mask = is_leaf[point_segments]
        if torch.any(is_leaf):
            leaf_node_parts.append(node_ids[is_leaf])
            leaf_point_parts.append(point_order[leaf_points_mask])
            leaf_length_parts.append(active_lengths[is_leaf])

        split_mask = ~is_leaf
        split_count = int(split_mask.sum().item())
        left = torch.full_like(active_lengths, -1)
        right = torch.full_like(active_lengths, -1)
        if split_count == 0:
            left_levels.append(left)
            right_levels.append(right)
            break

        split_nodes = node_ids[split_mask]
        child_base = node_cursor
        child_offsets = torch.arange(
            split_count, dtype=torch.long, device=device
        ) * 2
        left[split_mask] = child_base + child_offsets
        right[split_mask] = child_base + child_offsets + 1
        left_levels.append(left)
        right_levels.append(right)

        axes = torch.argmax(amax - amin, dim=1)
        point_axes = axes[point_segments]
        split_keys = points.gather(1, point_axes[:, None]).squeeze(1)
        key_min = amin.gather(1, axes[:, None]).squeeze(1)
        key_extent = (
            amax.gather(1, axes[:, None]).squeeze(1) - key_min
        ).clamp_min(torch.finfo(torch.float32).eps)
        normalized_keys = (
            split_keys - key_min[point_segments]
        ) / key_extent[point_segments]
        sort_keys = point_segments.to(torch.float64) + (
            normalized_keys.to(torch.float64) * (1.0 - 2.0**-24)
        )
        point_order = point_order[torch.argsort(sort_keys, stable=True)]

        split_lengths = active_lengths[split_mask]
        left_lengths = torch.div(split_lengths, 2, rounding_mode="floor")
        right_lengths = split_lengths - left_lengths
        active_lengths = torch.stack(
            [left_lengths, right_lengths], dim=1
        ).reshape(-1)
        active_parents = split_nodes.repeat_interleave(2)

        sorted_segments = torch.repeat_interleave(
            torch.arange(segment_count, device=device),
            count_levels[-1],
        )
        point_order = point_order[split_mask[sorted_segments]]
    else:
        raise RuntimeError("BVH construction exceeded max_depth")

    parents = torch.cat(parent_levels)
    depths = torch.cat(depth_levels)
    amins = torch.cat(amin_levels)
    amaxs = torch.cat(amax_levels)
    left_children = torch.cat(left_levels)
    right_children = torch.cat(right_levels)
    num_nodes = parents.numel()

    subtree_sizes = torch.ones(num_nodes, dtype=torch.long, device=device)
    for depth in range(int(depths[-1].item()), -1, -1):
        nodes = torch.nonzero(depths == depth, as_tuple=False).squeeze(1)
        internal = left_children[nodes] >= 0
        nodes = nodes[internal]
        if nodes.numel() > 0:
            subtree_sizes[nodes] += (
                subtree_sizes[left_children[nodes]]
                + subtree_sizes[right_children[nodes]]
            )

    preorder = torch.zeros(num_nodes, dtype=torch.long, device=device)
    for depth in range(int(depths[-1].item()) + 1):
        nodes = torch.nonzero(
            (depths == depth) & (left_children >= 0),
            as_tuple=False,
        ).squeeze(1)
        if nodes.numel() > 0:
            left = left_children[nodes]
            right = right_children[nodes]
            preorder[left] = preorder[nodes] + 1
            preorder[right] = preorder[left] + subtree_sizes[left]
    old_by_preorder = torch.argsort(preorder)
    preorder_of_old = torch.empty_like(old_by_preorder)
    preorder_of_old[old_by_preorder] = torch.arange(
        num_nodes, dtype=torch.long, device=device
    )

    leaf_nodes = torch.cat(leaf_node_parts)
    leaf_points = torch.cat(leaf_point_parts)
    leaf_lengths = torch.cat(leaf_length_parts)
    leaf_preorder = preorder_of_old[leaf_nodes]
    leaf_order = torch.argsort(leaf_preorder)
    leaf_nodes = leaf_nodes[leaf_order]
    leaf_lengths = leaf_lengths[leaf_order]
    gathered_leaf_points = torch.cat(leaf_point_parts)
    gathered_leaf_nodes = torch.repeat_interleave(
        torch.cat(leaf_node_parts),
        torch.cat(leaf_length_parts),
    )
    gathered_order = torch.argsort(preorder_of_old[gathered_leaf_nodes])
    exact_indices = gathered_leaf_points[gathered_order]

    rotmats = normalized_quat_to_rotmat(quats)
    covars = (
        rotmats
        @ torch.diag_embed(scales.square())
        @ rotmats.transpose(-1, -2)
    )
    exact_means = means[exact_indices]
    exact_covars = covars[exact_indices]
    exact_opacities = opacities[exact_indices].clamp(0.0, 1.0)
    exact_colors = colors[exact_indices]
    exact_volumes = scales[exact_indices].prod(dim=1)

    weights = exact_opacities
    weight_sums = _segment_reduce(weights, "sum", leaf_lengths)
    safe_sums = weight_sums.clamp_min(1e-12)
    weighted_means = _segment_reduce(
        exact_means * weights[:, None], "sum", leaf_lengths
    )
    leaf_means = weighted_means / safe_sums[:, None]
    zero_weight_leaves = weight_sums <= 1e-12
    if torch.any(zero_weight_leaves):
        uniform_means = _segment_reduce(exact_means, "mean", leaf_lengths)
        leaf_means[zero_weight_leaves] = uniform_means[zero_weight_leaves]
    repeated_leaf_means = torch.repeat_interleave(leaf_means, leaf_lengths, 0)
    delta = exact_means - repeated_leaf_means
    second_moments = exact_covars + delta[:, :, None] * delta[:, None, :]
    leaf_covars = _segment_reduce(
        second_moments * weights[:, None, None], "sum", leaf_lengths
    ) / safe_sums[:, None, None]
    if torch.any(zero_weight_leaves):
        uniform_covars = _segment_reduce(second_moments, "mean", leaf_lengths)
        leaf_covars[zero_weight_leaves] = uniform_covars[zero_weight_leaves]
    color_shape = exact_colors.shape[1:]
    flat_colors = exact_colors.reshape(n, -1)
    leaf_colors = _segment_reduce(
        flat_colors * weights[:, None], "sum", leaf_lengths
    ) / safe_sums[:, None]
    if torch.any(zero_weight_leaves):
        uniform_colors = _segment_reduce(flat_colors, "mean", leaf_lengths)
        leaf_colors[zero_weight_leaves] = uniform_colors[zero_weight_leaves]
    leaf_alpha = 1.0 - torch.exp(
        _segment_reduce(
            torch.log1p(-exact_opacities),
            "sum",
            leaf_lengths,
        )
    )
    leaf_mass = _segment_reduce(
        exact_opacities * exact_volumes, "sum", leaf_lengths
    )

    node_means = means.new_empty((num_nodes, 3))
    node_covars = means.new_empty((num_nodes, 3, 3))
    node_colors = colors.new_empty((num_nodes,) + color_shape)
    node_alpha = opacities.new_empty(num_nodes)
    node_mass = opacities.new_empty(num_nodes)
    node_weights = opacities.new_empty(num_nodes)
    node_means[leaf_nodes] = leaf_means
    node_covars[leaf_nodes] = leaf_covars
    node_colors[leaf_nodes] = leaf_colors.reshape((-1,) + color_shape)
    node_alpha[leaf_nodes] = leaf_alpha
    node_mass[leaf_nodes] = leaf_mass
    node_weights[leaf_nodes] = weight_sums

    max_built_depth = int(depths[-1].item())
    for depth in range(max_built_depth - 1, -1, -1):
        nodes = torch.nonzero(
            (depths == depth) & (left_children >= 0),
            as_tuple=False,
        ).squeeze(1)
        if nodes.numel() == 0:
            continue
        left = left_children[nodes]
        right = right_children[nodes]
        wa = node_weights[left]
        wb = node_weights[right]
        total = wa + wb
        safe_total = total.clamp_min(1e-12)
        mean = (
            wa[:, None] * node_means[left]
            + wb[:, None] * node_means[right]
        ) / safe_total[:, None]
        zero_weight = total <= 1e-12
        mean[zero_weight] = 0.5 * (
            node_means[left][zero_weight] + node_means[right][zero_weight]
        )
        da = node_means[left] - mean
        db = node_means[right] - mean
        covariance = (
            wa[:, None, None]
            * (node_covars[left] + da[:, :, None] * da[:, None, :])
            + wb[:, None, None]
            * (node_covars[right] + db[:, :, None] * db[:, None, :])
        ) / safe_total[:, None, None]
        covariance[zero_weight] = 0.5 * (
            node_covars[left][zero_weight] + node_covars[right][zero_weight]
        )
        color = (
            wa.reshape((-1,) + (1,) * len(color_shape))
            * node_colors[left]
            + wb.reshape((-1,) + (1,) * len(color_shape))
            * node_colors[right]
        ) / safe_total.reshape((-1,) + (1,) * len(color_shape))
        color[zero_weight] = 0.5 * (
            node_colors[left][zero_weight] + node_colors[right][zero_weight]
        )
        node_means[nodes] = mean
        node_covars[nodes] = covariance
        node_colors[nodes] = color
        node_alpha[nodes] = 1.0 - (
            1.0 - node_alpha[left]
        ) * (1.0 - node_alpha[right])
        node_mass[nodes] = node_mass[left] + node_mass[right]
        node_weights[nodes] = total

    symmetric_covars = 0.5 * (
        node_covars + node_covars.transpose(-1, -2)
    )
    eigvals, eigvecs = _eigh_symmetric_3x3(symmetric_covars)
    reflected = torch.linalg.det(eigvecs) < 0
    eigvecs[reflected, :, -1] *= -1
    proxy_scales = eigvals.clamp_min(1e-12).sqrt()
    proxy_quats = _rotmat_to_quat(eigvecs)
    if opacity_mode == "mass":
        proxy_opacities = (
            node_mass / proxy_scales.prod(dim=1).clamp_min(1e-12)
        ).clamp(0.0, 1.0)
    else:
        proxy_opacities = node_alpha

    reordered_parent = parents[old_by_preorder]
    reordered_parent = torch.where(
        reordered_parent >= 0,
        preorder_of_old[reordered_parent.clamp_min(0)],
        reordered_parent,
    )
    reordered_leaf_nodes = preorder_of_old[leaf_nodes]
    leaf_starts = torch.zeros(num_nodes, dtype=torch.long, device=device)
    leaf_lens = torch.zeros(num_nodes, dtype=torch.long, device=device)
    starts = torch.cat(
        [leaf_lengths.new_zeros(1), leaf_lengths.cumsum(0)[:-1]]
    )
    leaf_starts[reordered_leaf_nodes] = starts
    leaf_lens[reordered_leaf_nodes] = leaf_lengths

    root = _CachedLODRoot()
    root._node_center = 0.5 * (
        amins[old_by_preorder] + amaxs[old_by_preorder]
    )
    root._node_radius = 0.5 * torch.linalg.vector_norm(
        amaxs[old_by_preorder] - amins[old_by_preorder], dim=1
    )
    root._node_size = root._node_radius
    root._node_is_leaf = left_children[old_by_preorder] < 0
    root._node_parent = reordered_parent
    root._node_leaf_start = leaf_starts
    root._node_leaf_len = leaf_lens
    root._flat_proxy_means = node_means[old_by_preorder]
    root._flat_proxy_quats = proxy_quats[old_by_preorder]
    root._flat_proxy_scales = proxy_scales[old_by_preorder]
    root._flat_proxy_opacities = proxy_opacities[old_by_preorder]
    root._flat_proxy_colors = node_colors[old_by_preorder]
    root._flat_leaf_means = exact_means
    root._flat_leaf_quats = quats[exact_indices]
    root._flat_leaf_scales = scales[exact_indices]
    root._flat_leaf_opacities = opacities[exact_indices]
    root._flat_leaf_colors = exact_colors
    root._num_nodes = num_nodes
    root._leaf_index_device = device
    if retain_root_stats:
        # Keep unquantized moments, not the fp16 proxy, when joining subtrees.
        root._build_root_stats = tuple(
            tensor[:1].detach().cpu().clone()
            for tensor in (
                node_weights, node_means, node_covars, node_colors,
                node_alpha, node_mass,
            )
        )
    _cast_flat_fp16(root)
    return root


def _build_bvh_lod_cuda_flat(*args, **kwargs):
    return _build_bvh_lod_flat(*args, **kwargs)


def _build_bvh_lod_cpu_flat(*args, **kwargs):
    return _build_bvh_lod_flat(*args, **kwargs)


def _lod_cuda_sort_order(keys, device):
    stream = torch.cuda.current_stream(device)
    try:
        cuda_keys = torch.from_numpy(keys).to(device)
        return torch.argsort(cuda_keys, stable=True).cpu().numpy()
    finally:
        stream.synchronize()


def _meminfo_available_bytes() -> int | None:
    """Return MemAvailable, which counts reclaimable page cache unlike MemFree."""
    try:
        with open(_MEMINFO_PATH, encoding="ascii") as stream:
            for line in stream:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


_MEMINFO_PATH = "/proc/meminfo"


def _lod_host_memory() -> tuple[int, int]:
    """Return available/total physical bytes without an optional dependency."""
    if os.name == "nt":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                *[(name, ctypes.c_ulonglong) for name in (
                    "total", "available", "page_total", "page_available",
                    "virtual_total", "virtual_available", "extended",
                )],
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise ctypes.WinError()
        return int(status.available), int(status.total)
    page_size = os.sysconf("SC_PAGE_SIZE")
    available = _meminfo_available_bytes()
    if available is None:
        # Kernels without MemAvailable: free pages only (conservative).
        available = os.sysconf("SC_AVPHYS_PAGES") * page_size
    return available, os.sysconf("SC_PHYS_PAGES") * page_size


def _lod_host_budget() -> int:
    available, total = _lod_host_memory()
    # Aim below 90%, leaving extra room for drivers and untracked allocations.
    return available - int(total * 0.12) - 512 * 2**20


def _require_lod_host_memory(required: int, phase: str) -> None:
    budget = _lod_host_budget()
    if required > budget:
        raise MemoryError(
            f"LOD {phase} needs {required / 2**30:.2f} GiB additional RAM, "
            f"but only {max(0, budget) / 2**30:.2f} GiB is safe below 90% system usage. "
            "Close other memory-heavy processes or build on a larger-memory host."
        )


def _bvh_build_vram_budget(device, gpu_memory_percent: float = 0.0) -> int:
    if (
        isinstance(gpu_memory_percent, bool)
        or not isinstance(gpu_memory_percent, (int, float))
        or not np.isfinite(gpu_memory_percent)
        or not (gpu_memory_percent == 0 or 1 <= gpu_memory_percent <= 90)
    ):
        raise ValueError(
            "LOD GPU memory percentage must be 0 (auto) or finite between 1 and 90"
        )
    gpu_memory_percent = gpu_memory_percent or 80.0
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    reusable = (
        torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    )
    available = free_bytes + reusable
    # Reserve one extra percentage point below the device-wide target.
    reserve = int(total_bytes * ((101 - gpu_memory_percent) / 100))
    return max(0, available - max(512 * 2**20, reserve, available // 5))


@torch.no_grad()
def build_bvh_lod_batched(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    colors: Tensor,
    max_depth: int = 32,
    min_points_per_node: int = 64,
    opacity_mode: str = "union",
    *,
    max_points_per_batch: int = 0,
    cpu_workers: int = 0,
    gpu_memory_percent: float = 0.0,
    device: str = "cuda",
    progress: Optional[Callable[[int, int], None]] = None,
):
    """Build one ordinary flat BVH with CPU storage and bounded CUDA batches.

    Upper levels use the same longest-axis, stable median splits as the
    CUDA builder. Their frontier subtrees are built sequentially on CUDA and
    copied into one preallocated CPU cache in preorder. No pages, streaming
    format, point sampling, or additional proxy quantization is introduced.
    Host storage remains O(N); GPU working storage is O(max_points_per_batch).
    All batch limits are capped by current RAM/VRAM budgets and bounded pinned
    staging. Insufficient resident host capacity raises before allocation.
    gpu_memory_percent targets total device usage, including other processes,
    with an extra percentage point reserved; it is not a hard allocator quota.
    Values above the fixed 90 percent configuration ceiling are rejected.
    A zero GPU percentage uses an automatic 80 percent safety target.
    Lost headroom splits pending subtrees; CUDA OOM retries smaller subtrees
    or bounded CPU work without discarding completed results.
    A zero batch limit chooses a conservative limit from available VRAM. A
    zero worker count uses up to eight CPU workers for independent upper splits.
    """
    inputs = (means, quats, scales, opacities, colors)
    n = means.shape[0]
    if any(t.device.type != "cpu" or t.dtype != torch.float32 for t in inputs):
        raise ValueError("batched BVH inputs must be CPU float32 tensors")
    expected = ((n, 3), (n, 4), (n, 3), (n,))
    if any(t.shape != shape for t, shape in zip(inputs, expected)):
        raise ValueError("inconsistent BVH geometry/opacity shapes")
    if colors.ndim < 2 or colors.shape[0] != n or colors.shape[-1] != 3:
        raise ValueError("colors must have shape [N, ..., 3]")
    if n == 0 or max_depth < 0 or min_points_per_node < 1:
        raise ValueError("BVH requires points, nonnegative depth and positive leaf size")
    if opacity_mode not in ("union", "mass"):
        raise ValueError(f"unsupported opacity_mode: {opacity_mode!r}")
    if max_points_per_batch < 0 or cpu_workers < 0:
        raise ValueError("batch limit and CPU worker count must be nonnegative")
    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError("batched BVH requires a CUDA build device")
    requested_batch_points = max_points_per_batch
    auto_batch = max_points_per_batch == 0
    budget = _bvh_build_vram_budget(target, gpu_memory_percent)
    effective_percent = gpu_memory_percent or 80.0
    strategy = "auto" if gpu_memory_percent == 0 else "manual ceiling"
    print(
        f"[LOD] Build strategy={strategy}, GPU memory target={effective_percent:g}% "
        f"(device-wide); working budget={budget / 2**30:.2f} GiB",
        flush=True,
    )
    appearance_bytes = colors[0].numel() * colors.element_size()
    node_ratio = min(2.0, 4.0 / min_points_per_node)
    bytes_per_point = (
        512 + 2 * appearance_bytes
        + node_ratio * (256 + 3 * appearance_bytes)
    )
    safe_batch = max(1, int(budget / bytes_per_point)) if budget else 4096
    max_points_per_batch = (
        safe_batch if auto_batch else min(max_points_per_batch, safe_batch)
    )
    cpu_workers = cpu_workers or min(8, os.cpu_count() or 1)
    largest_leaf = n
    for _ in range(max_depth):
        if largest_leaf <= min_points_per_node:
            break
        largest_leaf = (largest_leaf + 1) // 2
    if (
        max_points_per_batch < largest_leaf
        and (auto_batch or requested_batch_points >= largest_leaf)
    ):
        max_points_per_batch = largest_leaf
    if largest_leaf > max_points_per_batch:
        raise ValueError(
            f"largest leaf ({largest_leaf} points) exceeds the batch limit "
            f"({max_points_per_batch}); increase max_depth or the batch limit"
        )

    @lru_cache(maxsize=None)
    def tree_size(count, depth):
        if depth == max_depth or count <= min_points_per_node:
            return 1
        half = count // 2
        return 1 + tree_size(half, depth + 1) + tree_size(count - half, depth + 1)

    num_nodes = tree_size(n, 0)
    input_width = sum(tensor[0].numel() for tensor in inputs)
    # Float32 means, FP16 attributes, and flat topology (excluding its alias).
    attribute_bytes = 12 + 2 * (input_width - 3)
    output_bytes = (n + num_nodes) * attribute_bytes + num_nodes * 41
    # Stable CPU sort: keys, int64 order/scratch, and indexed uint32 output.
    partition_bytes = n * (28 if n < 2**32 else 40)
    _require_lod_host_memory(output_bytes + partition_bytes, "BVH storage/sort")
    staging_budget = min(
        2 * 2**30, _lod_host_budget() - output_bytes - partition_bytes
    )
    staging_per_point = (
        8 * input_width + attribute_bytes
        + node_ratio * (attribute_bytes + 41) + 32
    )
    max_points_per_batch = min(
        max_points_per_batch, max(0, int(staging_budget / staging_per_point))
    )
    if largest_leaf > max_points_per_batch:
        raise MemoryError(
            "LOD pinned staging cannot fit the largest leaf below 90% RAM usage"
        )
    root = _CachedLODRoot()
    root._build_batch_points = max_points_per_batch
    root._build_cpu_workers = cpu_workers
    root._num_nodes = num_nodes
    root._leaf_index_device = torch.device("cpu")
    root._node_center = torch.empty((num_nodes, 3), dtype=torch.float32, device="cpu")
    root._node_radius = torch.empty(num_nodes, dtype=torch.float32, device="cpu")
    root._node_size = root._node_radius
    root._node_is_leaf = torch.empty(num_nodes, dtype=torch.bool, device="cpu")
    root._node_parent = torch.empty(num_nodes, dtype=torch.long, device="cpu")
    root._node_leaf_start = torch.zeros(num_nodes, dtype=torch.long, device="cpu")
    root._node_leaf_len = torch.zeros(num_nodes, dtype=torch.long, device="cpu")
    for kind, count in (("leaf", n), ("proxy", num_nodes)):
        for name, tensor in zip(
            ("means", "quats", "scales", "opacities", "colors"), inputs
        ):
            dtype = torch.float32 if name == "means" else torch.float16
            setattr(
                root, f"_flat_{kind}_{name}",
                torch.empty((count,) + tensor.shape[1:], dtype=dtype, device="cpu"),
            )
    xyz = means.detach().numpy()
    indices = np.arange(n, dtype=np.uint32 if n < 2**32 else np.int64)
    completed = 0

    subtree_timings = {
        "batch_gather_cpu": 0.0,
        "batch_prefetch_wait": 0.0,
        "batch_upload_gpu": 0.0,
        "batch_build_gpu": 0.0,
        "batch_build_cpu": 0.0,
        "batch_download_gpu": 0.0,
        "batch_output_cpu": 0.0,
    }

    @torch.no_grad()
    def prepare_batch(descriptor, buffer):
        started = time.perf_counter()
        ids = descriptor[0]
        gather = torch.from_numpy(ids.astype(np.int64))
        offset = 0
        prepared = []
        for tensor in inputs:
            shape = (len(ids),) + tensor.shape[1:]
            size = len(ids) * tensor[0].numel()
            destination = buffer[offset:offset + size].view(shape)
            torch.index_select(tensor, 0, gather, out=destination)
            prepared.append(destination)
            offset += size
        return prepared, offset, time.perf_counter() - started

    def build_gpu_subtree(descriptor, prepared, buffer, size, download_buffers):
        ids, depth, node_start, parent, exact_start = descriptor
        stream = torch.cuda.current_stream(target)
        try:
            events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            events[0].record(stream)
            uploaded = buffer[:size].to(target, non_blocking=True)
            batch = []
            offset = 0
            for tensor in prepared:
                batch.append(uploaded[offset:offset + tensor.numel()].view(tensor.shape))
                offset += tensor.numel()
            events[1].record(stream)
            subtree = _build_bvh_lod_cuda_flat(
                *batch, max_depth - depth, min_points_per_node, opacity_mode,
                retain_root_stats=True,
            )
            events[2].record(stream)
            downloaded = {}
            for attr in _LOD_FLAT_ATTRS:
                if attr == "_node_size":
                    continue
                tensor = getattr(subtree, attr)
                staging = download_buffers[attr][:len(tensor)]
                staging.copy_(tensor, non_blocking=True)
                downloaded[attr] = staging
            events[3].record(stream)
        finally:
            # Both upload and download DMA must finish before reusing buffers.
            stream.synchronize()
        for name, first, last in (
            ("batch_upload_gpu", 0, 1),
            ("batch_build_gpu", 1, 2),
            ("batch_download_gpu", 2, 3),
        ):
            subtree_timings[name] += events[first].elapsed_time(events[last]) / 1000
        return downloaded, subtree._build_root_stats

    def commit_subtree(descriptor, downloaded, root_stats):
        nonlocal completed
        ids, depth, node_start, parent, exact_start = descriptor
        output_started = time.perf_counter()
        for attr, values in downloaded.items():
            if attr == "_node_parent":
                values = torch.where(values >= 0, values + node_start, parent)
            elif attr == "_node_leaf_start":
                values = torch.where(
                    downloaded["_node_is_leaf"], values + exact_start, 0
                )
            start = exact_start if attr.startswith("_flat_leaf_") else node_start
            getattr(root, attr)[start:start + len(values)].copy_(values)
        subtree_timings["batch_output_cpu"] += time.perf_counter() - output_started
        stats[node_start] = root_stats
        completed += len(ids)
        if progress is not None:
            progress(completed, n)

    def partition(descriptor):
        ids = descriptor[0]
        count = len(ids)
        lower = np.full(3, np.inf, dtype=np.float32)
        upper = np.full(3, -np.inf, dtype=np.float32)
        scan_points = min(max_points_per_batch, 1 << 20)
        for start in range(0, count, scan_points):
            points = xyz[ids[start:start + scan_points]]
            lower = np.minimum(lower, points.min(axis=0))
            upper = np.maximum(upper, points.max(axis=0))
        axis = int(np.argmax(upper - lower))
        extent = max(upper[axis] - lower[axis], np.finfo(np.float32).eps)
        keys = np.empty(count, dtype=np.float32)
        for start in range(0, count, scan_points):
            end = min(start + scan_points, count)
            keys[start:end] = (xyz[ids[start:end], axis] - lower[axis]) / extent
        return lower, upper, keys

    def merge(left, right, node_start, parent, lower, upper):
        wa, mua, ca, cola, aa, ma = left
        wb, mub, cb, colb, ab, mb = right
        total = wa + wb
        safe = total.clamp_min(1e-12)
        zero = total <= 1e-12
        mean = (wa[:, None] * mua + wb[:, None] * mub) / safe[:, None]
        mean[zero] = 0.5 * (mua[zero] + mub[zero])
        da, db = mua - mean, mub - mean
        covariance = (
            wa[:, None, None] * (ca + da[:, :, None] * da[:, None, :])
            + wb[:, None, None] * (cb + db[:, :, None] * db[:, None, :])
        ) / safe[:, None, None]
        covariance[zero] = 0.5 * (ca[zero] + cb[zero])
        weight_shape = (-1,) + (1,) * (colors.ndim - 1)
        color = (
            wa.reshape(weight_shape) * cola + wb.reshape(weight_shape) * colb
        ) / safe.reshape(weight_shape)
        color[zero] = 0.5 * (cola[zero] + colb[zero])
        alpha = 1.0 - (1.0 - aa) * (1.0 - ab)
        mass = ma + mb
        eigenvalues, rotation = _eigh_symmetric_3x3(
            0.5 * (covariance + covariance.transpose(-1, -2))
        )
        rotation[torch.linalg.det(rotation) < 0, :, -1] *= -1
        proxy_scales = eigenvalues.clamp_min(1e-12).sqrt()
        proxy_opacity = (
            (mass / proxy_scales.prod(1).clamp_min(1e-12)).clamp(0, 1)
            if opacity_mode == "mass" else alpha
        )
        root._node_center[node_start] = torch.from_numpy(0.5 * (lower + upper))
        root._node_radius[node_start] = float(0.5 * np.linalg.norm(upper - lower))
        root._node_is_leaf[node_start] = False
        root._node_parent[node_start] = parent
        for name, tensor in zip(
            ("means", "quats", "scales", "opacities", "colors"),
            (mean, _rotmat_to_quat(rotation), proxy_scales, proxy_opacity, color),
        ):
            getattr(root, f"_flat_proxy_{name}")[node_start].copy_(tensor[0].cpu())
        return tuple(
            t.cpu() for t in (total, mean, covariance, color, alpha, mass)
        )

    started = time.perf_counter()
    active = [(indices, 0, 0, -1, 0)]
    frontier = []
    upper_nodes = []
    gpu_sort_nodes = 0
    gpu_sort_limit = n
    recovery = {"cuda_ooms": 0, "budget_splits": 0, "cpu_subtrees": 0}
    sort_seconds = 0.0
    with ThreadPoolExecutor(max_workers=cpu_workers) as executor:
        while active:
            splits = []
            for descriptor in active:
                if len(descriptor[0]) <= max_points_per_batch:
                    frontier.append(descriptor)
                else:
                    splits.append(descriptor)
            active = []
            # Siblings own disjoint slices of indices. No torch operations or
            # shared writes occur in workers, and results keep traversal order.
            # Do not let Executor.map retain a whole level of completed keys.
            def bounded_partitions():
                for start in range(0, len(splits), cpu_workers):
                    window = splits[start:start + cpu_workers]
                    _require_lod_host_memory(
                        sum(len(desc[0]) for desc in window) * 28, "upper partition"
                    )
                    yield from executor.map(partition, window)

            for descriptor, (lower, upper, keys) in zip(splits, bounded_partitions()):
                ids, depth, node_start, parent, exact_start = descriptor
                sort_started = time.perf_counter()
                # Only keys/indices go to CUDA, not all Gaussian attributes.
                # Large stable radix sorts need ~36 B/point on current torch;
                # budget 40 B/point plus the shared free-VRAM safety margin.
                if (
                    auto_batch
                    and len(ids) <= gpu_sort_limit
                    and len(ids) >= 1 << 18
                    and len(ids) * 40 <= _bvh_build_vram_budget(target, gpu_memory_percent)
                ):
                    sort_oom = False
                    try:
                        order = _lod_cuda_sort_order(keys, target)
                    except torch.cuda.OutOfMemoryError as exc:
                        print(
                            f"[LOD] CUDA OOM sorting {len(ids):,} points; "
                            f"retrying this sort on CPU: {exc}", flush=True,
                        )
                        sort_oom = True
                    if sort_oom:
                        recovery["cuda_ooms"] += 1
                        gpu_sort_limit = (len(ids) + 1) // 2
                        with torch.cuda.device(target):
                            torch.cuda.empty_cache()
                        order = np.argsort(keys, kind="stable")
                    else:
                        gpu_sort_nodes += 1
                else:
                    order = np.argsort(keys, kind="stable")
                ids[:] = ids[order]
                del keys, order
                sort_seconds += time.perf_counter() - sort_started
                half = len(ids) // 2
                left_start = node_start + 1
                right_start = left_start + tree_size(half, depth + 1)
                active.extend([
                    (ids[:half], depth + 1, left_start, node_start, exact_start),
                    (ids[half:], depth + 1, right_start, node_start, exact_start + half),
                ])
                upper_nodes.append(
                    (node_start, parent, left_start, right_start, lower, upper)
                )
    partition_seconds = time.perf_counter() - started
    if gpu_sort_nodes:
        # Root-sized radix-sort blocks are not the subtree builder's sizes.
        # Release this one-off workspace at the phase boundary, not per batch.
        with torch.cuda.device(target):
            torch.cuda.empty_cache()
    stats = {}
    started = time.perf_counter()
    frontier.sort(key=lambda desc: desc[-1])
    max_batch = max(len(desc[0]) for desc in frontier)
    _require_lod_host_memory(
        int(max_batch * staging_per_point), "pinned staging"
    )
    input_buffers = [
        torch.empty(
            max_batch * input_width, dtype=torch.float32, device="cpu",
            pin_memory=budget > 0,
        )
        for _ in range(min(2, len(frontier)))
    ]
    max_nodes = max(tree_size(len(desc[0]), desc[1]) for desc in frontier)
    download_buffers = {}
    for attr in _LOD_FLAT_ATTRS:
        if attr == "_node_size":
            continue
        tensor = getattr(root, attr)
        count = max_batch if attr.startswith("_flat_leaf_") else max_nodes
        download_buffers[attr] = torch.empty(
            (count,) + tensor.shape[1:], dtype=tensor.dtype,
            device="cpu", pin_memory=budget > 0,
        )
    recovery_batch_points = max_points_per_batch

    def process_subtree(descriptor, prepared, buffer, size):
        nonlocal recovery_batch_points
        ids, depth, node_start, parent, exact_start = descriptor
        count = len(ids)
        current_budget = _bvh_build_vram_budget(target, gpu_memory_percent)
        safe_points = max(0, int(current_budget / bytes_per_point))
        terminal = depth == max_depth or count <= min_points_per_node
        fits = count <= min(recovery_batch_points, safe_points)
        use_cpu = not fits and (terminal or (safe_points == 0 and count <= 4096))
        oom = False
        if fits:
            _require_lod_host_memory(0, "subtree build")
            try:
                downloaded, root_stats = build_gpu_subtree(
                    descriptor, prepared, buffer, size, download_buffers
                )
            except torch.cuda.OutOfMemoryError as exc:
                print(
                    f"[LOD] CUDA OOM building {count:,} points; "
                    f"retrying with smaller subtrees: {exc}", flush=True,
                )
                oom = True
            if not oom:
                commit_subtree(descriptor, downloaded, root_stats)
                return
            # Leave the exception scope so its traceback releases CUDA tensors.
            recovery["cuda_ooms"] += 1
            with torch.cuda.device(target):
                torch.cuda.empty_cache()
            recovery_batch_points = min(recovery_batch_points, (count + 1) // 2)
            use_cpu = terminal
        if use_cpu:
            _require_lod_host_memory(count * bytes_per_point, "CPU subtree fallback")
            print(f"[LOD] Building {count:,}-point subtree on CPU", flush=True)
            cpu_started = time.perf_counter()
            subtree = _build_bvh_lod_cpu_flat(
                *prepared, max_depth - depth, min_points_per_node, opacity_mode,
                retain_root_stats=True,
            )
            downloaded = {
                attr: getattr(subtree, attr) for attr in _LOD_FLAT_ATTRS
                if attr != "_node_size"
            }
            commit_subtree(descriptor, downloaded, subtree._build_root_stats)
            subtree_timings["batch_build_cpu"] += time.perf_counter() - cpu_started
            recovery["cpu_subtrees"] += 1
            return
        recovery["budget_splits"] += 1
        if not oom:
            recovery_batch_points = min(recovery_batch_points, max(1, safe_points))
        print(
            f"[LOD] Splitting pending {count:,}-point subtree; "
            f"current GPU batch ceiling={recovery_batch_points:,}", flush=True,
        )
        lower, upper, keys = partition(descriptor)
        ids[:] = ids[np.argsort(keys, kind="stable")]
        del keys
        half = count // 2
        left_start = node_start + 1
        right_start = left_start + tree_size(half, depth + 1)
        upper_nodes.append(
            (node_start, parent, left_start, right_start, lower, upper)
        )
        for child in (
            (ids[:half], depth + 1, left_start, node_start, exact_start),
            (ids[half:], depth + 1, right_start, node_start, exact_start + half),
        ):
            child_prepared, child_size, seconds = prepare_batch(child, buffer)
            subtree_timings["batch_gather_cpu"] += seconds
            process_subtree(child, child_prepared, buffer, child_size)

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="bvh-prefetch") as executor:
        future = executor.submit(prepare_batch, frontier[0], input_buffers[0])
        for index, descriptor in enumerate(frontier):
            wait_started = time.perf_counter()
            prepared, size, gather_seconds = future.result()
            subtree_timings["batch_prefetch_wait"] += time.perf_counter() - wait_started
            subtree_timings["batch_gather_cpu"] += gather_seconds
            if index + 1 < len(frontier):
                future = executor.submit(
                    prepare_batch, frontier[index + 1],
                    input_buffers[(index + 1) % len(input_buffers)],
                )
            buffer = input_buffers[index % len(input_buffers)]
            process_subtree(descriptor, prepared, buffer, size)
    download_buffers.clear()
    del prepared, input_buffers, buffer, future
    cuda_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for node_start, parent, left, right, lower, upper in reversed(upper_nodes):
        stats[node_start] = merge(
            stats.pop(left), stats.pop(right), node_start, parent, lower, upper
        )
    root._build_timings = {
        "upper_partition": partition_seconds,
        "upper_sort_in_partition": sort_seconds,
        "cuda_subtrees_and_copies": cuda_seconds,
        **subtree_timings,
        "upper_proxy_merge": time.perf_counter() - started,
    }
    root._build_gpu_sort_nodes = gpu_sort_nodes
    root._build_recovery = recovery
    print(f"[LOD] Automatic recovery: {recovery}", flush=True)
    return root


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
        "source_ply": getattr(root, "_source_ply", None),
    }
    for attr in _LOD_FLAT_ATTRS:
        blob[attr] = getattr(root, attr).detach().to("cpu").contiguous()

    path = _windows_extended_path(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=os.path.basename(path) + ".", suffix=".tmp",
        dir=parent, delete=False,
    ) as handle:
        tmp = handle.name
    try:
        torch.save(blob, tmp)
        os.replace(tmp, path)  # atomic on the same filesystem
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_lod_cache(
    path: str, device="cuda", source_ply_md5: Optional[str] = None,
) -> "_CachedLODRoot":
    """Reload a LOD cache written by :func:`save_lod_cache` onto ``device``.

    Returns a stand-in root accepted by :func:`select_lod`. Raises
    ``ValueError`` on a version mismatch and ``KeyError`` if the file is
    missing an expected tensor (either should trigger a rebuild by the caller).
    When supplied, ``source_ply_md5`` must match the persisted source metadata;
    this is checked before any cached tensors are uploaded to the device.
    """
    blob = torch.load(_windows_extended_path(path), map_location="cpu")
    version = blob.get("version") if isinstance(blob, dict) else None
    if version != _LOD_CACHE_VERSION:
        raise ValueError(
            f"LOD cache version mismatch: got {version!r}, "
            f"expected {_LOD_CACHE_VERSION}"
        )
    source_ply = blob.get("source_ply")
    if source_ply_md5 is not None and (
        not isinstance(source_ply, dict) or source_ply.get("md5") != source_ply_md5
    ):
        raise ValueError("LOD cache source PLY MD5 mismatch or missing provenance")
    dev = torch.device(device)
    root = _CachedLODRoot()
    root._source_ply = source_ply
    for attr in _LOD_FLAT_ATTRS:
        root.__dict__[attr] = blob[attr].to(dev)
    root._num_nodes = int(blob.get("num_nodes", root._node_center.shape[0]))
    root._leaf_index_device = dev
    # Older caches were saved entirely in fp32; downcast the fp16-safe
    # attributes so a reloaded tree matches a freshly built one.
    _cast_flat_fp16(root)
    return root
