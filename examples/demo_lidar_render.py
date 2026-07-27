"""LiDAR point-cloud rendering demo — rasterize a trained 3DGS scene via a
spinning LiDAR model and display the resulting depth-coloured point cloud.

Requirements
------------
- A trained checkpoint produced by ``simple_trainer.py`` (``ckpt_*.pt``).
- The ``gsplat`` package built with CUDA support (``has_camera_wrappers()``).
- ``matplotlib`` for visualisation.

Usage
-----
    # default sensor: generic 128-line LiDAR, sensor at scene origin
    python examples/demo_lidar_render.py --ckpt results/garden/ckpts/ckpt_30000_rank0.pt

    # waymo sensor, move the sensor 2 m above origin, look forward
    python examples/demo_lidar_render.py --ckpt path/to/ckpt.pt --sensor waymo --height 2.0

    # subsample columns for speed
    python examples/demo_lidar_render.py --ckpt path/to/ckpt.pt --col-step 4
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import gsplat
from gsplat import compute_lidar_angles_to_columns_map, compute_lidar_tiling
from gsplat.exporter import load_ply_to_splats
from gsplat.cuda._lidar import (
    RowOffsetStructuredSpinningLidarModelParameters,
    SpinningDirection,
)
from gsplat.cuda._wrapper import RowOffsetStructuredSpinningLidarModelParametersExt


def _save_pcd(
    path: Path,
    pts_x: np.ndarray,
    pts_y: np.ndarray,
    pts_z: np.ndarray,
    pts_rgb: np.ndarray,           # (M, 3) float32 in [0, 1]
    binary: bool = True,
) -> None:
    """Save points to a PCL-compatible PCD file (XYZRGB).

    RGB is packed as a single float32 following the PCL convention so the
    file can be opened directly in CloudCompare, Open3D, PCL viewer, etc.
    """
    n = len(pts_x)
    # Pack RGB: 0x00RRGGBB stored as little-endian float32 reinterpret
    r = (pts_rgb[:, 0] * 255).astype(np.uint8)
    g = (pts_rgb[:, 1] * 255).astype(np.uint8)
    b = (pts_rgb[:, 2] * 255).astype(np.uint8)
    rgb_int = (r.astype(np.uint32) << 16) | (g.astype(np.uint32) << 8) | b.astype(np.uint32)
    rgb_float = np.frombuffer(rgb_int.astype(np.uint32).tobytes(), dtype=np.float32)

    data_mode = "binary" if binary else "ascii"
    header = (
        "# .PCD v0.7 - Point Cloud Data\n"
        "VERSION 0.7\n"
        "FIELDS x y z rgb\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        f"DATA {data_mode}\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header.encode())
        if binary:
            pts = np.column_stack([
                pts_x.astype(np.float32),
                pts_y.astype(np.float32),
                pts_z.astype(np.float32),
                rgb_float,
            ])
            f.write(pts.tobytes())
        else:
            for x, y, z, c in zip(pts_x, pts_y, pts_z, rgb_float):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {c}\n".encode())


TEST_DATA = Path(__file__).resolve().parent.parent / "tests/sensors/test_data"
JSON_PATHS = {
    "generic": TEST_DATA / "row-offset-spinning-lidar-model-parameters.json",
    "waymo":   TEST_DATA / "row-offset-spinning-lidar-model-parameters-waymo.json",
}


# ─── helpers ──────────────────────────────────────────────────────────────────

def _load_ckpt(ckpt_path: str, device: torch.device):
    """Load splats from a simple_trainer checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    splats = ckpt["splats"]  # state dict

    means    = splats["means"].to(device)       # (N, 3)
    quats    = splats["quats"].to(device)        # (N, 4)  xyzw
    scales   = torch.exp(splats["scales"]).to(device)  # (N, 3) log-space in ckpt
    opacities = torch.sigmoid(splats["opacities"]).to(device)  # (N,) logits in ckpt

    # Colour: prefer sh0 (DC SH coefficient → RGB), fall back to "colors"
    if "sh0" in splats:
        # sh0 shape: (N, 1, 3); SH DC → linear RGB: C0 * sh0 + 0.5
        C0 = 0.28209479177387814
        colors = (C0 * splats["sh0"][:, 0, :] + 0.5).clamp(0, 1).to(device)
    else:
        colors = splats["colors"].to(device)    # (N, 3)

    return means, quats, scales, opacities, colors


def _load_ply(ply_path: str, device: torch.device):
    """Load splats from a standard 3DGS PLY file (e.g. point_cloud.ply)."""
    splats = load_ply_to_splats(ply_path)

    means     = splats["means"].to(device)                       # (N, 3)
    quats     = splats["quats"].to(device)                       # (N, 4)
    scales    = torch.exp(splats["scales"]).to(device)           # (N, 3) log-space in PLY
    opacities = torch.sigmoid(splats["opacities"]).to(device)    # (N,) logit in PLY

    # sh0: (N, 1, 3) → DC SH → linear RGB
    C0 = 0.28209479177387814
    colors = (C0 * splats["sh0"][:, 0, :] + 0.5).clamp(0, 1).to(device)

    return means, quats, scales, opacities, colors


def _build_lidar_params(config: str, device: torch.device):
    """Build RowOffsetStructuredSpinningLidarModelParametersExt from a JSON file."""
    path = JSON_PATHS[config]
    with path.open(encoding="utf-8") as f:
        p = json.load(f)

    row_elev  = torch.tensor(p["row_elevations_rad"],   dtype=torch.float32, device=device)
    col_az    = torch.tensor(p["column_azimuths_rad"],  dtype=torch.float32, device=device)

    raw_offsets = p.get("row_azimuth_offsets_rad")
    has_offsets = raw_offsets is not None and any(v != 0 for v in raw_offsets)
    row_offsets = (
        torch.tensor(raw_offsets, dtype=torch.float32, device=device)
        if has_offsets
        else torch.zeros(len(row_elev), dtype=torch.float32, device=device)
    )

    raw_dir  = str(p.get("spinning_direction", "cw")).lower()
    spin_dir = SpinningDirection.CLOCKWISE if raw_dir in ("cw", "0") \
               else SpinningDirection.COUNTER_CLOCKWISE
    freq_hz  = float(p.get("spinning_frequency_hz", 10.0))

    fov_vert_start = float(row_elev.max().item())
    fov_vert_span  = abs(fov_vert_start - float(row_elev.min().item()))

    base = RowOffsetStructuredSpinningLidarModelParameters(
        row_elevations_rad=row_elev,
        column_azimuths_rad=col_az,
        row_azimuth_offsets_rad=row_offsets,
        spinning_direction=spin_dir,
        spinning_frequency_hz=freq_hz,
    )

    print(f"  Sensor: {base.n_rows} rows × {base.n_columns} columns")
    print(f"  Vert FOV: [{math.degrees(fov_vert_start - fov_vert_span):.1f}°, "
          f"{math.degrees(fov_vert_start):.1f}°]")
    print(f"  Building acceleration structures …", end=" ", flush=True)

    a2c_map = compute_lidar_angles_to_columns_map(base)
    tiling  = compute_lidar_tiling(
        base,
        n_bins_elevation=16,
        max_pts_per_tile=64,         # tile_size=8 → 8*8
        resolution_elevation=1600,
        densification_factor_azimuth=8,
    )
    print("done")

    return RowOffsetStructuredSpinningLidarModelParametersExt(base, a2c_map, tiling)


def _make_viewmat(tx: float, ty: float, tz: float, device: torch.device) -> torch.Tensor:
    """World-to-sensor view matrix: sensor placed at (tx,ty,tz), looking forward (+X)."""
    # Simple translation only (no rotation), i.e. viewmat = [I | -t]
    viewmat = torch.eye(4, dtype=torch.float32, device=device)
    viewmat[0, 3] = -tx
    viewmat[1, 3] = -ty
    viewmat[2, 3] = -tz
    return viewmat.unsqueeze(0)   # (1, 4, 4)


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LiDAR depth rendering from 3DGS checkpoint")
    parser.add_argument("--ckpt",   help="Path to simple_trainer .pt checkpoint")
    parser.add_argument("--ply",    help="Path to 3DGS .ply file (alternative to --ckpt)")
    parser.add_argument("--sensor",   choices=["generic", "waymo"], default="generic")
    parser.add_argument("--height",   type=float, default=0.0,
                        help="Sensor height above scene origin (m)")
    parser.add_argument("--tx",       type=float, default=None,
                        help="Sensor X position in world frame")
    parser.add_argument("--ty",       type=float, default=None,
                        help="Sensor Y position in world frame")
    parser.add_argument("--auto-place", action="store_true",
                        help="Auto-place sensor outside scene bounding box (overrides --tx/--ty/--height)")
    parser.add_argument("--col-step", type=int, default=1,
                        help="Subsample every Nth column (default: 1 = all)")
    parser.add_argument("--save-pcd", action="store_true",
                        help="Save point cloud to PCD file")
    parser.add_argument("--pcd-out", default="lidar_render.pcd",
                        help="Output PCD path (default: lidar_render.pcd)")
    parser.add_argument("--pcd-ascii", action="store_true",
                        help="Write ASCII PCD instead of binary (larger but human-readable)")
    args = parser.parse_args()
    if not args.ckpt and not args.ply:
        parser.error("provide --ckpt or --ply")
    if args.ckpt and args.ply:
        parser.error("provide only one of --ckpt / --ply")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── 1. Load scene ──────────────────────────────────────────────────────
    if args.ckpt:
        print(f"Loading checkpoint: {args.ckpt}")
        means, quats, scales, opacities, colors = _load_ckpt(args.ckpt, device)
        scene_name = Path(args.ckpt).stem
    else:
        print(f"Loading PLY: {args.ply}")
        means, quats, scales, opacities, colors = _load_ply(args.ply, device)
        scene_name = Path(args.ply).stem
    print(f"  Gaussians: {means.shape[0]:,}")
    bbox_min = means.min(dim=0).values.cpu().tolist()
    bbox_max = means.max(dim=0).values.cpu().tolist()
    print(f"  Scene bbox X: [{bbox_min[0]:.2f}, {bbox_max[0]:.2f}]  "
          f"Y: [{bbox_min[1]:.2f}, {bbox_max[1]:.2f}]  "
          f"Z: [{bbox_min[2]:.2f}, {bbox_max[2]:.2f}]")
    center = [(bbox_min[i] + bbox_max[i]) / 2 for i in range(3)]
    radius = max(bbox_max[i] - bbox_min[i] for i in range(3)) / 2
    print(f"  Scene center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})  "
          f"approx radius: {radius:.2f}")

    # Auto-place: put sensor at scene_center + 2*radius along X, at mid-height
    if args.auto_place:
        args.tx     = center[0] + radius * 2.0
        args.ty     = center[1]
        args.height = center[2] + radius * 0.5
        print(f"  [auto-place] Sensor → X={args.tx:.2f}  Y={args.ty:.2f}  Z={args.height:.2f}")
    else:
        if args.tx is None:     args.tx = 0.0
        if args.ty is None:     args.ty = 0.0
        if args.tx == 0.0 and args.ty == 0.0 and args.height == 0.0:
            print("  [hint] Sensor is at origin — use --auto-place or "
                  "--tx {:.1f} to place it outside the scene".format(center[0] + radius * 2))

    # ── 2. Build LiDAR sensor ──────────────────────────────────────────────
    print(f"Building LiDAR sensor ({args.sensor}) …")
    lidar_ext = _build_lidar_params(args.sensor, device)

    n_rows = lidar_ext.n_rows
    n_cols = lidar_ext.n_columns

    # Column subsampling is applied post-render (lidar renderer always outputs n_cols columns)
    n_cols_render = (n_cols + args.col_step - 1) // args.col_step if args.col_step > 1 else n_cols
    if args.col_step > 1:
        print(f"  Subsampling columns (post-render): {n_cols} → {n_cols_render} (step={args.col_step})")

    # ── 3. Set up viewmat and Ks ───────────────────────────────────────────
    viewmat = _make_viewmat(args.tx, args.ty, args.height, device)  # (1, 4, 4)

    # For lidar the projection is handled entirely by the lidar model;
    # Ks is unused — pass identity (same as av_trainer.py)
    Ks = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0)  # (1, 3, 3)

    # ── 4. Rasterize ───────────────────────────────────────────────────────
    print("Rasterizing …")
    with torch.no_grad():
        renders, alphas, _meta = gsplat.rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat,
            Ks=Ks,
            width=n_cols,          # lidar renderer overrides width from lidar_coeffs.n_columns
            height=n_rows,
            camera_model="lidar",
            lidar_coeffs=lidar_ext,
            with_ut=True,             # required for lidar camera model
            with_eval3d=True,         # required for hit-distance mode
            packed=False,             # packed mode not supported with UT
            render_mode="RGB-Ed",     # expected hit-distance (more stable than RGB-d)
            global_z_order=False,     # spinning LiDAR has no single camera-Z axis
            near_plane=0.1,
            far_plane=200.0,
        )

    # renders: (1, H, n_cols, 4) — last channel is radial hit-distance
    render = renders[0].cpu().numpy()        # (H, n_cols, 4)
    rgb_img   = render[:, :, :3].clip(0, 1)  # (H, n_cols, 3)
    depth_img = render[:, :, 3]              # (H, n_cols)  radial distance in scene units
    alpha_img = alphas[0].squeeze(-1).cpu().numpy()  # (H, n_cols)

    # Post-render column subsampling
    if args.col_step > 1:
        rgb_img   = rgb_img[:, ::args.col_step, :]
        depth_img = depth_img[:, ::args.col_step]
        alpha_img = alpha_img[:, ::args.col_step]

    print(f"  Depth range (hit pixels): "
          f"{depth_img[depth_img > 0].min():.2f} – {depth_img.max():.2f} m")

    # ── 5. Convert depth map → 3D point cloud ─────────────────────────────
    # Use the azimuth / elevation tables to reconstruct rays
    row_elev = lidar_ext.row_elevations_rad.cpu().numpy()   # (n_rows,)
    col_az   = lidar_ext.column_azimuths_rad.cpu().numpy()  # (n_cols_orig,)

    if args.col_step > 1:
        col_az = col_az[::args.col_step]

    row_offsets_np = lidar_ext.row_azimuth_offsets_rad.cpu().numpy()  # (n_rows,)

    # Build ray directions from elevation + azimuth tables
    elev_grid = row_elev[:, None]   # (H, 1)
    az_grid   = col_az[None, :] + row_offsets_np[:, None]  # (H, W)

    dx = np.cos(elev_grid) * np.cos(az_grid)
    dy = np.cos(elev_grid) * np.sin(az_grid)
    dz = np.sin(elev_grid) * np.ones_like(az_grid)

    # world_point = sensor_origin + ray_dir * hit_distance
    depth = depth_img   # (H, W) — radial range
    hit   = (depth > 0) & (alpha_img > 0.05)

    pts_x = args.tx     + (dx * depth)[hit]
    pts_y = args.ty     + (dy * depth)[hit]
    pts_z = args.height + (dz * depth)[hit]
    pts_rgb = rgb_img[hit]       # (M, 3)

    print(f"  Point cloud: {hit.sum():,} points")

    # ── 5b. Save PCD ───────────────────────────────────────────────────────
    if args.save_pcd:
        pcd_path = Path(args.pcd_out)
        _save_pcd(pcd_path, pts_x, pts_y, pts_z, pts_rgb,
                  binary=not args.pcd_ascii)
        print(f"  PCD saved → {pcd_path.resolve()}")

    # ── 6. Visualise ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(
        f"LiDAR rendering  [{args.sensor}]  ·  {scene_name}", fontsize=12
    )

    # Left: depth image (range image)
    ax0 = fig.add_subplot(1, 3, 1)
    im = ax0.imshow(depth_img, cmap="turbo", origin="upper", aspect="auto",
                    vmin=0, vmax=np.percentile(depth_img[depth_img > 0], 95) if hit.any() else 1)
    plt.colorbar(im, ax=ax0, label="depth (m)")
    ax0.set_title("Range image")
    ax0.set_xlabel("column"); ax0.set_ylabel("row")

    # Middle: RGB image from lidar viewpoint (mostly dark, but shows coverage)
    ax1 = fig.add_subplot(1, 3, 2)
    ax1.imshow(rgb_img, origin="upper", aspect="auto")
    ax1.set_title("RGB (lidar view)")
    ax1.set_xlabel("column"); ax1.set_ylabel("row")

    # Right: top-down point cloud (XY), coloured by height Z
    ax2 = fig.add_subplot(1, 3, 3)
    sc = ax2.scatter(pts_x, pts_y, c=pts_z, cmap="viridis", s=0.3, alpha=0.6)
    plt.colorbar(sc, ax=ax2, label="Z (m)")
    ax2.set_title("Top-down point cloud")
    ax2.set_xlabel("X (m)"); ax2.set_ylabel("Y (m)")
    ax2.set_aspect("equal")

    plt.tight_layout()
    out_path = Path("lidar_render_demo.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved → {out_path.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
