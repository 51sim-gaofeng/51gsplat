"""Convert a gsplat checkpoint (.pt) to a standard 3DGS PLY file.

Usage:
    python examples/ckpt_to_ply.py --ckpt results/garden/ckpts/ckpt_29999_rank0.pt \
                                    --out garden.ply
"""

import argparse
import torch
from gsplat.exporter import export_splats


def main():
    parser = argparse.ArgumentParser(description="Convert gsplat ckpt to PLY")
    parser.add_argument("--ckpt", required=True, help="Path to ckpt_*.pt file")
    parser.add_argument("--out", required=True, help="Output .ply path")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.ckpt}")
    data = torch.load(args.ckpt, map_location="cpu")

    # ckpt layout: {"splats": {...}, "step": int, ...}
    if "splats" in data:
        splats = data["splats"]
    else:
        # fallback: the dict itself may be the splats
        splats = data

    required = {"means", "scales", "quats", "opacities"}
    missing = required - set(splats.keys())
    if missing:
        raise KeyError(f"Checkpoint is missing required splat keys: {missing}")

    # sh0 / shN may not exist in all ckpt variants
    sh0 = splats.get("sh0", None)
    shN = splats.get("shN", None)
    colors = splats.get("colors", None)

    if sh0 is None and colors is not None:
        # some trainers store raw RGB; convert to SH DC term
        from gsplat.utils import rgb_to_sh
        sh0 = rgb_to_sh(torch.sigmoid(colors).unsqueeze(1))
        shN = torch.empty([sh0.shape[0], 0, 3])
    elif sh0 is None:
        raise KeyError("Checkpoint has neither 'sh0' nor 'colors'; cannot export colors.")

    if shN is None:
        shN = torch.empty([sh0.shape[0], 0, 3])

    print(f"Gaussian count : {splats['means'].shape[0]:,}")
    print(f"SH degree      : {sh0.shape[1] + shN.shape[1] // 3}")

    export_splats(
        means=splats["means"],
        scales=splats["scales"],
        quats=splats["quats"],
        opacities=splats["opacities"],
        sh0=sh0,
        shN=shN,
        format="ply",
        save_to=args.out,
    )
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
