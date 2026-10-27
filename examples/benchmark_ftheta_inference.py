"""Compare ftheta UT rasterization with the stateful EWA inference renderer."""

import argparse
import importlib.util
import json
import math
from pathlib import Path
import statistics
import time

import torch


def measure(render, warmup, iterations):
    for _ in range(warmup):
        render()
    torch.cuda.synchronize()
    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        render()
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000)
    median = statistics.median(timings)
    return {
        "median_ms": median,
        "p95_ms": sorted(timings)[math.ceil(0.95 * len(timings)) - 1],
        "fps": 1000 / median,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-extension", type=Path)
    parser.add_argument("--count", type=int, default=100000)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    if min(args.count, args.width, args.height, args.iterations) <= 0 or args.warmup < 0:
        parser.error("counts and dimensions must be positive; warmup must be nonnegative")

    if args.native_extension:
        spec = importlib.util.spec_from_file_location(args.native_extension.stem, args.native_extension)
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        factory = native.GaussianInferenceRenderer
    else:
        from gsplat.experimental.render.kernels.gaussian_inference_ops import (
            create_native_gaussian_inference_renderer,
        )
        factory = create_native_gaussian_inference_renderer

    import gsplat
    from gsplat.rendering import rasterization
    from gsplat.cuda._wrapper import FThetaCameraDistortionParameters, FThetaPolynomialType

    generator = torch.Generator(device="cuda").manual_seed(42)
    means = torch.randn((args.count, 3), generator=generator, device="cuda")
    means[:, :2] *= 3
    means[:, 2] = means[:, 2].abs() * 3 + 3
    packed = torch.zeros((args.count, 8), device="cuda", dtype=torch.float16)
    packed[:, 0] = 1
    packed[:, 4:7] = 0.01
    packed[:, 7] = 0.5
    colors = torch.rand((args.count, 4), generator=generator, device="cuda", dtype=torch.float16)
    means_planar = means.T.contiguous()
    renderer = factory(means_planar, packed, colors, -1, 0)
    focal = min(args.width, args.height) / 2
    forward = (0, focal, 0, focal * 0.01, 0, 0)
    params = FThetaCameraDistortionParameters(
        reference_poly=FThetaPolynomialType.ANGLE_TO_PIXELDIST,
        angle_to_pixeldist_poly=forward,
        pixeldist_to_angle_poly=(0, 1 / focal, 0, 0, 0, 0),
        linear_cde=(1.01, 0.005, -0.003), max_angle=1.5,
    )
    native_params = [*forward, 1.01, 0.005, -0.003, 1.5]
    intrinsics = torch.tensor(
        [[focal, 0, args.width / 2 - 0.5], [0, focal, args.height / 2 - 0.5], [0, 0, 1]],
        device="cuda", dtype=torch.float32,
    )
    viewmat = torch.eye(4, device="cuda")
    output = torch.empty((1, args.height, args.width, 4), device="cuda", dtype=torch.float16)
    quats = packed[:, :4].float().contiguous()
    scales = packed[:, 4:7].float().contiguous()
    opacity = packed[:, 7].float().contiguous()
    rgb = colors[:, :3].float().contiguous()

    def old_render():
        return rasterization(
            means, quats, scales, opacity, rgb, viewmat[None], intrinsics[None],
            args.width, args.height, packed=False, with_ut=True, with_eval3d=False,
            camera_model="ftheta", ftheta_coeffs=params, tile_size=16,
        )

    def fast_render():
        renderer.render(
            means_planar, packed, colors, viewmat, intrinsics,
            args.width, args.height, 8, 0.01, 1e10, 0, 0.3, -1, 0, None, output,
            camera_model=3, ftheta_coeffs=native_params,
        )
        return output

    with torch.inference_mode():
        reference, reference_alpha, _ = old_render()
        fast = fast_render().float()
        mse = (reference - fast[..., :3]).square().mean().item()
        alpha_error = (reference_alpha - (1 - fast[..., 3:4])).abs()
        rounds = []
        for round_index in range(2):
            order = [("ut", old_render), ("inference", fast_render)]
            if round_index % 2:
                order.reverse()
            rounds.append({name: measure(render, args.warmup, args.iterations) for name, render in order})
        report = {
            "gpu": torch.cuda.get_device_name(), "gsplat": gsplat.__file__,
            "native_extension": str(args.native_extension) if args.native_extension else "installed",
            "count": args.count, "resolution": [args.width, args.height],
            "psnr_db": -10 * math.log10(max(mse, 1e-20)),
            "alpha_mean_error": alpha_error.mean().item(),
            "alpha_max_error": alpha_error.max().item(),
            "rounds": rounds,
        }
        print(json.dumps(report, indent=2))
        if not torch.isfinite(fast).all() or report["psnr_db"] < 35 or report["alpha_mean_error"] > 0.01:
            raise RuntimeError("Synthetic-scene correctness gate failed")
    renderer.release()


if __name__ == "__main__":
    main()