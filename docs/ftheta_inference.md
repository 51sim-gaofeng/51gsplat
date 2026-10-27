# F-Theta Inference

`GaussianInferenceRenderer.render` supports `camera_model="ftheta"` with
`ftheta_coeffs=FThetaCameraDistortionParameters(...)`. Rebuild the experimental
CUDA extension and install the updated Python wrapper together before using it.
The stateless inference API is unchanged.

```python
with torch.inference_mode():
    result = renderer.render(
        viewmat=viewmat, K=K, width=width, height=height,
        camera_model="ftheta", ftheta_coeffs=ftheta_params,
    )
rgb = result.frame[..., :3]
alpha = 1 - result.frame[..., 3:4]
```

The fast path reuses packed scenes, persistent buffers, fused SH evaluation,
indexed projection, and macro-tile rasterization. It evaluates the forward
polynomial and its analytic Jacobian rather than using UT projection. Parameters
are passed as host values; ftheta coefficients do not require a per-frame device
readback. Reuse the renderer across frames to avoid repacking the scene.

Optional `max_screen_radius` (pixels) fuses large-splat culling into the same
ftheta EWA projection. Zero disables it. `max_screen_radius_dist` limits culling
to camera-space Z at or below that distance; nonpositive values mean unlimited
distance. Neither option mutates packed scene opacity, so moving the camera or
changing the threshold cannot leave stale masks. Radii are the rounded EWA
bounding-box radii; the maximum of X/Y is compared to the threshold. This is not
UT-radius equivalence. Other camera models reject an enabled maximum radius.

Install the updated wrapper and rebuilt experimental extension together. The
native module exposes `ftheta_screen_cull`; the wrapper's
`_supports_ftheta_screen_cull()` checks it so applications can retain legacy
culling with old binaries. Existing callers that omit the options are unchanged.

## Constraints

- Only `ANGLE_TO_PIXELDIST` reference polynomials are supported. The six forward
  coefficients must have a zero constant term for a continuous optical axis.
- `linear_cde` must be invertible, all values finite, and `max_angle` positive.
- Intrinsic focal lengths are encoded in the polynomial, not taken from K.
  K supplies the principal point. The existing ftheta half-pixel offset is retained.
- Centers behind the camera or at/beyond `max_angle` are culled. Near/far clipping
  and depth sorting use camera-space Z, matching the existing UT projection.
- Output is float16 RGBT (T is transmittance), not RGB-alpha. Depth, training,
  rolling shutter, and external distortion are not added to inference.
- First-order EWA is not identical to UT or 3D evaluation. Large Gaussians,
  strong distortion, and field-of-view boundaries need scene-specific validation.
- The parent application's `render.ftheta_native_inference` switch enables
  opt-in 3dgs/4dgs routing after installation (see its `docs/FTHETA_INFERENCE.md`).
  Ftheta remains excluded from pinhole LOD selection. Do not reinterpret it as fisheye.

## Verification

From the gsplat directory, with the updated package installed:

```powershell
conda activate 4dgs
python -m pytest tests/experimental/render/test_ftheta_inference.py --noconftest -q
python examples/benchmark_ftheta_inference.py
```

The benchmark optionally accepts `--native-extension <absolute-path-to-pyd>` to
test a separately built native extension. The report identifies the core gsplat
package and explicit extension path. It compares UT + 2D rasterization (tile 16)
with stateful EWA inference (tile 8), including host dispatch and synchronization
but excluding packing. Two rounds reverse execution order, with 10 warmups and
30 samples per path. Synthetic correctness gates: RGB PSNR >= 35 dB, mean alpha
error <= 0.01, finite output.

Measured on RTX 5070, PyTorch 2.8.0+cu129, 100,000 synthetic Gaussians, 960x540:

| Round | UT median ms | Inference median ms | Speedup |
| --- | ---: | ---: | ---: |
| 1 | 0.5344 | 0.2575 | 2.07x |
| 2 | 0.6406 | 0.2996 | 2.14x |

RGB PSNR: 65.53 dB. Mean/max alpha error: 0.000448 / 0.01078.
These results used the installed core UT renderer and the newly compiled local
inference extension. They are not a same-revision full-package comparison, a
real-scene quality guarantee, or application-level FPS measurements.