import inspect
from types import SimpleNamespace

import pytest
import torch

from gsplat.experimental.render.components.gaussian_inference_renderer import (
    GaussianInferenceRenderer,
)
from gsplat.experimental.render.kernels.gaussian_inference_ops import (
    create_native_gaussian_inference_renderer,
)
from gsplat.cuda._wrapper import FThetaPolynomialType


def test_ftheta_coefficients_are_explicit_renderer_parameter():
    assert "ftheta_coeffs" in inspect.signature(
        GaussianInferenceRenderer.render
    ).parameters


def coefficients(**overrides):
    values = dict(
        reference_poly=FThetaPolynomialType.ANGLE_TO_PIXELDIST,
        angle_to_pixeldist_poly=(0, 80, 0, 0, 0, 0),
        linear_cde=(1, 0, 0),
        max_angle=1.5,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("overrides", [
    {"reference_poly": FThetaPolynomialType.PIXELDIST_TO_ANGLE},
    {"angle_to_pixeldist_poly": (1, 80, 0, 0, 0, 0)},
    {"angle_to_pixeldist_poly": (0, float("nan"), 0, 0, 0, 0)},
    {"linear_cde": (0, 0, 0)},
    {"max_angle": 0},
])
def test_invalid_coefficients(overrides):
    with pytest.raises(ValueError):
        GaussianInferenceRenderer._normalize_ftheta_coeffs(3, coefficients(**overrides))


def test_coefficient_model_contract():
    with pytest.raises(ValueError):
        GaussianInferenceRenderer._normalize_ftheta_coeffs(3, None)
    with pytest.raises(ValueError):
        GaussianInferenceRenderer._normalize_ftheta_coeffs(0, coefficients())
    assert GaussianInferenceRenderer._normalize_ftheta_coeffs(0, None) == []
    assert GaussianInferenceRenderer._normalize_ftheta_coeffs(3, coefficients()) == [
        0, 80, 0, 0, 0, 0, 1, 0, 0, 1.5
    ]


def make_native_scene(sh_degree, count=256):
    generator = torch.Generator(device="cuda").manual_seed(123)
    means = torch.randn((3, count), generator=generator, device="cuda")
    means[2] = means[2].abs() + 2
    packed = torch.zeros((count, 8), dtype=torch.float16, device="cuda")
    packed[:, 0] = 1
    packed[:, 4:7] = 0.02
    packed[:, 7] = 0.8
    color_shape = (count, 4) if sh_degree < 0 else (count, (sh_degree + 1) ** 2, 3)
    colors = torch.rand(color_shape, generator=generator, device="cuda", dtype=torch.float16)
    if 0 <= sh_degree < 3:
        colors = colors.float()
    native = create_native_gaussian_inference_renderer(means, packed, colors, sh_degree, 0)
    return native, means, packed, colors


def render_native(scene, camera_model, ftheta=None, indices=None, intrinsics=None, **options):
    native, means, packed, colors = scene
    if intrinsics is None:
        intrinsics = torch.tensor([[80, 0, 127.5], [0, 80, 127.5], [0, 0, 1]], device="cuda")
    output = torch.empty((1, 256, 256, 4), dtype=torch.float16, device="cuda")
    with torch.inference_mode():
        native.render(
            means, packed, colors, torch.eye(4, device="cuda"), intrinsics,
            256, 256, 8, 0.01, 100, 0, 0.3,
            -1 if colors.ndim == 2 else int(colors.shape[1] ** 0.5) - 1,
            0, None, output, camera_model=camera_model, active_indices=indices,
            **({"ftheta_coeffs": ftheta} if ftheta is not None else {}),
            **options,
        )
    return output


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("sh_degree", [-1, 1, 3])
@pytest.mark.parametrize("indexed", [False, True])
def test_equidistant_ftheta_matches_fisheye(sh_degree, indexed):
    scene = make_native_scene(sh_degree)
    indices = torch.arange(0, 256, 2, device="cuda", dtype=torch.int32) if indexed else None
    params = GaussianInferenceRenderer._normalize_ftheta_coeffs(3, coefficients())
    actual = render_native(scene, 3, params, indices)
    intrinsics = torch.tensor([[80, 0, 128], [0, 80, 128], [0, 0, 1]], device="cuda", dtype=torch.float32)
    expected = render_native(scene, 2, indices=indices, intrinsics=intrinsics)
    assert torch.isfinite(actual).all()
    assert actual[..., 3].min() < 0.9
    torch.testing.assert_close(actual[..., 3], expected[..., 3], atol=0.015, rtol=0.02)
    mse = (actual[..., :3].float() - expected[..., :3].float()).square().mean()
    assert mse < 1e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("point", [(0, 0, 3), (1, 0.5, 2), (-1, 0.7, 2)])
def test_polynomial_and_linear_projection_center(point):
    scene = make_native_scene(-1, count=1)
    scene[1][:, 0] = torch.tensor(point, device="cuda")
    params = [0, 80, 2, -1, 0.2, 0, 1.1, 0.05, -0.03, 1.5]
    image = render_native(scene, 3, params)
    point_tensor = torch.tensor(point, dtype=torch.float64)
    radial = point_tensor[:2].norm()
    theta = torch.atan2(radial, point_tensor[2])
    radius = sum(value * theta ** degree for degree, value in enumerate(params[:6]))
    offset = point_tensor[:2] / radial * radius if radial > 0 else torch.zeros_like(point_tensor[:2])
    linear = torch.tensor([[1.1, 0.05], [-0.03, 1]], dtype=torch.float64)
    expected = linear @ offset + 128
    alpha = 1 - image[0, ..., 3].float()
    peak = alpha.argmax().item()
    actual = torch.tensor([peak % 256 + 0.5, peak // 256 + 0.5])
    assert alpha.max() > 0.1
    torch.testing.assert_close(actual.double(), expected, atol=0.75, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("point", [(0, 0, -2), (3, 0, 1)])
def test_ftheta_culls_invalid_rays(point):
    scene = make_native_scene(-1, count=1)
    scene[1][:, 0] = torch.tensor(point, device="cuda")
    image = render_native(scene, 3, [0, 80, 0, 0, 0, 0, 1, 0, 0, 0.5])
    assert torch.all(image[..., 3] == 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("sh_degree", [-1, 1, 3])
@pytest.mark.parametrize("indexed", [False, True])
def test_ftheta_screen_cull_and_distance_restore(sh_degree, indexed):
    scene = make_native_scene(sh_degree, count=1)
    scene[1][:, 0] = torch.tensor([0, 0, 3], device="cuda")
    scene[2][:, 4:7] = 0.5
    indices = torch.zeros(1, device="cuda", dtype=torch.int32) if indexed else None
    params = [0, 80, 0, 0, 0, 0, 1, 0, 0, 1.5]
    baseline = render_native(scene, 3, params, indices)
    assert baseline[..., 3].min() < 0.9
    culled = render_native(scene, 3, params, indices,
                           max_screen_radius=5, max_screen_radius_dist=3)
    assert torch.all(culled[..., 3] == 1)
    restored = render_native(scene, 3, params, indices,
                             max_screen_radius=5, max_screen_radius_dist=2)
    torch.testing.assert_close(restored, baseline, rtol=0, atol=0)
    unlimited = render_native(scene, 3, params, indices,
                              max_screen_radius=5, max_screen_radius_dist=0)
    assert torch.all(unlimited[..., 3] == 1)
    disabled = render_native(scene, 3, params, indices,
                             max_screen_radius=0, max_screen_radius_dist=3)
    torch.testing.assert_close(disabled, baseline, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_stateful_wrapper_ftheta_render():
    from gsplat.scene import GaussianInferenceScene
    from gsplat.scene.sh_compression import SHCompressionMode
    from gsplat.cuda._wrapper import FThetaCameraDistortionParameters

    _, means, packed, colors = make_native_scene(-1)
    scene = GaussianInferenceScene("ftheta_test")
    scene.put("points", dict(
        means_planar=means, qso_packed=packed, colors_packed=colors,
        sh_degree=-1, sh_compression_mode=SHCompressionMode.NONE,
    ))
    params = FThetaCameraDistortionParameters(
        reference_poly=FThetaPolynomialType.ANGLE_TO_PIXELDIST,
        angle_to_pixeldist_poly=(0, 80, 0, 0, 0, 0),
        pixeldist_to_angle_poly=(0, 1 / 80, 0, 0, 0, 0),
        linear_cde=(1, 0, 0), max_angle=1.5,
    )
    with GaussianInferenceRenderer(scene) as renderer, torch.inference_mode():
        assert renderer._supports_ftheta_screen_cull()
        result = renderer.render(
            viewmat=torch.eye(4, device="cuda"),
            K=torch.tensor([[80, 0, 127.5], [0, 80, 127.5], [0, 0, 1]], device="cuda"),
            width=256, height=256, camera_model="ftheta", ftheta_coeffs=params,
            max_screen_radius=1000, max_screen_radius_dist=100,
        )
        assert result.frame.shape == (1, 256, 256, 4)
        assert torch.isfinite(result.frame).all()
        assert result.frame[..., 3].min() < 0.9