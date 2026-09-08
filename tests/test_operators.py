import math

import numpy as np
import pytest
import torch

from spenpy.core import (
    EncodingOperator,
    SPEN180Protocol,
    XSPENProtocol,
    coil_sensitivities,
)
from spenpy.sim import even_odd_phase


@pytest.mark.parametrize("kind", [SPEN180Protocol, XSPENProtocol])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_linearity_adjoint_and_independent_non_square_grids(kind, dtype):
    p = kind(acquisition_shape=(7, 6), r_value=5)
    coils = coil_sensitivities((11, 9), 3, dtype=dtype)
    phase = even_odd_phase(p, (11, 9), constant_rad=0.4, linear_rad=0.3)
    mask = torch.ones(7, 6)
    mask[2, :] = 0
    op = EncodingOperator(
        p, (11, 9), coil_maps=coils, dtype=dtype, phase_map_rad=phase, sample_mask=mask
    )
    x = torch.randn(2, 11, 9, dtype=dtype)
    z = torch.randn_like(x)
    y = torch.randn(2, 7, 6, 3, dtype=dtype)
    tol = 3e-6 if dtype == torch.complex64 else 1e-12
    torch.testing.assert_close(
        op((1 + 0.3j) * x - 0.2 * z),
        (1 + 0.3j) * op(x) - 0.2 * op(z),
        rtol=tol,
        atol=tol,
    )
    left = torch.vdot(op(x).flatten(), y.flatten())
    right = torch.vdot(x.flatten(), op.adjoint(y).flatten())
    torch.testing.assert_close(left, right, rtol=tol, atol=tol)
    assert op(x).shape == (2, 7, 6, 3)
    assert torch.count_nonzero(op(x)[:, 2]) == 0


@pytest.mark.parametrize("kind", [SPEN180Protocol, XSPENProtocol])
def test_image_gradient_matches_adjoint_and_finite_differences(kind):
    op = EncodingOperator(
        kind(acquisition_shape=(4, 3), r_value=2), (5, 4), dtype=torch.complex128
    )
    x = torch.randn(5, 4, dtype=torch.complex128, requires_grad=True)
    y = torch.randn(4, 3, 1, dtype=torch.complex128)
    loss = 0.5 * (op(x) - y).abs().square().sum()
    actual = torch.autograd.grad(loss, x)[0]
    torch.testing.assert_close(actual, op.adjoint(op(x) - y), rtol=1e-12, atol=1e-12)
    assert torch.autograd.gradcheck(op, (x,), fast_mode=True)


def test_xspen_matches_explicit_uniform_auxiliary_integral():
    p = XSPENProtocol(acquisition_shape=(6, 5), r_value=4)
    op = EncodingOperator(p, (9, 7), quadrature=1, dtype=torch.complex128)
    nz = 8192
    z = ((torch.arange(nz, dtype=torch.float64) + 0.5) / nz - 0.5) * p.slice_thickness_m
    q = (
        p.cross_term_cycles_m2 * op.y_m[None]
        + p.gamma_hz_t * p.gz_t_m * op.sample_times_s[:, None]
    )
    explicit = torch.exp(2j * math.pi * q[..., None] * z).mean(-1) * (6 / 9)
    torch.testing.assert_close(op.encoding_kernel, explicit, atol=1e-7, rtol=1e-7)
    expected_focus = (torch.arange(1, 7, dtype=torch.float64) / 6 - 0.5) * p.fov_m[0]
    torch.testing.assert_close(op.focus_positions_m, expected_focus)


def test_spen_pixel_integral_converges_to_fine_independent_reference():
    p = SPEN180Protocol(acquisition_shape=(6, 4), r_value=3)
    op = EncodingOperator(p, (16, 8), quadrature=64, dtype=torch.complex128)
    y = op.y_m.numpy()
    offsets = (np.arange(8192) + 0.5) / 8192 - 0.5
    yq = y[:, None] + offsets * p.fov_m[0] / 16
    focus = (np.arange(6) / 6 - 0.5) * p.fov_m[0]
    expected = (
        np.exp(
            1j
            * p.phase_coefficient_rad_m2
            * (yq[None] ** 2 - 2 * focus[:, None, None] * yq[None])
        ).mean(-1)
        * math.sqrt(6)
        / 16
    )
    np.testing.assert_allclose(
        op.encoding_kernel.numpy(), expected, atol=1e-5, rtol=1e-4
    )


def test_default_spen_integration_resolves_high_phase_variation():
    p = SPEN180Protocol(acquisition_shape=(96, 96), r_value=150)
    op = EncodingOperator(p, (64, 64), dtype=torch.complex128)
    # Independently integrate the most rapidly oscillating corner pixels.
    offsets = (np.arange(32768) + 0.5) / 32768 - 0.5
    for m, yidx in ((0, 63), (95, 0), (48, 32)):
        y = op.y_m[yidx].item() + offsets * p.fov_m[0] / 64
        focus = (m / 96 - 0.5) * p.fov_m[0]
        value = (
            np.exp(1j * p.phase_coefficient_rad_m2 * (y * y - 2 * focus * y)).mean()
            * math.sqrt(96)
            / 64
        )
        np.testing.assert_allclose(
            op.encoding_kernel[m, yidx].item(), value, atol=2e-10, rtol=2e-7
        )


def test_xspen_off_resonance_matches_legacy_phase_and_rf_masks():
    p = XSPENProtocol(acquisition_shape=(4, 3), r_value=4)
    field = torch.tensor([[40.0, -20.0, 0.0]] * 3, dtype=torch.float64)
    op = EncodingOperator(
        p, (3, 3), quadrature=1, b0_hz=field, n_z=64, dtype=torch.complex128
    )
    z = ((np.arange(64) + 0.5) / 64 - 0.5) * p.slice_thickness_m
    expected = np.zeros((4, 3, 3), dtype=np.complex128)
    for m, t in enumerate(op.sample_times_s.numpy()):
        for j, y in enumerate(op.y_m.numpy()):
            gy = p.gamma_hz_t * p.gy_t_m * y
            for ix in range(3):
                w = p.gamma_hz_t * p.gz_t_m * z + field[j, ix].item()
                mask = (
                    (abs(w) <= (1 - p.beta) * p.chirp_bandwidth_hz / 2)
                    & (abs(w - gy) <= p.chirp_bandwidth_hz / 2)
                    & (abs(w + gy) <= p.chirp_bandwidth_hz / 2)
                )
                phi = (-4 * p.chirp_duration_s / p.chirp_bandwidth_hz * gy + t) * w
                expected[m, j, ix] = (mask * np.exp(2j * np.pi * phi)).mean() * 4 / 3
    np.testing.assert_allclose(
        op.encoding_kernel.numpy(), expected, rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("kind", [SPEN180Protocol, XSPENProtocol])
def test_field_decay_and_phase_preserve_adjoint(kind):
    p = kind(acquisition_shape=(5, 4), r_value=3)
    opts = {
        "b0_hz": torch.linspace(-10, 10, 6).expand(7, 6),
        "t2_s": 0.08,
        "decay_times_s": torch.linspace(0, 0.02, 5),
    }
    if kind is SPEN180Protocol:
        opts["effective_times_s"] = torch.linspace(-0.01, 0.01, 5)
    op = EncodingOperator(p, (7, 6), dtype=torch.complex128, **opts)
    x = torch.randn(7, 6, dtype=torch.complex128)
    y = torch.randn(5, 4, 1, dtype=torch.complex128)
    torch.testing.assert_close(
        torch.vdot(op(x).flatten(), y.flatten()),
        torch.vdot(x.flatten(), op.adjoint(y).flatten()),
        atol=1e-12,
        rtol=1e-12,
    )


def test_coils_are_encoded_before_measurement():
    p = SPEN180Protocol(acquisition_shape=(8, 6), r_value=4)
    maps = coil_sensitivities((10, 9), 2)
    multi = EncodingOperator(p, (10, 9), coil_maps=maps)
    single = EncodingOperator(p, (10, 9))
    x = torch.randn(10, 9, dtype=torch.complex64)
    for c in range(2):
        torch.testing.assert_close(multi(x)[..., c], single(x * maps[..., c])[..., 0])


def test_missing_spen_refocusing_timing_is_rejected():
    with pytest.raises(ValueError, match="effective_times_s"):
        EncodingOperator(SPEN180Protocol(acquisition_shape=(4, 4)), b0_hz=20.0)


def test_whitened_operator_adjoint():
    from spenpy.core import CoilWhitenedOperator

    base = EncodingOperator(
        SPEN180Protocol(acquisition_shape=(6, 5), r_value=3),
        (8, 7),
        coil_maps=coil_sensitivities((8, 7), 2, dtype=torch.complex128),
        dtype=torch.complex128,
    )
    covariance = torch.tensor(
        [[2, 0.3 + 0.2j], [0.3 - 0.2j, 1]], dtype=torch.complex128
    )
    op = CoilWhitenedOperator(base, covariance)
    x = torch.randn(8, 7, dtype=torch.complex128)
    y = torch.randn(6, 5, 2, dtype=torch.complex128)
    torch.testing.assert_close(
        torch.vdot(op(x).flatten(), y.flatten()),
        torch.vdot(x.flatten(), op.adjoint(y).flatten()),
        atol=1e-12,
        rtol=1e-12,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("kind", [SPEN180Protocol, XSPENProtocol])
def test_cuda_forward_adjoint_and_image_gradient(kind):
    p = kind(acquisition_shape=(12, 9), r_value=7)
    op = EncodingOperator(p, (18, 15), coil_maps=coil_sensitivities((18, 15), 4)).to(
        "cuda"
    )
    x = torch.randn(2, 18, 15, device="cuda", dtype=torch.complex64, requires_grad=True)
    op(x).abs().square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert op.adjoint(op(x)).device.type == "cuda"
