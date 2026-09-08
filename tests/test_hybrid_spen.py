"""Hybrid SPEN-Diff2: independent signal equations and MATLAB references."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from spenpy import HybridSPENDiff2Protocol
from spenpy.core import EncodingOperator, calc_hybrid_spen_encoding, protocol_from_dict
from spenpy.io import load_simulation, save_simulation
from spenpy.recon import (
    calc_hybrid_spen_matrices,
    reconstruct,
    reconstruct_hybrid_spen,
    reconstruct_simulated_inva,
)
from spenpy.recon.phasemap import ro_fft
from spenpy.sim import simulate, simulate_hybrid_spen_diffusion


def test_native_signal_matches_legacy_matrix_and_centered_ifft(smooth_image):
    p = HybridSPENDiff2Protocol()
    image = smooth_image((60, 64)).to(torch.complex128)
    op = EncodingOperator(p, dtype=torch.complex128)
    a = calc_hybrid_spen_matrices(60, 0.18, 60)["a"]
    expected_ro = a @ image
    expected = torch.fft.fftshift(
        torch.fft.ifft(torch.fft.ifftshift(expected_ro, dim=1), dim=1), dim=1
    )
    torch.testing.assert_close(op(image)[..., 0], expected, atol=1e-12, rtol=1e-11)
    torch.testing.assert_close(
        ro_fft(op(image))[..., 0], expected_ro, atol=1e-12, rtol=1e-11
    )


def test_pixel_approximation_against_independent_complex_integral():
    p = HybridSPENDiff2Protocol()
    # HR pixels reduce the documented second-order approximation error.
    ny = 256
    a = calc_hybrid_spen_encoding(p, ny, dtype=torch.complex128).numpy()
    length = 18.0
    coefficient = 2 * np.pi * 60 / length**2
    nodes, weights = np.polynomial.legendre.leggauss(128)
    for m, j in [(0, 0), (0, 255), (29, 127), (59, 0), (59, 255)]:
        centre = ((j + 0.5) / ny - 0.5) * length
        y = centre + nodes * length / (2 * ny)
        linear = coefficient * length - 2 * coefficient * m * length / 60
        expected = (
            np.sum(weights * np.exp(1j * (coefficient * y**2 + linear * y)))
            * length
            / (2 * ny)
        )
        assert abs(a[m, j] - expected) < 2e-8


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_hr_multicoil_adjoint_and_image_gradient(device):
    p = HybridSPENDiff2Protocol(
        acquisition_shape=(8, 6), r_value=6, spen_shift_pixels=1.3
    )
    image = torch.randn(
        11, 9, device=device, dtype=torch.complex128, requires_grad=True
    )
    sample = simulate(image, p, num_coils=3, even_odd_constant_rad=0.7)
    data = torch.randn_like(sample.measurements)
    left = torch.vdot(sample.operator(image).flatten(), data.flatten())
    right = torch.vdot(image.flatten(), sample.operator.adjoint(data).flatten())
    torch.testing.assert_close(left, right, atol=1e-11, rtol=1e-11)
    loss = 0.5 * (sample.operator(image) - data).abs().square().sum()
    grad = torch.autograd.grad(loss, image)[0]
    torch.testing.assert_close(
        grad,
        sample.operator.adjoint(sample.operator(image) - data),
        atol=1e-11,
        rtol=1e-11,
    )


def test_position_phase_is_inverse_of_scanner_correction(smooth_image):
    p = HybridSPENDiff2Protocol(acquisition_shape=(12, 10), r_value=8)
    shifted = replace(p, spen_shift_pixels=2.4)
    image = smooth_image((18, 16)).double()
    nominal = EncodingOperator(p, image.shape, dtype=torch.complex128)(image)
    actual = EncodingOperator(shifted, image.shape, dtype=torch.complex128)(image)
    correction = torch.exp(
        4j
        * torch.pi
        * p.r_value
        / 12**2
        * 2.4
        * torch.arange(1, 13, dtype=torch.float64)
    )
    torch.testing.assert_close(
        actual * correction[:, None, None], nominal, atol=1e-12, rtol=1e-11
    )


def test_known_operator_cg_recovers_image(smooth_image):
    p = HybridSPENDiff2Protocol(acquisition_shape=(20, 18), r_value=15)
    image = smooth_image(p.acquisition_shape)
    sample = simulate(image, p, num_coils=3, even_odd_constant_rad=0.7)
    result = reconstruct(
        sample.acquisition,
        sample.operator,
        regularization=1e-7,
        max_iter=120,
        rtol=1e-6,
    )
    assert float((result.image - image).norm() / image.norm()) < 0.002


def test_simulation_reconstruction_uses_hybrid_pipeline_and_known_geometry(
    smooth_image,
):
    p = HybridSPENDiff2Protocol(
        acquisition_shape=(20, 18), r_value=15, spen_shift_pixels=1.25
    )
    sample = simulate(smooth_image((30, 28)), p, num_coils=3, even_odd_constant_rad=0.7)
    actual = reconstruct_simulated_inva(sample.acquisition, sample.operator)
    expected = reconstruct_hybrid_spen(sample.acquisition, shift_pixels=1.25)
    torch.testing.assert_close(actual.coil_images, expected.coil_images)
    assert actual.metadata["shift_calibration"] == "simulation_protocol"
    assert actual.phase_map_rad.shape == (9, 18)
    assert torch.isfinite(actual.magnitude).all()


def test_diffusion_tensor_contraction_and_gradients(smooth_image):
    p = HybridSPENDiff2Protocol(acquisition_shape=(8, 6), r_value=5)
    image = smooth_image((12, 10)).double().requires_grad_()
    d = torch.diag(
        torch.tensor([0.5e-3, 1e-3, 1.5e-3], dtype=torch.float64)
    ).requires_grad_()
    sample = simulate_hybrid_spen_diffusion(
        image, p, diffusion_tensor_mm2_s=d, num_coils=2
    )
    expected = (
        image[None]
        * torch.exp(torch.tensor([0, -0.3, -0.6, -0.9], dtype=torch.float64))[
            :, None, None
        ]
    )
    torch.testing.assert_close(sample.target.real, expected)
    assert sample.acquisition.axes == ("volume", "spen", "readout", "coil")
    assert sample.acquisition.select(volume=0).data.shape == (8, 6, 2)
    sample.measurements.abs().square().sum().backward()
    assert torch.isfinite(image.grad).all() and image.grad.norm() > 0
    assert torch.isfinite(d.grad).all() and d.grad.norm() > 0


def test_spatial_b_matrices_include_off_diagonal_contributions(smooth_image):
    p = HybridSPENDiff2Protocol(acquisition_shape=(6, 4), r_value=4)
    d = torch.tensor([[1e-3, 0.2e-3, 0], [0.2e-3, 1e-3, 0], [0, 0, 1e-3]])
    b = torch.zeros(2, 8, 1, 3, 3)
    direction = torch.tensor([1.0, 1.0, 0.0]) / np.sqrt(2)
    b[1] = 600 * direction[:, None] * direction[None]
    image = smooth_image((8, 7))
    sample = simulate_hybrid_spen_diffusion(
        image, p, diffusion_tensor_mm2_s=d, b_matrices_s_mm2=b, num_coils=1
    )
    torch.testing.assert_close(sample.target[1].real, image * np.exp(-0.72))
    assert sample.acquisition.metadata["diffusion"]["model"] == "supplied_b_matrices"


@pytest.mark.parametrize(
    "options",
    [
        {"diffusivity_mm2_s": -0.001},
        {"diffusion_tensor_mm2_s": [[1, 2, 0], [0, 1, 0], [0, 0, 1]]},
        {"b_matrices_s_mm2": -torch.eye(3)[None]},
        {"b_matrices_s_mm2": torch.eye(3).to(torch.complex64)[None]},
    ],
)
def test_invalid_diffusion_inputs_are_rejected(options):
    with pytest.raises(ValueError):
        simulate_hybrid_spen_diffusion(torch.ones(8, 6), **options)


def test_hybrid_snapshot_replays_volumes_exactly(tmp_path, smooth_image):
    p = HybridSPENDiff2Protocol(acquisition_shape=(8, 6), r_value=5)
    original = simulate_hybrid_spen_diffusion(
        smooth_image((12, 10)), p, snr_db=30, num_coils=2
    )
    saved = save_simulation(original, tmp_path / "hybrid.npz")
    restored = load_simulation(saved)
    assert restored.operator.protocol == p
    assert restored.acquisition.axes == original.acquisition.axes
    torch.testing.assert_close(
        restored.operator(restored.target), original.clean_measurements, rtol=0, atol=0
    )


def test_protocol_header_mapping_and_serialization():
    free = [0] * 24
    free[12:16] = [60, 14080, 20, 4096]
    yaps = {
        "tSequenceFileName": "%CustomerSeq%\\esrs_hyb_spen_Diff2",
        "sWiPMemBlock": {"alFree": free, "adFree": [600]},
        "sSliceArray": {
            "asSlice": [{"dPhaseFOV": 180, "dReadoutFOV": 192, "dThickness": 3}]
        },
        "sKSpace": {"lPhaseEncodingLines": 60, "lBaseResolution": 64},
        "sFastImaging": {"lEchoSpacing": 470},
        "alTE": [52000],
    }
    p = HybridSPENDiff2Protocol.from_siemens_header(yaps)
    assert p.chirp_duration_s == pytest.approx(0.01408)
    assert p.ro_refocusing_r_value == 20
    assert p.nominal_b_value_s_mm2 == 600
    assert protocol_from_dict(p.to_dict()) == p
    with pytest.raises(ValueError, match="Incomplete"):
        HybridSPENDiff2Protocol.from_siemens_header(
            {"tSequenceFileName": "esrs_hyb_spen_Diff2"}
        )


@pytest.mark.integration
def test_forward_kernel_window_matches_original_matlab_inva():
    h5py = pytest.importorskip("h5py")
    path = (
        Path(__file__).resolve().parents[2]
        / "xspen_recons/outputs/xspen_raw_reconstruction.mat"
    )
    if not path.exists():
        pytest.skip("Original MATLAB reference unavailable")
    p = HybridSPENDiff2Protocol()
    a = calc_hybrid_spen_encoding(p, 60, dtype=torch.complex128)
    distance = torch.arange(60)[:, None] - torch.arange(60)[None]
    sigma = 0.3 * 60**2 / (2 * p.r_value)
    actual = (a * torch.exp(-distance.double().square() / (2 * sigma**2))).mH
    with h5py.File(path) as f:
        v = f["InvA"][()].T
        expected = torch.from_numpy(v["real"] + 1j * v["imag"])
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-10)
