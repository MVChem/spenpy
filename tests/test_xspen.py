"""Tests for the minimal xSPEN forward model and reconstruction."""

import torch

from spenpy.sim import XSPENAcquisition, XSPENParameters, XSPENSimulator


def _phantom(simulator: XSPENSimulator) -> torch.Tensor:
    y = simulator.xspen_positions_cm[:, None]
    x = simulator.readout_positions_cm[None, :]
    return (
        torch.exp(-((y + 0.65) / 0.35).square() - ((x + 0.4) / 0.50).square())
        + 0.7
        * torch.exp(-((y - 0.65) / 0.25).square() - ((x - 0.55) / 0.40).square())
    )


def test_default_parameters_follow_legacy_matlab_sample_timing():
    params = XSPENParameters()
    simulator = XSPENSimulator(params, dtype=torch.float64)
    expected_focus = (
        torch.arange(1, params.n_xspen + 1, dtype=torch.float64)
        * params.fov_xspen_cm
        / params.n_xspen
        - params.fov_xspen_cm / 2
    )

    assert torch.allclose(
        simulator.focus_positions_cm,
        expected_focus,
        rtol=1e-12,
        atol=1e-12,
    )
    assert torch.isclose(
        simulator.sample_times_s[0],
        torch.tensor(
            params.readout_block_s - params.acquisition_time_s / 2,
            dtype=torch.float64,
        ),
    )
    assert torch.isclose(
        simulator.sample_times_s[-1],
        torch.tensor(params.acquisition_time_s / 2, dtype=torch.float64),
    )


def test_sinc_matrix_matches_explicit_auxiliary_slice_integration():
    params = XSPENParameters(n_xspen=12, n_readout=8, r_value=8.5)
    simulator = XSPENSimulator(params, dtype=torch.float64)

    gamma_gz = params.gamma_hz_per_gauss * params.gz_gauss_per_cm
    q = (
        simulator.cross_term_cycles_per_cm2
        * simulator.xspen_positions_cm.unsqueeze(0)
        + gamma_gz * simulator.sample_times_s.unsqueeze(1)
    )
    n_z = 8192
    dz = params.slice_thickness_cm / n_z
    z = (
        (torch.arange(n_z, dtype=torch.float64) + 0.5) * dz
        - params.slice_thickness_cm / 2
    )
    explicit = torch.exp(2j * torch.pi * q.unsqueeze(-1) * z).mean(dim=-1)

    assert torch.allclose(simulator.encoding_matrix, explicit, rtol=2e-7, atol=2e-7)


def test_acquire_and_reconstruct_support_batches():
    params = XSPENParameters(n_xspen=32, n_readout=24, r_value=32.0)
    simulator = XSPENSimulator(params)
    phantom = _phantom(simulator)
    batch = torch.stack((phantom, 0.6 * phantom), dim=0)

    acquisition = simulator.acquire(batch)
    reconstruction = simulator.reconstruct(acquisition)

    assert isinstance(acquisition, XSPENAcquisition)
    assert acquisition.raw_kspace.shape == (2, 32, 24)
    assert acquisition.raw_kspace.dtype == torch.complex64
    assert acquisition.localized_signal.shape == (2, 32, 24)
    assert reconstruction.shape == (2, 32, 24)
    assert torch.allclose(
        simulator.raw_to_localized(acquisition.raw_kspace),
        acquisition.localized_signal,
        rtol=1e-5,
        atol=1e-5,
    )

    relative_error = torch.linalg.vector_norm(
        reconstruction.real - batch
    ) / torch.linalg.vector_norm(batch)
    assert relative_error < 2e-3


def test_seeded_receiver_noise_is_reproducible():
    params = XSPENParameters(n_xspen=16, n_readout=12, r_value=16.0)
    simulator = XSPENSimulator(params)
    phantom = _phantom(simulator)

    first = simulator.acquire(phantom, noise_std=0.01, seed=17)
    second = simulator.acquire(phantom, noise_std=0.01, seed=17)
    different = simulator.acquire(phantom, noise_std=0.01, seed=18)

    assert torch.equal(first.raw_kspace, second.raw_kspace)
    assert not torch.equal(first.raw_kspace, different.raw_kspace)


def test_invalid_image_shape_is_rejected():
    simulator = XSPENSimulator(XSPENParameters(n_xspen=16, n_readout=12))

    try:
        simulator.acquire(torch.zeros(12, 16))
    except ValueError as exc:
        assert "must end with shape (16, 12)" in str(exc)
    else:  # pragma: no cover - documents the required failure
        raise AssertionError("shape mismatch should raise ValueError")
