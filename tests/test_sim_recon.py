import pytest
import torch

from spenpy.core import EncodingOperator, SPEN180Protocol, XSPENProtocol
from spenpy.recon import data_consistency, reconstruct
from spenpy.sim import NoiseModel, simulate, whiten_coils


@pytest.mark.parametrize("kind", [SPEN180Protocol, XSPENProtocol])
def test_simulation_reconstruction_and_gradients(kind, smooth_image):
    p = kind(acquisition_shape=(24, 20), r_value=12)
    x = smooth_image((24, 20)).requires_grad_()
    sample = simulate(x, p, num_coils=4, even_odd_constant_rad=0.3, snr_db=50, seed=18)
    assert sample.measurements.shape == (24, 20, 4)
    result = reconstruct(
        sample.acquisition, sample.operator, regularization=1e-3, max_iter=100
    )
    assert ((result.image - x).norm() / x.norm()).item() < 0.04
    assert result.relative_residual.item() < 1e-5
    result.image.abs().square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.norm() > 0


@pytest.mark.parametrize("kind", [SPEN180Protocol, XSPENProtocol])
def test_cg_matches_dense_complex_solution(kind):
    op = EncodingOperator(
        kind(acquisition_shape=(4, 3), r_value=3), (5, 4), dtype=torch.complex128
    )
    eye = torch.eye(20, dtype=torch.complex128).reshape(20, 5, 4)
    a = op(eye).reshape(20, -1).T
    y = torch.randn(2, 4, 3, 1, dtype=torch.complex128)
    prior = torch.randn(2, 5, 4, dtype=torch.complex128)
    lam = 0.3
    reference = torch.linalg.solve(
        a.mH @ a + lam * torch.eye(20, dtype=a.dtype),
        (y.reshape(2, -1) @ a.conj() + lam * prior.reshape(2, -1)).T,
    ).T.reshape(2, 5, 4)
    result = reconstruct(
        y, op, regularization=lam, prior=prior, max_iter=100, rtol=1e-11
    )
    torch.testing.assert_close(result.image, reference, rtol=1e-9, atol=1e-10)


def test_proximal_step_reduces_measurement_residual_and_preserves_gradient(
    smooth_image,
):
    sample = simulate(
        smooth_image((16, 12)),
        SPEN180Protocol(acquisition_shape=(12, 8), r_value=8),
        num_coils=2,
    )
    prior = torch.randn(16, 12, dtype=torch.complex64, requires_grad=True)
    corrected = data_consistency(
        prior, sample.measurements, sample.operator, mu=0.1, max_iter=60
    )
    assert (sample.operator(corrected) - sample.measurements).norm() < (
        sample.operator(prior) - sample.measurements
    ).norm()
    corrected.abs().square().mean().backward()
    assert torch.isfinite(prior.grad).all()


def test_mixed_zero_and_nonzero_batch_converges_without_nan(smooth_image):
    p = XSPENProtocol(acquisition_shape=(8, 6), r_value=8)
    sample = simulate(
        torch.stack([torch.zeros(8, 6), smooth_image((8, 6))]), p, num_coils=2
    )
    result = reconstruct(sample.measurements, sample.operator, max_iter=80)
    assert torch.isfinite(result.image).all()
    assert torch.count_nonzero(result.image[0]) == 0


def test_seed_is_local_and_reproducible(smooth_image):
    x = smooth_image((12, 10))
    p = XSPENProtocol(acquisition_shape=(8, 6), r_value=8)
    state = torch.random.get_rng_state().clone()
    a = simulate(x, p, num_coils=3, snr_db=20, seed=19)
    b = simulate(x, p, num_coils=3, snr_db=20, seed=19)
    c = simulate(x, p, num_coils=3, snr_db=20, seed=20)
    assert torch.equal(a.measurements, b.measurements)
    assert not torch.equal(a.measurements, c.measurements)
    assert torch.equal(state, torch.random.get_rng_state())


def test_noise_covariance_and_whitening():
    covariance = torch.tensor([[1, 0.3 + 0.2j], [0.3 - 0.2j, 2]], dtype=torch.complex64)
    clean = torch.zeros(400, 200, 2, dtype=torch.complex64)
    noisy, _ = NoiseModel(std=2, coil_covariance=covariance).add(clean, seed=4)
    flat = noisy.reshape(-1, 2)
    empirical = flat.T @ flat.conj() / flat.shape[0]
    torch.testing.assert_close(empirical, 4 * covariance, rtol=0.02, atol=0.04)
    white = whiten_coils(noisy, 4 * covariance).reshape(-1, 2)
    torch.testing.assert_close(
        white.T @ white.conj() / white.shape[0],
        torch.eye(2, dtype=white.dtype),
        rtol=0.02,
        atol=0.02,
    )


def test_noise_snr_and_zero_noise():
    clean = torch.ones(100, 200, 4, dtype=torch.complex64)
    noisy, _ = NoiseModel(snr_db=30).add(clean, seed=9)
    actual = 20 * torch.log10(
        clean.abs().square().mean().sqrt()
        / (noisy - clean).abs().square().mean().sqrt()
    )
    assert abs(actual.item() - 30) < 0.1
    zero, _ = NoiseModel().add(clean)
    assert torch.equal(zero, clean)


def test_masked_noise_snr_counts_only_acquired_samples():
    mask = torch.zeros(200, 200)
    mask[::4] = 1
    clean = torch.ones(200, 200, 4, dtype=torch.complex64) * mask[..., None]
    noisy, _ = NoiseModel(snr_db=30).add(clean, seed=9, sample_mask=mask)
    support = mask.bool()
    snr = 20 * torch.log10(clean[support].norm() / (noisy - clean)[support].norm())
    assert abs(snr.item() - 30) < 0.1
    assert torch.count_nonzero(noisy[~support]) == 0
    empty, _ = NoiseModel(snr_db=30).add(
        torch.zeros_like(clean), sample_mask=torch.zeros_like(mask)
    )
    assert torch.count_nonzero(empty) == 0


def test_real_phase_input_preserves_target_and_96_by_96_four_coils(smooth_image):
    magnitude = smooth_image((128, 120))
    sample = simulate(
        magnitude, acquisition_shape=(96, 96), object_phase_rad=0.4, num_coils=4
    )
    assert sample.measurements.shape == (96, 96, 4)
    torch.testing.assert_close(
        sample.target, magnitude.to(torch.complex64) * torch.exp(torch.tensor(0.4j))
    )


def test_cross_term_xspen_accepts_rectangular_acquisition(smooth_image):
    sample = simulate(
        smooth_image((80, 90)),
        XSPENProtocol(acquisition_shape=(60, 64), r_value=60),
        num_coils=32,
    )
    assert sample.measurements.shape == (60, 64, 32)
