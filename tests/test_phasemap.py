import math
from pathlib import Path

import numpy as np
import pytest
import torch

from spenpy import SPEN180Protocol, XSPENProtocol
from spenpy.core import Acquisition
from spenpy.recon import (
    calc_hybrid_spen_matrices,
    reconstruct_phasemap_inva,
    reconstruct_simulated_inva,
)
from spenpy.sim import simulate


@pytest.mark.parametrize("estimator", ["circular", "polynomial_unwrap"])
@pytest.mark.parametrize("constant", [0.7, 3.0])
def test_unknown_wrapped_phase_recovery(estimator, constant):
    # Independent low-resolution acquisition: odd/even encodings differ,
    # but both span the same image space. Phase crosses pi in the second case.
    rng = torch.Generator().manual_seed(107)
    n, k, c = 10, 28, 3
    a = torch.randn(28, n, generator=rng, dtype=torch.complex128)
    low = torch.randn(n, k, c, generator=rng, dtype=torch.complex128)
    x = torch.linspace(-1, 1, k, dtype=torch.float64)
    expected_phase = (constant + 0.5 * x).expand(n, -1)
    ro = torch.einsum("mn,nxc->mxc", a, low)
    ro[1::2] *= torch.exp(1j * expected_phase[0])[None, :, None]
    result = reconstruct_phasemap_inva(ro, torch.linalg.pinv(a), a, estimator=estimator)
    phase_error = torch.angle(torch.exp(1j * (result.phase_map_rad - expected_phase)))
    assert phase_error.abs().max() < 2e-4
    torch.testing.assert_close(result.coil_images, low, atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(result.corrected_ro_image[::2], ro[::2])


def test_supplied_phase_gradients_and_disabled_projection():
    rng = torch.Generator().manual_seed(5)
    a = torch.randn(12, 4, generator=rng, dtype=torch.complex128)
    data = torch.randn(
        12, 8, 2, generator=rng, dtype=torch.complex128, requires_grad=True
    )
    phase = torch.full((4, 8), 0.3, dtype=torch.float64, requires_grad=True)
    inv = torch.linalg.pinv(a)
    result = reconstruct_phasemap_inva(data, inv, a, phase_map_rad=phase)
    result.coil_images.abs().square().sum().backward()
    assert torch.isfinite(data.grad).all() and data.grad.norm() > 0
    assert torch.isfinite(phase.grad).all() and phase.grad.norm() > 0
    bypass = reconstruct_phasemap_inva(data, inv, a, correct_phase=False)
    torch.testing.assert_close(bypass.corrected_ro_image, data)
    torch.testing.assert_close(
        bypass.coil_images, torch.einsum("nm,mxc->nxc", inv, data)
    )


@pytest.mark.parametrize("protocol_class", [SPEN180Protocol, XSPENProtocol])
def test_simulation_baseline_uses_sequence_kernel(protocol_class, smooth_image):
    sample = simulate(
        smooth_image((48, 48)),
        protocol_class(acquisition_shape=(32, 32), r_value=24),
        num_coils=2,
    )
    result = reconstruct_simulated_inva(sample.acquisition, sample.operator)
    assert result.inv_a.shape == (32, 32)
    assert result.coil_images.shape == (32, 32, 2)
    assert torch.isfinite(result.magnitude).all()
    assert result.metadata["sequence"] == sample.operator.protocol.sequence
    assert result.metadata["inv_a_kind"] == "Gaussian-windowed adjoint"


def test_phase_baseline_rejects_unmatched_stages_and_mask(smooth_image):
    sample = simulate(
        smooth_image((16, 16)), SPEN180Protocol(acquisition_shape=(16, 16), r_value=10)
    )
    bad = Acquisition(sample.measurements, ("spen", "readout", "coil"), "coil_image")
    with pytest.raises(ValueError, match="stage"):
        reconstruct_simulated_inva(bad, sample.operator)
    sample.operator.sample_mask[0] = 0
    with pytest.raises(ValueError, match="fully sampled"):
        reconstruct_simulated_inva(sample.acquisition, sample.operator)


def test_zero_data_is_finite():
    a = torch.ones((8, 3), dtype=torch.complex64)
    result = reconstruct_phasemap_inva(
        torch.zeros(8, 5, 2, dtype=torch.complex64), a.mH, a
    )
    assert torch.count_nonzero(result.magnitude) == 0
    assert torch.count_nonzero(result.phase_map_rad) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_phase_reconstruction_matches_cpu():
    rng = torch.Generator().manual_seed(21)
    a = torch.randn(16, 6, generator=rng, dtype=torch.complex128)
    low = torch.randn(6, 10, 2, generator=rng, dtype=torch.complex128)
    data = torch.einsum("mn,nxc->mxc", a, low)
    data[1::2] *= np.exp(0.4j)
    cpu = reconstruct_phasemap_inva(data, torch.linalg.pinv(a), a)
    gpu = reconstruct_phasemap_inva(data.cuda(), torch.linalg.pinv(a).cuda(), a.cuda())
    torch.testing.assert_close(
        cpu.coil_images, gpu.coil_images.cpu(), atol=1e-5, rtol=1e-5
    )


@pytest.mark.integration
def test_siemens_matrices_match_saved_matlab_reference():
    h5py = pytest.importorskip("h5py")
    reference = (
        Path(__file__).resolve().parents[2]
        / "xspen_recons/outputs/xspen_raw_reconstruction.mat"
    )
    if not reference.exists():
        pytest.skip("Local MATLAB reference missing")
    matrices = calc_hybrid_spen_matrices(60, 0.18, 60)
    with h5py.File(reference) as handle:
        for key, name in [
            ("inv_a", "InvA"),
            ("inv_odd", "reconstructionInfo/phaseParameters/InvAHalfResOddEff"),
            ("inv_even", "reconstructionInfo/phaseParameters/InvAHalfResEvenEff"),
        ]:
            array = handle[name][()].T
            expected = torch.from_numpy(array["real"] + 1j * array["imag"])
            torch.testing.assert_close(matrices[key], expected, atol=1e-11, rtol=1e-10)
    assert math.isfinite(matrices["a"].abs().max())
