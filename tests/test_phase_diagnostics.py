"""Tests for optional PV360 phase-correction diagnostics."""

import torch

from spenpy.recon.phase import (
    PhaseCorrectionDiagnostics,
    apply_pv360_one_shot_phase_correction,
    even_odd_phase_fit_torch,
)


def _sample_inputs():
    torch.manual_seed(0)
    real = torch.randn(4, 6, 1, 1)
    imag = torch.randn(4, 6, 1, 1)
    roffted_data = torch.complex(real, imag)
    inv_a = torch.eye(4, dtype=torch.complex64)
    one_shot_odd_inv = torch.eye(2, dtype=torch.complex64)
    one_shot_even_inv = torch.eye(2, dtype=torch.complex64)
    return roffted_data, inv_a, one_shot_odd_inv, one_shot_even_inv


def test_phase_correction_default_return_is_tensor():
    corrected = apply_pv360_one_shot_phase_correction(
        *_sample_inputs(),
        optimize=False,
        smooth_motion_phase_between_shots=False,
    )

    assert isinstance(corrected, torch.Tensor)
    assert corrected.shape == (4, 6, 1, 1)


def test_phase_correction_can_return_diagnostics():
    corrected, diagnostics = apply_pv360_one_shot_phase_correction(
        *_sample_inputs(),
        optimize=False,
        smooth_motion_phase_between_shots=False,
        return_diagnostics=True,
    )

    assert corrected.shape == (4, 6, 1, 1)
    assert isinstance(diagnostics, PhaseCorrectionDiagnostics)
    assert len(diagnostics.first_pass) == 1
    assert len(diagnostics.refined_even_odd) == 1
    assert diagnostics.motion_between_shots == []

    first = diagnostics.first_pass[0]
    refined = diagnostics.refined_even_odd[0]
    assert first.phase_map.shape == (2, 6)
    assert first.phase_difference.shape == (2, 6)
    assert refined.smooth_phase.shape == (2, 6)
    assert refined.mask.shape == (2, 6)


def test_torch_even_odd_phase_fit_stays_on_torch():
    torch.manual_seed(1)
    phase = torch.linspace(-0.4, 0.4, 6).view(1, 6).expand(4, 6)
    odd = torch.ones(4, 6, 2, dtype=torch.complex64)
    even = odd * torch.exp(1j * phase[:, :, None])

    fit = even_odd_phase_fit_torch(odd, even, max_iter=20)

    assert isinstance(fit.smooth_phase, torch.Tensor)
    assert fit.smooth_phase.shape == (4, 6)
    assert fit.phase_difference.shape == (4, 6)
    assert fit.mask.shape == (4, 6)
