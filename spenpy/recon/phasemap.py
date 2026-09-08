"""Referenceless odd/even phase correction followed by a matrix InvA.

The matrix API keeps the historical low-resolution decode/correct/re-encode
operation explicit. InvA is supplied by the caller: it may be a windowed
adjoint (scanner baseline) or a truncated pseudoinverse (ideal simulation).
It must never replace the physical operator's adjoint in data consistency.
"""

from dataclasses import dataclass, field

import torch

from ..core import Acquisition


@dataclass
class PhaseMapInvAResult:
    coil_images: torch.Tensor
    magnitude: torch.Tensor
    ro_image: torch.Tensor
    corrected_ro_image: torch.Tensor
    phase_map_rad: torch.Tensor
    inv_a: torch.Tensor
    coefficients: torch.Tensor
    metadata: dict = field(default_factory=dict)


def _frame(data, allowed_stages):
    if isinstance(data, Acquisition):
        if data.stage not in allowed_stages:
            raise ValueError(f"Expected stage in {allowed_stages}, got {data.stage}")
        extra = set(data.axes) - {"spen", "readout", "coil"}
        if extra:
            raise ValueError(f"Select one frame first; remaining axes: {sorted(extra)}")
        data = data.permute("spen", "readout", "coil").data
    if data.ndim != 3 or not data.is_complex():
        raise ValueError("Expected complex [spen, readout, coil] for one frame")
    if not torch.isfinite(data).all():
        raise ValueError("Input signal must be finite")
    return data


def ro_fft(data):
    """Historical centred, unnormalised RO FFT for scanner data."""
    return torch.fft.fftshift(
        torch.fft.fft(torch.fft.ifftshift(data, dim=1), dim=1), dim=1
    )


def _apply(matrix, data):
    return torch.einsum("ym,mxc->yxc", matrix, data)


def _circular_phase_fit(odd, even):
    """Torch-only quadratic circular regression; no gradient through fitting."""
    odd, even = odd.detach(), even.detach()
    mo = odd.abs().square().sum(-1).sqrt()
    me = even.abs().square().sum(-1).sqrt()
    cross = (even * odd.conj()).sum(-1)
    threshold = 2 * torch.std(me - mo, unbiased=False)
    mask = (mo > threshold) & (me > threshold)
    if mask.any():
        mask &= cross.abs() >= torch.quantile(cross.abs()[mask], 0.2)
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, mo.shape[0], device=mo.device, dtype=mo.dtype),
        torch.linspace(-1, 1, mo.shape[1], device=mo.device, dtype=mo.dtype),
        indexing="ij",
    )
    terms = torch.stack(
        [torch.ones_like(x), x, y, x.square(), x * y, y.square()], dim=-1
    )
    weights = torch.minimum(me, mo) * mask
    weights /= weights.max().clamp_min(1e-8)
    coeffs = mo.new_zeros(6)
    if mask.sum() < 6:
        return mo.new_zeros(mo.shape), coeffs, False
    design, target, weight = terms[mask], torch.angle(cross)[mask], weights[mask]
    coeffs[0] = torch.atan2(
        (weight * target.sin()).sum(), (weight * target.cos()).sum()
    )
    with torch.enable_grad():
        coeffs.requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [coeffs], lr=0.8, max_iter=200, line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad()
            loss = -(weight * torch.cos(target - design @ coeffs)).sum() / weight.sum()
            loss.backward()
            return loss

        optimizer.step(closure)
    coeffs = coeffs.detach()
    return terms @ coeffs, coeffs, True


def reconstruct_phasemap_inva(
    ro_image,
    inv_a,
    a_half,
    *,
    inv_odd=None,
    inv_even=None,
    phase_map_rad=None,
    correct_phase=True,
    estimator="circular",
    polynomial_order=2,
    rtol=1e-6,
):
    """Reconstruct one frame from complex RO-transformed data.

    ``inv_a`` is [output_y, sample], ``a_half`` is [sample, low_y].
    The estimated map is even-minus-odd phase on [low_y, readout], shared
    across coils. Corrected even rows are A_even @ (exp(-i*map) * low_even),
    matching the Siemens algorithm, including its low-resolution projection.
    Zero-based rows 1,3,... are corrected. Odd rows remain unchanged.

    ``circular`` uses a Torch circular quadratic fit. ``polynomial_unwrap`` reproduces
    the historical polynomial/unwrap estimator on CPU (requires scipy).
    Phase estimation is detached. Matrix application remains differentiable
    with respect to the signal or an explicitly supplied phase map.
    """
    data = _frame(ro_image, {"ro_image"})
    m, nx, _ = data.shape
    if m % 2 or m < 4:
        raise ValueError("PhaseMap requires an even number of at least four SPEN rows")
    inv_a = torch.as_tensor(inv_a, device=data.device, dtype=data.dtype)
    a_half = torch.as_tensor(a_half, device=data.device, dtype=data.dtype)
    if inv_a.ndim != 2 or inv_a.shape[1] != m or inv_a.shape[0] < 1:
        raise ValueError("inv_a must have shape [output_y, spen_samples]")
    if a_half.ndim != 2 or a_half.shape[0] != m or not 1 <= a_half.shape[1] <= m // 2:
        raise ValueError("a_half must have shape [spen_samples, low_y <= samples/2]")
    if not torch.isfinite(inv_a).all() or not torch.isfinite(a_half).all():
        raise ValueError("Reconstruction matrices must be finite")
    if rtol <= 0 or not torch.isfinite(torch.tensor(rtol)):
        raise ValueError("rtol must be finite and positive")
    if estimator not in {"circular", "polynomial_unwrap"}:
        raise ValueError("estimator must be circular or polynomial_unwrap")
    if polynomial_order not in (1, 2) or (
        estimator == "circular" and polynomial_order != 2
    ):
        raise ValueError("Use order 1/2 for polynomial_unwrap or order 2 for circular")
    a_odd, a_even = a_half[::2], a_half[1::2]
    inverses = []
    for supplied, matrix in ((inv_odd, a_odd), (inv_even, a_even)):
        inverse = (
            torch.linalg.pinv(matrix, rtol=rtol)
            if supplied is None
            else torch.as_tensor(supplied, device=data.device, dtype=data.dtype)
        )
        if inverse.shape != matrix.T.shape or not torch.isfinite(inverse).all():
            raise ValueError("Half-resolution inverse has invalid shape or values")
        inverses.append(inverse)
    low_odd = _apply(inverses[0], data[::2])
    low_even = _apply(inverses[1], data[1::2])
    coeffs = data.real.new_zeros((polynomial_order + 1) * (polynomial_order + 2) // 2)
    phase = data.real.new_zeros((a_half.shape[1], nx))
    fit_valid = False
    if phase_map_rad is not None:
        if not correct_phase:
            raise ValueError("phase_map_rad requires correct_phase=True")
        phase = torch.as_tensor(
            phase_map_rad, device=data.device, dtype=data.real.dtype
        )
        if phase.shape != (a_half.shape[1], nx) or not torch.isfinite(phase).all():
            raise ValueError("phase_map_rad must be finite [low_y, readout]")
        fit_valid = True
    elif correct_phase and data.abs().max() > 0:
        if estimator == "circular":
            phase, coeffs, fit_valid = _circular_phase_fit(low_odd, low_even)
        else:
            from .hybrid_spen import _polynomial_unwrap_phase_fit

            phase, coeffs, fit_valid = _polynomial_unwrap_phase_fit(
                low_even, low_odd, polynomial_order
            )
    corrected = data.clone()
    if correct_phase:
        corrected[1::2] = _apply(a_even, low_even * torch.exp(-1j * phase)[..., None])
    images = _apply(inv_a, corrected)
    return PhaseMapInvAResult(
        images,
        images.abs().square().sum(-1).sqrt(),
        data,
        corrected,
        phase,
        inv_a,
        coeffs,
        {
            "phase_estimator": estimator,
            "phase_fit_valid": bool(fit_valid),
            "phase_corrected": correct_phase,
            "phase_map_supplied": phase_map_rad is not None,
            "phase_convention": "even minus odd; correction exp(-i * phase)",
            "half_resolution": a_half.shape[1],
            "coil_combination": "RSS",
        },
    )
