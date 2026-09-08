"""Referenceless odd/even phase correction and windowed InvA reconstruction.

The phase is estimated with distinct odd/even reconstruction matrices, then
applied directly to the ORIGINAL even ROFFT lines. No low-resolution signal
replacement or PE pseudoinverse is involved in this reconstruction path.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .phasemap import PhaseMapInvAResult, _apply, _frame


def _evidence(odd, even):
    cross = (even * odd.conj()).sum(-1)
    phase = cross.angle().float()
    mo = odd.abs().square().sum(-1).sqrt()
    me = even.abs().square().sum(-1).sqrt()
    threshold = 2 * torch.std(me - mo)
    mask = (mo > threshold) & (me > threshold) & torch.isfinite(phase)
    if not mask.any():
        raise ValueError("No reliable pixels remain for phase fitting")
    mask &= cross.abs() >= torch.quantile(cross.abs()[mask], 0.2)
    weights = mask.float() * torch.minimum(mo, me).float()
    weights /= weights.max().clamp_min(1e-8)
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, phase.shape[0], device=phase.device),
        torch.linspace(-1, 1, phase.shape[1], device=phase.device),
        indexing="ij",
    )
    return phase, mask, weights, x, y


def _quadratic(odd, even):
    phase, mask, weights, x, y = _evidence(odd, even)
    design = torch.stack([torch.ones_like(x), x, y, x * x, x * y, y * y], dim=-1)
    keep = weights > 0
    if keep.sum() < 6:
        raise ValueError("Not enough valid pixels for the six-parameter phase fit")
    coeffs = phase.new_zeros(6)
    coeffs[0] = torch.atan2(
        (weights * phase.sin()).sum(), (weights * phase.cos()).sum()
    )
    with torch.enable_grad():
        coeffs.requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [coeffs], lr=0.8, max_iter=200, line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad()
            loss = (
                -(weights[keep] * (phase[keep] - design[keep] @ coeffs).cos()).sum()
                / weights.sum()
            )
            loss.backward()
            return loss

        optimizer.step(closure)
    coeffs = coeffs.detach()
    polynomial = design @ coeffs
    residual = torch.atan2((phase - polynomial).sin(), (phase - polynomial).cos())
    # Notebook 09 smooths the wrapped residual with the binary mask, not the
    # fit weights. Both convolutions use zero padding and an 11x11 box.
    kernel = phase.new_ones((1, 1, 11, 11))
    numerator = F.conv2d((residual * mask)[None, None], kernel, padding=5)[0, 0]
    denominator = F.conv2d(mask.float()[None, None], kernel, padding=5)[0, 0]
    return polynomial + numerator / denominator.clamp_min(1e-6), coeffs, mask


class _TinyPhase(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 1)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, coords):
        return self.net(coords).squeeze(-1)


def _tiny(odd, even, steps):
    raw, mask, weights, x, y = _evidence(odd, even)
    coords = torch.stack([x, y], dim=-1).reshape(-1, 2)
    # Match notebook 19's initialization without modifying caller RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        model = _TinyPhase().to(raw.device)
    with torch.enable_grad():
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-2, weight_decay=1e-4)
        for _ in range(steps + 1):
            optimizer.zero_grad(set_to_none=True)
            phase = model(coords).reshape(raw.shape)
            cross = (even * torch.exp(-1j * phase[..., None]) * odd.conj()).sum(-1)
            alignment = 1 - (
                weights * cross.real / cross.abs().clamp_min(1e-8)
            ).sum() / weights.sum().clamp_min(1e-8)
            dx, dy = phase[:, 1:] - phase[:, :-1], phase[1:] - phase[:-1]
            smooth = (
                torch.atan2(dx.sin(), dx.cos()).square().mean()
                + torch.atan2(dy.sin(), dy.cos()).square().mean()
            )
            (alignment + 2e-3 * smooth).backward()
            optimizer.step()
        phase = model(coords).reshape(raw.shape).detach()
    return phase, raw.new_empty(0), mask


def reconstruct_even_odd_inva(
    ro_image,
    inv_a,
    odd_inv_a,
    even_inv_a,
    *,
    estimator="quadratic",
    coil_combination="adaptive",
    tiny_steps=500,
    phase_map_rad=None,
):
    """Estimate an odd/even phase map and correct the acquired even RO lines.

    Input is [PE, RO, coil], already transformed along RO. Full InvA is
    [PE,PE]; odd/even InvA are [PE/2,PE/2]. Output retains the native PE/RO
    grid. Phase estimation is detached; correction/matrix application can
    differentiate through the signal or an explicitly supplied phase map.
    """
    data = _frame(ro_image, {"ro_image"})
    m, k, _ = data.shape
    if m < 4 or m % 2 or k < 2:
        raise ValueError("Requires an even PE size >= 4 and RO size >= 2")
    if estimator not in {"quadratic", "tiny"}:
        raise ValueError("estimator must be quadratic or tiny")
    if coil_combination not in {"adaptive", "rss"}:
        raise ValueError("coil_combination must be adaptive or rss")
    if not isinstance(tiny_steps, int) or tiny_steps < 0:
        raise ValueError("tiny_steps must be a nonnegative integer")
    matrices = []
    for value, size in ((inv_a, m), (odd_inv_a, m // 2), (even_inv_a, m // 2)):
        value = torch.as_tensor(value, device=data.device, dtype=data.dtype)
        if value.shape != (size, size) or not torch.isfinite(value).all():
            raise ValueError(f"Expected a finite {size}x{size} InvA matrix")
        matrices.append(value)
    inv_a, odd_inv_a, even_inv_a = matrices
    odd = _apply(odd_inv_a, data[::2]).detach()
    even = _apply(even_inv_a, data[1::2]).detach()
    coeffs = data.real.new_empty(0)
    mask = torch.zeros((m // 2, k), dtype=torch.bool, device=data.device)
    if phase_map_rad is not None:
        phase = torch.as_tensor(
            phase_map_rad, device=data.device, dtype=data.real.dtype
        )
        if phase.shape != (m // 2, k) or not torch.isfinite(phase).all():
            raise ValueError("phase_map_rad must be finite [PE/2,RO]")
    elif data.abs().max() == 0:
        phase = data.real.new_zeros((m // 2, k))
    elif estimator == "quadratic":
        phase, coeffs, mask = _quadratic(odd, even)
    else:
        phase, coeffs, mask = _tiny(odd, even, tiny_steps)
    corrected = data.clone()
    corrected[1::2] *= torch.exp(-1j * phase)[..., None]
    images = _apply(inv_a, corrected)
    if coil_combination == "adaptive":
        from .._legacy.utils.coil_combine import coil_combine_batched

        combined = coil_combine_batched(images[None, :, :, None, :])[0, :, :, 0]
        magnitude = combined.abs()
    else:
        magnitude = images.abs().square().sum(-1).sqrt()
    return PhaseMapInvAResult(
        images,
        magnitude,
        data,
        corrected,
        phase,
        inv_a,
        coeffs,
        {
            "phase_estimator": estimator,
            "phase_correction": "direct multiplication on original even ROFFT lines",
            "phase_mask_fraction": float(mask.float().mean()),
            "coil_combination": coil_combination,
            "phase_map_supplied": phase_map_rad is not None,
            "inv_a_kind": "Gaussian-windowed adjoint",
            "output_grid": "native acquisition grid",
        },
    )
