"""Torch coil maps with sum-of-squares magnitude normalization."""

import math

import torch


def coil_sensitivities(
    shape, num_coils=4, *, device=None, dtype=torch.complex64, seed=0
):
    """Return smooth synthetic sensitivities [y,x,coil], not calibrated maps."""
    if num_coils < 1 or len(shape) != 2 or min(shape) < 1:
        raise ValueError("Positive image dimensions and coil count required")
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    y, x = torch.meshgrid(
        *(torch.linspace(-1, 1, n, device=device, dtype=real_dtype) for n in shape),
        indexing="ij",
    )
    if num_coils == 1:
        return torch.ones((*shape, 1), device=device, dtype=dtype)
    generator = torch.Generator(device=device or "cpu").manual_seed(seed)
    angles = torch.arange(num_coils, device=device, dtype=real_dtype) * (
        2 * math.pi / num_coils
    )
    cy, cx = 0.8 * angles.sin(), 0.8 * angles.cos()
    mag = (
        torch.exp(
            -((y[..., None] - cy) ** 2 + (x[..., None] - cx) ** 2) / (2 * 0.85**2)
        )
        + 0.05
    )
    mag = mag / mag.square().sum(-1, keepdim=True).sqrt()
    offsets = (
        torch.rand(num_coils, device=device, dtype=real_dtype, generator=generator)
        * 2
        * math.pi
    )
    phase = offsets + 0.4 * (y[..., None] * cx - x[..., None] * cy)
    return (mag * torch.exp(1j * phase)).to(dtype)
