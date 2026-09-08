"""Receiver noise is separate from the deterministic measurement operator."""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NoiseModel:
    """Proper complex Gaussian noise.

    std is absolute complex RMS for unit coil covariance (each real component
    has std/sqrt(2)). snr_db instead sets total signal/noise RMS, per batch
    sample. SNR scaling is detached from the image. Covariance is E[n n^H].
    """

    std: float = 0.0
    snr_db: float | None = None
    coil_covariance: torch.Tensor | None = None

    def __post_init__(self):
        if not math.isfinite(self.std) or self.std < 0:
            raise ValueError("Noise std must be finite and nonnegative")
        if self.snr_db is not None and (
            not math.isfinite(self.snr_db) or self.std != 0
        ):
            raise ValueError("Specify finite snr_db OR std, not both")

    def add(self, clean, *, seed=0, sample_mask=None):
        if not clean.is_complex() or clean.ndim < 3:
            raise ValueError("Expected complex [...,PE,RO,coil]")
        coils = clean.shape[-1]
        if sample_mask is None:
            support = torch.ones(
                clean.shape[-3:-1], device=clean.device, dtype=torch.bool
            )
        else:
            mask = torch.as_tensor(sample_mask, device=clean.device)
            if not torch.isfinite(mask).all() or ((mask < 0) | (mask > 1)).any():
                raise ValueError("sample_mask must lie in [0,1]")
            support = torch.broadcast_to(mask > 0, clean.shape[-3:-1])
        covariance = (
            torch.eye(coils, device=clean.device, dtype=clean.dtype)
            if self.coil_covariance is None
            else torch.as_tensor(
                self.coil_covariance, device=clean.device, dtype=clean.dtype
            )
        )
        if covariance.shape != (coils, coils) or not torch.allclose(
            covariance, covariance.mH
        ):
            raise ValueError("coil_covariance must be Hermitian [coil,coil]")
        chol = torch.linalg.cholesky(covariance)
        scale = torch.as_tensor(self.std, device=clean.device, dtype=clean.real.dtype)
        if self.snr_db is not None:
            power = (clean.detach().abs().square() * support[..., None]).sum(
                dim=(-3, -2, -1), keepdim=True
            )
            rms = (power / (support.sum() * coils).clamp_min(1)).sqrt()
            scale = (
                rms
                * 10 ** (-self.snr_db / 20)
                / (covariance.diagonal().real.mean().sqrt())
            )
        generator = torch.Generator(device=clean.device).manual_seed(seed)
        re = torch.randn(
            clean.shape,
            device=clean.device,
            dtype=clean.real.dtype,
            generator=generator,
        )
        im = torch.randn(
            clean.shape,
            device=clean.device,
            dtype=clean.real.dtype,
            generator=generator,
        )
        noise = (
            torch.einsum(
                "dc,...mkc->...mkd", chol, torch.complex(re, im) / math.sqrt(2)
            )
            * scale
        )
        noise = noise * support[..., None]
        return clean + noise, {
            "seed": seed,
            "std": self.std,
            "snr_db": self.snr_db,
            "scale": scale,
            "coil_covariance": covariance,
        }


def whiten_coils(data, covariance):
    """Whiten the last coil axis using the supplied covariance."""
    covariance = torch.as_tensor(covariance, device=data.device, dtype=data.dtype)
    chol = torch.linalg.cholesky(covariance)
    return torch.linalg.solve_triangular(
        chol, data.reshape(-1, data.shape[-1]).T, upper=False
    ).T.reshape(data.shape)
