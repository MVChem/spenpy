"""Minimal ideal xSPEN acquisition and reconstruction.

This module implements the compact signal equation used by the legacy
``xSPEN1D.m`` simulation.  It is intentionally smaller than the configurable
SPEN simulator: the first version models a single coil, an on-resonance
object, ideal chirps, and a uniform auxiliary slice.

The two crossed chirps leave a bilinear y-z phase.  During acquisition, the
z gradient sweeps a stationary point through y.  Integrating the auxiliary z
coordinate over a uniform slice produces a sinc localization kernel.  That
kernel is used as an explicit forward matrix, which also makes a small
regularized reconstruction possible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class XSPENParameters:
    """Sequence and image parameters for the minimal xSPEN model.

    The defaults mirror ``xSPEN1D.m`` where practical.  ``beta`` is called
    ``belta`` in the MATLAB source; it partitions the chirp bandwidth between
    the xSPEN (y) and auxiliary (z) gradients.
    """

    n_xspen: int = 64
    n_readout: int = 64
    fov_xspen_cm: float = 4.0
    fov_readout_cm: float = 4.0
    slice_thickness_cm: float = 0.8
    r_value: float = 64.0
    beta: float = 0.5
    spectral_width_hz: float = 250_000.0
    blip_duration_s: float = 80e-6
    gamma_hz_per_gauss: float = 4275.707747

    def __post_init__(self) -> None:
        if self.n_xspen <= 0 or self.n_readout <= 0:
            raise ValueError("n_xspen and n_readout must be positive")
        if self.fov_xspen_cm <= 0 or self.fov_readout_cm <= 0:
            raise ValueError("field of view values must be positive")
        if self.slice_thickness_cm <= 0:
            raise ValueError("slice_thickness_cm must be positive")
        if self.r_value <= 0:
            raise ValueError("r_value must be positive")
        if not 0 < self.beta < 1:
            raise ValueError("beta must be strictly between 0 and 1")
        if self.spectral_width_hz <= 0:
            raise ValueError("spectral_width_hz must be positive")
        if self.blip_duration_s < 0:
            raise ValueError("blip_duration_s cannot be negative")
        if self.gamma_hz_per_gauss <= 0:
            raise ValueError("gamma_hz_per_gauss must be positive")

    @property
    def readout_block_s(self) -> float:
        """Duration of one readout plus its blip."""
        return self.n_readout / self.spectral_width_hz + self.blip_duration_s

    @property
    def acquisition_time_s(self) -> float:
        """Total duration of the xSPEN acquisition train."""
        return self.readout_block_s * self.n_xspen

    @property
    def chirp_duration_s(self) -> float:
        """Duration of either chirp in the ideal sequence relation."""
        return self.acquisition_time_s / (4.0 * self.beta)

    @property
    def chirp_bandwidth_hz(self) -> float:
        """Chirp bandwidth derived from ``R = bandwidth * duration``."""
        return self.r_value / self.chirp_duration_s

    @property
    def gy_gauss_per_cm(self) -> float:
        """xSPEN-axis chirp gradient."""
        return (
            self.beta
            * self.chirp_bandwidth_hz
            / self.gamma_hz_per_gauss
            / self.fov_xspen_cm
        )

    @property
    def gz_gauss_per_cm(self) -> float:
        """Auxiliary-axis chirp/acquisition gradient."""
        return (
            (1.0 - self.beta)
            * self.chirp_bandwidth_hz
            / self.gamma_hz_per_gauss
            / self.slice_thickness_cm
        )


@dataclass
class XSPENAcquisition:
    """Outputs of :meth:`XSPENSimulator.acquire`.

    Leading batch dimensions, if present, are preserved.  The final two axes
    always have layout ``[..., xspen_sample, readout]``.
    """

    input_image: torch.Tensor
    raw_kspace: torch.Tensor
    localized_signal: torch.Tensor
    encoding_matrix: torch.Tensor
    xspen_positions_cm: torch.Tensor
    focus_positions_cm: torch.Tensor
    sample_times_s: torch.Tensor
    parameters: XSPENParameters


def _centered_fft(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.fft.fftshift(
        torch.fft.fft(torch.fft.ifftshift(x, dim=dim), dim=dim, norm="ortho"),
        dim=dim,
    )


def _centered_ifft(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.fft.fftshift(
        torch.fft.ifft(torch.fft.ifftshift(x, dim=dim), dim=dim, norm="ortho"),
        dim=dim,
    )


class XSPENSimulator:
    """Small matrix-based xSPEN simulator.

    The input is an image with shape ``[..., n_xspen, n_readout]``.  xSPEN
    localization is applied along the penultimate axis and a centered FFT is
    applied along the conventional readout axis.  The returned raw data are
    ideal, sorted k-space; EPI zigzag order and odd/even phase errors are not
    part of this first model.
    """

    def __init__(
        self,
        parameters: XSPENParameters | None = None,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64")
        self.parameters = parameters or XSPENParameters()
        self.device = torch.device(device)
        self.dtype = dtype
        self.complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128

        p = self.parameters
        self.xspen_positions_cm = self._voxel_centers(p.n_xspen, p.fov_xspen_cm)
        self.readout_positions_cm = self._voxel_centers(p.n_readout, p.fov_readout_cm)
        self.sample_times_s = (
            (torch.arange(p.n_xspen, device=self.device, dtype=self.dtype) + 1.0)
            * p.readout_block_s
            - p.acquisition_time_s / 2.0
        )

        gamma_gy = p.gamma_hz_per_gauss * p.gy_gauss_per_cm
        gamma_gz = p.gamma_hz_per_gauss * p.gz_gauss_per_cm
        self.cross_term_cycles_per_cm2 = (
            -4.0 * (p.chirp_duration_s / p.chirp_bandwidth_hz) * gamma_gy * gamma_gz
        )

        # q has cycles/cm along the auxiliary z direction.  Integrating
        # exp(i*2*pi*q*z) over a uniform, centered slice yields sinc(q*Lz).
        q = (
            self.cross_term_cycles_per_cm2 * self.xspen_positions_cm.unsqueeze(0)
            + gamma_gz * self.sample_times_s.unsqueeze(1)
        )
        self.encoding_matrix = torch.sinc(q * p.slice_thickness_cm).to(self.complex_dtype)
        self.focus_positions_cm = (
            -gamma_gz * self.sample_times_s / self.cross_term_cycles_per_cm2
        )

    def _voxel_centers(self, count: int, fov_cm: float) -> torch.Tensor:
        step = fov_cm / count
        return (
            (torch.arange(count, device=self.device, dtype=self.dtype) + 0.5) * step
            - fov_cm / 2.0
        )

    def _prepare_image(self, image: torch.Tensor) -> torch.Tensor:
        p = self.parameters
        image = torch.as_tensor(image, device=self.device)
        if image.ndim < 2:
            raise ValueError("image must have shape [..., n_xspen, n_readout]")
        if image.shape[-2:] != (p.n_xspen, p.n_readout):
            raise ValueError(
                "image must end with shape "
                f"({p.n_xspen}, {p.n_readout}); got {tuple(image.shape)}"
            )
        return image.to(self.complex_dtype)

    @torch.no_grad()
    def acquire(
        self,
        image: torch.Tensor,
        *,
        noise_std: float = 0.0,
        seed: int | None = None,
    ) -> XSPENAcquisition:
        """Simulate an ideal xSPEN acquisition.

        Args:
            image: Real or complex tensor shaped
                ``[..., n_xspen, n_readout]``.
            noise_std: Complex RMS noise as a fraction of the maximum raw
                signal magnitude in each input image.
            seed: Optional local random seed used only for receiver noise.
        """
        if noise_std < 0:
            raise ValueError("noise_std cannot be negative")
        image_c = self._prepare_image(image)
        localized = torch.einsum("ay,...yx->...ax", self.encoding_matrix, image_c)
        raw = _centered_fft(localized, dim=-1)

        if noise_std:
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(int(seed))
            scale = raw.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(
                torch.finfo(self.dtype).eps
            )
            component_std = float(noise_std) / math.sqrt(2.0)
            real = torch.randn(
                raw.shape,
                dtype=self.dtype,
                device=self.device,
                generator=generator,
            )
            imag = torch.randn(
                raw.shape,
                dtype=self.dtype,
                device=self.device,
                generator=generator,
            )
            raw = raw + component_std * scale * (real + 1j * imag)

        return XSPENAcquisition(
            input_image=image_c,
            raw_kspace=raw.to(self.complex_dtype),
            localized_signal=localized.to(self.complex_dtype),
            encoding_matrix=self.encoding_matrix,
            xspen_positions_cm=self.xspen_positions_cm,
            focus_positions_cm=self.focus_positions_cm,
            sample_times_s=self.sample_times_s,
            parameters=self.parameters,
        )

    def raw_to_localized(self, raw_kspace: torch.Tensor) -> torch.Tensor:
        """Apply the conventional readout IFFT to ideal sorted raw data."""
        p = self.parameters
        raw = torch.as_tensor(raw_kspace, device=self.device).to(self.complex_dtype)
        if raw.ndim < 2 or raw.shape[-2:] != (p.n_xspen, p.n_readout):
            raise ValueError(
                "raw_kspace must end with shape "
                f"({p.n_xspen}, {p.n_readout}); got {tuple(raw.shape)}"
            )
        return _centered_ifft(raw, dim=-1).to(self.complex_dtype)

    def reconstruction_matrix(self, regularization: float = 1e-3) -> torch.Tensor:
        """Return a Tikhonov inverse of the xSPEN localization matrix.

        ``regularization`` is dimensionless and relative to the square of the
        largest singular value.  Set it to zero for a truncated pseudoinverse.
        """
        if regularization < 0:
            raise ValueError("regularization cannot be negative")
        u, singular_values, vh = torch.linalg.svd(self.encoding_matrix, full_matrices=False)
        if regularization == 0:
            cutoff = (
                max(self.encoding_matrix.shape)
                * torch.finfo(self.dtype).eps
                * singular_values[0]
            )
            filt = torch.where(
                singular_values > cutoff,
                singular_values.reciprocal(),
                torch.zeros_like(singular_values),
            )
        else:
            lam = float(regularization) * singular_values[0].square()
            filt = singular_values / (singular_values.square() + lam)
        return vh.mH @ torch.diag(filt.to(self.complex_dtype)) @ u.mH

    @torch.no_grad()
    def reconstruct(
        self,
        raw_kspace: torch.Tensor | XSPENAcquisition,
        *,
        regularization: float = 1e-3,
    ) -> torch.Tensor:
        """Reconstruct an image from ideal xSPEN raw k-space."""
        if isinstance(raw_kspace, XSPENAcquisition):
            raw_kspace = raw_kspace.raw_kspace
        localized = self.raw_to_localized(raw_kspace)
        inverse = self.reconstruction_matrix(regularization=regularization)
        return torch.einsum("ya,...ax->...yx", inverse, localized).to(self.complex_dtype)


__all__ = ["XSPENAcquisition", "XSPENParameters", "XSPENSimulator"]
