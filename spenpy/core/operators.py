"""Deterministic Torch forward/adjoint operators on independent image grids.

Images: [..., y, x]. Measurements: [..., SPEN sample, RO sample, coil].
Positive readout Fourier sign follows the legacy image-to-kspace IFFT sign.
Image pixels are at voxel centres. Pixel quadrature and normalization are
explicit; vendor FFT origins/scaling must be calibrated before real-data DC.
"""

import math
from functools import lru_cache

import numpy as np
import torch
from torch import nn

from .protocol import HybridSPENDiff2Protocol, SPEN180Protocol, XSPENProtocol


def voxel_centers(n, length, *, device, dtype):
    return ((torch.arange(n, device=device, dtype=dtype) + 0.5) / n - 0.5) * length


@lru_cache(maxsize=32)
def _pixel_quadrature(n):
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return nodes / 2, weights / 2


class EncodingOperator(nn.Module):
    """Shared multi-coil encoding. All nuisance maps are fixed during forward.

    The fast model uses one encoding/B0 time per readout line and a separate
    Fourier readout. It does not model evolution within each RO line.
    """

    def __init__(
        self,
        protocol,
        image_shape=None,
        *,
        coil_maps=None,
        device=None,
        dtype=torch.complex64,
        quadrature=None,
        phase_map_rad=None,
        sample_mask=None,
        b0_hz=None,
        effective_times_s=None,
        t2_s=None,
        decay_times_s=None,
        slice_profile=None,
        n_z=128,
    ):
        super().__init__()
        if dtype not in (torch.complex64, torch.complex128):
            raise ValueError("dtype must be complex64 or complex128")
        self.protocol = protocol
        self.image_shape = tuple(image_shape or protocol.acquisition_shape)
        if len(self.image_shape) != 2 or any(
            not isinstance(n, int) or n < 1 for n in self.image_shape
        ):
            raise ValueError("image_shape must be two positive integers")
        if quadrature is None:
            # Resolve the fastest chirp variation within each image pixel.
            quadrature = max(
                16, math.ceil(2 * math.pi * protocol.r_value / self.image_shape[0]) + 8
            )
        if not isinstance(quadrature, int) or quadrature < 1:
            raise ValueError("quadrature must be None or a positive integer")
        self.quadrature = quadrature
        if device is None and isinstance(coil_maps, torch.Tensor):
            device = coil_maps.device
        rdtype = torch.float64 if dtype == torch.complex128 else torch.float32
        ny, nx = self.image_shape
        m, k = protocol.acquisition_shape
        y = voxel_centers(ny, protocol.fov_m[0], device=device, dtype=rdtype)
        x = voxel_centers(nx, protocol.fov_m[1], device=device, dtype=rdtype)
        self.register_buffer("y_m", y)
        self.register_buffer("x_m", x)
        if coil_maps is None:
            coil_maps = torch.ones((ny, nx, 1), device=device, dtype=dtype)
        coil_maps = torch.as_tensor(coil_maps, device=device).to(dtype)
        if (
            coil_maps.ndim != 3
            or coil_maps.shape[:2] != (ny, nx)
            or coil_maps.shape[2] < 1
        ):
            raise ValueError("coil_maps must be [image_y,image_x,coil]")
        if not torch.isfinite(coil_maps).all():
            raise ValueError("coil_maps must be finite")
        self.register_buffer("coil_maps", coil_maps)

        offsets_np, weights_np = _pixel_quadrature(quadrature)
        offsets = torch.as_tensor(offsets_np, device=device, dtype=rdtype)
        weights = torch.as_tensor(weights_np, device=device, dtype=rdtype)
        yq = y[:, None] + offsets * protocol.fov_m[0] / ny
        if isinstance(protocol, (SPEN180Protocol, HybridSPENDiff2Protocol)):
            if slice_profile is not None:
                raise ValueError("slice_profile is only supported by xSPEN")
            focus = (
                (
                    torch.arange(m, device=device, dtype=rdtype)
                    + getattr(protocol, "focus_offset_voxels", 0.0)
                )
                / m
                - 0.5
            ) * protocol.fov_m[0]
            a = protocol.phase_coefficient_rad_m2
            kernel = (
                torch.exp(
                    1j * a * (yq[None] ** 2 - 2 * focus[:, None, None] * yq[None])
                )
                * weights
            ).sum(-1)
            kernel = kernel * (math.sqrt(m) / ny)
            if isinstance(protocol, HybridSPENDiff2Protocol):
                from .hybrid_spen import calc_hybrid_spen_encoding

                kernel = calc_hybrid_spen_encoding(
                    protocol, ny, device=device, dtype=dtype
                )
                # Inverse of the historical position-phase correction.
                ramp = torch.exp(
                    -4j
                    * math.pi
                    * protocol.r_value
                    / m**2
                    * protocol.spen_shift_pixels
                    * torch.arange(1, m + 1, device=device, dtype=rdtype)
                )
                kernel = kernel * ramp[:, None]
            times = (
                torch.arange(m, device=device, dtype=rdtype) - (m - 1) / 2
            ) * protocol.echo_spacing_s
            if b0_hz is not None:
                if effective_times_s is None:
                    raise ValueError(
                        "SPEN B0 requires explicit effective_times_s for the refocusing timing"
                    )
                field = self._map(b0_hz, (ny, nx), device, rdtype, "b0_hz")
                teff = self._map(
                    effective_times_s, (m,), device, rdtype, "effective_times_s"
                )
                kernel = kernel[..., None] * torch.exp(
                    2j * math.pi * teff[:, None, None] * field
                )
            elif effective_times_s is not None:
                raise ValueError("effective_times_s requires b0_hz")
        elif isinstance(protocol, XSPENProtocol):
            if effective_times_s is not None:
                raise ValueError("xSPEN derives its B0 timing from the crossed chirps")
            times = (
                torch.arange(m, device=device, dtype=rdtype) + 1
            ) * protocol.echo_spacing_s - protocol.acquisition_time_s / 2
            focus = (
                -protocol.gamma_hz_t
                * protocol.gz_t_m
                * times
                / protocol.cross_term_cycles_m2
            )
            kernel = self._xspen_kernel(
                protocol, yq, weights, times, nx, b0_hz, slice_profile, n_z
            )
            kernel = kernel * (m / ny)
        else:
            raise TypeError(
                "Use SPEN180Protocol, XSPENProtocol or HybridSPENDiff2Protocol"
            )
        if t2_s is not None:
            if decay_times_s is None:
                raise ValueError("t2_s requires nonnegative decay_times_s")
            t2 = self._map(t2_s, (ny, nx), device, rdtype, "t2_s")
            td = self._map(decay_times_s, (m,), device, rdtype, "decay_times_s")
            if (t2 <= 0).any() or (td < 0).any():
                raise ValueError("T2 must be positive and decay times nonnegative")
            if kernel.ndim == 2:
                kernel = kernel[..., None]
            kernel = kernel * torch.exp(-td[:, None, None] / t2)
        elif decay_times_s is not None:
            raise ValueError("decay_times_s requires t2_s")
        self.register_buffer("encoding_kernel", kernel.to(dtype))
        self.register_buffer("sample_times_s", times)
        self.register_buffer("focus_positions_m", focus)

        # Pixel integrals of a positive-sign Fourier basis; not image resizing.
        freqs = (
            torch.arange(k, device=device, dtype=rdtype) - k // 2
        ) / protocol.fov_m[1]
        ro = (
            torch.exp(2j * math.pi * freqs[:, None] * x[None])
            * torch.sinc(freqs[:, None] * protocol.fov_m[1] / nx)
            * (math.sqrt(k) / nx)
        )
        if isinstance(protocol, HybridSPENDiff2Protocol):
            # Centred IFFT on the native grid, matching FFTXSpace2KSpace.m.
            # On an independent HR grid this is Fourier quadrature/truncation.
            # This legacy discrete RO basis does not include pixel apodization.
            centre_offset = (0.5 + k // 2 - k / 2) * protocol.fov_m[1] / k
            ro = (
                torch.exp(2j * math.pi * freqs[:, None] * (x[None] - centre_offset))
                / nx
            )
        self.register_buffer("readout_matrix", ro.to(dtype))
        phase = (
            torch.zeros((m, nx), device=device, dtype=rdtype)
            if phase_map_rad is None
            else self._map(phase_map_rad, (m, nx), device, rdtype, "phase_map_rad")
        )
        self.register_buffer("phase_factor", torch.exp(1j * phase).to(dtype))
        mask = (
            torch.ones((m, k), device=device, dtype=rdtype)
            if sample_mask is None
            else self._map(sample_mask, (m, k), device, rdtype, "sample_mask")
        )
        if ((mask < 0) | (mask > 1)).any():
            raise ValueError("sample_mask values must lie in [0,1]")
        self.register_buffer("sample_mask", mask)

    @staticmethod
    def _map(value, shape, device, dtype, name):
        out = torch.as_tensor(value, device=device, dtype=dtype)
        try:
            out = torch.broadcast_to(out, shape)
        except RuntimeError as exc:
            raise ValueError(f"{name} must broadcast to {shape}") from exc
        if not torch.isfinite(out).all():
            raise ValueError(f"{name} must be finite")
        return out

    @staticmethod
    def _xspen_kernel(p, yq, y_weights, times, nx, b0_hz, profile, n_z):
        q = (
            p.cross_term_cycles_m2 * yq[None]
            + p.gamma_hz_t * p.gz_t_m * times[:, None, None]
        )
        if b0_hz is None and profile is None:
            return (
                (torch.sinc(q * p.slice_thickness_m) * y_weights)
                .sum(-1)
                .to(torch.complex128 if yq.dtype == torch.float64 else torch.complex64)
            )
        if not isinstance(n_z, int) or n_z < 2:
            raise ValueError("n_z must be at least 2")
        z = voxel_centers(n_z, p.slice_thickness_m, device=yq.device, dtype=yq.dtype)
        weights = (
            torch.ones_like(z)
            if profile is None
            else torch.as_tensor(profile, device=z.device, dtype=z.dtype)
        )
        if (
            weights.shape != (n_z,)
            or not torch.isfinite(weights).all()
            or (weights < 0).any()
            or weights.sum() <= 0
        ):
            raise ValueError(
                "slice_profile must be nonnegative [n_z] with positive sum"
            )
        weights = weights / weights.sum()
        if b0_hz is None:
            return (
                (torch.exp(2j * math.pi * q[..., None] * z) * weights).sum(-1)
                * y_weights
            ).sum(-1)
        # Explicit auxiliary integration, including ideal rectangular RF masks.
        # Object and B0 are constant within each in-plane voxel and along z.
        field = EncodingOperator._map(
            b0_hz, (yq.shape[0], nx), yq.device, yq.dtype, "b0_hz"
        )
        gy_y = p.gamma_hz_t * p.gy_t_m * yq[:, :, None]
        accum = 0
        for zi, wi in zip(z, weights):
            w = p.gamma_hz_t * p.gz_t_m * zi + field[:, None, :]
            mask = (
                (w.abs() <= (1 - p.beta) * p.chirp_bandwidth_hz / 2)
                & ((w - gy_y).abs() <= p.chirp_bandwidth_hz / 2)
                & ((w + gy_y).abs() <= p.chirp_bandwidth_hz / 2)
            )
            phase = (
                times[:, None, None, None]
                - 4 * p.chirp_duration_s / p.chirp_bandwidth_hz * gy_y[None]
            ) * w[None]
            accum = accum + wi * (
                torch.exp(2j * math.pi * phase)
                * mask[None]
                * y_weights[None, None, :, None]
            ).sum(-2)
        return accum

    @property
    def measurement_shape(self):
        return (*self.protocol.acquisition_shape, self.coil_maps.shape[-1])

    def forward(self, image):
        image = torch.as_tensor(image, device=self.coil_maps.device).to(
            self.coil_maps.dtype
        )
        if image.ndim < 2 or tuple(image.shape[-2:]) != self.image_shape:
            raise ValueError(f"image must end in {self.image_shape}")
        coils = image[..., None] * self.coil_maps
        if self.encoding_kernel.ndim == 2:
            localized = torch.einsum("my,...yxc->...mxc", self.encoding_kernel, coils)
        else:
            localized = torch.einsum("myx,...yxc->...mxc", self.encoding_kernel, coils)
        localized = localized * self.phase_factor[..., None]
        return (
            torch.einsum("kx,...mxc->...mkc", self.readout_matrix, localized)
            * self.sample_mask[..., None]
        )

    def adjoint(self, measurements):
        data = torch.as_tensor(measurements, device=self.coil_maps.device).to(
            self.coil_maps.dtype
        )
        if data.ndim < 3 or tuple(data.shape[-3:]) != self.measurement_shape:
            raise ValueError(f"measurements must end in {self.measurement_shape}")
        data = data * self.sample_mask[..., None]
        localized = torch.einsum("kx,...mkc->...mxc", self.readout_matrix.conj(), data)
        localized = localized * self.phase_factor.conj()[..., None]
        if self.encoding_kernel.ndim == 2:
            coils = torch.einsum(
                "my,...mxc->...yxc", self.encoding_kernel.conj(), localized
            )
        else:
            coils = torch.einsum(
                "myx,...mxc->...yxc", self.encoding_kernel.conj(), localized
            )
        return (coils * self.coil_maps.conj()).sum(-1)

    def normal(self, image):
        return self.adjoint(self(image))

    def data_gradient(self, image, measurements):
        """Gradient of 0.5 * ||Ax-y||^2 under the complex real-inner-product convention."""
        return self.adjoint(self(image) - measurements)

    def local_psf(self, index):
        impulse = torch.zeros(
            self.image_shape, device=self.coil_maps.device, dtype=self.coil_maps.dtype
        )
        impulse[index] = 1
        return self.normal(impulse)


class SPEN180Operator(EncodingOperator):
    def __init__(self, protocol=None, image_shape=None, **kwargs):
        p = protocol or SPEN180Protocol()
        if not isinstance(p, SPEN180Protocol):
            raise TypeError("SPEN180Operator requires SPEN180Protocol")
        super().__init__(p, image_shape, **kwargs)


class XSPENOperator(EncodingOperator):
    def __init__(self, protocol=None, image_shape=None, **kwargs):
        p = protocol or XSPENProtocol()
        if not isinstance(p, XSPENProtocol):
            raise TypeError("XSPENOperator requires XSPENProtocol")
        super().__init__(p, image_shape, **kwargs)


class HybridSPENDiff2Operator(EncodingOperator):
    def __init__(self, protocol=None, image_shape=None, **kwargs):
        p = protocol or HybridSPENDiff2Protocol()
        if not isinstance(p, HybridSPENDiff2Protocol):
            raise TypeError("HybridSPENDiff2Operator requires HybridSPENDiff2Protocol")
        super().__init__(p, image_shape, **kwargs)


SPENOperator = SPEN180Operator
