"""Spatial super-resolution matrices and reconstruction of simulated acquisitions."""

import math

import torch

from ..core import (
    EncodingOperator,
    HybridSPENDiff2Protocol,
    SPEN180Protocol,
    XSPENProtocol,
)
from .even_odd import reconstruct_even_odd_inva
from .phasemap import _frame


def calc_inva_matrices(
    protocol, *, device="cpu", dtype=torch.complex64, gaussian_width=0.8
):
    """Native-grid full/odd/even windowed InvA for the two ideal encodings.

    SPEN uses calcInvA exactly as in the notebooks, in centimetres. xSPEN
    applies the same windowed-adjoint construction to its actual crossed-chirp
    kernel and focus positions; this is not the quadratic Siemens sequence.
    """
    from .._legacy.core.matrix import calcInvA

    if isinstance(protocol, HybridSPENDiff2Protocol):
        raise TypeError(
            "Use calc_hybrid_spen_matrices for Hybrid SPEN-Diff2's rectangular half-resolution matrices"
        )
    if not isinstance(protocol, (SPEN180Protocol, XSPENProtocol)):
        raise TypeError("Expected SPEN180Protocol or XSPENProtocol")
    if dtype not in (torch.complex64, torch.complex128):
        raise ValueError("dtype must be complex64 or complex128")
    m, k = protocol.acquisition_shape
    if m < 4 or m % 2:
        raise ValueError("Requires an even number of PE samples >= 4")
    if not math.isfinite(gaussian_width) or gaussian_width <= 0:
        raise ValueError("gaussian_width must be finite and positive")
    if isinstance(protocol, SPEN180Protocol):
        a = torch.tensor(
            protocol.phase_coefficient_rad_m2 / 10000,
            dtype=torch.float64,
            device=device,
        )
        offset = protocol.focus_offset_voxels
        result = []
        for size, relative in (
            (m, offset),
            (m // 2, offset / 2),
            (m // 2, (offset + 1) / 2),
        ):
            inv, _ = calcInvA(
                a,
                protocol.fov_m[0] * 100,
                size,
                0,
                1,
                relative,
                gaussian_width,
                stable_integrals=True,
            )
            result.append(inv.to(dtype))
        return tuple(result)
    if not isinstance(protocol, XSPENProtocol):
        raise TypeError("Expected SPEN180Protocol or XSPENProtocol")
    matrices = []
    for size, parity in ((m, None), (m // 2, 0), (m // 2, 1)):
        operator = EncodingOperator(
            protocol, image_shape=(size, k), device=device, dtype=dtype
        )
        kernel, focus = operator.encoding_kernel, operator.focus_positions_m
        if parity is not None:
            kernel, focus = kernel[parity::2], focus[parity::2]
        distance = size / protocol.fov_m[0] * (focus[:, None] - operator.y_m[None])
        sigma = gaussian_width * size**2 / (2 * protocol.r_value)
        matrices.append((kernel * torch.exp(-distance.square() / (2 * sigma**2))).mH)
    return tuple(matrices)


def reconstruct_simulated_inva(
    data, operator, *, estimator="quadratic", gaussian_width=0.8
):
    """Decode a simulated SPEN180, xSPEN or Hybrid SPEN-Diff2 acquisition.

    The calibrated unitary DFT accounts for voxel-centre origin. No image-grid
    upsampling, PE pseudoinverse, or fitted intensity scaling is performed.
    SPEN signal units are converted to the centimetre-integral matrix convention.
    """
    signal = _frame(data, {"uniform_kspace"})
    if not isinstance(operator, EncodingOperator):
        raise TypeError("Expected EncodingOperator")
    if tuple(signal.shape) != operator.measurement_shape:
        raise ValueError("Signal shape does not match the operator")
    if operator.encoding_kernel.ndim != 2 or not torch.all(operator.sample_mask == 1):
        raise ValueError("Requires separable, fully sampled ideal encoding")
    # Reject altered 2D kernels (e.g. slice profiles) whose protocol alone does
    # not describe the simulated physics used to build these InvA matrices.
    nominal = EncodingOperator(
        operator.protocol,
        operator.image_shape,
        device=operator.coil_maps.device,
        dtype=operator.coil_maps.dtype,
    )
    if not torch.allclose(
        operator.encoding_kernel, nominal.encoding_kernel, atol=1e-6, rtol=1e-5
    ):
        raise ValueError("InvA calibration requires the nominal protocol kernel")
    if isinstance(operator.protocol, HybridSPENDiff2Protocol):
        from .hybrid_spen import reconstruct_hybrid_spen

        # Simulated geometry is known, as in the CG operator. The odd/even
        # phase is still estimated from the measurements. The scan-wide
        # automatic shift estimator is intended for measured acquisitions;
        # its phase ambiguities need not identify a synthetic phantom's origin.
        if estimator != "quadratic":
            raise ValueError(
                "Hybrid SPEN-Diff2 uses its historical polynomial phase estimator"
            )
        result = reconstruct_hybrid_spen(
            data, shift_pixels=operator.protocol.spen_shift_pixels
        )
        result.metadata["shift_calibration"] = "simulation_protocol"
        return result
    signal = signal.to(operator.encoding_kernel)
    m, k, _ = signal.shape
    rdtype, device = signal.real.dtype, signal.device
    frequency = torch.arange(k, device=device, dtype=rdtype) - k // 2
    centres = (torch.arange(k, device=device, dtype=rdtype) + 0.5) / k - 0.5
    inverse_ro = torch.exp(
        -2j * math.pi * centres[:, None] * frequency[None]
    ) / math.sqrt(k)
    ro = torch.einsum("xk,mkc->mxc", inverse_ro, signal)
    if isinstance(operator.protocol, SPEN180Protocol):
        ro = ro * (operator.protocol.fov_m[0] * 100 / math.sqrt(m))
    matrices = calc_inva_matrices(
        operator.protocol,
        device=device,
        dtype=signal.dtype,
        gaussian_width=gaussian_width,
    )
    result = reconstruct_even_odd_inva(ro, *matrices, estimator=estimator)
    result.metadata.update(
        sequence=operator.protocol.sequence,
        gaussian_width=gaussian_width,
        readout_transform="calibrated unitary Fourier inverse on native RO grid",
        encoding_model=operator.protocol.sequence,
    )
    return result
