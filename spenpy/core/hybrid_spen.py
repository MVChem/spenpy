"""Hybrid SPEN-Diff2 PE encoding in the archived Siemens SR convention."""

import math

import torch

from .protocol import HybridSPENDiff2Protocol


def calc_hybrid_spen_encoding(
    protocol, image_y, *, device="cpu", dtype=torch.complex64
):
    """Return A[PE sample, image pixel], with integrals expressed in cm.

    A integrates exp(i*(a*y**2 + (b+k_m)*y)) using the second-order
    intra-pixel expansion from CalcSRMatrixApprox.m. Image and acquisition
    grids are independent. No window, inverse, phase fit or target is used.
    """
    from .._legacy.core.matrix import calcSRMatrixApprox

    if not isinstance(protocol, HybridSPENDiff2Protocol):
        raise TypeError("Expected HybridSPENDiff2Protocol")
    if not isinstance(image_y, int) or isinstance(image_y, bool) or image_y < 1:
        raise ValueError("image_y must be a positive integer")
    if dtype not in (torch.complex64, torch.complex128):
        raise ValueError("Expected a complex dtype")
    # Build in float64, including the stable small-argument moment limits.
    length = protocol.fov_m[0] * 100
    phase = 2 * math.pi * protocol.r_value
    a = phase / length**2
    m = protocol.acquisition_shape[0]
    borders = torch.linspace(
        -length / 2, length / 2, image_y + 1, device=device, dtype=torch.float64
    )
    ky = -2 * a * torch.arange(m, device=device, dtype=torch.float64) * length / m
    return calcSRMatrixApprox(
        phase, image_y, ky, borders, a * length, stable_integrals=True
    )[0].to(dtype)
