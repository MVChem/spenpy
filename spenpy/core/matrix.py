"""SPEN encoding matrix: A and its weighted adjoint (InvA).

Ported from calcInvA.m and CalcSRMatrixApprox.m.
"""

import torch
from spenpy.core.sinc import math_sinc


def _real_dtype_from(*values) -> torch.dtype:
    for value in values:
        if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
            return value.dtype
    return torch.float64


def _device_from(*values) -> torch.device:
    for value in values:
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.device("cpu")


@torch.no_grad()
def calcSRMatrixApprox(
    MaxPhase: float,
    NumPixels: int,
    k: torch.Tensor,
    Partitions: torch.Tensor,
    b: float | None = None,
    ZeroThreshold: float | None = None,
):
    """Compute the approximate SPEN encoding matrix A.

    Matches MATLAB CalcSRMatrixApprox.m exactly.

    Returns:
        A: encoding matrix [NumKs, NumPixels]
        ADerivative: derivative of A
        IdxPositions: pixel center positions
        PartitionsUsed: partition borders
    """
    dtype = _real_dtype_from(k, Partitions)
    device = _device_from(k, Partitions)
    MaxPhase = torch.as_tensor(MaxPhase, dtype=dtype, device=device)
    k = torch.as_tensor(k, dtype=dtype, device=device)
    Partitions = torch.as_tensor(Partitions, dtype=dtype, device=device)

    DefaultZeroThreshold = 10 * torch.finfo(dtype).eps

    if ZeroThreshold is None:
        ZeroThreshold = DefaultZeroThreshold

    aEffective = MaxPhase / (Partitions[-1] - Partitions[0]) ** 2

    if b is None:
        # MATLAB default: sample at stationary point, middle of pixel.
        b = -(2 * aEffective * Partitions[0] * (1 - 1 / NumPixels) + k[0])
    else:
        b = torch.as_tensor(b, dtype=dtype, device=device)

    Partitions = Partitions.reshape(-1, 1)
    IdxPositions = (Partitions[:-1] + Partitions[1:]) / 2
    delta = Partitions[1:] - IdxPositions

    k = k.reshape(-1, 1)
    NumKs = len(k)

    deltaMat = delta.view(1, -1).expand(NumKs, -1)
    IdxPosMat = IdxPositions.view(1, -1).expand(NumKs, -1)
    kMat = k.expand(-1, NumPixels)

    LinCoeffMat = 2 * aEffective * IdxPosMat + b + kMat
    LinCoeff_x_delta_Mat = LinCoeffMat * deltaMat

    SincInput = LinCoeff_x_delta_Mat
    ExpInput = aEffective * IdxPosMat**2 + b * IdxPosMat + kMat * IdxPosMat

    HighOrder2 = 2 * (
        (LinCoeff_x_delta_Mat**2 - 2) * torch.sin(LinCoeff_x_delta_Mat)
        + 2 * LinCoeff_x_delta_Mat * torch.cos(LinCoeff_x_delta_Mat)
    ) / (LinCoeffMat**3)

    ZeroLinCoeffMatIdxs = torch.abs(LinCoeffMat) < ZeroThreshold
    HighOrder2[ZeroLinCoeffMatIdxs] = (2 / 3) * deltaMat[ZeroLinCoeffMatIdxs] ** 3

    DerivativeOrder1 = (
        2j / LinCoeffMat.to(torch.complex128 if dtype == torch.float64 else torch.complex64) ** 2
        * (
            torch.sin(LinCoeff_x_delta_Mat)
            - LinCoeff_x_delta_Mat * torch.cos(LinCoeff_x_delta_Mat)
        )
    )

    A = torch.exp(1j * ExpInput) * (
        (2 * deltaMat) * math_sinc(SincInput) + 1j * aEffective * HighOrder2
    )
    ADerivative = torch.exp(1j * ExpInput) * DerivativeOrder1

    return A, ADerivative, IdxPositions, Partitions


@torch.no_grad()
def calcInvA(
    a_rad2cmsqr: float,
    LPE: float,
    NumPE: int,
    ShiftPE: float,
    SPENAcquireSign: int,
    ky1RelativePos: float,
    GaussRelativeWidth: float,
):
    """Build the SPEN super-resolution encoding matrix and weighted adjoint.

    Ported from calcInvA.m.

    Returns:
        InvA: weighted adjoint reconstruction operator [NumPE, NumPE]
        AFinal: encoding matrix
    """
    dtype = _real_dtype_from(a_rad2cmsqr, LPE, ShiftPE, ky1RelativePos, GaussRelativeWidth)
    device = _device_from(a_rad2cmsqr, LPE, ShiftPE, ky1RelativePos, GaussRelativeWidth)
    a_rad2cmsqr = torch.as_tensor(a_rad2cmsqr, dtype=dtype, device=device)
    LPE = torch.as_tensor(LPE, dtype=dtype, device=device)
    ShiftPE = torch.as_tensor(ShiftPE, dtype=dtype, device=device)
    ky1RelativePos = torch.as_tensor(ky1RelativePos, dtype=dtype, device=device)
    GaussRelativeWidth = torch.as_tensor(GaussRelativeWidth, dtype=dtype, device=device)

    MaxPhase = a_rad2cmsqr * LPE**2

    NumPixels = NumPE
    NumPixelsFinal = NumPE

    Partitions = SPENAcquireSign * torch.linspace(
        -LPE.item() / 2, LPE.item() / 2, NumPixels + 1, dtype=dtype, device=device
    ) + ShiftPE / 10
    PartitionsFinal = SPENAcquireSign * torch.linspace(
        -LPE.item() / 2, LPE.item() / 2, NumPixelsFinal + 1, dtype=dtype, device=device
    ) + ShiftPE / 10

    ky = (
        -2
        * SPENAcquireSign
        * a_rad2cmsqr
        * torch.arange(NumPE, dtype=dtype, device=device)
        * LPE
        / NumPE
    )

    b = -ky[0] + -2 * a_rad2cmsqr * (
        Partitions[0] + (Partitions[1] - Partitions[0]) * ky1RelativePos
    )

    AFinal = calcSRMatrixApprox(
        MaxPhase, NumPixelsFinal, ky, PartitionsFinal, b
    )[0]

    GaussWeightVar = (GaussRelativeWidth * torch.pi * NumPixelsFinal**2 / MaxPhase) ** 2

    yk = -(b + ky) / (2 * a_rad2cmsqr)
    yPixels = (PartitionsFinal[:-1] + PartitionsFinal[1:]) / 2

    DistMat = NumPixelsFinal / LPE * (yk.unsqueeze(1) - yPixels.unsqueeze(0))
    GaussWeight = torch.exp(-DistMat**2 / (2 * GaussWeightVar))

    AGaussWeighted = AFinal * GaussWeight

    # InvA is the conjugate transpose (weighted adjoint), NOT a matrix inverse
    InvA = AGaussWeighted.conj().T

    return InvA, AFinal
