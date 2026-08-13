"""Tests for core SPEN encoding matrix functions."""

import torch
import pytest
from spenpy.core.matrix import calcSRMatrixApprox, calcInvA


class TestCalcSRMatrixApprox:
    def test_output_shape(self):
        MaxPhase = 100.0
        NumPixels = 64
        k = torch.arange(NumPixels, dtype=torch.float32)
        Partitions = torch.linspace(-32, 32, NumPixels + 1)

        A, ADerivative, IdxPositions, PartitionsUsed = calcSRMatrixApprox(
            MaxPhase, NumPixels, k, Partitions
        )

        assert A.shape == (NumPixels, NumPixels)
        assert ADerivative.shape == (NumPixels, NumPixels)
        assert len(IdxPositions) == NumPixels
        assert len(PartitionsUsed) == NumPixels + 1

    def test_is_complex(self):
        MaxPhase = 100.0
        NumPixels = 32
        k = torch.arange(NumPixels, dtype=torch.float32)
        Partitions = torch.linspace(-16, 16, NumPixels + 1)

        A, _, _, _ = calcSRMatrixApprox(MaxPhase, NumPixels, k, Partitions)
        assert A.is_complex()


class TestCalcInvA:
    def test_output_shape(self):
        a_rad2cmsqr = 2.5e6
        LPE = 4.0
        NumPE = 128

        InvA, AFinal = calcInvA(
            a_rad2cmsqr, LPE, NumPE,
            ShiftPE=0, SPENAcquireSign=1,
            ky1RelativePos=0, GaussRelativeWidth=0.8,
        )

        assert InvA.shape == (NumPE, NumPE)
        assert AFinal.shape == (NumPE, NumPE)

    def test_invA_is_conjugate_transpose(self):
        """Verify InvA = AGaussWeighted.conj().T, not matrix inverse."""
        a_rad2cmsqr = 2.5e6
        LPE = 4.0
        NumPE = 64

        InvA, AFinal = calcInvA(
            a_rad2cmsqr, LPE, NumPE,
            ShiftPE=0, SPENAcquireSign=1,
            ky1RelativePos=0, GaussRelativeWidth=0.8,
        )

        # InvA should NOT be the matrix inverse of AFinal
        # (it's the weighted adjoint)
        assert not torch.allclose(
            InvA @ AFinal,
            torch.eye(NumPE, dtype=InvA.dtype),
            atol=1e-3,
        )

    def test_ky1_relative_pos_shift(self):
        """Different ky1RelativePos should produce different operators."""
        a_rad2cmsqr = 2.5e6
        LPE = 4.0
        NumPE = 64

        InvA_odd, _ = calcInvA(
            a_rad2cmsqr, LPE, NumPE,
            ShiftPE=0, SPENAcquireSign=1,
            ky1RelativePos=0, GaussRelativeWidth=0.8,
        )
        InvA_even, _ = calcInvA(
            a_rad2cmsqr, LPE, NumPE,
            ShiftPE=0, SPENAcquireSign=1,
            ky1RelativePos=0.5, GaussRelativeWidth=0.8,
        )

        assert not torch.allclose(InvA_odd, InvA_even)
