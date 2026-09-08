"""Zero-filling utilities matching ParaVision conventions.

Ported from Zero_filling_PV6.m and RM_Zero_filling_PV6.m.
"""

import torch


def _matlab_round(x):
    """MATLAB-style round: 0.5 rounds up (away from zero for positive)."""
    import math
    return int(math.floor(x + 0.5))


def zero_filling_pv6(kField: torch.Tensor, ZF: int) -> torch.Tensor:
    """Zero-fill k-space along dim 0 to size ZF.

    Preserves symmetry: data is centered in the larger array.
    MATLAB uses 1-based indexing, so the conversion to Python 0-based
    requires careful handling of the trunc/2 boundaries.
    """
    ZF = max(ZF, kField.shape[0])
    orig_size = kField.shape[0]
    kFieldZF = torch.zeros(ZF, kField.shape[1], dtype=kField.dtype, device=kField.device)
    trunc = ZF - orig_size
    half = _matlab_round(trunc / 2)
    if trunc % 2 == 1:
        # MATLAB: kFieldZF(half:end-half, :)
        # Python: kFieldZF[half-1 : ZF-half, :]
        kFieldZF[half - 1:ZF - half, :] = kField
    else:
        # MATLAB: kFieldZF(half+1:end-half, :)
        # Python: kFieldZF[half : ZF-half, :]
        kFieldZF[half:ZF - half, :] = kField
    return kFieldZF


def rm_zero_filling_pv6(kField: torch.Tensor, size_orig: int) -> torch.Tensor:
    """Remove zero-filling along dim 0, cropping back to size_orig."""
    size_orig = min(size_orig, kField.shape[0])
    trunc = kField.shape[0] - size_orig
    half = _matlab_round(trunc / 2)
    if trunc % 2 == 1:
        return kField[half - 1:size_orig + half - 1, :]
    else:
        return kField[half:size_orig + half, :]
