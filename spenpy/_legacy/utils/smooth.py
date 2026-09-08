"""2D smoothing utilities.

Ported from smooth2a.m -- mean filter over a (2*Nr+1) x (2*Nc+1) rectangle,
ignoring NaN values.
"""

import torch
import torch.nn.functional as F


def smooth2a(matrix_in: torch.Tensor, Nr: int, Nc: int | None = None) -> torch.Tensor:
    """Smooth a 2D array using a mean filter, ignoring NaNs.

    Args:
        matrix_in: [rows, cols] tensor
        Nr: smoothing window half-size along rows
        Nc: smoothing window half-size along columns (defaults to Nr)
    """
    if Nc is None:
        Nc = Nr

    rows, cols = matrix_in.shape
    A = torch.isnan(matrix_in)
    data = matrix_in.clone()
    data[A] = 0

    kernel_h = 2 * Nr + 1
    kernel_w = 2 * Nc + 1
    kernel = torch.ones(1, 1, kernel_h, kernel_w, device=data.device, dtype=data.dtype)

    data_4d = data.unsqueeze(0).unsqueeze(0)
    a_4d = A.float().unsqueeze(0).unsqueeze(0)

    smoothed = F.conv2d(data_4d, kernel, padding=(Nr, Nc)).squeeze()
    count = F.conv2d((~A).float().unsqueeze(0).unsqueeze(0), kernel, padding=(Nr, Nc)).squeeze()

    count[A] = float("nan")
    return smoothed / count
