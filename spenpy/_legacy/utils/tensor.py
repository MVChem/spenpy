"""Tensor utilities.

Ported from MultMatTensor.m -- applies a matrix to the first dimension of a tensor.
"""

import torch


def mult_mat_tensor(M: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Apply matrix M to dimension 0 of tensor T.

    MATLAB: Out(:,:,i,j,k,l) = M * T(:,:,i,j,k,l)

    Uses torch.einsum for efficiency instead of nested loops.
    """
    # M: [out_dim, in_dim], T: [in_dim, ...]
    # Result: [out_dim, ...]
    ndim = T.dim()
    if ndim == 2:
        return M @ T
    # For higher dimensions, use einsum
    # M has indices (m, n), T has indices (n, a, b, c, ...)
    # Result has indices (m, a, b, c, ...)
    letters = "abcdefghijklmnop"
    t_indices = letters[1:ndim]  # skip 'a' which is used by M's contracted dim
    expr = f"ma,a{t_indices}->m{t_indices}"
    return torch.einsum(expr, M, T)
