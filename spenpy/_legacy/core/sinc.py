import torch


def math_sinc(x: torch.Tensor) -> torch.Tensor:
    """MATLAB ``MathSinc.m``: ``sin(x) / x`` with value 1 at zero."""
    out = torch.ones_like(x)
    nonzero = x != 0
    out[nonzero] = torch.sin(x[nonzero]) / x[nonzero]
    return out
