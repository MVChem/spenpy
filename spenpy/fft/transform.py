"""FFT transforms matching MATLAB conventions.

MATLAB: fftshift(fft(ifftshift(x, dim), [], dim), dim)  -- kspace to image
MATLAB: fftshift(ifft(ifftshift(x, dim), [], dim), dim)  -- image to kspace
"""

import torch


def fftshift_ifftshift_fft(x: torch.Tensor, dim: int) -> torch.Tensor:
    """MATLAB equivalent: fftshift(fft(ifftshift(x, dim), [], dim), dim).

    Converts k-space to image-space along a single dimension.
    """
    return torch.fft.fftshift(
        torch.fft.fft(torch.fft.ifftshift(x, dim=dim), dim=dim), dim=dim
    )


def fftshift_ifftshift_ifft(x: torch.Tensor, dim: int) -> torch.Tensor:
    """MATLAB equivalent: fftshift(ifft(ifftshift(x, dim), [], dim), dim).

    Converts image-space to k-space along a single dimension.
    """
    return torch.fft.fftshift(
        torch.fft.ifft(torch.fft.ifftshift(x, dim=dim), dim=dim), dim=dim
    )


def fft_kspace_to_xspace(x: torch.Tensor, dim: int) -> torch.Tensor:
    """K-space to image-space transform along `dim`."""
    return fftshift_ifftshift_fft(x, dim)


def fft_xspace_to_kspace(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Image-space to k-space transform along `dim`."""
    return fftshift_ifftshift_ifft(x, dim)
