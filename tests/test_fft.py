"""Tests for FFT transform functions."""

import torch
import numpy as np
from spenpy.fft.transform import fft_kspace_to_xspace, fft_xspace_to_kspace


class TestFFTConventions:
    def test_roundtrip(self):
        """K-space -> image -> K-space should recover original."""
        x = torch.randn(64, 128, dtype=torch.complex64)
        img = fft_kspace_to_xspace(x, dim=0)
        x_back = fft_xspace_to_kspace(img, dim=0)
        assert torch.allclose(x, x_back, atol=1e-5)

    def test_kspace_to_xspace_shape(self):
        x = torch.randn(32, 64, dtype=torch.complex64)
        img = fft_kspace_to_xspace(x, dim=0)
        assert img.shape == x.shape

    def test_batch_roundtrip(self):
        x = torch.randn(4, 32, 64, dtype=torch.complex64)
        img = fft_kspace_to_xspace(x, dim=1)
        x_back = fft_xspace_to_kspace(img, dim=1)
        assert torch.allclose(x, x_back, atol=1e-5)

    def test_matches_numpy_ifft(self):
        """Verify fftshift(fft(ifftshift(x))) matches numpy."""
        np.random.seed(42)
        x_np = np.random.randn(64) + 1j * np.random.randn(64)

        # NumPy equivalent of MATLAB fftshift(fft(ifftshift(x)))
        x_ifftshift = np.fft.ifftshift(x_np)
        x_fft = np.fft.fft(x_ifftshift)
        result_np = np.fft.fftshift(x_fft)

        # PyTorch
        x_torch = torch.from_numpy(x_np)
        result_torch = fft_kspace_to_xspace(x_torch, dim=0)

        assert torch.allclose(result_torch, torch.from_numpy(result_np), atol=1e-10)
