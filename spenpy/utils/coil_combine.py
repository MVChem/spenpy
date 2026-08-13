"""Coil combination for multi-coil MRI data.

Ported from coilCombinebao.m -- Walsh et al. adaptive reconstruction.
"""

import torch
import torch.nn.functional as F


def coil_combine(im1: torch.Tensor) -> torch.Tensor:
    """Combine multi-coil complex images using adaptive reconstruction.

    Input shape:  [sx, sy], [sx, sy, N_coils], or [sx, sy, N_images, N_coils]
    Output shape: [sx, sy] or [sx, sy, N_images]

    Based on: Walsh DO, Gmitro AF, Marcellin MW. Adaptive reconstruction
    of phased array MR imagery. Magn Reson Med 2000;43:682-690.
    """
    if im1.dim() == 2:
        return im1
    if im1.dim() == 3:
        im = im1.unsqueeze(2)
        squeeze_n = True
    elif im1.dim() == 4:
        im = im1
        squeeze_n = False
    else:
        raise ValueError("coil_combine expects a 2D, 3D, or 4D tensor")

    sx, sy, n_images, n_coils = im.shape
    if n_coils == 1:
        out = im[..., 0]
        return out[..., 0] if squeeze_n else out

    filtsize = 7

    rs = torch.zeros(sx, sy, n_coils, n_coils, dtype=im.dtype, device=im.device)
    kernel = torch.ones(1, 1, filtsize, filtsize, dtype=im.real.dtype, device=im.device)

    for kc1 in range(n_coils):
        for kc2 in range(n_coils):
            acc = torch.zeros(sx, sy, dtype=im.dtype, device=im.device)
            for kn in range(n_images):
                prod = im[:, :, kn, kc1] * torch.conj(im[:, :, kn, kc2])
                smoothed_real = F.conv2d(
                    prod.real.unsqueeze(0).unsqueeze(0), kernel, padding=filtsize // 2
                ).squeeze()
                smoothed_imag = F.conv2d(
                    prod.imag.unsqueeze(0).unsqueeze(0), kernel, padding=filtsize // 2
                ).squeeze()
                acc = acc + smoothed_real + 1j * smoothed_imag
            rs[:, :, kc1, kc2] = acc

    im2 = torch.zeros(sx, sy, n_images, dtype=im.dtype, device=im.device)
    for kx in range(sx):
        for ky in range(sy):
            R_mat = rs[kx, ky, :, :]
            U, _, _ = torch.linalg.svd(R_mat)
            myfilt = U[:, 0]
            samples = im[kx, ky, :, :].transpose(0, 1)
            im2[kx, ky, :] = myfilt.conj() @ samples

    return im2[:, :, 0] if squeeze_n else im2


def coil_combine_batched(
    images: torch.Tensor,
    batch_size: int | None = None,
) -> torch.Tensor:
    """Vectorized Walsh adaptive combination for batches of image stacks.

    Args:
        images: Complex tensor shaped
            ``[batch, x, y, n_images, n_coils]``.
        batch_size: Number of leading-axis items processed at once.  Use this
            to limit memory for multi-slice volumes.

    Returns:
        Complex tensor shaped ``[batch, x, y, n_images]``.

    This is mathematically equivalent to :func:`coil_combine`, but performs
    the per-pixel SVD as a batched operation.  That makes whole-volume Siemens
    reconstruction practical while retaining slice-specific coil weights.
    """
    if images.dim() != 5:
        raise ValueError(
            "coil_combine_batched expects [batch, x, y, n_images, n_coils]"
        )
    if not torch.is_complex(images):
        raise ValueError("coil_combine_batched expects complex-valued images")

    batch, sx, sy, _n_images, n_coils = images.shape
    if n_coils == 1:
        return images[..., 0]

    if batch_size is None:
        batch_size = batch
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    filtsize = 7
    kernel = torch.ones(
        1,
        1,
        filtsize,
        filtsize,
        dtype=images.real.dtype,
        device=images.device,
    )
    output = torch.empty(
        batch,
        sx,
        sy,
        images.shape[3],
        dtype=images.dtype,
        device=images.device,
    )

    for start in range(0, batch, batch_size):
        stop = min(start + batch_size, batch)
        block = images[start:stop]

        covariance = torch.einsum(
            "bxyic,bxyid->bxycd",
            block,
            block.conj(),
        )
        covariance_nchw = covariance.permute(0, 3, 4, 1, 2).reshape(
            -1,
            1,
            sx,
            sy,
        )
        smoothed = torch.complex(
            F.conv2d(
                covariance_nchw.real,
                kernel,
                padding=filtsize // 2,
            ),
            F.conv2d(
                covariance_nchw.imag,
                kernel,
                padding=filtsize // 2,
            ),
        )
        smoothed = smoothed.reshape(
            stop - start,
            n_coils,
            n_coils,
            sx,
            sy,
        ).permute(0, 3, 4, 1, 2)

        left_vectors, _, _ = torch.linalg.svd(smoothed, full_matrices=False)
        weights = left_vectors[..., :, 0]
        output[start:stop] = torch.einsum(
            "bxyc,bxyic->bxyi",
            weights.conj(),
            block,
        )

    return output
