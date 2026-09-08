"""Torch complex CG and proximal data consistency for diffusion/flow guidance."""

import math
from dataclasses import dataclass

import torch

from ..core import Acquisition


@dataclass
class ReconstructionResult:
    image: torch.Tensor
    iterations: int
    relative_residual: torch.Tensor
    converged: torch.Tensor


def conjugate_gradient(matvec, rhs, *, initial=None, max_iter=100, rtol=1e-6, atol=0.0):
    """Solve independent Hermitian positive-definite systems over the last 2 axes.

    Leading axes are batches. Gradients pass through the executed iterations;
    this is not an implicit-differentiation solver. Stop decisions are detached.
    """
    if rhs.ndim < 2 or max_iter < 1 or rtol < 0 or atol < 0:
        raise ValueError("Invalid CG dimensions or stopping parameters")
    x = torch.zeros_like(rhs) if initial is None else initial.clone()
    if x.shape != rhs.shape:
        raise ValueError("initial must have the same shape as rhs")
    dot = lambda a, b: (a.conj() * b).sum(dim=(-2, -1), keepdim=True).real
    r = rhs - matvec(x)
    p = r
    rr = dot(r, r)
    norm = dot(rhs, rhs).sqrt()
    threshold = torch.maximum(norm * rtol, torch.full_like(norm, atol))
    tiny = torch.finfo(rhs.real.dtype).tiny
    iterations = 0
    for _ in range(max_iter):
        active = rr.sqrt().detach() > threshold.detach()
        if not active.any():
            break
        ap = matvec(p)
        denom = dot(p, ap)
        if ((denom <= 0) & active).any():
            raise RuntimeError(
                "CG encountered a non-positive curvature; check the adjoint and regularization"
            )
        alpha = torch.where(active, rr / denom.clamp_min(tiny), 0.0)
        x = x + alpha * p
        r = r - alpha * ap
        rr_next = dot(r, r)
        beta = torch.where(active, rr_next / rr.clamp_min(tiny), 0.0)
        p = r + beta * p
        rr = rr_next
        iterations += 1
    # Report true residual, not only the recursively updated residual.
    final_residual = rhs - matvec(x)
    actual = dot(final_residual, final_residual).sqrt()
    return ReconstructionResult(
        x,
        iterations,
        (actual / norm.clamp_min(tiny)).squeeze(-1).squeeze(-1),
        (actual <= threshold).squeeze(-1).squeeze(-1),
    )


def reconstruct(
    data,
    operator,
    *,
    regularization=1e-3,
    prior=None,
    initial=None,
    max_iter=100,
    rtol=1e-6,
):
    """Solve ||Ax-y||² + regularization*||x-prior||², with zero prior by default.

    Noise whitening must be applied to both the operator and measurements.
    regularization is in this operator's squared-singular-value units.
    """
    if not math.isfinite(regularization) or regularization < 0:
        raise ValueError("regularization must be finite and nonnegative")
    if isinstance(data, Acquisition):
        if data.stage != "uniform_kspace":
            raise ValueError(
                "This solver needs uniform_kspace matching the supplied operator; calibrate scanner stages explicitly"
            )
        if data.axes[-3:] != ("spen", "readout", "coil"):
            raise ValueError("Permute acquisition to [...,spen,readout,coil]")
        data = data.data
    rhs = operator.adjoint(data)
    if prior is not None:
        prior = torch.as_tensor(prior, device=rhs.device, dtype=rhs.dtype)
        if prior.shape != rhs.shape:
            raise ValueError("prior must match the reconstructed image shape")
        rhs = rhs + regularization * prior
    return conjugate_gradient(
        lambda x: operator.normal(x) + regularization * x,
        rhs,
        initial=initial,
        max_iter=max_iter,
        rtol=rtol,
    )


def data_consistency(image, data, operator, *, mu=0.1, max_iter=30, rtol=1e-6):
    """Proximal physics step anchored to a flow/diffusion image estimate."""
    if mu <= 0:
        raise ValueError("mu must be positive")
    return reconstruct(
        data,
        operator,
        regularization=mu,
        prior=image,
        initial=image,
        max_iter=max_iter,
        rtol=rtol,
    ).image


def rss(coil_images, dim=-1):
    return coil_images.abs().square().sum(dim).sqrt()
