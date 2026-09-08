"""Hybrid SPEN-Diff2 diffusion volumes with explicit b-matrix attenuation."""

import torch

from ..core import Acquisition, HybridSPENDiff2Protocol
from .simulator import simulate


def _symmetric_psd(value, name):
    value = value.detach()
    if value.shape[-2:] != (3, 3) or not torch.isfinite(value).all():
        raise ValueError(f"{name} must end in finite [3,3] matrices")
    scale = value.abs().amax().clamp_min(torch.finfo(value.dtype).tiny)
    if not torch.allclose(value, value.mT, atol=float(scale) * 1e-6, rtol=1e-6):
        raise ValueError(f"{name} must be symmetric")
    if (torch.linalg.eigvalsh(value) < -scale * 1e-6).any():
        raise ValueError(f"{name} must be positive semidefinite")


def simulate_hybrid_spen_diffusion(
    image,
    protocol=None,
    *,
    diffusion_tensor_mm2_s=None,
    diffusivity_mm2_s=None,
    b_matrices_s_mm2=None,
    **simulation_options,
):
    """Simulate b0 / DWI-RO / DWI-PE / DWI-SS, or explicit b-matrices.

    Attenuation is exp(-sum_ij B_ij D_ij), with axes (RO, PE, SS).
    D may be [3,3] or [y,x,3,3]; scalar/map diffusivity makes an isotropic D.
    B may be [volume,3,3] or [volume,y,x,3,3] (spatial axes may broadcast).
    Explicit B can include the encoding/imaging gradients' diffusion terms.
    The default B contains only nominal orthogonal PGSE weighting. It does
    not infer the actual spatial b-matrix from R, TE or a protocol name.

    The target contains diffusion-weighted complex objects on the input HR
    grid. All volumes share an encoding operator. Gradients propagate to
    image and D. Receive coils default to 32; other options go to simulate.
    """
    protocol = protocol or HybridSPENDiff2Protocol()
    if not isinstance(protocol, HybridSPENDiff2Protocol):
        raise TypeError("Expected HybridSPENDiff2Protocol")
    image = torch.as_tensor(image, device=simulation_options.get("device"))
    if image.ndim != 2:
        raise ValueError("Select one 2D baseline image")
    rdtype = (
        torch.float64
        if image.dtype in (torch.float64, torch.complex128)
        else torch.float32
    )
    if diffusion_tensor_mm2_s is not None and diffusivity_mm2_s is not None:
        raise ValueError("Supply a diffusion tensor OR isotropic diffusivity")
    if diffusion_tensor_mm2_s is None:
        d = torch.as_tensor(
            0.8e-3 if diffusivity_mm2_s is None else diffusivity_mm2_s,
            device=image.device,
        )
        if d.is_complex():
            raise ValueError("Diffusivity must be real")
        d = d.to(rdtype)
        d = torch.broadcast_to(d, image.shape)
        tensor = d[..., None, None] * torch.eye(3, device=image.device, dtype=rdtype)
    else:
        supplied = torch.as_tensor(diffusion_tensor_mm2_s, device=image.device)
        if supplied.is_complex():
            raise ValueError("Diffusion tensor must be real")
        tensor = torch.broadcast_to(supplied.to(rdtype), (*image.shape, 3, 3))
    _symmetric_psd(tensor, "Diffusion tensor")
    nominal = b_matrices_s_mm2 is None
    if nominal:
        b = torch.zeros((4, 3, 3), device=image.device, dtype=rdtype)
        b[1:] = (
            torch.diag_embed(torch.eye(3, device=image.device, dtype=rdtype))
            * protocol.nominal_b_value_s_mm2
        )
    else:
        b = torch.as_tensor(b_matrices_s_mm2, device=image.device)
        if b.is_complex():
            raise ValueError("b-matrices must be real")
        b = b.to(rdtype)
    if b.ndim not in (3, 5) or b.shape[0] < 1:
        raise ValueError("b-matrices must be [volume,3,3] or [volume,y,x,3,3]")
    _symmetric_psd(b, "b-matrices")
    spatial_b = b[:, None, None] if b.ndim == 3 else b
    spatial_b = torch.broadcast_to(spatial_b, (b.shape[0], *image.shape, 3, 3))
    attenuation = torch.exp(-(spatial_b * tensor).sum(dim=(-1, -2)))
    simulation_options.setdefault("num_coils", 32)
    sample = simulate(image[None] * attenuation, protocol, **simulation_options)
    labels = (
        ["b0", "DWI-RO", "DWI-PE/SPEN", "DWI-SS"]
        if nominal
        else [f"volume {i + 1}" for i in range(b.shape[0])]
    )
    metadata = dict(sample.acquisition.metadata)
    metadata["diffusion"] = {
        "axes": ["RO", "PE", "SS"],
        "model": "nominal_orthogonal_pgse" if nominal else "supplied_b_matrices",
        "b_matrices_s_mm2": b.detach(),
        "volume_labels": labels,
        "intrinsic_diffusion_calibrated": False,
    }
    sample.acquisition = Acquisition(
        sample.measurements,
        ("volume", "spen", "readout", "coil"),
        sample.acquisition.stage,
        metadata,
    )
    return sample
