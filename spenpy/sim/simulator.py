"""Image + protocol + nuisance settings -> Torch measurements and truth."""

from dataclasses import dataclass
from typing import Any

import torch

from ..core import (
    Acquisition,
    EncodingOperator,
    HybridSPENDiff2Protocol,
    SPEN180Protocol,
    XSPENProtocol,
    coil_sensitivities,
)
from .noise import NoiseModel


@dataclass
class SimulatedSample:
    acquisition: Acquisition
    target: torch.Tensor
    clean_measurements: torch.Tensor
    operator: EncodingOperator
    noise_metadata: dict[str, Any]

    @property
    def measurements(self):
        return self.acquisition.data

    @property
    def parameters(self):
        return self.acquisition.metadata

    def as_training_dict(self):
        """Tensor-only batch fields; keep the operator outside DataLoader collation."""
        return {
            "target": self.target,
            "measurements": self.measurements,
            "clean_measurements": self.clean_measurements,
            "coil_maps": self.operator.coil_maps,
        }


def even_odd_phase(
    protocol,
    image_shape,
    *,
    constant_rad=0.0,
    linear_rad=0.0,
    device=None,
    dtype=torch.float32,
):
    """Fixed phase on zero-based odd lines. Linear term spans [-linear,+linear]."""
    m = protocol.acquisition_shape[0]
    x = torch.linspace(-1, 1, image_shape[1], device=device, dtype=dtype)
    odd = (torch.arange(m, device=device) % 2).to(dtype)
    return odd[:, None] * (constant_rad + linear_rad * x[None])


def simulate(
    image,
    protocol=None,
    *,
    sequence="spen",
    acquisition_shape=None,
    num_coils=4,
    coil_maps=None,
    noise_std=0.0,
    snr_db=None,
    noise_model=None,
    seed=0,
    object_phase_rad=None,
    even_odd_constant_rad=0.0,
    even_odd_linear_rad=0.0,
    device=None,
    dtype=None,
    operator_options=None,
):
    """Simulate SPEN or xSPEN without dropping image gradients.

    image has shape [...,image_y,image_x], independent of acquisition size.
    Real images have zero phase unless object_phase_rad is supplied. All
    batches share a protocol/operator; randomize protocols by separate calls.
    """
    image = torch.as_tensor(image, device=device)
    if image.ndim < 2 or not torch.isfinite(image).all():
        raise ValueError("image must be finite [...,image_y,image_x]")
    if dtype is None:
        dtype = (
            torch.complex128
            if image.dtype in (torch.float64, torch.complex128)
            else torch.complex64
        )
    target = image.to(dtype)
    if object_phase_rad is not None:
        if image.is_complex():
            raise ValueError("Supply complex image OR magnitude plus object_phase_rad")
        phase = torch.as_tensor(
            object_phase_rad, device=image.device, dtype=target.real.dtype
        )
        if phase.ndim > target.ndim or not torch.isfinite(phase).all():
            raise ValueError(
                "object_phase_rad must be finite and cannot introduce new batch axes"
            )
        target = target * torch.exp(1j * torch.broadcast_to(phase, target.shape))
    if protocol is None:
        cls = {
            "spen": SPEN180Protocol,
            "spen180": SPEN180Protocol,
            "xspen": XSPENProtocol,
            "hybrid_spen_diff2": HybridSPENDiff2Protocol,
        }.get(sequence.lower())
        if cls is None:
            raise ValueError(
                "sequence must be 'spen180', 'xspen' or 'hybrid_spen_diff2'"
            )
        protocol = cls(
            **(
                {"acquisition_shape": tuple(acquisition_shape)}
                if acquisition_shape
                else {}
            )
        )
    elif acquisition_shape is not None:
        raise ValueError(
            "Specify acquisition_shape inside protocol when supplying protocol"
        )
    if coil_maps is None:
        coil_maps = coil_sensitivities(
            tuple(image.shape[-2:]),
            num_coils,
            device=image.device,
            dtype=dtype,
            seed=seed,
        )
    options = dict(operator_options or {})
    if even_odd_constant_rad or even_odd_linear_rad:
        if "phase_map_rad" in options:
            raise ValueError("Specify phase_map_rad OR even_odd parameters")
        options["phase_map_rad"] = even_odd_phase(
            protocol,
            image.shape[-2:],
            constant_rad=even_odd_constant_rad,
            linear_rad=even_odd_linear_rad,
            device=image.device,
            dtype=target.real.dtype,
        )
    operator = EncodingOperator(
        protocol,
        tuple(image.shape[-2:]),
        coil_maps=coil_maps,
        device=image.device,
        dtype=dtype,
        **options,
    )
    clean = operator(target)
    if noise_model is not None and (noise_std != 0 or snr_db is not None):
        raise ValueError("Specify noise_model OR noise_std/snr_db")
    model = noise_model or NoiseModel(std=noise_std, snr_db=snr_db)
    measurements, noise_meta = model.add(
        clean, seed=seed + 1, sample_mask=operator.sample_mask
    )
    axes = tuple(f"batch{i}" for i in range(target.ndim - 2)) + (
        "spen",
        "readout",
        "coil",
    )
    metadata = {
        "protocol": protocol.to_dict(),
        "image_shape": list(image.shape[-2:]),
        "seed": seed,
        "noise_std": model.std,
        "snr_db": model.snr_db,
        "model": "ideal_line_encoding",
        "readout_fourier_sign": 1,
        "grid_origin": "voxel_centres",
        "normalization": "quadrature_integral",
        "pixel_quadrature": operator.quadrature,
        "operator_options": options,
        "calibrated_to_scanner": False,
    }
    if isinstance(protocol, HybridSPENDiff2Protocol):
        metadata.update(
            model="hybrid_spen_diff2_quadratic_sr",
            sequence_name=protocol.sequence_name,
            phase_fov_m=protocol.fov_m[0],
            readout_fov_m=protocol.fov_m[1],
            r_value=protocol.r_value,
            normalization="PE voxel integral in cm; centred RO IFFT",
            pe_integration="CalcSRMatrixApprox second-order pixel moments",
            readout_basis="centred discrete Fourier; no RO pixel apodization",
            simulated=True,
        )
    return SimulatedSample(
        Acquisition(measurements, axes, "uniform_kspace", metadata),
        target,
        clean,
        operator,
        noise_meta,
    )
