"""Bruker adapter around the preserved PV5/PV360 binary reader."""

from pathlib import Path

import numpy as np
import torch

from .._legacy.bruker.param import read_pv_param
from ..core import Acquisition, SPEN180Protocol


def _scalar(path, name, default=None):
    value = read_pv_param(str(path), name)
    if value is None:
        return default
    return np.asarray(value).reshape(-1)[0].item()


def protocol_from_bruker(scan_dir):
    """Extract quadratic SPEN parameters. Never classify xSPEN as quadratic SPEN."""
    path = Path(scan_dir)
    method = str(read_pv_param(str(path), "Method"))
    if "xspen" in method.lower() or "spen" not in method.lower():
        raise ValueError(f"A quadratic SPEN protocol cannot be inferred from {method}")
    if int(_scalar(path, "NSegments", 1)) != 1:
        raise NotImplementedError(
            "Protocol extraction currently supports single-shot SPEN"
        )
    matrix = read_pv_param(str(path), "PVM_Matrix")
    fov = read_pv_param(str(path), "PVM_Fov")
    gy = _scalar(path, "SpenGyGaussStren")
    tp = _scalar(path, "SpatEncDuration")
    esp = _scalar(path, "PVM_EpiEchoSpacing")
    if any(v is None for v in (matrix, fov, gy, tp, esp)):
        raise ValueError("Missing matrix, FOV, chirp or echo-spacing parameters")
    # FOV: mm -> m; gradient: gauss/cm -> T/m; duration: ms -> s.
    shape = (int(matrix[1]), int(matrix[0]))
    fov_m = (float(fov[1]) / 1000, float(fov[0]) / 1000)
    r = abs(float(gy) * 0.01 * float(tp) * 0.001 * fov_m[0] * 42_574_000)
    return SPEN180Protocol(shape, fov_m, r, float(esp) / 1000)


def read_bruker(
    scan_dir, *, regrid=False, traj_dir=None, regrid_flavor="pv360", device="cpu"
):
    """Read single-echo SPEN as [spen,readout,coil,slice,volume].

    Binary unpacking and reflected-line sorting happen in the reference
    reader. Optional regridding is explicit and recorded in the data stage.
    Multi-echo and non-SPEN acquisitions are rejected rather than mislabelled.
    """
    from .._legacy.bruker.raw import read_bruker_kspace_pv360_fid_multichannel

    path = Path(scan_dir).expanduser().resolve()
    method = str(read_pv_param(str(path), "Method"))
    if "spen" not in method.lower():
        raise ValueError(f"This adapter requires a SPEN/xSPEN method; got {method}")
    if int(_scalar(path, "PVM_NEchoImages", 1)) != 1:
        raise NotImplementedError(
            "The canonical Bruker adapter currently supports single-echo data"
        )
    coils = int(_scalar(path, "PVM_EncNReceivers", 1))
    raw = read_bruker_kspace_pv360_fid_multichannel(str(path))
    if raw.ndim != 5 or raw.shape[-1] != 1 or raw.shape[3] % coils:
        raise ValueError(f"Unsupported Bruker frame layout {raw.shape}")
    trajectory = read_pv_param(str(traj_dir or path), "PVM_EpiTrajAdjkx")
    if regrid:
        if trajectory is None or not np.any(np.asarray(trajectory) > 0):
            raise ValueError("Regridding requested without an available trajectory")
        from .._legacy.recon.spen_recon import _regrid_readout_if_needed

        matrix = list(read_pv_param(str(path), "PVM_Matrix"))[:2]
        raw = _regrid_readout_if_needed(
            raw,
            str(path),
            int(_scalar(path, "NSegments", 1)),
            matrix,
            traj_dir=traj_dir,
            regrid_flavor=regrid_flavor,
        )
    # Legacy receiver frames flatten [volume,coil] in MATLAB column order.
    ro, pe, slices, frames, _ = raw.shape
    data = (
        raw[..., 0]
        .reshape(ro, pe, slices, frames // coils, coils, order="F")
        .transpose(1, 0, 4, 2, 3)
    )
    metadata = {
        "source": str(path),
        "vendor": "bruker",
        "sequence_name": method,
        "regridded": regrid,
        "readout_reflections_corrected": True,
        "trajectory_available": trajectory is not None,
        "calibrated_to_simulator": False,
    }
    if trajectory is not None:
        metadata["readout_trajectory"] = np.asarray(trajectory).reshape(-1).tolist()
    try:
        metadata["protocol"] = protocol_from_bruker(path).to_dict()
    except (ValueError, NotImplementedError) as exc:
        metadata["protocol_extraction_note"] = str(exc)
    return Acquisition(
        torch.from_numpy(np.ascontiguousarray(data.astype(np.complex64))).to(device),
        ("spen", "readout", "coil", "slice", "volume"),
        "regridded_kspace" if regrid else "sorted_samples",
        metadata,
    )
