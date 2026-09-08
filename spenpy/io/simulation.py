"""Portable simulation snapshots, including the exact fixed operator buffers."""

import json
from pathlib import Path

import numpy as np
import torch

from ..core import Acquisition, EncodingOperator, protocol_from_dict
from ..sim import SimulatedSample
from .files import _json_default


def save_simulation(sample, path):
    """Save truth, noisy/clean measurements and the operator without pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "target": sample.target,
        "data": sample.measurements,
        "clean": sample.clean_measurements,
    }
    arrays.update(
        {"operator__" + k: v for k, v in sample.operator.state_dict().items()}
    )
    header = {
        "version": 1,
        "protocol": sample.operator.protocol.to_dict(),
        "axes": sample.acquisition.axes,
        "stage": sample.acquisition.stage,
        "metadata": sample.parameters,
        "noise_metadata": sample.noise_metadata,
    }
    values = {k: v.detach().cpu().resolve_conj().numpy() for k, v in arrays.items()}
    with path.open("wb") as f:
        np.savez_compressed(
            f, **values, header=np.asarray(json.dumps(header, default=_json_default))
        )
    return path


def load_simulation(path, *, device="cpu"):
    """Restore exactly the stored forward operator, including field-dependent kernels.

    Metadata is descriptive JSON; arrays used by the actual operator retain
    their original precision. No Python class instances are deserialized.
    """
    with np.load(path, allow_pickle=False) as f:
        header = json.loads(str(f["header"].item()))
        if header.get("version") != 1:
            raise ValueError("Unsupported simulation snapshot version")
        arrays = {
            k: torch.from_numpy(np.array(f[k], copy=True)).to(device)
            for k in f.files
            if k != "header"
        }
    protocol = protocol_from_dict(header["protocol"])
    maps = arrays["operator__coil_maps"]
    op = EncodingOperator(
        protocol,
        tuple(arrays["target"].shape[-2:]),
        coil_maps=maps,
        device=device,
        dtype=maps.dtype,
        quadrature=header["metadata"].get("pixel_quadrature"),
    )
    stored = {
        k.removeprefix("operator__"): v
        for k, v in arrays.items()
        if k.startswith("operator__")
    }
    if set(stored) != set(op.state_dict()):
        raise ValueError("Operator snapshot keys do not match this library version")
    for key, value in stored.items():
        expected = getattr(op, key)
        if key == "encoding_kernel":
            allowed = {
                (protocol.acquisition_shape[0], op.image_shape[0]),
                (protocol.acquisition_shape[0], *op.image_shape),
            }
            if tuple(value.shape) not in allowed:
                raise ValueError("Invalid encoding kernel dimensions")
        elif value.shape != expected.shape:
            raise ValueError(f"Invalid snapshot shape for {key}")
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite operator buffer: {key}")
        setattr(op, key, value)
    acquisition = Acquisition(
        arrays["data"], tuple(header["axes"]), header["stage"], header["metadata"]
    )
    return SimulatedSample(
        acquisition, arrays["target"], arrays["clean"], op, header["noise_metadata"]
    )
