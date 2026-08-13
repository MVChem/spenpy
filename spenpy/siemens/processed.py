"""Reconstruct Siemens SPEN volumes from processed ICE MATLAB exports.

The supported files contain a ``SignalFixedPostROFFTPostSR_*`` variable with
the layout::

    [phase, readout, coil, slice, repetition, set/direction]

The variable name records that readout FFT and SPEN super-resolution have
already been applied.  This module therefore performs the remaining receiver
combination and repetition reduction; it deliberately does not apply a second
SPEN inverse matrix.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.io import loadmat, whosmat

from spenpy.utils.coil_combine import coil_combine_batched


@dataclass
class SiemensProcessedRecon:
    """Reconstructed volumes and scanner metadata."""

    source: Path
    signal_name: str
    signal_shape: tuple[int, ...]
    adaptive_complex: np.ndarray
    adaptive_magnitude: np.ndarray
    rss_magnitude: np.ndarray
    rotated_locations_mm: np.ndarray
    voxel_size_mm: tuple[float, float, float]
    protocol_name: str
    sequence_name: str
    b_value: float | None
    device: str


def _get_field(value: Any, dotted_path: str, default: Any = None) -> Any:
    current = value
    for field in dotted_path.split("."):
        if current is None or not hasattr(current, field):
            return default
        current = getattr(current, field)
    return current


def _find_signal_name(path: Path) -> str:
    names = [
        name
        for name, _, _ in whosmat(path)
        if name.startswith("SignalFixedPostROFFTPostSR")
    ]
    if len(names) != 1:
        raise ValueError(
            f"Expected one SignalFixedPostROFFTPostSR* variable in {path}; "
            f"found {names}"
        )
    return names[0]


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return resolved


def _scanner_metadata(
    protocol: Any,
    signal_shape: tuple[int, ...],
) -> tuple[tuple[float, float, float], str, str, float | None]:
    first_slice = _get_field(protocol, "sSliceArray.asSlice")
    if isinstance(first_slice, np.ndarray):
        first_slice = first_slice.reshape(-1)[0]

    phase_fov = float(_get_field(first_slice, "dPhaseFOV", signal_shape[0]))
    readout_fov = float(_get_field(first_slice, "dReadoutFOV", signal_shape[1]))
    thickness = float(_get_field(first_slice, "dThickness", 1.0))
    voxel_size = (
        phase_fov / signal_shape[0],
        readout_fov / signal_shape[1],
        thickness,
    )

    protocol_name = str(_get_field(protocol, "tProtocolName", ""))
    sequence_name = str(_get_field(protocol, "tSequenceFileName", ""))
    raw_b_value = _get_field(protocol, "sWiPMemBlock.adFree")
    try:
        b_value = float(np.asarray(raw_b_value).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        b_value = None
    if b_value is None:
        if re.search(r"b[_ -]?zero", protocol_name, flags=re.IGNORECASE):
            b_value = 0.0
        else:
            match = re.search(r"b(?:val)?[_ -]?(\d+(?:\.\d+)?)", protocol_name)
            if match:
                b_value = float(match.group(1))
    return voxel_size, protocol_name, sequence_name, b_value


def reconstruct_processed_mat(
    path: str | Path,
    *,
    device: str = "auto",
    batch_slices: int = 5,
) -> SiemensProcessedRecon:
    """Combine coils and repetitions in a processed Siemens SPEN MAT file."""
    source = Path(path).expanduser().resolve()
    signal_name = _find_signal_name(source)
    payload = loadmat(
        source,
        variable_names=[signal_name, "RotatedLocs", "mrprot"],
        squeeze_me=True,
        struct_as_record=False,
    )
    signal = np.asarray(payload[signal_name])
    if signal.ndim == 5:
        signal = signal[..., np.newaxis]
    if signal.ndim != 6:
        raise ValueError(
            "Expected [phase, readout, coil, slice, repetition, set], "
            f"got {signal.shape}"
        )
    if not np.iscomplexobj(signal):
        raise ValueError(f"{signal_name} is expected to be complex-valued")

    original_shape = tuple(int(value) for value in signal.shape)
    phase, readout, coils, slices, repetitions, sets = original_shape
    del phase, readout

    rss_per_repetition = np.sqrt(
        np.sum(np.abs(signal) ** 2, axis=2, dtype=np.float32)
    )
    rss_magnitude = np.mean(rss_per_repetition, axis=3, dtype=np.float32)

    # [PE, RO, coil, slice, repetition, set]
    # -> [slice, PE, RO, repetition * set, coil]
    adaptive_input = np.transpose(signal, (3, 0, 1, 4, 5, 2)).reshape(
        slices,
        original_shape[0],
        original_shape[1],
        repetitions * sets,
        coils,
    )
    resolved_device = _resolve_device(device)
    adaptive_tensor = torch.from_numpy(
        np.ascontiguousarray(adaptive_input.astype(np.complex64, copy=False))
    ).to(resolved_device)
    with torch.inference_mode():
        adaptive = coil_combine_batched(
            adaptive_tensor,
            batch_size=batch_slices,
        )
    adaptive_np = adaptive.cpu().numpy().reshape(
        slices,
        original_shape[0],
        original_shape[1],
        repetitions,
        sets,
    )
    adaptive_complex = np.transpose(adaptive_np, (1, 2, 0, 3, 4))
    adaptive_magnitude = np.mean(
        np.abs(adaptive_complex),
        axis=3,
        dtype=np.float32,
    )

    locations = np.asarray(
        payload.get("RotatedLocs", np.zeros((3, slices), dtype=np.float64)),
        dtype=np.float64,
    ).reshape(3, slices)
    voxel_size, protocol_name, sequence_name, b_value = _scanner_metadata(
        payload.get("mrprot"),
        original_shape,
    )

    return SiemensProcessedRecon(
        source=source,
        signal_name=signal_name,
        signal_shape=original_shape,
        adaptive_complex=adaptive_complex.astype(np.complex64, copy=False),
        adaptive_magnitude=adaptive_magnitude.astype(np.float32, copy=False),
        rss_magnitude=rss_magnitude.astype(np.float32, copy=False),
        rotated_locations_mm=locations,
        voxel_size_mm=voxel_size,
        protocol_name=protocol_name,
        sequence_name=sequence_name,
        b_value=b_value,
        device=str(resolved_device),
    )


def _save_montage(
    result: SiemensProcessedRecon,
    path: Path,
    *,
    set_index: int,
    slice_indices: tuple[int, ...] | None = None,
    columns: int = 10,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patheffects
    from matplotlib.font_manager import FontProperties

    volume = result.adaptive_magnitude
    n_slices = volume.shape[2]
    n_sets = volume.shape[3]
    if not 0 <= set_index < n_sets:
        raise IndexError(f"Set {set_index} is out of range for {n_sets} sets")
    if slice_indices is None:
        slice_indices = tuple(range(n_slices))
    if not slice_indices or any(index < 0 or index >= n_slices for index in slice_indices):
        raise IndexError(f"Invalid slice indices {slice_indices}")
    if columns < 1:
        raise ValueError("columns must be positive")

    rows = math.ceil(len(slice_indices) / columns)
    panel_width = 1.15
    panel_height = panel_width * volume.shape[1] / volume.shape[0]
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(
            panel_width * columns,
            panel_height * rows,
        ),
        squeeze=False,
        facecolor="black",
        gridspec_kw={"wspace": 0.01, "hspace": 0.01},
    )
    fig.subplots_adjust(
        left=0.002,
        right=0.998,
        bottom=0.002,
        top=0.998,
        wspace=0.01,
        hspace=0.01,
    )
    label_font = FontProperties(
        family="Times New Roman",
        weight="bold",
        size=8,
    )
    label_effect = [
        patheffects.withStroke(linewidth=1.25, foreground="black")
    ]

    scale = float(np.percentile(volume[..., set_index], 99.5))
    if scale <= 0:
        scale = 1.0
    for panel_index, slice_index in enumerate(slice_indices):
        row, column = divmod(panel_index, columns)
        axis = axes[row, column]
        axis.set_facecolor("black")
        axis.imshow(
            np.rot90(volume[:, :, slice_index, set_index]),
            cmap="gray",
            vmin=0,
            vmax=scale,
            aspect="equal",
            interpolation="nearest",
        )
        axis.text(
            0.025,
            0.98,
            f"SET {set_index}  SLICE {slice_index + 1}",
            transform=axis.transAxes,
            color="white",
            fontproperties=label_font,
            horizontalalignment="left",
            verticalalignment="top",
            path_effects=label_effect,
        )
        axis.set_axis_off()
    for panel_index in range(len(slice_indices), rows * columns):
        row, column = divmod(panel_index, columns)
        axes[row, column].set_axis_off()

    fig.savefig(
        path,
        dpi=180,
        facecolor="black",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)


def save_processed_reconstruction(
    result: SiemensProcessedRecon,
    output_dir: str | Path,
    *,
    save_complex: bool = False,
) -> dict[str, Path]:
    """Save NPZ data, a compact montage, and JSON metadata."""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    prefix = result.source.stem

    npz_path = output / f"{prefix}__spenpy_recon.npz"
    arrays: dict[str, np.ndarray] = {
        "adaptive_magnitude": result.adaptive_magnitude,
        "rss_magnitude": result.rss_magnitude,
        "rotated_locations_mm": result.rotated_locations_mm,
        "voxel_size_mm": np.asarray(result.voxel_size_mm),
    }
    if save_complex:
        arrays["adaptive_complex_per_repetition"] = result.adaptive_complex
    np.savez_compressed(npz_path, **arrays)

    montage_paths: dict[str, Path] = {}
    slice_count = result.adaptive_magnitude.shape[2]
    for set_index in range(result.adaptive_magnitude.shape[3]):
        montage_path = (
            output / f"{prefix}__set_{set_index}__slices_1-{slice_count}.png"
        )
        _save_montage(result, montage_path, set_index=set_index)
        montage_paths[f"montage_set_{set_index}"] = montage_path

    metadata_path = output / f"{prefix}__metadata.json"
    metadata = {
        "source": str(result.source),
        "signal_variable": result.signal_name,
        "signal_shape": list(result.signal_shape),
        "axis_order_input": [
            "phase",
            "readout",
            "coil",
            "slice",
            "repetition",
            "set_or_direction",
        ],
        "output_shape": list(result.adaptive_magnitude.shape),
        "axis_order_output": ["phase", "readout", "slice", "set_or_direction"],
        "voxel_size_mm": list(result.voxel_size_mm),
        "protocol_name": result.protocol_name,
        "sequence_name": result.sequence_name,
        "b_value": result.b_value,
        "device": result.device,
        "input_stage": "post_readout_fft_post_spen_sr",
        "processing": [
            "SPENPy Walsh adaptive receiver combination per slice",
            "magnitude average over repetitions",
            "RSS magnitude reference",
        ],
        "saved_outputs": [
            "npz",
            "one montage per set",
            "metadata",
        ],
        "montage_slice_numbers": list(range(1, slice_count + 1)),
        "saved_complex": save_complex,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return {
        "npz": npz_path,
        **montage_paths,
        "metadata": metadata_path,
    }
