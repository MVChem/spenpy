"""Experimental reconstruction of Siemens hybrid-SPEN Twix raw data.

The Siemens files used to develop this adapter contain one image acquisition
for every ``(line, slice, repetition, set)`` tuple.  Each acquisition holds
four receiver channels and a two-times oversampled readout.

Two reconstruction modes are provided:

``physics``
    Correct reflected readouts, Fourier transform/crop the readout, and apply
    SPENPy's analytical weighted-adjoint (``InvA``) along the SPEN dimension.
    This path needs no processed image, but Siemens' proprietary phase
    correction is not reproduced.

``calibrated``
    Estimate one complex PE operator from a paired
    ``SignalFixedPostROFFTPostSR_*`` MATLAB export.  The operator can then be
    applied to another acquisition with the same Q/protocol.  Calibration and
    reconstruction sources are stored in the output metadata so this mode
    cannot be confused with a raw-only reconstruction.
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
from scipy.io import loadmat

from spenpy.core.matrix import calcInvA
from spenpy.siemens.processed import (
    _find_signal_name,
    _resolve_device,
    _scanner_metadata,
)
from spenpy.utils.coil_combine import coil_combine_batched


AXIS_ORDER = ("phase", "readout", "coil", "slice", "repetition", "set")


@dataclass
class SiemensROFFTData:
    """Organized Siemens data after readout FFT and oversampling removal."""

    source: Path
    signal: np.ndarray
    raw_shape: tuple[int, ...]
    reflected_acquisitions: int


@dataclass
class SiemensRawCalibration:
    """A PE reconstruction operator estimated from one paired raw/MAT scan."""

    matrix: np.ndarray
    source_dat: Path
    source_mat: Path
    q_value: float | None
    ridge: float
    readout_stride: int
    slice_stride: int
    repetition_index: int
    set_index: int
    signal_name: str


@dataclass
class SiemensRawRecon:
    """Raw-derived reconstructed coil data and combined image volumes."""

    source: Path
    method: str
    coil_signal: np.ndarray
    adaptive_complex: np.ndarray
    adaptive_magnitude: np.ndarray
    rss_magnitude: np.ndarray
    voxel_size_mm: tuple[float, float, float]
    rotated_locations_mm: np.ndarray
    protocol_name: str
    sequence_name: str
    b_value: float | None
    q_value: float | None
    device: str
    method_metadata: dict[str, Any]


def q_value_from_name(path: str | Path) -> float | None:
    """Return the ``Q<number>`` value embedded in a scan filename."""
    match = re.search(r"(?:^|[_-])Q(\d+(?:\.\d+)?)", Path(path).name, re.IGNORECASE)
    return float(match.group(1)) if match else None


def _as_six_dimensional(signal: np.ndarray) -> np.ndarray:
    out = np.asarray(signal)
    if out.ndim == 5:
        out = out[..., np.newaxis]
    if out.ndim != 6:
        raise ValueError(f"Expected axes {AXIS_ORDER}, got shape {out.shape}")
    return out


def read_twix_rofft(
    path: str | Path,
    *,
    output_readout: int | None = None,
) -> SiemensROFFTData:
    """Read image MDBs, correct REFLECT, perform centered FFT, and crop RO.

    The crop defaults to the central half of the acquired samples, matching
    the two-times Siemens readout oversampling in the supplied scans.
    """
    try:
        import twixtools
    except ImportError as exc:
        raise RuntimeError(
            "Siemens raw support requires twixtools; install twixtools>=0.24"
        ) from exc

    source = Path(path).expanduser().resolve()
    measurements = twixtools.read_twix(
        str(source),
        parse_geometry=False,
        parse_data=True,
    )
    measurement = measurements[-1]
    scans = [mdb for mdb in measurement["mdb"] if mdb.is_image_scan()]
    if not scans:
        raise ValueError(f"No image acquisitions found in {source}")

    first = scans[0]
    channels, acquired_readout = map(int, first.data.shape)
    phase = max(int(mdb.mdh.Counter.Lin) for mdb in scans) + 1
    slices = max(int(mdb.mdh.Counter.Sli) for mdb in scans) + 1
    repetitions = max(int(mdb.mdh.Counter.Rep) for mdb in scans) + 1
    sets = max(int(mdb.mdh.Counter.Set) for mdb in scans) + 1
    raw_shape = (phase, acquired_readout, channels, slices, repetitions, sets)

    raw = np.zeros(raw_shape, dtype=np.complex64)
    occupied = np.zeros((phase, slices, repetitions, sets), dtype=bool)
    reflected = 0
    for mdb in scans:
        counter = mdb.mdh.Counter
        index = (
            int(counter.Lin),
            int(counter.Sli),
            int(counter.Rep),
            int(counter.Set),
        )
        if occupied[index]:
            raise ValueError(f"Duplicate image acquisition {index} in {source}")
        data = np.asarray(mdb.data).T
        if "MDH_REFLECT" in mdb.get_active_flags():
            data = data[::-1]
            reflected += 1
        raw[index[0], :, :, index[1], index[2], index[3]] = data
        occupied[index] = True

    if not np.all(occupied):
        missing = int(occupied.size - np.count_nonzero(occupied))
        raise ValueError(f"{source} is missing {missing} image acquisitions")

    if output_readout is None:
        if acquired_readout % 2:
            raise ValueError(
                "Cannot infer two-times readout oversampling from odd sample "
                f"count {acquired_readout}"
            )
        output_readout = acquired_readout // 2
    if not 0 < output_readout <= acquired_readout:
        raise ValueError(
            f"Invalid output readout {output_readout} for {acquired_readout} samples"
        )

    roffted = np.fft.fftshift(
        np.fft.fft(np.fft.ifftshift(raw, axes=1), axis=1),
        axes=1,
    )
    start = (acquired_readout - output_readout) // 2
    stop = start + output_readout
    signal = roffted[:, start:stop].astype(np.complex64, copy=False)
    return SiemensROFFTData(
        source=source,
        signal=signal,
        raw_shape=raw_shape,
        reflected_acquisitions=reflected,
    )


def apply_pe_operator(signal: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a square PE operator to a six-dimensional coil signal."""
    signal_6d = _as_six_dimensional(signal)
    operator = np.asarray(matrix)
    if operator.shape != (signal_6d.shape[0], signal_6d.shape[0]):
        raise ValueError(
            f"Operator {operator.shape} does not match PE size {signal_6d.shape[0]}"
        )
    return np.einsum(
        "ab,brcsnk->arcsnk",
        operator,
        signal_6d,
        optimize=True,
    ).astype(np.complex64, copy=False)


def fit_raw_calibration(
    dat_path: str | Path,
    mat_path: str | Path,
    *,
    ridge: float = 1e-3,
    readout_stride: int = 2,
    slice_stride: int = 2,
    repetition_index: int = 0,
    set_index: int = 0,
) -> SiemensRawCalibration:
    """Fit a global complex PE operator from a paired raw/processed scan.

    Alternating readout points and slices are used by default.  This leaves
    complementary samples available for an internal holdout check and reduces
    the chance that the operator memorizes one anatomical image.
    """
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    if readout_stride < 1 or slice_stride < 1:
        raise ValueError("training strides must be positive")

    source_dat = Path(dat_path).expanduser().resolve()
    source_mat = Path(mat_path).expanduser().resolve()
    roffted = read_twix_rofft(source_dat)
    signal_name = _find_signal_name(source_mat)
    target = _as_six_dimensional(
        loadmat(source_mat, variable_names=[signal_name])[signal_name]
    )
    if target.shape != roffted.signal.shape:
        raise ValueError(
            f"Raw ROFFT shape {roffted.signal.shape} does not match "
            f"{signal_name} shape {target.shape}"
        )
    if repetition_index >= target.shape[4] or set_index >= target.shape[5]:
        raise IndexError("Calibration repetition/set index is out of range")

    x = roffted.signal[
        :, ::readout_stride, :, ::slice_stride, repetition_index, set_index
    ].reshape(target.shape[0], -1)
    y = target[
        :, ::readout_stride, :, ::slice_stride, repetition_index, set_index
    ].reshape(target.shape[0], -1)
    x = x.astype(np.complex128, copy=False)
    y = y.astype(np.complex128, copy=False)

    gram = x @ x.conj().T
    cross = y @ x.conj().T
    ridge_scale = float(np.trace(gram).real / gram.shape[0])
    regularized = gram + ridge * ridge_scale * np.eye(gram.shape[0])
    matrix = np.linalg.solve(regularized.T, cross.T).T
    return SiemensRawCalibration(
        matrix=matrix.astype(np.complex64),
        source_dat=source_dat,
        source_mat=source_mat,
        q_value=q_value_from_name(source_dat),
        ridge=float(ridge),
        readout_stride=int(readout_stride),
        slice_stride=int(slice_stride),
        repetition_index=int(repetition_index),
        set_index=int(set_index),
        signal_name=signal_name,
    )


def save_raw_calibration(
    calibration: SiemensRawCalibration,
    path: str | Path,
) -> Path:
    """Save a calibration operator and its provenance without pickle."""
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source_dat": str(calibration.source_dat),
        "source_mat": str(calibration.source_mat),
        "q_value": calibration.q_value,
        "ridge": calibration.ridge,
        "readout_stride": calibration.readout_stride,
        "slice_stride": calibration.slice_stride,
        "repetition_index": calibration.repetition_index,
        "set_index": calibration.set_index,
        "signal_name": calibration.signal_name,
    }
    np.savez_compressed(
        output,
        matrix=calibration.matrix,
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    return output


def load_raw_calibration(path: str | Path) -> SiemensRawCalibration:
    """Load a calibration written by :func:`save_raw_calibration`."""
    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as payload:
        matrix = np.asarray(payload["matrix"], dtype=np.complex64)
        metadata = json.loads(str(payload["metadata_json"].item()))
    return SiemensRawCalibration(
        matrix=matrix,
        source_dat=Path(metadata["source_dat"]),
        source_mat=Path(metadata["source_mat"]),
        q_value=metadata["q_value"],
        ridge=float(metadata["ridge"]),
        readout_stride=int(metadata["readout_stride"]),
        slice_stride=int(metadata["slice_stride"]),
        repetition_index=int(metadata["repetition_index"]),
        set_index=int(metadata["set_index"]),
        signal_name=str(metadata["signal_name"]),
    )


def siemens_physics_operator(
    q_value: float,
    phase: int,
    *,
    phase_fov_cm: float = 12.0,
    phase_factor_pi: float = 2.0,
    acquire_sign: int = -1,
    gauss_relative_width: float = 0.2,
) -> np.ndarray:
    """Build the current raw-only Siemens SPEN approximation.

    ``Q`` is the chirp time-bandwidth product.  The supplied scans support
    ``MaxPhase ~= 2*pi*Q`` as the best raw-only initialization.  The custom
    Siemens ICE phase correction remains unavailable, so this operator is an
    experimental weighted-adjoint rather than a parity reconstruction.
    """
    if q_value <= 0 or phase <= 0 or phase_fov_cm <= 0:
        raise ValueError("Q, phase size, and phase FOV must be positive")
    max_phase = float(phase_factor_pi) * np.pi * float(q_value)
    a_rad2cmsqr = max_phase / float(phase_fov_cm) ** 2
    inv_a, _ = calcInvA(
        a_rad2cmsqr,
        float(phase_fov_cm),
        int(phase),
        0.0,
        int(acquire_sign),
        0.0,
        float(gauss_relative_width),
    )
    return inv_a.cpu().resolve_conj().numpy().astype(np.complex64)


def _combine_coils(
    coil_signal: np.ndarray,
    *,
    device: str,
    batch_slices: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    signal = _as_six_dimensional(coil_signal)
    phase, readout, coils, slices, repetitions, sets = signal.shape
    rss_per_repetition = np.sqrt(
        np.sum(np.abs(signal) ** 2, axis=2, dtype=np.float32)
    )
    rss_magnitude = np.mean(rss_per_repetition, axis=3, dtype=np.float32)

    adaptive_input = np.transpose(signal, (3, 0, 1, 4, 5, 2)).reshape(
        slices,
        phase,
        readout,
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
        phase,
        readout,
        repetitions,
        sets,
    )
    adaptive_complex = np.transpose(adaptive_np, (1, 2, 0, 3, 4))
    adaptive_magnitude = np.mean(
        np.abs(adaptive_complex),
        axis=3,
        dtype=np.float32,
    )
    return (
        adaptive_complex.astype(np.complex64, copy=False),
        adaptive_magnitude.astype(np.float32, copy=False),
        rss_magnitude.astype(np.float32, copy=False),
        str(resolved_device),
    )


def reconstruct_raw_dat(
    path: str | Path,
    *,
    calibration: SiemensRawCalibration | None = None,
    q_value: float | None = None,
    metadata_mat: str | Path | None = None,
    device: str = "auto",
    batch_slices: int = 5,
    phase_factor_pi: float = 2.0,
    gauss_relative_width: float = 0.2,
    allow_q_mismatch: bool = False,
) -> SiemensRawRecon:
    """Reconstruct a Siemens Twix file by calibrated or raw-only physics mode."""
    roffted = read_twix_rofft(path)
    scan_q = q_value if q_value is not None else q_value_from_name(roffted.source)
    if calibration is not None:
        if (
            not allow_q_mismatch
            and scan_q is not None
            and calibration.q_value is not None
            and not np.isclose(scan_q, calibration.q_value)
        ):
            raise ValueError(
                f"Scan Q={scan_q:g} does not match calibration "
                f"Q={calibration.q_value:g}"
            )
        operator = calibration.matrix
        method = "calibrated_global_pe_operator"
        method_metadata: dict[str, Any] = {
            "calibration_dat": str(calibration.source_dat),
            "calibration_mat": str(calibration.source_mat),
            "calibration_signal": calibration.signal_name,
            "ridge": calibration.ridge,
            "readout_stride": calibration.readout_stride,
            "slice_stride": calibration.slice_stride,
            "repetition_index": calibration.repetition_index,
            "set_index": calibration.set_index,
        }
    else:
        if scan_q is None:
            raise ValueError("Physics mode requires Q or a filename containing _Q<number>")
        operator = siemens_physics_operator(
            scan_q,
            roffted.signal.shape[0],
            phase_factor_pi=phase_factor_pi,
            gauss_relative_width=gauss_relative_width,
        )
        method = "raw_only_spenpy_weighted_adjoint"
        method_metadata = {
            "phase_factor_pi": float(phase_factor_pi),
            "gauss_relative_width": float(gauss_relative_width),
            "siemens_ice_phase_correction_reproduced": False,
        }

    coil_signal = apply_pe_operator(roffted.signal, operator)
    adaptive_complex, adaptive_magnitude, rss_magnitude, resolved_device = (
        _combine_coils(
            coil_signal,
            device=device,
            batch_slices=batch_slices,
        )
    )

    slices = coil_signal.shape[3]
    locations = np.zeros((3, slices), dtype=np.float64)
    voxel_size = (
        120.0 / coil_signal.shape[0],
        400.0 / coil_signal.shape[1],
        2.5,
    )
    protocol_name = ""
    sequence_name = ""
    b_value = None
    if metadata_mat is not None:
        metadata_payload = loadmat(
            Path(metadata_mat).expanduser().resolve(),
            variable_names=["RotatedLocs", "mrprot"],
            squeeze_me=True,
            struct_as_record=False,
        )
        locations = np.asarray(
            metadata_payload.get("RotatedLocs", locations),
            dtype=np.float64,
        ).reshape(3, slices)
        voxel_size, protocol_name, sequence_name, b_value = _scanner_metadata(
            metadata_payload.get("mrprot"),
            tuple(int(value) for value in coil_signal.shape),
        )

    method_metadata.update(
        {
            "raw_shape": list(roffted.raw_shape),
            "roffted_shape": list(roffted.signal.shape),
            "reflected_acquisitions": roffted.reflected_acquisitions,
        }
    )
    return SiemensRawRecon(
        source=roffted.source,
        method=method,
        coil_signal=coil_signal,
        adaptive_complex=adaptive_complex,
        adaptive_magnitude=adaptive_magnitude,
        rss_magnitude=rss_magnitude,
        voxel_size_mm=voxel_size,
        rotated_locations_mm=locations,
        protocol_name=protocol_name,
        sequence_name=sequence_name,
        b_value=b_value,
        q_value=scan_q,
        device=resolved_device,
        method_metadata=method_metadata,
    )


def _magnitude_correlation(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    a -= np.mean(a)
    b -= np.mean(b)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def validate_against_processed_mat(
    result: SiemensRawRecon,
    path: str | Path,
) -> dict[str, Any]:
    """Score a completed raw reconstruction against a processed MAT reference."""
    reference_path = Path(path).expanduser().resolve()
    signal_name = _find_signal_name(reference_path)
    reference = _as_six_dimensional(
        loadmat(reference_path, variable_names=[signal_name])[signal_name]
    )
    if reference.shape != result.coil_signal.shape:
        raise ValueError(
            f"Reference {reference.shape} does not match reconstruction "
            f"{result.coil_signal.shape}"
        )
    complex_denominator = np.linalg.norm(result.coil_signal) * np.linalg.norm(reference)
    complex_correlation = (
        float(abs(np.vdot(result.coil_signal, reference)) / complex_denominator)
        if complex_denominator
        else 0.0
    )
    per_set = [
        _magnitude_correlation(
            np.abs(result.coil_signal[..., index]),
            np.abs(reference[..., index]),
        )
        for index in range(reference.shape[5])
    ]
    return {
        "reference": str(reference_path),
        "reference_signal": signal_name,
        "magnitude_correlation": _magnitude_correlation(
            np.abs(result.coil_signal),
            np.abs(reference),
        ),
        "magnitude_correlation_per_set": per_set,
        "absolute_complex_correlation": complex_correlation,
    }


def _save_montage(
    result: SiemensRawRecon,
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
    if not 0 <= set_index < volume.shape[3]:
        raise IndexError(f"Set {set_index} is out of range for {volume.shape[3]} sets")
    if slice_indices is None:
        slice_indices = tuple(range(volume.shape[2]))
    if not slice_indices or any(
        index < 0 or index >= volume.shape[2] for index in slice_indices
    ):
        raise IndexError(f"Invalid slice indices {slice_indices}")
    if columns < 1:
        raise ValueError("columns must be positive")

    # rot90([PE, RO]) produces a tall RO-by-PE panel.  Match the subplot cell
    # to that aspect ratio so images fill a compact grid without distortion.
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
    scale = scale if scale > 0 else 1.0
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


def save_raw_reconstruction(
    result: SiemensRawRecon,
    output_dir: str | Path,
    *,
    save_complex: bool = False,
    validation: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Save raw-derived NPZ, a compact montage, and provenance JSON."""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    prefix = f"{result.source.stem}__raw_{result.method}"

    npz_path = output / f"{prefix}.npz"
    arrays: dict[str, np.ndarray] = {
        "adaptive_magnitude": result.adaptive_magnitude,
        "rss_magnitude": result.rss_magnitude,
        "rotated_locations_mm": result.rotated_locations_mm,
        "voxel_size_mm": np.asarray(result.voxel_size_mm),
    }
    if save_complex:
        arrays["adaptive_complex_per_repetition"] = result.adaptive_complex
        arrays["reconstructed_coil_signal"] = result.coil_signal
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
        "method": result.method,
        "q_value": result.q_value,
        "coil_signal_shape": list(result.coil_signal.shape),
        "axis_order_coil_signal": list(AXIS_ORDER),
        "output_shape": list(result.adaptive_magnitude.shape),
        "axis_order_output": ["phase", "readout", "slice", "set_or_direction"],
        "voxel_size_mm": list(result.voxel_size_mm),
        "protocol_name": result.protocol_name,
        "sequence_name": result.sequence_name,
        "b_value": result.b_value,
        "device": result.device,
        "method_metadata": result.method_metadata,
        "processing": [
            "reverse MDH_REFLECT readouts",
            "centered readout FFT",
            "remove two-times readout oversampling",
            "apply PE reconstruction operator",
            "SPENPy Walsh adaptive receiver combination",
            "magnitude average over repetitions",
        ],
        "saved_outputs": [
            "npz",
            "one montage per set",
            "metadata",
        ],
        "montage_slice_numbers": list(range(1, slice_count + 1)),
        "validation": validation,
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
