#!/usr/bin/env python3
"""Visual and numeric verification for the PV360 reconstruction path.

This script is intentionally standalone, because it writes inspection artifacts
and can optionally run a pv360.m-like export pass. It is not collected by
pytest.

Examples:
    /home/data1/musong/workspace/python/spen_recons/.venv/bin/python \
        tests/verify_pv360_reconstruction.py --spen-index 1

    /home/data1/musong/workspace/python/spen_recons/.venv/bin/python \
        tests/verify_pv360_reconstruction.py --export-pv360 --spen-index 1

    /home/data1/musong/workspace/python/spen_recons/.venv/bin/python \
        tests/verify_pv360_reconstruction.py --export-pv360 --all-spen
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from spenpy.cli.pv360 import process_rare_epi, process_spen, read_datalist
from spenpy.recon import orient_pv360_spen_image, reconstruct_odd_segments


DEFAULT_DATA_DIR = Path(
    "/home/data1/musong/workspace/python/spen_recons/spen_matlab/data/"
    "20240321_204022_lxj_spen_mouse_240321_1_1_1"
)
DEFAULT_MATLAB_EXPORT_DIR = Path(
    "/home/data1/musong/workspace/python/spen_recons/spen_matlab/export_data/pv360"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "pv360_verify_outputs"


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().resolve_conj().numpy()
    return np.asarray(value)


def _middle_2d(value: Any) -> np.ndarray:
    arr = np.squeeze(_to_numpy(value))
    while arr.ndim > 2:
        arr = np.take(arr, arr.shape[-1] // 2, axis=-1)
    if arr.ndim != 2:
        raise ValueError(f"Expected an image-like array, got shape {arr.shape}")
    return arr


def _display_abs(value: Any) -> np.ndarray:
    arr = np.abs(_middle_2d(value)).astype(np.float64)
    if not np.any(arr):
        return arr
    scale = np.percentile(arr[arr > 0], 99.5)
    if scale <= 0:
        scale = float(np.max(arr))
    if scale > 0:
        arr = np.clip(arr / scale, 0, 1)
    return arr


def _complex_corr(a: Any, b: Any) -> float:
    x = _middle_2d(a).reshape(-1)
    y = _middle_2d(b).reshape(-1)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return 0.0
    return float(abs(np.vdot(x, y)) / denom)


def _abs_corr(a: Any, b: Any) -> float:
    x = np.abs(_middle_2d(a)).reshape(-1)
    y = np.abs(_middle_2d(b)).reshape(-1)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return 0.0
    return float(np.vdot(x, y).real / denom)


def _mag_nrmse(a: Any, b: Any) -> float:
    x = np.abs(_middle_2d(a))
    y = np.abs(_middle_2d(b))
    denom = np.linalg.norm(y)
    if denom == 0:
        return float("inf")
    return float(np.linalg.norm(x - y) / denom)


def _max_rel_abs_diff(a: Any, b: Any) -> float:
    x = np.abs(_middle_2d(a))
    y = np.abs(_middle_2d(b))
    denom = np.max(y)
    if denom == 0:
        return float("inf")
    return float(np.max(np.abs(x - y)) / denom)


def _metric_block(py_value: Any, matlab_value: Any) -> dict[str, Any]:
    py_img = _middle_2d(py_value)
    matlab_img = _middle_2d(matlab_value)
    return {
        "python_shape": list(py_img.shape),
        "matlab_shape": list(matlab_img.shape),
        "complex_corr_phase_insensitive": _complex_corr(py_img, matlab_img),
        "abs_corr": _abs_corr(py_img, matlab_img),
        "abs_nrmse": _mag_nrmse(py_img, matlab_img),
        "max_relative_abs_diff": _max_rel_abs_diff(py_img, matlab_img),
    }


def _load_matlab_spen(matlab_export_dir: Path, spen_index: int) -> dict[str, Any] | None:
    mat_path = matlab_export_dir / f"ratbrain_SPEN_96_{spen_index}.mat"
    if not mat_path.exists():
        return None
    return scipy.io.loadmat(mat_path)


def _save_python_spen_mat(
    mat_dir: Path,
    spen_index: int,
    image_spen: np.ndarray,
    recon,
) -> Path:
    mat_dir.mkdir(parents=True, exist_ok=True)
    path = mat_dir / f"python_ratbrain_SPEN_96_{spen_index}.mat"
    scipy.io.savemat(
        path,
        {
            "Imag_low": _to_numpy(recon.imag_low),
            "Imag_origin": _to_numpy(recon.imag_origin),
            "Image_SPEN": _to_numpy(image_spen),
            "SPEN_AZ": {key: _to_numpy(value) for key, value in recon.spen_az.items()},
        },
    )
    return path


def _plot_spen_panel(
    fig_dir: Path,
    spen_index: int,
    scan_id: int,
    py_images: dict[str, Any],
    matlab_images: dict[str, Any] | None,
) -> Path:
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / f"spen_{spen_index:03d}_scan_{scan_id}_verification.png"

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    panels: list[tuple[str, Any]] = [
        ("Python Imag_low", py_images["Imag_low"]),
        ("Python Image_SPEN", py_images["Image_SPEN"]),
        ("Python Imag_origin", py_images["Imag_origin"]),
    ]

    if matlab_images is not None:
        py_recon = _middle_2d(py_images["Image_SPEN"])
        mat_recon = _middle_2d(matlab_images["Image_SPEN"])
        recon_diff = np.abs(py_recon) - np.abs(mat_recon)
        panels.extend(
            [
                ("MATLAB Imag_low", matlab_images["Imag_low"]),
                ("MATLAB Image_SPEN", matlab_images["Image_SPEN"]),
                ("Abs diff: Python - MATLAB", recon_diff),
            ]
        )
    else:
        panels.extend(
            [
                ("No MATLAB reference", np.zeros_like(_middle_2d(py_images["Imag_low"]))),
                ("No MATLAB reference", np.zeros_like(_middle_2d(py_images["Image_SPEN"]))),
                ("No MATLAB reference", np.zeros_like(_middle_2d(py_images["Imag_origin"]))),
            ]
        )

    for ax, (title, image) in zip(axes.ravel(), panels):
        if "diff" in title.lower():
            arr = _middle_2d(image).astype(np.float64)
            vmax = np.percentile(np.abs(arr), 99.0)
            if vmax <= 0:
                vmax = 1.0
            im = ax.imshow(arr, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(_display_abs(image), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"PV360 SPEN verification: index {spen_index}, scan {scan_id}")
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)
    return fig_path


def verify_single_spen(
    data_dir: Path,
    matlab_export_dir: Path,
    output_dir: Path,
    spen_index: int,
    compare_matlab: bool,
) -> dict[str, Any]:
    rare_id, epi_id, spen_ids = read_datalist(str(data_dir))
    del rare_id, epi_id

    if spen_index < 1 or spen_index > len(spen_ids):
        raise ValueError(f"SPEN index must be in [1, {len(spen_ids)}], got {spen_index}")

    scan_id = spen_ids[spen_index - 1]
    scan_dir = data_dir / str(scan_id)
    recon = reconstruct_odd_segments(str(scan_dir))
    image_spen = orient_pv360_spen_image(recon.images)

    py_images = {
        "Imag_low": _to_numpy(recon.imag_low),
        "Imag_origin": _to_numpy(recon.imag_origin),
        "Image_SPEN": _to_numpy(image_spen),
    }
    matlab_images = _load_matlab_spen(matlab_export_dir, spen_index) if compare_matlab else None

    mat_path = _save_python_spen_mat(output_dir / "mat", spen_index, image_spen, recon)
    fig_path = _plot_spen_panel(output_dir / "figures", spen_index, scan_id, py_images, matlab_images)

    metrics: dict[str, Any] = {
        "spen_index": spen_index,
        "scan_id": scan_id,
        "python_mat_file": str(mat_path),
        "figure": str(fig_path),
    }
    if matlab_images is not None:
        metrics["matlab_reference_file"] = str(
            matlab_export_dir / f"ratbrain_SPEN_96_{spen_index}.mat"
        )
        metrics["comparisons"] = {
            key: _metric_block(py_images[key], matlab_images[key])
            for key in ("Imag_low", "Imag_origin", "Image_SPEN")
            if key in matlab_images
        }

    metrics_path = output_dir / f"spen_{spen_index:03d}_metrics.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metrics["metrics_file"] = str(metrics_path)
    return metrics


def export_pv360_like(
    data_dir: Path,
    output_dir: Path,
    spen_index: int,
    all_spen: bool,
) -> list[Path]:
    rare_id, epi_id, spen_ids = read_datalist(str(data_dir))
    mat_dir = output_dir / "pv360_like_mat"
    mat_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    process_rare_epi(str(data_dir), rare_id, str(mat_dir), "ratbrain_RARE.mat", "Image_RARE")
    process_rare_epi(str(data_dir), epi_id, str(mat_dir), "ratbrain_EPI.mat", "image_EPI")
    written.extend([mat_dir / "ratbrain_RARE.mat", mat_dir / "ratbrain_EPI.mat"])

    selected = range(1, len(spen_ids) + 1) if all_spen else [spen_index]
    for idx in selected:
        process_spen(str(data_dir), spen_ids[idx - 1], str(mat_dir), idx)
        written.append(mat_dir / f"ratbrain_SPEN_96_{idx}.mat")

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--matlab-export-dir", type=Path, default=DEFAULT_MATLAB_EXPORT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--spen-index", type=int, default=1)
    parser.add_argument("--no-matlab-compare", action="store_true")
    parser.add_argument(
        "--export-pv360",
        action="store_true",
        help="Also export .mat files using the Python pv360.m-like pipeline.",
    )
    parser.add_argument(
        "--all-spen",
        action="store_true",
        help="With --export-pv360, reconstruct all SPEN scans instead of one selected scan.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.export_pv360:
        written = export_pv360_like(args.data_dir, args.output_dir, args.spen_index, args.all_spen)
        print(f"Wrote {len(written)} pv360-like .mat files under {args.output_dir / 'pv360_like_mat'}")

    metrics = verify_single_spen(
        args.data_dir,
        args.matlab_export_dir,
        args.output_dir,
        args.spen_index,
        compare_matlab=not args.no_matlab_compare,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
