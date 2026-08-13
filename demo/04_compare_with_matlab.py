#!/usr/bin/env python3
"""Demo 04 — compare a Python SPEN .mat against the MATLAB reference.

For a chosen SPEN index, this script:

1. Reads the Python output ``ratbrain_SPEN_96_<i>.mat`` (or, with
   ``--reconstruct``, regenerates it on the fly via
   ``reconstruct_odd_segments``).
2. Reads the MATLAB reference ``ratbrain_SPEN_96_<i>.mat`` produced by
   the original ``pv360.m``.
3. Computes magnitude correlation, complex correlation, NRMSE and
   max-relative-|abs| difference for ``Imag_low`` / ``Imag_origin`` /
   ``Image_SPEN``.
4. Renders a 2x3 panel: top row = Python, bottom row = MATLAB; plus an
   absolute-difference colormap for ``Image_SPEN``.

> Note on orientation: the Python ``.mat`` files emitted by
> ``spenpy.cli.pv360.process_spen`` apply ``flip(flip(.,1),2)`` to all
> three images so the figure panels share an orientation. The MATLAB
> reference applies that flip only to ``Image_SPEN``. This script
> therefore re-applies the same flip to the MATLAB ``Imag_low`` /
> ``Imag_origin`` before computing similarity metrics so the comparison
> is apples-to-apples.

Examples:

    # Compare an existing python output against MATLAB:
    python spenpy/demo/04_compare_with_matlab.py --spen-index 1 --python-export-dir /tmp/pv360_python --matlab-export-dir /home/data1/musong/workspace/python/spen_recons/spen_matlab/export_data/pv360

    # Reconstruct on the fly (uses spen_matlab/data/...) and compare:
    python spenpy/demo/04_compare_with_matlab.py --spen-index 1 --reconstruct
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


DEFAULT_DATA_DIR = Path(
    "/home/data1/musong/workspace/python/spen_recons/spen_matlab/data/"
    "20240321_204022_lxj_spen_mouse_240321_1_1_1"
)
DEFAULT_MATLAB_EXPORT_DIR = Path(
    "/home/data1/musong/workspace/python/spen_recons/spen_matlab/export_data/pv360"
)


def _orient(arr: np.ndarray) -> np.ndarray:
    """Match the Python pipeline orientation: flip(flip(.,1),2)."""
    return np.flip(np.flip(np.asarray(arr), axis=0), axis=1)


def _to_2d(arr: np.ndarray) -> np.ndarray:
    a = np.squeeze(np.asarray(arr))
    while a.ndim > 2:
        a = np.take(a, a.shape[-1] // 2, axis=-1)
    return a


def _scale_abs(arr: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    a = np.abs(_to_2d(arr)).astype(np.float64)
    pos = a[a > 0]
    if not pos.size:
        return a
    s = np.percentile(pos, percentile)
    s = s if s > 0 else float(np.max(a))
    return np.clip(a / s, 0, 1) if s > 0 else a


def _abs_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.abs(_to_2d(a)).reshape(-1)
    y = np.abs(_to_2d(b)).reshape(-1)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.vdot(x, y).real / denom) if denom else 0.0


def _complex_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = _to_2d(a).reshape(-1)
    y = _to_2d(b).reshape(-1)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(abs(np.vdot(x, y)) / denom) if denom else 0.0


def _nrmse_abs(a: np.ndarray, b: np.ndarray) -> float:
    x = np.abs(_to_2d(a))
    y = np.abs(_to_2d(b))
    denom = np.linalg.norm(y)
    return float(np.linalg.norm(x - y) / denom) if denom else float("inf")


def _max_rel_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    x = np.abs(_to_2d(a))
    y = np.abs(_to_2d(b))
    denom = np.max(y)
    return float(np.max(np.abs(x - y)) / denom) if denom else float("inf")


def _metrics_block(py_value: np.ndarray, mat_value: np.ndarray) -> dict[str, Any]:
    return {
        "python_shape": list(_to_2d(py_value).shape),
        "matlab_shape": list(_to_2d(mat_value).shape),
        "abs_corr": _abs_corr(py_value, mat_value),
        "complex_corr": _complex_corr(py_value, mat_value),
        "abs_nrmse": _nrmse_abs(py_value, mat_value),
        "max_relative_abs_diff": _max_rel_abs_diff(py_value, mat_value),
    }


def _reconstruct_python_mat(spen_index: int, file_dir: Path) -> dict[str, np.ndarray]:
    from spenpy.cli.pv360 import read_datalist
    from spenpy.recon import orient_pv360_spen_image, reconstruct_odd_segments

    _, _, spen_ids = read_datalist(str(file_dir))
    if spen_index < 1 or spen_index > len(spen_ids):
        raise SystemExit(
            f"--spen-index must be in [1, {len(spen_ids)}], got {spen_index}"
        )
    scan_dir = file_dir / str(spen_ids[spen_index - 1])

    recon = reconstruct_odd_segments(str(scan_dir))

    def _np(value):
        if hasattr(value, "detach"):
            return value.detach().cpu().resolve_conj().numpy()
        return np.asarray(value)

    return {
        "Imag_low": orient_pv360_spen_image(_np(recon.imag_low)),
        "Imag_origin": orient_pv360_spen_image(_np(recon.imag_origin)),
        "Image_SPEN": orient_pv360_spen_image(recon.images),
    }


def _load_python_mat(python_export_dir: Path, spen_index: int) -> dict[str, np.ndarray]:
    path = python_export_dir / f"ratbrain_SPEN_96_{spen_index}.mat"
    if not path.exists():
        raise SystemExit(
            f"Python .mat not found: {path}. Run demo 01 or 02 first, or use --reconstruct."
        )
    mat = scipy.io.loadmat(path)
    return {k: mat[k] for k in ("Imag_low", "Imag_origin", "Image_SPEN")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--matlab-export-dir",
        type=Path,
        default=DEFAULT_MATLAB_EXPORT_DIR,
        help=f"Directory containing MATLAB ratbrain_SPEN_96_<i>.mat (default: {DEFAULT_MATLAB_EXPORT_DIR}).",
    )
    parser.add_argument(
        "--python-export-dir",
        type=Path,
        default=Path("/tmp/pv360_python"),
        help="Directory containing Python ratbrain_SPEN_96_<i>.mat (default: /tmp/pv360_python).",
    )
    parser.add_argument(
        "--reconstruct",
        action="store_true",
        help="Reconstruct the Python output on the fly instead of reading --python-export-dir.",
    )
    parser.add_argument("--spen-index", type=int, default=1)
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("/tmp/pv360_python_compare"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.reconstruct:
        print("Reconstructing Python output on the fly ...")
        py = _reconstruct_python_mat(args.spen_index, args.file_dir)
    else:
        py = _load_python_mat(args.python_export_dir, args.spen_index)

    matlab_path = args.matlab_export_dir / f"ratbrain_SPEN_96_{args.spen_index}.mat"
    if not matlab_path.exists():
        raise SystemExit(f"MATLAB reference not found: {matlab_path}")
    mat = scipy.io.loadmat(matlab_path)

    # Re-apply the orientation correction to the MATLAB low/origin so that the
    # comparison is apples-to-apples (the Python pipeline already does this).
    mat_oriented = {
        "Imag_low": _orient(mat["Imag_low"]),
        "Imag_origin": _orient(mat["Imag_origin"]),
        "Image_SPEN": np.asarray(mat["Image_SPEN"]),
    }

    metrics = {
        "spen_index": args.spen_index,
        "matlab_file": str(matlab_path),
        "comparisons": {
            key: _metrics_block(py[key], mat_oriented[key])
            for key in ("Imag_low", "Imag_origin", "Image_SPEN")
        },
    }
    metrics_path = args.output / f"spen_{args.spen_index:03d}_compare.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    # ---- 2x3 panel: Python (top) vs MATLAB (bottom) + diff ------------------
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5), constrained_layout=True)
    for ax, key in zip(axes[0], ("Imag_low", "Imag_origin", "Image_SPEN")):
        im = ax.imshow(_scale_abs(py[key]), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Python {key}")
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    diff = np.abs(_to_2d(py["Image_SPEN"])) - np.abs(_to_2d(mat_oriented["Image_SPEN"]))
    diff_vmax = float(np.percentile(np.abs(diff), 99.0)) or 1.0

    for ax, key in zip(axes[1][:2], ("Imag_low", "Imag_origin")):
        im = ax.imshow(_scale_abs(mat_oriented[key]), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"MATLAB {key}")
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1][2]
    im = ax.imshow(diff, cmap="coolwarm", vmin=-diff_vmax, vmax=diff_vmax)
    ax.set_title("|Python| - |MATLAB| (Image_SPEN)")
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Python vs MATLAB — SPEN index {args.spen_index}",
        fontsize=12,
    )
    fig_path = args.output / f"spen_{args.spen_index:03d}_compare.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    print()
    print(f"Saved metrics : {metrics_path}")
    print(f"Saved figure  : {fig_path}")


if __name__ == "__main__":
    main()
