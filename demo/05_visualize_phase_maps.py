#!/usr/bin/env python3
"""Demo 05 - visualize PV360 SPEN phase-correction diagnostics.

The MATLAB reconstruction has useful debugging plots around even/odd phase
correction, especially ``SmoothPhase``, ``Mask`` and
``angle(ImgEven .* conj(ImgOdd))``. This script exposes the same kind of
inspection from the Python reconstruction path.

Examples:
    python spenpy/demo/05_visualize_phase_maps.py --spen-index 1
    python spenpy/demo/05_visualize_phase_maps.py --spen-index 1 --image-index 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spenpy.bruker.param import read_pv_param
from spenpy.cli.pv360 import read_datalist
from spenpy.recon import orient_pv360_spen_image, reconstruct_odd_segments
from spenpy.recon.phase import EvenOddPhaseFit, PhaseCorrectionDiagnostics


DEFAULT_DATA_DIR = Path(
    "/home/data1/musong/workspace/python/spen_recons/spen_matlab/data/"
    "20240321_204022_lxj_spen_mouse_240321_1_1_1"
)


def _to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().resolve_conj().numpy()
    return np.asarray(value)


def _scale_abs(arr: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    a = np.abs(np.asarray(arr)).astype(np.float64)
    pos = a[a > 0]
    if not pos.size:
        return a
    scale = np.percentile(pos, percentile)
    scale = scale if scale > 0 else float(np.max(a))
    return np.clip(a / scale, 0, 1) if scale > 0 else a


def _orient_2d(arr: np.ndarray | None) -> np.ndarray | None:
    if arr is None:
        return None
    return np.flip(np.flip(np.asarray(arr), axis=0), axis=1)


def _wrap_phase(phase: np.ndarray | None) -> np.ndarray | None:
    if phase is None:
        return None
    return np.angle(np.exp(1j * np.asarray(phase, dtype=np.float64)))


def _fit_at(
    fits: list[EvenOddPhaseFit],
    image_index: int,
    stage_name: str,
) -> EvenOddPhaseFit | None:
    if not fits:
        return None
    if image_index < 0 or image_index >= len(fits):
        raise SystemExit(
            f"--image-index must be in [0, {len(fits) - 1}] for {stage_name}, "
            f"got {image_index}"
        )
    return fits[image_index]


def _empty_panel(ax, title: str) -> None:
    ax.set_title(title, fontsize=10)
    ax.text(0.5, 0.5, "not available", ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()


def _show_magnitude(fig, ax, image: np.ndarray, title: str) -> None:
    im = ax.imshow(_scale_abs(image), cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _show_phase(fig, ax, phase: np.ndarray | None, title: str) -> None:
    phase = _wrap_phase(_orient_2d(phase))
    if phase is None:
        _empty_panel(ax, title)
        return

    im = ax.imshow(phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([-np.pi, 0, np.pi])
    cbar.set_ticklabels(["-pi", "0", "pi"])


def _show_mask(fig, ax, mask: np.ndarray | None, title: str) -> None:
    mask = _orient_2d(mask)
    if mask is None:
        _empty_panel(ax, title)
        return

    im = ax.imshow(mask.astype(np.float64), cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _add_fit_arrays(data: dict[str, np.ndarray], prefix: str, fit: EvenOddPhaseFit | None) -> None:
    if fit is None:
        return
    data[f"{prefix}_coeffs"] = np.asarray(fit.coeffs)
    data[f"{prefix}_phase_map"] = np.asarray(fit.phase_map)
    data[f"{prefix}_smooth_phase"] = np.asarray(fit.smooth_phase)
    data[f"{prefix}_mask"] = np.asarray(fit.mask)
    if fit.phase_difference is not None:
        data[f"{prefix}_phase_difference"] = np.asarray(fit.phase_difference)


def _save_phase_arrays(
    npz_path: Path,
    diagnostics: PhaseCorrectionDiagnostics,
    first_fit: EvenOddPhaseFit | None,
    refined_fit: EvenOddPhaseFit | None,
    motion_fit: EvenOddPhaseFit | None,
) -> None:
    arrays: dict[str, np.ndarray] = {}
    _add_fit_arrays(arrays, "first_pass", first_fit)
    _add_fit_arrays(arrays, "refined_even_odd", refined_fit)
    _add_fit_arrays(arrays, "motion_between_shots", motion_fit)
    if diagnostics.optimized_even_phase_coeffs:
        arrays["optimized_even_phase_coeffs"] = np.stack(
            [np.asarray(coeffs) for coeffs in diagnostics.optimized_even_phase_coeffs],
            axis=0,
        )
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--file-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Bruker experiment directory (default: {DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--spen-index",
        type=int,
        default=1,
        help="1-based SPEN index inside the datalist (default: 1).",
    )
    parser.add_argument(
        "--image-index",
        type=int,
        default=0,
        help="Zero-based receiver/image index before coil combination (default: 0).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("/tmp/pv360_python_phase"),
        help="Output directory for PNG + NPZ files (default: /tmp/pv360_python_phase).",
    )
    parser.add_argument(
        "--no-motion-phase",
        action="store_true",
        help="Disable the final between-shot/motion phase correction panel.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rare_id, epi_id, spen_ids = read_datalist(str(args.file_dir))
    del rare_id, epi_id

    if args.spen_index < 1 or args.spen_index > len(spen_ids):
        raise SystemExit(
            f"--spen-index must be in [1, {len(spen_ids)}], got {args.spen_index}"
        )

    scan_id = spen_ids[args.spen_index - 1]
    scan_dir = args.file_dir / str(scan_id)
    n_segments = read_pv_param(str(scan_dir), "NSegments")

    print(f"SPEN index   : {args.spen_index} (scan id = {scan_id})")
    print(f"Image index  : {args.image_index}")
    print(f"NSegments    : {n_segments}")
    print("Running reconstruction with phase diagnostics ...")

    recon = reconstruct_odd_segments(
        str(scan_dir),
        return_phase_diagnostics=True,
        smooth_motion_phase_between_shots=not args.no_motion_phase,
    )
    if not recon.phase_diagnostics:
        raise SystemExit(
            "No phase diagnostics were collected. This usually means the scan "
            "did not use the one-shot odd-segment phase-correction path."
        )

    diagnostics = recon.phase_diagnostics[-1]
    first_fit = _fit_at(diagnostics.first_pass, args.image_index, "first-pass phase")
    refined_fit = _fit_at(diagnostics.refined_even_odd, args.image_index, "refined phase")
    motion_fit = _fit_at(
        diagnostics.motion_between_shots,
        args.image_index,
        "motion/between-shot phase",
    )

    image_spen = orient_pv360_spen_image(recon.images)
    imag_low = orient_pv360_spen_image(_to_numpy(recon.imag_low))
    imag_origin = orient_pv360_spen_image(_to_numpy(recon.imag_origin))

    args.output.mkdir(parents=True, exist_ok=True)
    fig_path = args.output / f"spen_{args.spen_index:03d}_phase_maps.png"
    npz_path = args.output / f"spen_{args.spen_index:03d}_phase_maps.npz"

    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5), constrained_layout=True)
    _show_magnitude(fig, axes[0, 0], imag_origin, "|Imag_origin|")
    _show_magnitude(fig, axes[0, 1], imag_low, "|Imag_low|")
    _show_magnitude(fig, axes[0, 2], image_spen, "|Image_SPEN|")
    _show_mask(
        fig,
        axes[0, 3],
        refined_fit.mask if refined_fit is not None else None,
        "Refined phase mask",
    )
    _show_phase(
        fig,
        axes[1, 0],
        refined_fit.phase_difference if refined_fit is not None else None,
        "Odd/even phase diff",
    )
    _show_phase(
        fig,
        axes[1, 1],
        first_fit.phase_map if first_fit is not None else None,
        "First-pass phase map",
    )
    _show_phase(
        fig,
        axes[1, 2],
        refined_fit.smooth_phase if refined_fit is not None else None,
        "Refined SmoothPhase",
    )
    _show_phase(
        fig,
        axes[1, 3],
        motion_fit.smooth_phase if motion_fit is not None else None,
        "Motion SmoothPhase",
    )

    fig.suptitle(
        f"PV360 phase diagnostics - SPEN {args.spen_index} "
        f"(scan {scan_id}, image {args.image_index})",
        fontsize=12,
    )
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    _save_phase_arrays(npz_path, diagnostics, first_fit, refined_fit, motion_fit)

    print()
    print("Phase panels are display-wrapped to [-pi, pi].")
    print(f"Saved figure : {fig_path}")
    print(f"Saved arrays : {npz_path}")


if __name__ == "__main__":
    main()
