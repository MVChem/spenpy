"""Full PV360 script equivalent.

This module mirrors the top-level workflow in ``spen_matlab/pv360.m``:

1. read ``datalist.txt``
2. process RARE and EPI ``2dseq`` images
3. reconstruct every listed SPEN scan
4. save MATLAB-compatible ``.mat`` files
5. save visual PNG summaries for quick inspection

Run:
    python -m spenpy.cli.pv360_full --file-dir /path/to/bruker/study

For a quick smoke test:
    python -m spenpy.cli.pv360_full --spen-index 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.io

from spenpy.bruker.param import read_pv_param
from spenpy.cli.pv360 import process_rare_epi, process_spen, read_datalist


DEFAULT_DATA_DIR = Path(
    "/home/data1/musong/workspace/python/spen_recons/spen_matlab/data/20240321_204022_lxj_spen_mouse_240321_1_1_1"
)


def _matplotlib(show: bool):
    import matplotlib

    if not show:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _middle_image(data: np.ndarray) -> np.ndarray:
    arr = np.squeeze(np.asarray(data))
    while arr.ndim > 2:
        arr = np.take(arr, arr.shape[-1] // 2, axis=-1)
    if arr.ndim != 2:
        raise ValueError(f"Expected image-like data, got shape {arr.shape}")
    return arr


def _scale_abs(data: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    arr = np.abs(_middle_image(data)).astype(np.float64)
    positive = arr[arr > 0]
    if positive.size == 0:
        return arr
    scale = np.percentile(positive, percentile)
    if scale <= 0:
        scale = float(np.max(positive))
    return np.clip(arr / scale, 0, 1) if scale > 0 else arr


def _volume_slices(data: np.ndarray, max_slices: int = 16) -> list[np.ndarray]:
    arr = np.squeeze(np.asarray(data))
    if arr.ndim == 2:
        return [arr]
    if arr.ndim > 3:
        while arr.ndim > 3:
            arr = np.take(arr, arr.shape[-1] // 2, axis=-1)
    if arr.ndim != 3:
        return [_middle_image(arr)]

    n_slices = arr.shape[-1]
    if n_slices <= max_slices:
        idxs = np.arange(n_slices)
    else:
        idxs = np.round(np.linspace(0, n_slices - 1, max_slices)).astype(int)
    return [arr[:, :, idx] for idx in idxs]


def save_volume_figure(
    data: np.ndarray,
    figure_path: Path,
    title: str,
    show: bool = False,
) -> Path:
    """Save a MATLAB ``imshow3Dfull``-style slice overview."""
    plt = _matplotlib(show)
    slices = _volume_slices(data)
    n = len(slices)
    cols = min(6, n)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.set_axis_off()

    for idx, image in enumerate(slices):
        ax = axes.ravel()[idx]
        ax.imshow(_scale_abs(image), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"slice {idx + 1}", fontsize=9)

    fig.suptitle(title)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return figure_path


def save_spen_figure(
    mat_path: Path,
    figure_path: Path,
    title: str,
    show: bool = False,
) -> Path:
    """Save a low-resolution/origin/reconstruction panel for one SPEN scan."""
    plt = _matplotlib(show)
    mat = scipy.io.loadmat(mat_path)

    panels = [
        ("Imag_low", mat.get("Imag_low")),
        ("Imag_origin", mat.get("Imag_origin")),
        ("Image_SPEN", mat.get("Image_SPEN")),
    ]
    panels = [(name, value) for name, value in panels if value is not None]
    if not panels:
        raise ValueError(f"No SPEN image variables found in {mat_path}")

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.2))
    axes = np.atleast_1d(axes)
    for ax, (name, image) in zip(axes, panels):
        im = ax.imshow(_scale_abs(image), cmap="gray", vmin=0, vmax=1)
        ax.set_title(name)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return figure_path


SpenItem = tuple[int, int, int | None, str]


def _is_spen_scan(file_dir: Path, scan_id: int) -> bool:
    return read_pv_param(str(file_dir / str(scan_id)), "SpenGyGaussStren") is not None


def _detect_pv_version(file_dir: Path, spen_tail: list[int]) -> str:
    """Infer whether ``datalist.txt`` uses PV360 or PV5 scan ordering."""
    if len(spen_tail) >= 3:
        first_is_spen = _is_spen_scan(file_dir, spen_tail[0])
        second_is_spen = _is_spen_scan(file_dir, spen_tail[1])
        third_is_spen = _is_spen_scan(file_dir, spen_tail[2])
        if first_is_spen and not second_is_spen and third_is_spen:
            return "pv5"
    return "pv360"


def _build_spen_items(
    file_dir: Path,
    epi_id: int,
    spen_tail: list[int],
    pv_version: str,
) -> tuple[str, list[SpenItem]]:
    if pv_version == "auto":
        pv_version = _detect_pv_version(file_dir, spen_tail)
    if pv_version not in {"pv360", "pv5"}:
        raise ValueError(f"Unsupported pv_version: {pv_version}")

    if pv_version == "pv360":
        return pv_version, [
            (spen_index, scan_id, None, "pv360")
            for spen_index, scan_id in enumerate(spen_tail, start=1)
        ]

    pair_source = [epi_id, *spen_tail]
    items: list[SpenItem] = []
    for spen_index, offset in enumerate(range(0, len(pair_source) - 1, 2), start=1):
        traj_id = pair_source[offset]
        scan_id = pair_source[offset + 1]
        items.append((spen_index, scan_id, traj_id, "pv5"))
    return pv_version, items


def _selected_spen_items(
    spen_items: list[SpenItem],
    spen_indices: Iterable[int] | None,
    max_spen: int | None,
) -> list[SpenItem]:
    indexed = list(spen_items)
    if spen_indices:
        wanted = set(spen_indices)
        indexed = [item for item in indexed if item[0] in wanted]
    if max_spen is not None:
        indexed = indexed[:max_spen]
    return indexed


def run_pv360_full(
    file_dir: str | Path = DEFAULT_DATA_DIR,
    export_dir: str | Path | None = None,
    figure_dir: str | Path | None = None,
    save_figures: bool = True,
    show: bool = False,
    spen_indices: Iterable[int] | None = None,
    max_spen: int | None = None,
    skip_existing: bool = False,
    continue_on_error: bool = False,
    pv_version: str = "auto",
) -> dict[str, object]:
    """Run a full ``pv360.m``-like export and visualization pass."""
    file_dir = Path(file_dir)
    if export_dir is None:
        export_dir = Path.cwd() / "export_data" / "pv360"
    else:
        export_dir = Path(export_dir)
    if figure_dir is None:
        figure_dir = export_dir / "figures"
    else:
        figure_dir = Path(figure_dir)

    if not file_dir.is_dir():
        raise NotADirectoryError(f"Directory not found: {file_dir}")

    export_dir.mkdir(parents=True, exist_ok=True)
    if save_figures:
        figure_dir.mkdir(parents=True, exist_ok=True)

    rare_id, epi_id, spen_ids = read_datalist(str(file_dir))
    detected_pv_version, spen_items = _build_spen_items(file_dir, epi_id, spen_ids, pv_version)
    selected_spen = _selected_spen_items(spen_items, spen_indices, max_spen)

    print(f"Input directory: {file_dir}")
    print(f"Export directory: {export_dir}")
    if save_figures:
        print(f"Figure directory: {figure_dir}")
    print(f"RARE scan ID: {rare_id}")
    print(f"EPI scan ID: {epi_id}")
    print(f"PV reconstruction mode: {detected_pv_version}")
    print(f"SPEN scan IDs selected: {[scan_id for _, scan_id, _, _ in selected_spen]}")
    if detected_pv_version == "pv5":
        print(
            "PV5 trajectory pairs: "
            f"{[(traj_id, scan_id) for _, scan_id, traj_id, _ in selected_spen]}"
        )

    outputs: dict[str, object] = {
        "file_dir": str(file_dir),
        "export_dir": str(export_dir),
        "figure_dir": str(figure_dir) if save_figures else None,
        "rare_scan_id": rare_id,
        "epi_scan_id": epi_id,
        "pv_version": detected_pv_version,
        "spen": [],
        "errors": [],
    }

    print("\nProcessing RARE data...")
    rare_mat = export_dir / "ratbrain_RARE.mat"
    if skip_existing and rare_mat.exists():
        rare_data = scipy.io.loadmat(rare_mat)["Image_RARE"]
        print(f"  Skipping existing: {rare_mat}")
    else:
        rare_data = process_rare_epi(
            str(file_dir), rare_id, str(export_dir), "ratbrain_RARE.mat", "Image_RARE"
        )
    if save_figures:
        save_volume_figure(rare_data, figure_dir / "ratbrain_RARE.png", "RARE Image", show)

    print("\nProcessing EPI data...")
    epi_mat = export_dir / "ratbrain_EPI.mat"
    if skip_existing and epi_mat.exists():
        epi_data = scipy.io.loadmat(epi_mat)["image_EPI"]
        print(f"  Skipping existing: {epi_mat}")
    else:
        epi_data = process_rare_epi(
            str(file_dir), epi_id, str(export_dir), "ratbrain_EPI.mat", "image_EPI"
        )
    if save_figures:
        save_volume_figure(epi_data, figure_dir / "ratbrain_EPI.png", "EPI Image", show)

    print("\nProcessing SPEN data series...")
    for spen_index, scan_id, traj_id, recon_flavor in selected_spen:
        spen_mat = export_dir / f"ratbrain_SPEN_96_{spen_index}.mat"
        try:
            if skip_existing and spen_mat.exists():
                print(f"  Skipping existing SPEN {spen_index}: {spen_mat}")
            else:
                process_spen(
                    str(file_dir),
                    scan_id,
                    str(export_dir),
                    spen_index,
                    traj_scan_id=traj_id,
                    recon_flavor=recon_flavor,
                )

            fig_path = None
            if save_figures:
                fig_path = save_spen_figure(
                    spen_mat,
                    figure_dir / f"ratbrain_SPEN_96_{spen_index}.png",
                    f"SPEN Result {spen_index} (scan {scan_id})",
                    show,
                )
            outputs["spen"].append(
                {
                    "spen_index": spen_index,
                    "scan_id": scan_id,
                    "trajectory_scan_id": traj_id,
                    "mat_file": str(spen_mat),
                    "figure": str(fig_path) if fig_path is not None else None,
                }
            )
        except Exception as exc:
            entry = {
                "spen_index": spen_index,
                "scan_id": scan_id,
                "trajectory_scan_id": traj_id,
                "error": repr(exc),
            }
            outputs["errors"].append(entry)
            if not continue_on_error:
                raise
            print(f"  ERROR SPEN {spen_index} scan {scan_id}: {exc}")

    summary_path = export_dir / "pv360_full_summary.json"
    summary_path.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    outputs["summary_file"] = str(summary_path)

    print(f"\nDone. Files exported to {export_dir}")
    if save_figures:
        print(f"Figures exported to {figure_dir}")
    print(f"Summary: {summary_path}")
    return outputs


def _parse_indices(values: list[str] | None) -> list[int] | None:
    if not values:
        return None
    indices: list[int] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                indices.append(int(part))
    return indices


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a full pv360.m-like reconstruction.")
    parser.add_argument(
        "--file-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Bruker experiment directory. Default: {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output directory for .mat files. Default: ./export_data/pv360",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=None,
        help="Output directory for PNG figures. Default: <output>/figures",
    )
    parser.add_argument(
        "--spen-index",
        action="append",
        default=None,
        help="Process selected 1-based SPEN index. Can be repeated or comma-separated.",
    )
    parser.add_argument(
        "--max-spen",
        type=int,
        default=None,
        help="Process only the first N selected SPEN scans.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--show", action="store_true", help="Show figures interactively.")
    parser.add_argument(
        "--pv-version",
        choices=["auto", "pv360", "pv5"],
        default="auto",
        help="Interpret datalist/regridding as PV360 or PV5. Default: auto.",
    )
    args = parser.parse_args()

    run_pv360_full(
        file_dir=args.file_dir,
        export_dir=args.output,
        figure_dir=args.figure_dir,
        save_figures=not args.no_figures,
        show=args.show,
        spen_indices=_parse_indices(args.spen_index),
        max_spen=args.max_spen,
        skip_existing=args.skip_existing,
        continue_on_error=args.continue_on_error,
        pv_version=args.pv_version,
    )


if __name__ == "__main__":
    main()
