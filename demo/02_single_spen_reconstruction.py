#!/usr/bin/env python3
"""Demo 02 — reconstruct a single SPEN scan and save / visualise it.

Equivalent to *one iteration* of the ``for ispen = 1:length(SPEN_datalist)``
loop in ``spen_matlab/pv360.m``::

    spen_dir   = fullfile(file_dir, num2str(SPEN_datalist(ispen)), filesep);
    NSegments  = ReadPVParam(spen_dir, 'NSegments');
    [images, Imag_origin, Imag_low, SPEN_AZ] = ...
        Function_Process_NewPE_SPEN_OddNumWithMask_bruker_PV360(spen_dir);
    Image_SPEN = flip(flip(images, 1), 2);
    save(fullfile(export_dir, save_name), 'Imag_low', 'Imag_origin', 'Image_SPEN', 'SPEN_AZ');
    figure('Name', ['SPEN Result ', num2str(ispen)]);
    imagesc(abs(Image_SPEN));

Use this script when you want to reconstruct a *single* SPEN scan and
inspect the intermediate variables (``Imag_low``, ``Imag_origin``,
``Image_SPEN``, ``SPEN_AZ``) directly in Python rather than going via
``.mat`` files.

Examples:
    python spenpy/demo/02_single_spen_reconstruction.py --spen-index 1
    python spenpy/demo/02_single_spen_reconstruction.py --spen-index 1 --output /tmp/single_spen
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from spenpy.bruker.param import read_pv_param
from spenpy.cli.pv360 import read_datalist
from spenpy.recon import orient_pv360_spen_image, reconstruct_odd_segments


DEFAULT_DATA_DIR = Path(
    "/home/data1/musong/workspace/python/spen_recons/spen_matlab/data/"
    "20240321_204022_lxj_spen_mouse_240321_1_1_1"
)


def _to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().resolve_conj().numpy()
    return np.asarray(value)


def _scale_abs(arr: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    a = np.abs(arr).astype(np.float64)
    pos = a[a > 0]
    if not pos.size:
        return a
    s = np.percentile(pos, percentile)
    s = s if s > 0 else float(np.max(a))
    return np.clip(a / s, 0, 1) if s > 0 else a


def _save_panel(
    figure_path: Path,
    imag_low: np.ndarray,
    imag_origin: np.ndarray,
    image_spen: np.ndarray,
    title: str,
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    for ax, img, name in zip(
        axes,
        [imag_low, imag_origin, image_spen],
        ["Imag_low", "Imag_origin", "Image_SPEN"],
    ):
        im = ax.imshow(_scale_abs(img), cmap="gray", vmin=0, vmax=1)
        ax.set_title(name)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    return figure_path


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
        "--output",
        "-o",
        type=Path,
        default=Path("/tmp/pv360_python_single"),
        help="Output directory for .mat + figure (default: /tmp/pv360_python_single).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rare_id, epi_id, spen_ids = read_datalist(str(args.file_dir))
    if args.spen_index < 1 or args.spen_index > len(spen_ids):
        raise SystemExit(
            f"--spen-index must be in [1, {len(spen_ids)}], got {args.spen_index}"
        )

    scan_id = spen_ids[args.spen_index - 1]
    scan_dir = args.file_dir / str(scan_id)
    n_segments = read_pv_param(str(scan_dir), "NSegments")

    print(f"RARE scan id : {rare_id}")
    print(f"EPI scan id  : {epi_id}")
    print(f"SPEN index   : {args.spen_index} (scan id = {scan_id})")
    print(f"NSegments    : {n_segments}")
    print()
    print("Running reconstruct_odd_segments ...")

    recon = reconstruct_odd_segments(str(scan_dir))

    image_spen = orient_pv360_spen_image(recon.images)
    imag_low = orient_pv360_spen_image(_to_numpy(recon.imag_low))
    imag_origin = orient_pv360_spen_image(_to_numpy(recon.imag_origin))

    print(f"  Imag_low shape    : {imag_low.shape}")
    print(f"  Imag_origin shape : {imag_origin.shape}")
    print(f"  Image_SPEN shape  : {image_spen.shape}")
    print()

    args.output.mkdir(parents=True, exist_ok=True)
    mat_path = args.output / f"ratbrain_SPEN_96_{args.spen_index}.mat"
    fig_path = args.output / f"ratbrain_SPEN_96_{args.spen_index}.png"

    scipy.io.savemat(
        mat_path,
        {
            "Imag_low": imag_low,
            "Imag_origin": imag_origin,
            "Image_SPEN": image_spen,
            "SPEN_AZ": {key: _to_numpy(value) for key, value in recon.spen_az.items()},
            "NSegments": int(n_segments) if n_segments is not None else 1,
        },
    )
    _save_panel(
        fig_path,
        imag_low,
        imag_origin,
        image_spen,
        title=f"SPEN Result {args.spen_index} (scan {scan_id})",
    )

    print(f"Saved .mat   : {mat_path}")
    print(f"Saved figure : {fig_path}")


if __name__ == "__main__":
    main()
