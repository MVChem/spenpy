#!/usr/bin/env python3
"""Demo 03 — inspect each stage of the SPEN reconstruction pipeline.

This goes one level deeper than ``02_single_spen_reconstruction.py``: it
opens up the pipeline and saves a multi-panel figure showing every
intermediate stage on the same SPEN scan, similar to the diagnostic
``figure; imagesc(...)`` blocks scattered through
``Function_Process_NewPE_SPEN_OddNumWithMask_bruker_PV360.m``.

Stages shown (5-panel figure):

    1. Raw k-space magnitude (channel 0, log-scaled)
       MATLAB analogue: ``imagesc(abs(kField(:,:,1,1,1,1)))``

    2. K-space after 1D EPI trajectory regridding (channel 0, log-scaled)
       MATLAB analogue: end of the ``oneD_regriding_PV360`` block

    3. Imag_origin (readout-FFT + coil combine, no SR correction)
       MATLAB analogue: ``coilCombinebao(FFTKSpace2XSpace(CmplxData, 2))``

    4. |tmpInvAZ| — the weighted adjoint encoding matrix used by SPEN
       MATLAB analogue: ``calcInvA(...)`` output

    5. Image_SPEN (SR-corrected, oriented)
       MATLAB analogue: ``flip(flip(images, 1), 2)`` after MultMatTensor

Examples:
    python spenpy/demo/03_inspect_intermediate_steps.py --spen-index 1
    python spenpy/demo/03_inspect_intermediate_steps.py --spen-index 5 --output /tmp/steps
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spenpy.bruker.param import read_pv_param
from spenpy.bruker.raw import read_bruker_kspace_pv360_fid_multichannel
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


def _first_2d_slice(arr: np.ndarray) -> np.ndarray:
    """Return a 2D (RO, PE)-like slice from any-rank numpy array."""
    a = np.asarray(arr)
    while a.ndim > 2:
        a = a[..., 0] if a.shape[-1] < a.shape[0] else np.take(a, 0, axis=-1)
    return a


def _log_abs(x: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    a = np.abs(_first_2d_slice(x)).astype(np.float64)
    a = np.log1p(a / max(a.max(), eps))
    return a / max(a.max(), 1e-12)


def _scale_abs(x: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    a = np.abs(_first_2d_slice(x)).astype(np.float64)
    pos = a[a > 0]
    if not pos.size:
        return a
    s = np.percentile(pos, percentile)
    s = s if s > 0 else float(np.max(a))
    return np.clip(a / s, 0, 1) if s > 0 else a


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--spen-index", type=int, default=1)
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("/tmp/pv360_python_steps"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rare_id, epi_id, spen_ids = read_datalist(str(args.file_dir))
    del rare_id, epi_id

    if args.spen_index < 1 or args.spen_index > len(spen_ids):
        raise SystemExit(
            f"--spen-index must be in [1, {len(spen_ids)}], got {args.spen_index}"
        )

    scan_id = spen_ids[args.spen_index - 1]
    scan_dir = args.file_dir / str(scan_id)

    print(f"SPEN index : {args.spen_index} (scan id = {scan_id})")
    print(f"Scan dir   : {scan_dir}")

    n_segments = read_pv_param(str(scan_dir), "NSegments")
    matrix = read_pv_param(str(scan_dir), "PVM_Matrix")
    print(f"NSegments  : {n_segments}")
    print(f"PVM_Matrix : {matrix}")
    print()

    # --- Stage 1: raw k-space (channel 0). -----------------------------------
    print("Stage 1/5: raw k-space ...")
    kfield_raw = read_bruker_kspace_pv360_fid_multichannel(str(scan_dir))
    kfield_raw_2d = _first_2d_slice(np.squeeze(kfield_raw))
    print(f"  raw kfield shape: {kfield_raw.shape}, slice shown: {kfield_raw_2d.shape}")

    # --- Stage 2-5: full pipeline + regridded k-space ------------------------
    print("Stage 2/5: 1D regridding + reconstruction (this can take a while) ...")
    recon = reconstruct_odd_segments(str(scan_dir))

    # ``recon.kfield`` is the post-regridding, post-zero-fill complex k-space
    # used internally by reconstruct_odd_segments (shape [RO, PE, sl, arr, echo]).
    kfield_regrid_2d = _first_2d_slice(np.squeeze(recon.kfield))
    print(f"  regridded kfield shape: {recon.kfield.shape}")

    print("Stage 3/5: Imag_origin (readout FFT, no SR) ...")
    imag_origin = orient_pv360_spen_image(_to_numpy(recon.imag_origin))

    print("Stage 4/5: |tmpInvAZ| encoding matrix ...")
    inv_a = _to_numpy(recon.spen_az["tmpInvAZ"])

    print("Stage 5/5: Image_SPEN (SR + orient) ...")
    image_spen = orient_pv360_spen_image(recon.images)

    # --- 5-panel figure -------------------------------------------------------
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.6), constrained_layout=True)

    panels = [
        (axes[0], _log_abs(kfield_raw_2d),
         f"1) Raw k-space (log|.|)\n{kfield_raw_2d.shape}", "viridis"),
        (axes[1], _log_abs(kfield_regrid_2d),
         f"2) After 1D regridding (log|.|)\n{kfield_regrid_2d.shape}", "viridis"),
        (axes[2], _scale_abs(imag_origin),
         f"3) Imag_origin (FFT-only)\n{imag_origin.shape}", "gray"),
        (axes[3], _scale_abs(inv_a),
         f"4) |tmpInvAZ|\n{inv_a.shape}", "magma"),
        (axes[4], _scale_abs(image_spen),
         f"5) Image_SPEN\n{image_spen.shape}", "gray"),
    ]
    for ax, data, title, cmap in panels:
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"SPEN intermediate stages — index {args.spen_index} (scan {scan_id})",
        fontsize=12,
    )

    fig_path = args.output / f"spen_{args.spen_index:03d}_intermediate.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    print()
    print(f"Saved figure : {fig_path}")


if __name__ == "__main__":
    main()
