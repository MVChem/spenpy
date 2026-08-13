#!/usr/bin/env python3
"""Demo 01 — full pv360.m mimic.

This is the Python equivalent of running ``spen_matlab/pv360.m`` from
start to finish. It reads ``datalist.txt``, processes the RARE and EPI
2dseq images, then reconstructs every SPEN scan listed in the datalist
and saves both ``.mat`` files and PNG figures.

Annotated mapping to the original MATLAB script:

    | MATLAB block in pv360.m                                | Python equivalent |
    |--------------------------------------------------------|-------------------|
    | datalist = textread('datalist.txt', '%u');             | read_datalist     |
    | imageObj = ImageDataObject(rare_pdata); ...            | process_rare_epi  |
    | imageObj = ImageDataObject(epi_pdata);  ...            | process_rare_epi  |
    | for ispen = 1:length(SPEN_datalist)                    | run_pv360_full    |
    |     [images, Imag_origin, Imag_low, SPEN_AZ] = ...     | reconstruct_odd_segments |
    |     Image_SPEN = flip(flip(images, 1), 2);             | orient_pv360_spen_image |
    |     save(fullfile(export_dir, save_name), ...);        | scipy.io.savemat  |
    |     figure('Name', ['SPEN Result ', num2str(ispen)]);  | save_spen_figure  |
    | end                                                    |                   |

Examples:

    # Smallest possible invocation, defaults to the demo dataset and
    # writes to ``./export_data/pv360`` next to the current working dir.
    python spenpy/demo/01_run_pv360_pipeline.py

    # Mimic pv360.m and dump everything to /tmp/pv360_python:
    python spenpy/demo/01_run_pv360_pipeline.py --output /tmp/pv360_python

    # Reconstruct only the first three SPEN scans (faster smoke test):
    python spenpy/demo/01_run_pv360_pipeline.py \\
        --output /tmp/pv360_python --max-spen 3

    # Reconstruct only specific SPEN indices:
    python spenpy/demo/01_run_pv360_pipeline.py --file-dir /home/data1/musong/workspace/python/spen_recons/spen_matlab/data/20231208_103719_lxj_spen_mouse4_231208_2_1_1 --output /tmp/pv360_python --spen-index 1,5,10
    
    # Reconstruct all SPEN indices:
    python spenpy/demo/01_run_pv360_pipeline.py --file-dir /home/data1/musong/workspace/python/spen_recons/spen_matlab/data/20231208_114947_lxj_spen_mouse5_2312085_2_1_1 --output /tmp/pv360_python
    
    # pv5 data
    python spenpy/demo/01_run_pv360_pipeline.py --file-dir /home/data1/musong/workspace/python/spen_recons/spen_matlab/data/lxj_spen_230907.lJ2 --output /tmp/pv5_python
    
    # unknown PV version (auto-detect)
    python spenpy/demo/01_run_pv360_pipeline.py --file-dir /home/data1/musong/workspace/python/spen_recons/spen_matlab/data/lxj_SPEN_TEST_230401.ja2 --output /tmp/unknown_pv_python
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from spenpy.cli.pv360_full import DEFAULT_DATA_DIR, run_pv360_full


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mimic spen_matlab/pv360.m end-to-end.")
    parser.add_argument(
        "--file-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Bruker experiment directory (default: {DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("/tmp/pv360_python"),
        help="Output directory for .mat + figures (default: /tmp/pv360_python).",
    )
    parser.add_argument(
        "--spen-index",
        action="append",
        default=None,
        help="Process only the given 1-based SPEN index. Can be repeated or "
        "comma-separated (e.g. --spen-index 1,5).",
    )
    parser.add_argument(
        "--max-spen",
        type=int,
        default=None,
        help="Process only the first N selected SPEN scans (handy for smoke tests).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip scans whose .mat already exists in --output.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with remaining scans if one fails.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip PNG generation (only emit .mat files).",
    )
    parser.add_argument(
        "--pv-version",
        choices=["auto", "pv360", "pv5"],
        default="auto",
        help="Interpret datalist/regridding as PV360 or PV5 (default: auto).",
    )
    return parser.parse_args()


def _parse_indices(values: list[str] | None) -> list[int] | None:
    if not values:
        return None
    out: list[int] = []
    for v in values:
        for part in v.split(","):
            part = part.strip()
            if part:
                out.append(int(part))
    return out


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Bruker dataset : {args.file_dir}")
    print(f"Export folder  : {args.output}")
    print(f"Figures folder : {args.output / 'figures'}")
    print()

    summary = run_pv360_full(
        file_dir=args.file_dir,
        export_dir=args.output,
        figure_dir=args.output / "figures",
        save_figures=not args.no_figures,
        show=False,
        spen_indices=_parse_indices(args.spen_index),
        max_spen=args.max_spen,
        skip_existing=args.skip_existing,
        continue_on_error=args.continue_on_error,
        pv_version=args.pv_version,
    )

    print()
    print(f"Done. Summary written to {summary['summary_file']}")
    if summary.get("errors"):
        print(f"There were {len(summary['errors'])} errors. See the summary JSON for details.")


if __name__ == "__main__":
    main()
