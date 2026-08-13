#!/usr/bin/env python3
"""Reconstruct processed Siemens SPEN MAT files with SPENPy coil combination."""

from __future__ import annotations

import argparse
from pathlib import Path

from spenpy.siemens import (
    reconstruct_processed_mat,
    save_processed_reconstruction,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_DIR / "data"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "recon_spenpy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Processed Siemens MAT files. Defaults to all MAT files in ../data.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto, cpu, cuda, or cuda:N (default: auto).",
    )
    parser.add_argument(
        "--batch-slices",
        type=int,
        default=5,
        help="Number of slices combined per batch (default: 5).",
    )
    parser.add_argument(
        "--save-complex",
        action="store_true",
        help="Also store the complex adaptive result for every repetition.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = args.files or sorted(DEFAULT_DATA_DIR.glob("*.mat"))
    if not files:
        raise SystemExit(f"No .mat files found in {DEFAULT_DATA_DIR}")

    for index, path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {path}")
        result = reconstruct_processed_mat(
            path,
            device=args.device,
            batch_slices=args.batch_slices,
        )
        print(f"  signal: {result.signal_name} {result.signal_shape}")
        print(f"  output: {result.adaptive_magnitude.shape}")
        print(f"  voxel size: {result.voxel_size_mm} mm")
        print(f"  device: {result.device}")
        saved = save_processed_reconstruction(
            result,
            args.output,
            save_complex=args.save_complex,
        )
        for name, output_path in saved.items():
            print(f"  {name}: {output_path}")


if __name__ == "__main__":
    main()
