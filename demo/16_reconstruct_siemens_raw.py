#!/usr/bin/env python3
"""Reconstruct Siemens hybrid-SPEN Twix raw data.

Examples
--------
Fit a Q60 calibration from the right-breast b0 pair::

    python 16_reconstruct_siemens_raw.py calibrate raw.dat processed.mat \
        -o q60_calibration.npz

Apply it to held-out Q60 scans and optionally score after reconstruction::

    python 16_reconstruct_siemens_raw.py reconstruct q60_calibration.npz scan.dat \
        --metadata-dir ../data --reference-dir ../data --output ../recon_siemens_raw

Run the current raw-only analytical approximation::

    python 16_reconstruct_siemens_raw.py physics scan.dat \
        --metadata-dir ../data --output ../recon_siemens_raw_physics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spenpy.siemens import (
    fit_raw_calibration,
    load_raw_calibration,
    reconstruct_raw_dat,
    save_raw_calibration,
    save_raw_reconstruction,
    validate_against_processed_mat,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_DIR / "data"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "recon_siemens_raw"


def _paired_mat(dat_path: Path, directory: Path | None) -> Path | None:
    candidate = (directory / f"{dat_path.stem}.mat") if directory else dat_path.with_suffix(".mat")
    return candidate.resolve() if candidate.exists() else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibrate = subparsers.add_parser(
        "calibrate",
        help="Fit a transferable PE operator from one paired DAT/MAT scan.",
    )
    calibrate.add_argument("dat", type=Path)
    calibrate.add_argument("mat", type=Path)
    calibrate.add_argument("--output", "-o", type=Path, required=True)
    calibrate.add_argument("--ridge", type=float, default=1e-3)
    calibrate.add_argument("--readout-stride", type=int, default=2)
    calibrate.add_argument("--slice-stride", type=int, default=2)
    calibrate.add_argument("--repetition", type=int, default=0)
    calibrate.add_argument("--set", dest="set_index", type=int, default=0)

    reconstruct = subparsers.add_parser(
        "reconstruct",
        help="Apply a saved calibrated operator to one or more Twix scans.",
    )
    reconstruct.add_argument("calibration", type=Path)
    reconstruct.add_argument("files", nargs="+", type=Path)
    _add_reconstruction_options(reconstruct)
    reconstruct.add_argument(
        "--allow-q-mismatch",
        action="store_true",
        help="Apply a calibration to a different Q value (normally unsafe).",
    )

    physics = subparsers.add_parser(
        "physics",
        help="Use raw-only SPENPy weighted-adjoint reconstruction.",
    )
    physics.add_argument("files", nargs="+", type=Path)
    _add_reconstruction_options(physics)
    physics.add_argument(
        "--q",
        type=float,
        help="Override Q; otherwise infer it from each _Q<number> filename.",
    )
    physics.add_argument(
        "--phase-factor-pi",
        type=float,
        default=2.0,
        help="Use MaxPhase = this * pi * Q (default: 2).",
    )
    physics.add_argument(
        "--gauss-relative-width",
        type=float,
        default=0.2,
        help="SPENPy weighted-adjoint Gaussian width (default: 0.2).",
    )
    return parser.parse_args()


def _add_reconstruction_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        help="Directory containing same-stem MAT files; only metadata is read.",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        help=(
            "After reconstruction, score against same-stem processed MAT files. "
            "References are not read until the reconstruction is complete."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-slices", type=int, default=5)
    parser.add_argument("--save-complex", action="store_true")


def _run_calibrate(args: argparse.Namespace) -> None:
    calibration = fit_raw_calibration(
        args.dat,
        args.mat,
        ridge=args.ridge,
        readout_stride=args.readout_stride,
        slice_stride=args.slice_stride,
        repetition_index=args.repetition,
        set_index=args.set_index,
    )
    path = save_raw_calibration(calibration, args.output)
    print(f"saved calibration: {path}")
    print(f"operator: {calibration.matrix.shape}")
    print(f"Q: {calibration.q_value}")
    print(f"source DAT: {calibration.source_dat}")
    print(f"source MAT: {calibration.source_mat}")


def _run_reconstructions(args: argparse.Namespace) -> None:
    calibration = (
        load_raw_calibration(args.calibration)
        if args.command == "reconstruct"
        else None
    )
    for index, path in enumerate(args.files, start=1):
        source = path.expanduser().resolve()
        metadata_mat = _paired_mat(source, args.metadata_dir)
        print(f"[{index}/{len(args.files)}] {source}")
        result = reconstruct_raw_dat(
            source,
            calibration=calibration,
            q_value=getattr(args, "q", None),
            metadata_mat=metadata_mat,
            device=args.device,
            batch_slices=args.batch_slices,
            phase_factor_pi=getattr(args, "phase_factor_pi", 2.0),
            gauss_relative_width=getattr(args, "gauss_relative_width", 0.2),
            allow_q_mismatch=getattr(args, "allow_q_mismatch", False),
        )
        validation = None
        reference_mat = _paired_mat(source, args.reference_dir)
        if args.reference_dir is not None and reference_mat is None:
            print("  reference: not found")
        elif reference_mat is not None and args.reference_dir is not None:
            validation = validate_against_processed_mat(result, reference_mat)
            print(
                "  validation magnitude correlation: "
                f"{validation['magnitude_correlation']:.6f}"
            )
            print(
                "  validation per set: "
                + ", ".join(
                    f"{value:.6f}"
                    for value in validation["magnitude_correlation_per_set"]
                )
            )
            print(
                "  validation absolute complex correlation: "
                f"{validation['absolute_complex_correlation']:.6f}"
            )

        saved = save_raw_reconstruction(
            result,
            args.output,
            save_complex=args.save_complex,
            validation=validation,
        )
        print(f"  method: {result.method}")
        print(f"  output: {result.adaptive_magnitude.shape}")
        print(f"  Q: {result.q_value}")
        print(f"  files: {json.dumps({k: str(v) for k, v in saved.items()})}")


def main() -> None:
    args = parse_args()
    if args.command == "calibrate":
        _run_calibrate(args)
    else:
        _run_reconstructions(args)


if __name__ == "__main__":
    main()
