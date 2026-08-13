#!/usr/bin/env python3
"""Test script for pv360 Bruker data pipeline.

Usage:
    python test_pv360.py /path/to/bruker/experiment_dir
    python test_pv360.py /path/to/bruker/experiment_dir --output /tmp/export

Or import individual components:
    from test_pv360 import test_image_reader, test_raw_reader, test_param_reader
"""

import sys
import os
import argparse
import numpy as np

# Ensure spenpy is importable when running as a script
sys.path.insert(0, os.path.dirname(__file__))

from spenpy.bruker.param import read_pv_param
from spenpy.bruker.image import read_bruker_2dseq
from spenpy.bruker.raw import read_bruker_kspace_pv360_fid_multichannel


def test_param_reader(fid_dir: str):
    """Read a few key parameters and print them."""
    print(f"\n--- Parameter Reader Test ---")
    print(f"Directory: {fid_dir}")

    params = ["NSegments", "PVM_Matrix", "PVM_NReceivers", "PVM_NEchoes",
              "VisuCoreSize", "VisuCoreWordType"]
    for p in params:
        val = read_pv_param(fid_dir, p)
        print(f"  {p}: {val}")


def test_image_reader(pdata_dir: str):
    """Read a 2dseq image and report shape/type."""
    print(f"\n--- Image Reader Test (2dseq) ---")
    print(f"Directory: {pdata_dir}")

    data = read_bruker_2dseq(pdata_dir)
    print(f"  Shape: {data.shape}")
    print(f"  Dtype: {data.dtype}")
    print(f"  Min:   {np.min(np.abs(data)):.4e}")
    print(f"  Max:   {np.max(np.abs(data)):.4e}")
    print(f"  Has NaN: {np.any(np.isnan(data))}")
    return data


def test_raw_reader(fid_dir: str):
    """Read raw k-space and report shape/type."""
    print(f"\n--- Raw Reader Test (fid/rawdata) ---")
    print(f"Directory: {fid_dir}")

    data = read_bruker_kspace_pv360_fid_multichannel(fid_dir)
    print(f"  Shape: {data.shape}")
    print(f"  Dtype: {data.dtype}")
    print(f"  Min:   {np.min(np.abs(data)):.4e}")
    print(f"  Max:   {np.max(np.abs(data)):.4e}")
    print(f"  Has NaN: {np.any(np.isnan(data))}")
    return data


def test_full_pipeline(file_dir: str, export_dir: str = None):
    """Run the full pv360 pipeline."""
    print(f"\n{'='*60}")
    print(f"Full PV360 Pipeline Test")
    print(f"{'='*60}")

    from spenpy.cli.pv360 import run_pv360
    run_pv360(file_dir, export_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test pv360 Bruker data pipeline")
    parser.add_argument("file_dir", help="Path to Bruker experiment directory")
    parser.add_argument("--output", "-o", default=None, help="Output directory")
    parser.add_argument("--component", choices=["param", "image", "raw", "full"],
                        default="full", help="Which component to test")
    parser.add_argument("--scan-id", type=int, default=1,
                        help="Scan ID to test (default: 1)")
    args = parser.parse_args()

    file_dir = args.file_dir.rstrip("/")

    if args.component == "param":
        test_param_reader(file_dir)
    elif args.component == "image":
        pdata_dir = os.path.join(file_dir, str(args.scan_id), "pdata", "1")
        test_image_reader(pdata_dir)
    elif args.component == "raw":
        fid_dir = os.path.join(file_dir, str(args.scan_id))
        test_raw_reader(fid_dir)
    elif args.component == "full":
        test_full_pipeline(file_dir, args.output)
