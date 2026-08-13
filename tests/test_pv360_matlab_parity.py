"""MATLAB parity checks for the PV360 Bruker readers.

These tests compare Python reader outputs with MATLAB outputs produced from the
same source files.

Run directly without pytest:
    /home/data1/musong/workspace/python/spen_recons/.venv/bin/python \
        tests/test_pv360_matlab_parity.py
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import scipy.io

from spenpy.bruker.raw import read_bruker_kspace_pv360_fid_multichannel
from spenpy.cli.pv360 import process_rare_epi, process_spen, read_datalist


DATA_DIR = Path(
    "/home/data1/musong/workspace/2026/03/17/spen_matlab/data/"
    "20240321_204022_lxj_spen_mouse_240321_1_1_1"
)
MATLAB_EXPORT_DIR = Path("/home/data1/musong/workspace/2026/03/17/spen_matlab/export_data/pv360")
MATLAB_CODE_DIR = Path("/home/data1/musong/workspace/python/spen_recons/spen_matlab")


def _require_reference_environment() -> None:
    missing = []
    if shutil.which("matlab") is None:
        missing.append("matlab executable")
    if not DATA_DIR.exists():
        missing.append(str(DATA_DIR))
    if not MATLAB_CODE_DIR.exists():
        missing.append(str(MATLAB_CODE_DIR))
    if not MATLAB_EXPORT_DIR.exists():
        missing.append(str(MATLAB_EXPORT_DIR))
    if missing:
        raise RuntimeError("Missing MATLAB parity dependencies: " + ", ".join(missing))


def _matlab_kfield(scan_id: int) -> np.ndarray:
    """Run MATLAB raw reader and return complex kField as double precision."""
    with tempfile.NamedTemporaryFile(suffix=".mat") as tmp:
        matlab_cmd = (
            f"cd('{MATLAB_CODE_DIR}'); "
            "addpath(genpath('spen')); "
            f"kField = ReadBrukerkSpace_PV360_fid_multichannel('{DATA_DIR}/{scan_id}/'); "
            "kReal = real(kField); "
            "kImag = imag(kField); "
            f"save('{tmp.name}', 'kReal', 'kImag');"
        )
        subprocess.run(["matlab", "-batch", matlab_cmd], check=True)
        mat = scipy.io.loadmat(tmp.name)
    return mat["kReal"] + 1j * mat["kImag"]


def test_datalist_matches_matlab_pv360_expectation():
    rare_id, epi_id, spen_ids = read_datalist(str(DATA_DIR))

    assert rare_id == 4
    assert epi_id == 6
    assert spen_ids[0] == 15
    assert spen_ids[-1] == 57
    assert len(spen_ids) == 43


def test_rare_epi_exports_match_existing_matlab_pv360_outputs():
    with tempfile.TemporaryDirectory() as tmp:
        rare_py = process_rare_epi(str(DATA_DIR), 4, tmp, "ratbrain_RARE.mat", "Image_RARE")
        epi_py = process_rare_epi(str(DATA_DIR), 6, tmp, "ratbrain_EPI.mat", "image_EPI")

    rare_mat = scipy.io.loadmat(MATLAB_EXPORT_DIR / "ratbrain_RARE.mat")["Image_RARE"]
    epi_mat = scipy.io.loadmat(MATLAB_EXPORT_DIR / "ratbrain_EPI.mat")["image_EPI"]

    assert rare_py.shape == rare_mat.shape == (128, 128, 11)
    assert epi_py.shape == epi_mat.shape == (96, 96, 11)
    np.testing.assert_array_equal(rare_py, rare_mat)
    np.testing.assert_array_equal(epi_py, epi_mat)


def _assert_spen_raw_reader_matches_matlab(scan_id: int):
    kfield_py = read_bruker_kspace_pv360_fid_multichannel(str(DATA_DIR / str(scan_id)))
    kfield_py = np.squeeze(kfield_py, axis=-1)
    kfield_mat = _matlab_kfield(scan_id)

    assert kfield_py.shape == kfield_mat.shape == (96, 96, 1, 4)
    np.testing.assert_array_equal(kfield_py, kfield_mat)
    print(f"OK SPEN raw scan {scan_id}: exact match, shape={kfield_py.shape}")


def _abs_corr(a: np.ndarray, b: np.ndarray) -> float:
    a_abs = np.abs(np.asarray(a)).reshape(-1)
    b_abs = np.abs(np.asarray(b)).reshape(-1)
    denom = np.linalg.norm(a_abs) * np.linalg.norm(b_abs)
    if denom == 0:
        return 0.0
    return float(np.vdot(a_abs, b_abs).real / denom)


def test_spen_raw_reader_matches_matlab_first_scan():
    _assert_spen_raw_reader_matches_matlab(15)


def test_spen_raw_reader_matches_matlab_last_scan():
    _assert_spen_raw_reader_matches_matlab(57)


def test_spen_reconstruction_export_has_matlab_variable_names():
    with tempfile.TemporaryDirectory() as tmp:
        process_spen(str(DATA_DIR), 15, tmp, 1)
        sample = scipy.io.loadmat(Path(tmp) / "ratbrain_SPEN_96_1.mat")

        assert sample["Imag_low"].shape == (96, 96)
        assert sample["Imag_origin"].shape == (96, 96)
        assert sample["Image_SPEN"].shape == (96, 96)
        assert "SPEN_AZ" in sample
        assert int(sample["NSegments"].squeeze()) == 1

        matlab_sample = scipy.io.loadmat(MATLAB_EXPORT_DIR / "ratbrain_SPEN_96_1.mat")
        assert _abs_corr(sample["Imag_low"], matlab_sample["Imag_low"]) > 0.999
        assert _abs_corr(sample["Imag_origin"], matlab_sample["Imag_origin"]) > 0.999
        assert _abs_corr(sample["Image_SPEN"], matlab_sample["Image_SPEN"]) > 0.999


def main() -> None:
    _require_reference_environment()
    test_datalist_matches_matlab_pv360_expectation()
    print("OK datalist")
    test_rare_epi_exports_match_existing_matlab_pv360_outputs()
    print("OK RARE/EPI exports: exact match")
    test_spen_raw_reader_matches_matlab_first_scan()
    test_spen_raw_reader_matches_matlab_last_scan()
    test_spen_reconstruction_export_has_matlab_variable_names()
    print("OK SPEN reconstruction export")


if __name__ == "__main__":
    main()
