from pathlib import Path

import numpy as np

from spenpy.siemens.raw import (
    SiemensRawCalibration,
    apply_pe_operator,
    load_raw_calibration,
    q_value_from_name,
    save_raw_calibration,
    siemens_physics_operator,
)


def test_q_value_from_name():
    assert q_value_from_name("scan_Q60_R.dat") == 60.0
    assert q_value_from_name("scan-q38.5-test.dat") == 38.5
    assert q_value_from_name("scan_without_q.dat") is None


def test_apply_pe_operator_preserves_non_pe_axes():
    rng = np.random.default_rng(3)
    signal = (
        rng.normal(size=(4, 3, 2, 2, 1, 2))
        + 1j * rng.normal(size=(4, 3, 2, 2, 1, 2))
    ).astype(np.complex64)
    matrix = np.diag(np.arange(1, 5)).astype(np.complex64)
    actual = apply_pe_operator(signal, matrix)
    expected = signal * np.arange(1, 5)[:, None, None, None, None, None]
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_calibration_round_trip(tmp_path: Path):
    calibration = SiemensRawCalibration(
        matrix=np.eye(3, dtype=np.complex64) * (1 + 2j),
        source_dat=Path("/tmp/source.dat"),
        source_mat=Path("/tmp/source.mat"),
        q_value=60.0,
        ridge=1e-3,
        readout_stride=2,
        slice_stride=2,
        repetition_index=0,
        set_index=0,
        signal_name="SignalFixedPostROFFTPostSR_R",
    )
    path = save_raw_calibration(calibration, tmp_path / "calibration.npz")
    loaded = load_raw_calibration(path)
    np.testing.assert_array_equal(loaded.matrix, calibration.matrix)
    assert loaded.q_value == 60.0
    assert loaded.source_dat == calibration.source_dat
    assert loaded.signal_name == calibration.signal_name


def test_physics_operator_shape_and_finiteness():
    operator = siemens_physics_operator(60.0, 12)
    assert operator.shape == (12, 12)
    assert operator.dtype == np.complex64
    assert np.all(np.isfinite(operator))
