import numpy as np
import pytest
import torch

from spenpy.core import Acquisition, XSPENProtocol, protocol_from_dict
from spenpy.io import (
    load_image,
    read_acquisition,
    read_mat,
    save_acquisition,
)


def test_named_selection_and_portable_round_trip(tmp_path):
    a = Acquisition(
        torch.randn(6, 8, 4, 2, 3, dtype=torch.complex64),
        ("spen", "readout", "coil", "slice", "volume"),
        "regridded_kspace",
        {"r_value": 60},
    )
    p = save_acquisition(a, tmp_path / "scan.npz")
    b = read_acquisition(p)
    assert b.axes == a.axes and b.stage == a.stage and b.metadata == a.metadata
    torch.testing.assert_close(b.select(slice=1, volume=2).data, a.data[:, :, :, 1, 2])
    assert b.select(slice=0, volume=0).permute(
        "coil", "spen", "readout"
    ).data.shape == (4, 6, 8)


def test_mat_v73_complex_and_axis_order(tmp_path):
    h5py = pytest.importorskip("h5py")
    p = tmp_path / "scan.mat"
    rng = np.random.default_rng(1)
    logical = (
        rng.normal(size=(6, 8, 3, 2, 4)) + 1j * rng.normal(size=(6, 8, 3, 2, 4))
    ).astype(np.complex64)
    packed = np.empty(logical.T.shape, dtype=[("real", "<f4"), ("imag", "<f4")])
    packed["real"] = logical.T.real
    packed["imag"] = logical.T.imag
    with h5py.File(p, "w") as f:
        f["SignalFixedPostROFFTPostSR"] = packed
    result = read_mat(p)
    assert result.stage == "coil_image"
    np.testing.assert_array_equal(result.data.numpy(), logical)


def test_mat_v5_complex(tmp_path):
    sio = pytest.importorskip("scipy.io")
    p = tmp_path / "scan.mat"
    data = np.arange(24).reshape(2, 3, 4) + 2j
    sio.savemat(p, {"SignalFixedPostROFFTPostSR": data})
    np.testing.assert_array_equal(read_mat(p).data.numpy(), data)


def test_image_loading_keeps_shape_and_complex_phase(tmp_path):
    p = tmp_path / "image.npy"
    a = (np.arange(30).reshape(5, 6) + 3j).astype(np.complex64)
    np.save(p, a)
    np.testing.assert_array_equal(load_image(p).numpy(), a)


def test_volume_image_requires_explicit_slice(tmp_path):
    p = tmp_path / "image.npy"
    np.save(p, np.ones((8, 9, 5)))
    with pytest.raises(ValueError, match="Select a 2D"):
        load_image(p)
    assert load_image(p, slice_index=2).shape == (8, 9)


def test_protocol_serialization():
    p = XSPENProtocol(acquisition_shape=(60, 64), r_value=60)
    assert protocol_from_dict(p.to_dict()) == p


def test_stage_cannot_be_inferred_from_complex_shape():
    with pytest.raises(ValueError, match="complex"):
        Acquisition(torch.ones(8, 9, 4), ("spen", "readout", "coil"), "uniform_kspace")


def test_simulation_replay_including_b0_kernel(tmp_path, smooth_image):
    from spenpy.io import load_simulation, save_simulation
    from spenpy.sim import simulate

    original = simulate(
        smooth_image((10, 8)),
        XSPENProtocol(acquisition_shape=(8, 6), r_value=5),
        num_coils=3,
        snr_db=25,
        operator_options={"b0_hz": 10.0, "n_z": 32},
        seed=23,
    )
    restored = load_simulation(save_simulation(original, tmp_path / "simulation.npz"))
    torch.testing.assert_close(
        restored.operator(restored.target), original.clean_measurements, rtol=0, atol=0
    )
    torch.testing.assert_close(
        restored.measurements, original.measurements, rtol=0, atol=0
    )
    x = restored.target.clone().requires_grad_()
    restored.operator(x).abs().square().sum().backward()
    assert torch.isfinite(x.grad).all()
