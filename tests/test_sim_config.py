"""Tests for the YAML-configurable SPEN simulator."""

from pathlib import Path

import torch
import pytest

from spenpy.recon.spen_recon import reconstruct_odd_segments
from spenpy.sim import SimulatedScannerRaw, SpenSim, load_sim_config
from spenpy.spen import spen


def test_sim_config_dict_preserves_legacy_return_shapes():
    cfg = {
        "scanner": {"L": [2.0, 2.0], "acq_point": [32, 32], "oversample_pe": 4},
        "randomization": {"seed": 123},
        "artifacts": {
            "even_odd": {
                "constant_range_rad": [0.1, 0.1],
                "linear_range_rad_per_cm": [0.0, 0.0],
                "object_phase_scale_range_rad": [0.0, 0.0],
            },
            "noise": {"complex_std": [0.0, 0.0]},
        },
    }
    sim = SpenSim(config=cfg)
    H = torch.rand(2, 32, 32)

    corrupted, phase_map, good_lr = sim.sim(
        H,
        return_phase_map=True,
        return_good_lr_image=True,
    )

    assert corrupted.shape == (2, 32, 32)
    assert corrupted.dtype == torch.complex64
    assert phase_map.shape == (2, 16, 32)
    assert phase_map.dtype == torch.float32
    assert good_lr.shape == (2, 32, 32)
    assert good_lr.dtype == torch.complex64


def test_sim_can_load_yaml_and_return_metadata(tmp_path: Path):
    cfg_path = tmp_path / "sim.yaml"
    cfg_path.write_text(
        """
scanner:
  L: [2.0, 2.0]
  acq_point: [24, 24]
  oversample_pe: 4
randomization:
  seed: 5
artifacts:
  b0:
    enabled: true
    coef_ranges_cm:
      - [0.001, 0.001]
      - [0.0, 0.0]
      - [0.0, 0.0]
      - [0.0, 0.0]
  shot_phase:
    enabled: true
    poly_coeff_ranges_rad:
      - [0.2, 0.2]
      - [0.0, 0.0]
      - [0.0, 0.0]
      - [0.0, 0.0]
      - [0.0, 0.0]
      - [0.0, 0.0]
  noise:
    complex_std: [0.0, 0.0]
""",
        encoding="utf-8",
    )

    sim = SpenSim.from_yaml(cfg_path)
    out, phase_map, meta = sim.sim(torch.rand(1, 24, 24), return_phase_map=True, return_metadata=True)

    assert out.shape == (1, 24, 24)
    assert phase_map.shape == (1, 12, 24)
    assert meta["phase_map_true"].shape == (1, 12, 24)
    assert meta["shot_phase_map"].shape == (1, 1, 24, 24)
    assert meta["sampled"]["b0_coeffs_cm"][0][0] == pytest.approx(0.001)


def test_legacy_spen_name_accepts_config():
    sim = spen(
        L=[2.0, 2.0],
        acq_point=[24, 24],
        config={"scanner": {"oversample_pe": 4}},
        seed=1,
    )
    out = sim.sim(torch.rand(1, 24, 24))
    assert out.shape == (1, 24, 24)


def test_packaged_scanner_like_config_loads():
    cfg = load_sim_config(Path("spenpy/configs/scanner_like.yaml"))
    assert cfg["scanner"]["acq_point"] == [96, 96]
    assert cfg["artifacts"]["b0"]["enabled"] is True


def test_sim_scanner_raw_returns_multicoil_recon_ready_kfield():
    cfg = {
        "scanner": {
            "L": [2.0, 2.0],
            "acq_point": [16, 16],
            "oversample_pe": 4,
            "gauss_relative_width": 0.8,
        },
        "scanner_raw": {
            "num_coils": 2,
            "coil_sensitivity": {"phase_winding_range_rad": [0.0, 0.0]},
            "receiver": {
                "gain_range": [1.0, 1.0],
                "phase_range_rad": [0.0, 0.0],
            },
        },
        "randomization": {"seed": 7},
        "artifacts": {
            "even_odd": {"enabled": False},
            "noise": {"complex_std": [0.0, 0.0]},
        },
    }
    sim = SpenSim(config=cfg)
    sample = sim.sim_scanner_raw(torch.rand(2, 16, 16))

    assert isinstance(sample, SimulatedScannerRaw)
    assert sample.kfield.shape == (2, 16, 16, 1, 2, 1)
    assert sample.raw_rofft_coils.shape == (2, 16, 16, 2)
    assert sample.raw_rofft.shape == (2, 16, 16)
    assert sample.good_lr_coils.shape == (2, 16, 16, 2)
    assert sample.good_lr.shape == (2, 16, 16)
    assert sample.phase_map_true.shape == (2, 8, 16)
    assert sample.inv_a.shape == (16, 16)
    assert sample.a_final.shape == (16, 16)
    assert sample.metadata["bruker_params"]["PVM_EncNReceivers"] == 2
    assert sample.kfield_for_recon(0).shape == (16, 16, 1, 2, 1)


def _write_simulated_bruker_stub(scan_dir: Path, params: dict[str, object]) -> None:
    scan_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for key, value in params.items():
        if isinstance(value, (list, tuple)):
            flat = " ".join(str(v) for v in value)
            lines.append(f"##${key}=( {len(value)} )\n{flat}")
        else:
            lines.append(f"##${key}={value}")
    (scan_dir / "method").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (scan_dir / "acqp").write_text("", encoding="utf-8")


def test_sim_scanner_raw_kfield_round_trips_through_reconstruction(tmp_path: Path):
    cfg = {
        "scanner": {
            "L": [2.0, 2.0],
            "acq_point": [12, 12],
            "oversample_pe": 4,
            "gauss_relative_width": 0.8,
        },
        "scanner_raw": {
            "num_coils": 2,
            "coil_sensitivity": {"phase_winding_range_rad": [0.0, 0.0]},
            "receiver": {
                "gain_range": [1.0, 1.0],
                "phase_range_rad": [0.0, 0.0],
            },
        },
        "randomization": {"seed": 11},
        "artifacts": {
            "even_odd": {"enabled": False},
            "noise": {"complex_std": [0.0, 0.0]},
        },
    }
    sim = SpenSim(config=cfg)
    sample = sim.sim_scanner_raw(torch.rand(1, 12, 12))
    scan_dir = tmp_path / "scan"
    _write_simulated_bruker_stub(scan_dir, sample.metadata["bruker_params"])

    recon = reconstruct_odd_segments(
        str(scan_dir),
        kfield=sample.kfield_for_recon(0),
        process_with_pre_phase_corr=False,
    )

    assert recon.kfield.shape == (12, 12, 1, 2, 1)
    assert recon.imag_origin.shape == sample.raw_rofft[0].shape
    assert torch.allclose(
        recon.imag_origin.abs().to(torch.float32),
        sample.raw_rofft[0].abs().to(torch.float32),
        rtol=1e-4,
        atol=1e-4,
    )
