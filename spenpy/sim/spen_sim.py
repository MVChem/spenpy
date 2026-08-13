"""YAML-configurable SPEN forward simulation.

The simulator keeps the original ``spen(...).sim(...)`` return contract but
adds scanner-like randomization hooks that are useful for synthetic training
data.  The defaults are intentionally close to the historical single-shot
path; richer behavior is enabled through a YAML profile.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from spenpy.bruker.param import read_pv_param
from spenpy.core.matrix import calcInvA
from spenpy.fft.transform import fft_kspace_to_xspace, fft_xspace_to_kspace
from spenpy.utils.coil_combine import coil_combine


DEFAULT_SIM_CONFIG: dict[str, Any] = {
    "version": 1,
    "scanner": {
        "L": [4.0, 4.0],
        "acq_point": [256, 256],
        "nseg": 1,
        "chirp_rvalue": 120.0,
        "tblip": 128e-6,
        "gamma_hz": 4257.4,
        "sw_hz": 250000.0,
        "oversample_pe": 16,
        "a_sign": -1,
        "gauss_relative_width": 0.9,
    },
    "randomization": {
        "seed": None,
    },
    "scanner_raw": {
        "num_coils": 4,
        "coil_combine": "adaptive",
        "coil_sensitivity": {
            "enabled": True,
            "radius": 0.65,
            "width": 0.85,
            "floor": 0.05,
            "phase_winding_range_rad": [-0.45, 0.45],
        },
        "receiver": {
            "gain_range": [0.90, 1.10],
            "phase_range_rad": [-math.pi, math.pi],
        },
    },
    "artifacts": {
        "b0": {
            "enabled": False,
            "coef_ranges_cm": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        },
        "shot_phase": {
            "enabled": False,
            "poly_coeff_ranges_rad": [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            "smooth_std_range_rad": [0.0, 0.0],
            "smooth_grid": 6,
        },
        "even_odd": {
            "enabled": True,
            "apply_when_nseg_odd": True,
            "constant_range_rad": [-math.pi, math.pi],
            "linear_range_rad_per_cm": [-math.pi, math.pi],
            "quadratic_range_rad_per_cm2": [0.0, 0.0],
            "object_phase_scale_range_rad": [2 * math.pi, 2 * math.pi],
            "smooth_std_range_rad": [0.0, 0.0],
            "estimate_error_std_rad": 0.0,
        },
        "trajectory": {
            "segment_shift_range_cm": [0.0, 0.0],
            "readout_shift_range_px": [0.0, 0.0],
            "phase_shift_range_px": [0.0, 0.0],
            "line_dropout_probability": 0.0,
            "line_dropout_width": 1,
        },
        "intensity": {
            "gain_range": [1.0, 1.0],
            "bias_field_std_range": [0.0, 0.0],
            "bias_grid": 5,
            "gamma_range": [1.0, 1.0],
        },
        "noise": {
            "complex_std": [0.0, 0.0],
            "relative_to_signal": False,
            "kspace_spike_probability": 0.0,
            "kspace_spike_scale": [0.0, 0.0],
        },
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_sim_config(config: str | Path | dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a simulation config and merge it over ``DEFAULT_SIM_CONFIG``."""
    if config is None:
        return deepcopy(DEFAULT_SIM_CONFIG)
    if isinstance(config, (str, Path)):
        try:
            import yaml
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency declared
            raise ModuleNotFoundError(
                "PyYAML is required to load SPEN simulator YAML configs."
            ) from exc
        with open(config, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Simulation config must be a YAML mapping: {config}")
        return _deep_merge(DEFAULT_SIM_CONFIG, loaded)
    if isinstance(config, dict):
        return _deep_merge(DEFAULT_SIM_CONFIG, config)
    raise TypeError(f"Unsupported simulation config type: {type(config)!r}")


def save_sim_config(config: dict[str, Any], path: str | Path) -> None:
    """Write a simulation config as YAML."""
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency declared
        raise ModuleNotFoundError("PyYAML is required to write simulator YAML configs.") from exc
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def _as_list(value: Any, default: list[Any]) -> list[Any]:
    if value is None:
        return list(default)
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _as_float(value: Any, default: float = 0.0) -> float:
    vals = _as_list(value, [default])
    return float(vals[0]) if vals else default


def _as_int(value: Any, default: int = 1) -> int:
    vals = _as_list(value, [default])
    return int(vals[0]) if vals else default


def _range_tuple(value: Any, default: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float]:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        v = float(value)
        return v, v
    vals = list(value)
    if len(vals) == 0:
        return default
    if len(vals) == 1:
        v = float(vals[0])
        return v, v
    return float(vals[0]), float(vals[1])


def _normalize_abs(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    flat = torch.abs(x).reshape(x.shape[0], -1)
    scale = flat.max(dim=1).values.clamp_min(eps)
    return x / scale.view(-1, *([1] * (x.ndim - 1)))


@dataclass
class SimulatedScannerRaw:
    """Synthetic scanner-like raw sample produced by :meth:`SpenSim.sim_scanner_raw`.

    Tensor layout is batch-first for training.  ``kfield`` intentionally keeps
    Bruker/reconstruction axis order after the batch dimension:
    ``[B, RO, PE, slice, receiver, echo]``.
    """

    object_gt: torch.Tensor
    kfield: torch.Tensor
    raw_rofft_coils: torch.Tensor
    raw_rofft: torch.Tensor
    good_lr_coils: torch.Tensor
    good_lr: torch.Tensor
    phase_map_true: torch.Tensor
    phase_map_estimate: torch.Tensor
    shot_phase_map: torch.Tensor
    inv_a: torch.Tensor
    a_final: torch.Tensor
    coil_sensitivities: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def kfield_for_recon(self, index: int = 0) -> torch.Tensor:
        """Return one sample as ``[RO, PE, slice, receiver, echo]``."""
        return self.kfield[index]

    def kfield_numpy(self, index: int = 0) -> np.ndarray:
        """Return one sample as a NumPy array accepted by ``reconstruct_odd_segments``."""
        return self.kfield_for_recon(index).detach().cpu().resolve_conj().numpy()

    def as_training_dict(self) -> dict[str, Any]:
        """Return the common tensors used by raw-signal training scripts."""
        return {
            "object_gt": self.object_gt,
            "raw_rofft": self.raw_rofft,
            "raw_rofft_coils": self.raw_rofft_coils,
            "kfield": self.kfield,
            "good_lr": self.good_lr,
            "good_lr_coils": self.good_lr_coils,
            "phase_map_true": self.phase_map_true,
            "phase_map_estimate": self.phase_map_estimate,
            "shot_phase_map": self.shot_phase_map,
            "inv_a": self.inv_a,
            "a_final": self.a_final,
            "coil_sensitivities": self.coil_sensitivities,
            "metadata": self.metadata,
        }


class SpenSim:
    """SPEN forward simulator with optional YAML-driven randomization.

    Parameters remain backward-compatible with the original simulator.  Passing
    ``config=...`` or using :meth:`from_yaml` enables richer scanner-like
    artifacts without changing the old ``sim`` return tuple.
    """

    def __init__(
        self,
        L: list[float] | tuple[float, float] | None = None,
        acq_point: list[int] | tuple[int, int] | None = None,
        nseg: int | None = None,
        chirp_rvalue: float | None = None,
        tblip: float | None = None,
        gamma_hz: float | None = None,
        device: str = "cpu",
        noise_level: float | None = None,
        config: str | Path | dict[str, Any] | None = None,
        seed: int | None = None,
    ):
        self.config = load_sim_config(config)
        scanner = self.config["scanner"]

        if L is not None:
            scanner["L"] = [float(v) for v in L]
        if acq_point is not None:
            scanner["acq_point"] = [int(v) for v in acq_point]
        if nseg is not None:
            scanner["nseg"] = int(nseg)
        if chirp_rvalue is not None:
            scanner["chirp_rvalue"] = float(chirp_rvalue)
        if tblip is not None:
            scanner["tblip"] = float(tblip)
        if gamma_hz is not None:
            scanner["gamma_hz"] = float(gamma_hz)
        if noise_level is not None:
            self.config["artifacts"]["noise"]["complex_std"] = [float(noise_level), float(noise_level)]
        if seed is not None:
            self.config["randomization"]["seed"] = int(seed)

        self.L = [float(v) for v in scanner["L"]]
        self.acq_point = [int(v) for v in scanner["acq_point"]]
        self.nseg = int(scanner["nseg"])
        if self.nseg <= 0:
            raise ValueError("nseg must be positive")
        if self.acq_point[1] % self.nseg != 0:
            raise ValueError("acq_point[1] must be divisible by nseg")

        self.chirp_rvalue = float(scanner["chirp_rvalue"])
        self.tblip = float(scanner["tblip"])
        self.gamma_hz = float(scanner["gamma_hz"])
        self.device = device
        self.sw = float(scanner["sw_hz"])
        self.oversample_pe = int(scanner["oversample_pe"])
        self.a_sign = int(scanner["a_sign"])
        self.gauss_relative_width = float(scanner["gauss_relative_width"])
        self.noise_level = float(_range_tuple(self.config["artifacts"]["noise"]["complex_std"])[1])

        self.N = [self.acq_point[0], self.acq_point[1] * self.oversample_pe]
        self.x = torch.linspace(-self.L[0] / 2, self.L[0] / 2, self.N[0], device=device)
        self.y = torch.linspace(-self.L[1] / 2, self.L[1] / 2, self.N[1], device=device)
        self.Ydire_inhomo_coef = torch.zeros(4, device=device)

        self.one_shot_num = self.acq_point[1] / self.nseg
        self.Ta = (self.acq_point[0] / self.sw + self.tblip) * self.one_shot_num
        self.chirp_tp = self.Ta / 2

        self.procpar_struct = {
            "np": self.acq_point[0],
            "ne": 1,
            "nv": self.acq_point[1],
            "nseg": self.nseg,
            "nf": self.acq_point[1],
            "arraydim": 1,
            "Rvol": self.chirp_rvalue,
            "lpe": self.L[0],
            "lro": self.L[1],
            "ppe": 0,
            "Tp": self.chirp_tp,
            "Gchip": self.chirp_rvalue / self.chirp_tp / self.L[1] / self.gamma_hz,
        }

        self.rfwdth = self.procpar_struct["Tp"]
        self.GPEe = self.procpar_struct["Gchip"]
        self.alfa = (
            self.a_sign
            * 2
            * math.pi
            * self.gamma_hz
            * self.GPEe
            * self.rfwdth
            / self.procpar_struct["lpe"]
        )

        seed_value = self.config["randomization"].get("seed")
        self._generator = None
        if seed_value is not None:
            self._generator = torch.Generator(device=device)
            self._generator.manual_seed(int(seed_value))

    @classmethod
    def from_yaml(cls, path: str | Path, **kwargs: Any) -> "SpenSim":
        """Build a simulator from a YAML config file."""
        return cls(config=path, **kwargs)

    @classmethod
    def from_bruker_scan(
        cls,
        scan_dir: str | Path,
        config: str | Path | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "SpenSim":
        """Initialize scanner dimensions from a Bruker SPEN ``method`` file.

        The YAML/default config still controls artifact randomization.  Bruker
        values populate the deterministic sequence geometry when available.
        """
        cfg = load_sim_config(config)
        scanner = cfg["scanner"]
        scan_dir = str(scan_dir)

        matrix = [int(v) for v in _as_list(read_pv_param(scan_dir, "PVM_Matrix"), scanner["acq_point"])]
        while len(matrix) < 2:
            matrix.append(matrix[-1])
        fov_cm = [float(v) for v in _as_list(read_pv_param(scan_dir, "PVM_FovCm"), scanner["L"])]
        while len(fov_cm) < 2:
            fov_cm.append(fov_cm[-1])

        scanner["acq_point"] = matrix[:2]
        scanner["L"] = fov_cm[:2]
        scanner["nseg"] = _as_int(read_pv_param(scan_dir, "NSegments"), scanner["nseg"])

        spen_gy = read_pv_param(scan_dir, "SpenGyGaussStren")
        tp_ms = read_pv_param(scan_dir, "SpatEncDuration")
        if spen_gy is not None and tp_ms is not None:
            tp_s = _as_float(tp_ms) / 1000
            scanner["chirp_rvalue"] = _as_float(spen_gy) * tp_s * scanner["L"][1] * scanner["gamma_hz"]

        echo_spacing_ms = read_pv_param(scan_dir, "PVM_EpiEchoSpacing")
        if echo_spacing_ms is not None:
            scanner["tblip"] = _as_float(echo_spacing_ms) / 1000

        return cls(config=cfg, **kwargs)

    def get_InvA(self):
        """Get the reconstruction operator."""
        inv_a, a_final = calcInvA(
            self.alfa,
            self.L[0],
            self.N[0],
            0,
            self.a_sign,
            0,
            self.gauss_relative_width,
        )
        return inv_a.to(torch.complex64), a_final.to(torch.complex64)

    def sample_config(self) -> dict[str, Any]:
        """Return the effective merged config used by this simulator."""
        return deepcopy(self.config)

    def _rand(self, shape: tuple[int, ...] = ()) -> torch.Tensor:
        return torch.rand(shape, device=self.device, generator=self._generator)

    def _randn(self, shape: tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.randn(shape, device=self.device, dtype=dtype, generator=self._generator)

    def _uniform(self, value: Any) -> float:
        lo, hi = _range_tuple(value)
        if lo == hi:
            return lo
        return (lo + (hi - lo) * self._rand(()).item())

    def _uniform_tensor(self, value: Any, shape: tuple[int, ...]) -> torch.Tensor:
        lo, hi = _range_tuple(value)
        if lo == hi:
            return torch.full(shape, float(lo), device=self.device)
        return lo + (hi - lo) * self._rand(shape)

    def _sample_coeffs(self, ranges: list[Any], n: int) -> torch.Tensor:
        coeffs = [self._uniform(ranges[i] if i < len(ranges) else [0.0, 0.0]) for i in range(n)]
        return torch.tensor(coeffs, device=self.device, dtype=torch.float32)

    def _prepare_input(self, H: torch.Tensor) -> torch.Tensor:
        if H.ndim == 2:
            H = H.unsqueeze(0)
        if H.ndim != 3:
            raise ValueError("H must have shape [B, H, W] or [H, W]")
        H = H.to(self.device)

        def interp(x: torch.Tensor) -> torch.Tensor:
            return F.interpolate(
                x.permute(0, 2, 1).unsqueeze(1),
                size=(self.acq_point[0], self.acq_point[1] * self.oversample_pe),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        if torch.is_complex(H):
            return interp(H.real.float()) + 1j * interp(H.imag.float())
        return interp(H.float())

    def _poly2_phase(self, coeffs: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xg = x.view(-1, 1)
        yg = y.view(1, -1)
        terms = [
            torch.ones_like(xg * yg),
            xg.expand(-1, y.numel()),
            yg.expand(x.numel(), -1),
            xg.square().expand(-1, y.numel()),
            (xg * yg),
            yg.square().expand(x.numel(), -1),
        ]
        return sum(coeffs[i] * terms[i] for i in range(min(len(coeffs), len(terms))))

    def _smooth_random_map(self, batch: int, h: int, w: int, std: float, grid: int) -> torch.Tensor:
        if std == 0:
            return torch.zeros((batch, h, w), device=self.device)
        grid = max(2, int(grid))
        noise = self._randn((batch, 1, grid, grid))
        smooth = F.interpolate(noise, size=(h, w), mode="bicubic", align_corners=False).squeeze(1)
        smooth = smooth - smooth.mean(dim=(1, 2), keepdim=True)
        smooth_std = smooth.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        return smooth / smooth_std * std

    def _apply_intensity_model(self, Hb: torch.Tensor) -> torch.Tensor:
        intensity = self.config["artifacts"]["intensity"]
        mag = torch.abs(Hb) if torch.is_complex(Hb) else Hb.clamp_min(0)

        gamma = self._uniform(intensity.get("gamma_range", [1.0, 1.0]))
        if gamma != 1.0:
            mag_max = mag.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
            mag = (mag / mag_max).clamp_min(0).pow(gamma) * mag_max
            if torch.is_complex(Hb):
                Hb = mag * torch.exp(1j * torch.angle(Hb))
            else:
                Hb = mag

        bias_std = self._uniform(intensity.get("bias_field_std_range", [0.0, 0.0]))
        if bias_std:
            bias = self._smooth_random_map(
                Hb.shape[0],
                Hb.shape[1],
                Hb.shape[2],
                bias_std,
                intensity.get("bias_grid", 5),
            )
            Hb = Hb * torch.exp(bias)

        gain = self._uniform(intensity.get("gain_range", [1.0, 1.0]))
        return Hb * gain

    def _b0_coeffs(self) -> torch.Tensor:
        b0 = self.config["artifacts"]["b0"]
        if not b0.get("enabled", False):
            return self.Ydire_inhomo_coef
        return self._sample_coeffs(b0.get("coef_ranges_cm", []), 4)

    def _encode_clean_lr(
        self,
        Hb: torch.Tensor,
        phase_batch: int | None = None,
        repeats_per_phase: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        batch = Hb.shape[0]
        phase_batch = batch if phase_batch is None else int(phase_batch)
        repeats_per_phase = max(1, int(repeats_per_phase))
        if phase_batch * repeats_per_phase != batch:
            raise ValueError(
                "phase_batch * repeats_per_phase must equal the encoded batch size"
            )
        n_pe_per_seg = self.acq_point[1] // self.nseg
        encoded = torch.zeros(
            (batch, self.acq_point[1], self.acq_point[0]),
            dtype=torch.complex64,
            device=self.device,
        )
        good_lr = torch.zeros_like(encoded)
        shot_phase = torch.zeros(
            (batch, self.nseg, self.acq_point[0], n_pe_per_seg),
            dtype=torch.float32,
            device=self.device,
        )
        sampled: dict[str, Any] = {"segment_shift_cm": [], "b0_coeffs_cm": [], "shot_phase_coeffs_rad": []}

        traj = self.config["artifacts"]["trajectory"]
        shot = self.config["artifacts"]["shot_phase"]

        for k in range(self.nseg):
            seg_shift = self._uniform(traj.get("segment_shift_range_cm", [0.0, 0.0]))
            sampled["segment_shift_cm"].append(seg_shift)

            start = -self.L[1] / 2 + k * self.L[1] / self.acq_point[1] + seg_shift
            step = self.L[1] / n_pe_per_seg
            temp_yacq = start + torch.arange(n_pe_per_seg, device=self.device) * step

            coeff = self._b0_coeffs()
            sampled["b0_coeffs_cm"].append([float(v) for v in coeff.detach().cpu()])
            b0y = sum(coeff[i] * self.y**i for i in range(4))
            b0acq = sum(coeff[i] * temp_yacq**i for i in range(4))

            y_grid, temp_yacq_grid = torch.meshgrid(self.y, temp_yacq, indexing="ij")
            b0y_grid, b0acq_grid = torch.meshgrid(b0y, b0acq, indexing="ij")
            part1 = (y_grid + b0y_grid) - (temp_yacq_grid + b0acq_grid)
            part2 = temp_yacq_grid + b0acq_grid
            exp_term = torch.exp(1j * self.alfa * (part1.square() - part2.square())).to(torch.complex64)

            acquired = torch.matmul(Hb.to(torch.complex64), exp_term)
            good_lr[:, k::self.nseg, :] = acquired.permute(0, 2, 1)

            if shot.get("enabled", False):
                coeffs = self._sample_coeffs(shot.get("poly_coeff_ranges_rad", []), 6)
                smooth_std = self._uniform(shot.get("smooth_std_range_rad", [0.0, 0.0]))
                phase = self._poly2_phase(coeffs, self.x, temp_yacq).unsqueeze(0)
                smooth_phase = self._smooth_random_map(
                    phase_batch,
                    self.acq_point[0],
                    n_pe_per_seg,
                    smooth_std,
                    shot.get("smooth_grid", 6),
                )
                if repeats_per_phase > 1:
                    smooth_phase = smooth_phase.repeat_interleave(repeats_per_phase, dim=0)
                phase = phase + smooth_phase
                acquired = acquired * torch.exp(1j * phase)
                shot_phase[:, k, :, :] = phase
                sampled["shot_phase_coeffs_rad"].append([float(v) for v in coeffs.detach().cpu()])
            else:
                sampled["shot_phase_coeffs_rad"].append([0.0] * 6)

            encoded[:, k::self.nseg, :] = acquired.permute(0, 2, 1)

        return encoded, good_lr, {"shot_phase_map": shot_phase, "sampled": sampled}

    def _build_coil_sensitivities(
        self,
        batch: int,
        num_coils: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        raw_cfg = self.config.get("scanner_raw", {})
        coil_cfg = raw_cfg.get("coil_sensitivity", {})
        if num_coils <= 1 or not coil_cfg.get("enabled", True):
            return torch.ones(
                (batch, max(1, num_coils), height, width),
                dtype=torch.complex64,
                device=self.device,
            )

        x = torch.linspace(-1.0, 1.0, height, device=self.device).view(1, height, 1)
        y = torch.linspace(-1.0, 1.0, width, device=self.device).view(1, 1, width)
        angles = torch.linspace(
            0.0,
            2.0 * torch.pi * (num_coils - 1) / num_coils,
            num_coils,
            device=self.device,
        ).view(num_coils, 1, 1)
        radius = float(coil_cfg.get("radius", 0.65))
        width_scale = max(float(coil_cfg.get("width", 0.85)), 1e-3)
        floor = float(coil_cfg.get("floor", 0.05))
        cx = radius * torch.cos(angles)
        cy = radius * torch.sin(angles)

        dist2 = (x - cx).square() + (y - cy).square()
        mag = torch.exp(-dist2 / (2.0 * width_scale * width_scale)) + floor
        mag = mag / torch.sqrt(mag.square().sum(dim=0, keepdim=True).clamp_min(1e-8))

        winding = self._uniform(coil_cfg.get("phase_winding_range_rad", [-0.45, 0.45]))
        phase = angles + winding * (x * cy - y * cx)
        sens = mag.to(torch.complex64) * torch.exp(1j * phase.to(torch.complex64))
        return sens.unsqueeze(0).expand(batch, -1, -1, -1).clone()

    def _apply_receiver_model(self, coil_data: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raw_cfg = self.config.get("scanner_raw", {})
        receiver = raw_cfg.get("receiver", {})
        batch, _pe, _ro, num_coils = coil_data.shape
        gain = self._uniform_tensor(receiver.get("gain_range", [1.0, 1.0]), (batch, 1, 1, num_coils))
        phase = self._uniform_tensor(
            receiver.get("phase_range_rad", [0.0, 0.0]),
            (batch, 1, 1, num_coils),
        )
        factor = gain.to(torch.complex64) * torch.exp(1j * phase.to(torch.complex64))
        return coil_data * factor, {"gain": gain.detach().cpu(), "phase_rad": phase.detach().cpu()}

    def _coil_combine_batch(self, coil_data: torch.Tensor) -> torch.Tensor:
        """Combine ``[B, PE, RO, receiver]`` data using the configured method."""
        raw_cfg = self.config.get("scanner_raw", {})
        mode = str(raw_cfg.get("coil_combine", "adaptive")).lower()
        if coil_data.shape[-1] == 1:
            return coil_data[..., 0]
        if mode == "adaptive":
            combined = [coil_combine(coil_data[idx]) for idx in range(coil_data.shape[0])]
            return torch.stack(combined, dim=0)
        if mode in {"rss", "sos"}:
            return torch.sqrt(coil_data.abs().square().sum(dim=-1)).to(torch.complex64)
        if mode == "first":
            return coil_data[..., 0]
        raise ValueError(f"Unsupported scanner_raw.coil_combine mode: {mode}")

    def _scanner_recon_matrices(self) -> tuple[torch.Tensor, torch.Tensor]:
        inv_a, a_final = calcInvA(
            self.alfa,
            self.L[0],
            self.acq_point[1],
            0,
            -self.a_sign,
            0,
            self.gauss_relative_width,
        )
        return inv_a.to(self.device).to(torch.complex64), a_final.to(self.device).to(torch.complex64)

    def simulated_bruker_params(self, num_coils: int | None = None) -> dict[str, Any]:
        """Return minimal Bruker-like params for reconstructing simulated k-space."""
        if num_coils is None:
            num_coils = int(self.config.get("scanner_raw", {}).get("num_coils", 1))
        return {
            "PVM_Matrix": [int(self.acq_point[0]), int(self.acq_point[1])],
            "PVM_Fov": [float(self.L[1]) * 10.0, float(self.L[0]) * 10.0],
            "PVM_SPackArrNSlices": [1],
            "PVM_ObjOrderList": [0],
            "PVM_EncNReceivers": int(num_coils),
            "PVM_NEchoImages": 1,
            "NSegments": int(self.nseg),
            "SpenGyGaussStren": float(self.GPEe),
            "SpatEncDuration": float(self.rfwdth) * 1000.0,
            "PVM_SPackArrPhase1Offset": 0.0,
        }

    def _object_low_frequency_map(self, H: torch.Tensor, h: int, w: int) -> torch.Tensor:
        base = torch.abs(H.to(self.device))
        if base.ndim == 2:
            base = base.unsqueeze(0)
        base = F.interpolate(base.float().unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False).squeeze(1)
        blur_h = max(3, min(31, (h // 8) * 2 + 1))
        blur_w = max(3, min(31, (w // 8) * 2 + 1))
        base = F.avg_pool2d(base.unsqueeze(1), kernel_size=(blur_h, blur_w), stride=1, padding=(blur_h // 2, blur_w // 2)).squeeze(1)
        base = base - base.amin(dim=(1, 2), keepdim=True)
        base = base / base.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        return base

    def _build_even_odd_phase(self, H: torch.Tensor, target_h: int, target_w: int) -> tuple[torch.Tensor, torch.Tensor]:
        even_odd = self.config["artifacts"]["even_odd"]
        batch = H.shape[0] if H.ndim == 3 else 1
        if not even_odd.get("enabled", True):
            z = torch.zeros((batch, target_h, target_w), device=self.device)
            return z, z
        if even_odd.get("apply_when_nseg_odd", True) and self.nseg % 2 == 0:
            z = torch.zeros((batch, target_h, target_w), device=self.device)
            return z, z

        x = torch.linspace(-self.L[0] / 2, self.L[0] / 2, target_w, device=self.device)
        constant = self._uniform(even_odd.get("constant_range_rad", [0.0, 0.0]))
        linear = self._uniform(even_odd.get("linear_range_rad_per_cm", [0.0, 0.0]))
        quadratic = self._uniform(even_odd.get("quadratic_range_rad_per_cm2", [0.0, 0.0]))
        phase = constant + linear * x + quadratic * x.square()
        phase = phase.view(1, 1, target_w).expand(batch, target_h, target_w).clone()

        object_scale = self._uniform(even_odd.get("object_phase_scale_range_rad", [0.0, 0.0]))
        if object_scale:
            phase = phase + object_scale * self._object_low_frequency_map(H, target_h, target_w)

        smooth_std = self._uniform(even_odd.get("smooth_std_range_rad", [0.0, 0.0]))
        if smooth_std:
            phase = phase + self._smooth_random_map(batch, target_h, target_w, smooth_std, 6)

        estimate = phase.clone()
        err_std = float(even_odd.get("estimate_error_std_rad", 0.0) or 0.0)
        if err_std:
            estimate = estimate + self._randn(tuple(estimate.shape)) * err_std
        return phase.float(), estimate.float()

    def _fourier_shift(self, x: torch.Tensor, shift_px: float, dim: int) -> torch.Tensor:
        if shift_px == 0:
            return x
        n = x.shape[dim]
        freq = torch.fft.fftfreq(n, device=x.device)
        shape = [1] * x.ndim
        shape[dim] = n
        ramp = torch.exp(-2j * torch.pi * shift_px * freq.reshape(shape))
        k = torch.fft.fft(x, dim=dim)
        return torch.fft.ifft(k * ramp, dim=dim).to(torch.complex64)

    def _apply_trajectory_model(self, encoded: torch.Tensor) -> torch.Tensor:
        traj = self.config["artifacts"]["trajectory"]
        read_shift = self._uniform(traj.get("readout_shift_range_px", [0.0, 0.0]))
        phase_shift = self._uniform(traj.get("phase_shift_range_px", [0.0, 0.0]))
        encoded = self._fourier_shift(encoded, read_shift, dim=2)
        encoded = self._fourier_shift(encoded, phase_shift, dim=1)
        return encoded

    def _add_noise_and_kspace_events(self, kspace: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        noise = self.config["artifacts"]["noise"]
        std = self._uniform(noise.get("complex_std", [0.0, 0.0]))
        if noise.get("relative_to_signal", False):
            signal_scale = torch.abs(kspace).reshape(kspace.shape[0], -1).std(dim=1).clamp_min(1e-8)
            std_tensor = signal_scale.view(-1, 1, 1) * std
        else:
            std_tensor = torch.as_tensor(std, device=self.device)
        if std:
            real = self._randn(tuple(kspace.shape))
            imag = self._randn(tuple(kspace.shape))
            kspace = kspace + (real + 1j * imag).to(torch.complex64) * std_tensor

        spike_probability = float(noise.get("kspace_spike_probability", 0.0) or 0.0)
        spikes = 0
        if spike_probability and self._rand(()).item() < spike_probability:
            scale = self._uniform(noise.get("kspace_spike_scale", [0.0, 0.0]))
            batch_idx = torch.arange(kspace.shape[0], device=self.device)
            pe_idx = torch.randint(0, kspace.shape[1], (kspace.shape[0],), device=self.device, generator=self._generator)
            ro_idx = torch.randint(0, kspace.shape[2], (kspace.shape[0],), device=self.device, generator=self._generator)
            phase = torch.exp(1j * 2 * torch.pi * self._rand((kspace.shape[0],)))
            kspace[batch_idx, pe_idx, ro_idx] += scale * phase
            spikes = int(kspace.shape[0])

        traj = self.config["artifacts"]["trajectory"]
        dropout_probability = float(traj.get("line_dropout_probability", 0.0) or 0.0)
        dropped_lines = 0
        if dropout_probability and self._rand(()).item() < dropout_probability:
            width = max(1, int(traj.get("line_dropout_width", 1)))
            center = int(torch.randint(0, kspace.shape[1], (), device=self.device, generator=self._generator).item())
            lo = max(0, center - width // 2)
            hi = min(kspace.shape[1], lo + width)
            kspace[:, lo:hi, :] = 0
            dropped_lines = hi - lo

        return kspace, {"noise_std": float(std), "spikes": spikes, "dropped_lines": dropped_lines}

    @torch.no_grad()
    def get_phase_map(self, H: torch.Tensor, noise_level: float = 0.0) -> torch.Tensor:
        """Generate an even/odd phase-map estimate for ``H``.

        ``noise_level`` is an additional estimate perturbation in radians,
        kept for compatibility with the historical API.
        """
        phase_true, phase_est = self._build_even_odd_phase(H, self.acq_point[1] // 2, self.acq_point[0])
        if noise_level:
            phase_est = phase_est + self._randn(tuple(phase_est.shape)) * float(noise_level)
        return phase_est

    @torch.no_grad()
    def sim(
        self,
        H: torch.Tensor,
        return_phase_map: bool = False,
        return_good_lr_image: bool = False,
        return_metadata: bool = False,
    ):
        """Forward SPEN simulation.

        Args:
            H: input image ``[B, H, W]`` or ``[H, W]``.
            return_phase_map: include the even-line phase estimate.
            return_good_lr_image: include the clean low-resolution SPEN image.
            return_metadata: include truth maps and sampled artifact values.
        """
        if H.ndim == 2:
            H = H.unsqueeze(0)
        H = H.to(self.device)
        Hb = self._prepare_input(H)
        Hb = self._apply_intensity_model(Hb)

        encoded, good_lr_image, encode_meta = self._encode_clean_lr(Hb)
        good_lr_image = _normalize_abs(good_lr_image)

        encoded = self._apply_trajectory_model(encoded)
        phase_true, phase_estimate = self._build_even_odd_phase(H, encoded.shape[1] // 2, encoded.shape[2])
        if phase_true.numel():
            encoded[:, 1::2, :] = encoded[:, 1::2, :] * torch.exp(1j * phase_true)

        kspace = fft_xspace_to_kspace(encoded, dim=1)
        kspace = _normalize_abs(kspace)
        kspace, noise_meta = self._add_noise_and_kspace_events(kspace)
        final_rxyacq_rofft = fft_kspace_to_xspace(kspace, dim=1)

        metadata = {
            "config": self.sample_config(),
            "sampled": encode_meta["sampled"],
            "noise": noise_meta,
            "phase_map_true": phase_true,
            "phase_map_estimate": phase_estimate,
            "shot_phase_map": encode_meta["shot_phase_map"],
        }

        outputs: list[Any] = [final_rxyacq_rofft]
        if return_phase_map:
            outputs.append(phase_estimate)
        if return_good_lr_image:
            outputs.append(good_lr_image)
        if return_metadata:
            outputs.append(metadata)
        return tuple(outputs) if len(outputs) > 1 else outputs[0]

    @torch.no_grad()
    def sim_scanner_raw(
        self,
        H: torch.Tensor,
        num_coils: int | None = None,
    ) -> SimulatedScannerRaw:
        """Simulate a scanner-like multi-coil raw SPEN acquisition.

        This path starts from the same object model as :meth:`sim`, but keeps
        receiver channels separate and exports Bruker/reconstruction-shaped
        k-space.  The returned ``kfield`` can be passed to
        ``spenpy.recon.spen_recon.reconstruct_odd_segments(..., kfield=...)``
        with matching parameters from :meth:`simulated_bruker_params`.
        """
        if H.ndim == 2:
            H = H.unsqueeze(0)
        if H.ndim != 3:
            raise ValueError("H must have shape [B, H, W] or [H, W]")
        H = H.to(self.device)
        batch = H.shape[0]

        raw_cfg = self.config.get("scanner_raw", {})
        if num_coils is None:
            num_coils = int(raw_cfg.get("num_coils", 1))
        num_coils = max(1, int(num_coils))

        Hb = self._prepare_input(H)
        Hb = self._apply_intensity_model(Hb)
        coil_sens = self._build_coil_sensitivities(
            batch,
            num_coils,
            Hb.shape[1],
            Hb.shape[2],
        )
        hb_coils = Hb.to(torch.complex64).unsqueeze(1) * coil_sens
        hb_flat = hb_coils.reshape(batch * num_coils, Hb.shape[1], Hb.shape[2])

        encoded_flat, good_flat, encode_meta = self._encode_clean_lr(
            hb_flat,
            phase_batch=batch,
            repeats_per_phase=num_coils,
        )
        pe, ro = self.acq_point[1], self.acq_point[0]
        encoded_coils = (
            encoded_flat.reshape(batch, num_coils, pe, ro)
            .permute(0, 2, 3, 1)
            .contiguous()
        )
        good_lr_coils = (
            good_flat.reshape(batch, num_coils, pe, ro)
            .permute(0, 2, 3, 1)
            .contiguous()
        )

        encoded_coils, receiver_meta = self._apply_receiver_model(encoded_coils)
        receiver_gain = receiver_meta["gain"].to(self.device)
        receiver_phase = receiver_meta["phase_rad"].to(self.device)
        receiver_factor = receiver_gain.to(torch.complex64) * torch.exp(
            1j * receiver_phase.to(torch.complex64)
        )
        good_lr_coils = good_lr_coils * receiver_factor

        encoded_flat = encoded_coils.permute(0, 3, 1, 2).reshape(batch * num_coils, pe, ro)
        encoded_flat = self._apply_trajectory_model(encoded_flat)
        encoded_coils = (
            encoded_flat.reshape(batch, num_coils, pe, ro)
            .permute(0, 2, 3, 1)
            .contiguous()
        )

        phase_true, phase_estimate = self._build_even_odd_phase(H, pe // 2, ro)
        if phase_true.numel():
            encoded_coils[:, 1::2, :, :] = encoded_coils[:, 1::2, :, :] * torch.exp(
                1j * phase_true.unsqueeze(-1).to(torch.complex64)
            )

        pe_kspace_flat = fft_xspace_to_kspace(
            encoded_coils.permute(0, 3, 1, 2).reshape(batch * num_coils, pe, ro),
            dim=1,
        )
        pe_kspace_flat, noise_meta = self._add_noise_and_kspace_events(pe_kspace_flat)
        raw_rofft_flat = fft_kspace_to_xspace(pe_kspace_flat, dim=1)
        raw_rofft_coils = (
            raw_rofft_flat.reshape(batch, num_coils, pe, ro)
            .permute(0, 2, 3, 1)
            .contiguous()
        )

        raw_rofft = self._coil_combine_batch(raw_rofft_coils)
        good_lr = self._coil_combine_batch(good_lr_coils)

        ro_kspace = fft_xspace_to_kspace(raw_rofft_coils, dim=2)
        kfield = (
            ro_kspace.permute(0, 2, 1, 3)
            .unsqueeze(3)
            .unsqueeze(-1)
            .contiguous()
        )

        shot_phase = encode_meta["shot_phase_map"].reshape(
            batch,
            num_coils,
            self.nseg,
            self.acq_point[0],
            self.acq_point[1] // self.nseg,
        )[:, 0]
        inv_a, a_final = self._scanner_recon_matrices()
        metadata = {
            "config": self.sample_config(),
            "sampled": encode_meta["sampled"],
            "noise": noise_meta,
            "receiver": receiver_meta,
            "bruker_params": self.simulated_bruker_params(num_coils=num_coils),
            "kfield_layout": "[B, RO, PE, slice, receiver, echo]",
            "raw_rofft_coils_layout": "[B, PE, RO, receiver]",
        }

        return SimulatedScannerRaw(
            object_gt=H.detach().clone(),
            kfield=kfield.to(torch.complex64),
            raw_rofft_coils=raw_rofft_coils.to(torch.complex64),
            raw_rofft=raw_rofft.to(torch.complex64),
            good_lr_coils=good_lr_coils.to(torch.complex64),
            good_lr=good_lr.to(torch.complex64),
            phase_map_true=phase_true.float(),
            phase_map_estimate=phase_estimate.float(),
            shot_phase_map=shot_phase.float(),
            inv_a=inv_a,
            a_final=a_final,
            coil_sensitivities=coil_sens.to(torch.complex64),
            metadata=metadata,
        )

    def sim_raw(self, H: torch.Tensor, num_coils: int | None = None) -> SimulatedScannerRaw:
        """Alias for :meth:`sim_scanner_raw`."""
        return self.sim_scanner_raw(H, num_coils=num_coils)


__all__ = [
    "DEFAULT_SIM_CONFIG",
    "SimulatedScannerRaw",
    "SpenSim",
    "load_sim_config",
    "save_sim_config",
]
