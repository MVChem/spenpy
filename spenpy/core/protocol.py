"""Explicit sequence parameters. Spatial units are metres; times are seconds."""

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SPEN180Protocol:
    """Single-shot SPEN with a 180-degree encoding chirp.

    Both shapes and FOVs use (SPEN/y, readout/x) order. R is the chirp
    time-bandwidth product. The coefficient is sign * 2*pi*R/FOV_y**2.
    """

    acquisition_shape: tuple[int, int] = (96, 96)
    fov_m: tuple[float, float] = (0.016, 0.016)
    r_value: float = 150.0
    echo_spacing_s: float = 0.0004608
    encoding_sign: int = -1
    focus_offset_voxels: float = 0.0

    def __post_init__(self):
        _validate(self)
        if self.encoding_sign not in (-1, 1):
            raise ValueError("encoding_sign must be -1 or +1")
        if not math.isfinite(self.focus_offset_voxels):
            raise ValueError("focus_offset_voxels must be finite")

    @property
    def sequence(self):
        return "spen"

    @property
    def phase_coefficient_rad_m2(self):
        return self.encoding_sign * 2 * math.pi * self.r_value / self.fov_m[0] ** 2

    def to_dict(self):
        return dict(sequence=self.sequence, **asdict(self))


@dataclass(frozen=True)
class XSPENProtocol:
    """Ideal crossed-chirp xSPEN, following local xSPEN1D.m.

    Tp=Ta/(4*beta), BW=R/Tp. A uniform auxiliary slice can be integrated
    analytically. This parameterization describes this sequence family;
    a Siemens sequence name alone does not establish model equivalence.
    """

    acquisition_shape: tuple[int, int] = (64, 64)
    fov_m: tuple[float, float] = (0.04, 0.04)
    r_value: float = 64.0
    echo_spacing_s: float = 0.000336
    slice_thickness_m: float = 0.008
    beta: float = 0.5
    gamma_hz_t: float = 42_577_478.92

    def __post_init__(self):
        _validate(self)
        if not math.isfinite(self.beta) or not 0 < self.beta < 1:
            raise ValueError("beta must be between zero and one")
        if not math.isfinite(self.slice_thickness_m) or self.slice_thickness_m <= 0:
            raise ValueError("slice_thickness_m must be positive")
        if not math.isfinite(self.gamma_hz_t) or self.gamma_hz_t <= 0:
            raise ValueError("gamma_hz_t must be positive")

    @property
    def sequence(self):
        return "xspen"

    @property
    def acquisition_time_s(self):
        return self.acquisition_shape[0] * self.echo_spacing_s

    @property
    def chirp_duration_s(self):
        return self.acquisition_time_s / (4 * self.beta)

    @property
    def chirp_bandwidth_hz(self):
        return self.r_value / self.chirp_duration_s

    @property
    def gy_t_m(self):
        return self.beta * self.chirp_bandwidth_hz / (self.gamma_hz_t * self.fov_m[0])

    @property
    def gz_t_m(self):
        return (
            (1 - self.beta)
            * self.chirp_bandwidth_hz
            / (self.gamma_hz_t * self.slice_thickness_m)
        )

    @property
    def cross_term_cycles_m2(self):
        return (
            -4
            * self.chirp_duration_s
            / self.chirp_bandwidth_hz
            * self.gamma_hz_t
            * self.gy_t_m
            * self.gamma_hz_t
            * self.gz_t_m
        )

    def to_dict(self):
        return dict(sequence=self.sequence, **asdict(self))


@dataclass(frozen=True)
class HybridSPENDiff2Protocol:
    """Reduced signal model for Siemens esrs_hyb_spen_Diff2.

    Defaults come from MID253, not from its xSPEN directory/figure title.
    PE uses the historical quadratic CalcSRMatrixApprox voxel integral;
    RO uses the centred discrete Fourier convention of FFTXSpace2KSpace.
    RF durations, slice thickness and TE describe the acquisition. They do
    not imply a Bloch simulation or automatic diffusion/B0 calibration.
    """

    acquisition_shape: tuple[int, int] = (60, 64)
    fov_m: tuple[float, float] = (0.180, 0.192)
    r_value: float = 60.0
    echo_spacing_s: float = 470e-6
    chirp_duration_s: float = 14.080e-3
    ro_refocusing_duration_s: float = 4.096e-3
    ro_refocusing_r_value: float = 20.0
    slice_thickness_m: float = 3e-3
    echo_time_s: float = 52e-3
    nominal_b_value_s_mm2: float = 600.0
    spen_shift_pixels: float = 0.0

    def __post_init__(self):
        _validate(self)
        for name in (
            "chirp_duration_s",
            "ro_refocusing_duration_s",
            "slice_thickness_m",
            "echo_time_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("ro_refocusing_r_value", "nominal_b_value_s_mm2"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        if not math.isfinite(self.spen_shift_pixels):
            raise ValueError("spen_shift_pixels must be finite")

    @property
    def sequence(self):
        return "hybrid_spen_diff2"

    @property
    def sequence_name(self):
        return "esrs_hyb_spen_Diff2"

    @property
    def phase_coefficient_rad_m2(self):
        return 2 * math.pi * self.r_value / self.fov_m[0] ** 2

    def to_dict(self):
        return dict(sequence=self.sequence, **asdict(self))

    @classmethod
    def from_siemens_header(cls, yaps):
        """Read this sequence's explicit WIP mapping from a parsed MeasYaps.

        WIP indices follow esrs_hyb_spen_Diff2.cpp, eSeqSpecialParameters.
        The protocol name (e.g. Ortho900) is not used to infer b-values.
        """
        if "esrs_hyb_spen_Diff2" not in str(yaps.get("tSequenceFileName", "")):
            raise ValueError("Expected esrs_hyb_spen_Diff2 MeasYaps")
        try:
            free = yaps["sWiPMemBlock"]["alFree"]
            doubles = yaps["sWiPMemBlock"]["adFree"]
            sl = yaps["sSliceArray"]["asSlice"][0]
            space = yaps["sKSpace"]
            return cls(
                acquisition_shape=(
                    int(space["lPhaseEncodingLines"]),
                    int(space["lBaseResolution"]),
                ),
                fov_m=(float(sl["dPhaseFOV"]) / 1000, float(sl["dReadoutFOV"]) / 1000),
                r_value=float(free[12]),
                echo_spacing_s=float(yaps["sFastImaging"]["lEchoSpacing"]) * 1e-6,
                chirp_duration_s=float(free[13]) * 1e-6,
                ro_refocusing_r_value=float(free[14]),
                ro_refocusing_duration_s=float(free[15]) * 1e-6,
                slice_thickness_m=float(sl["dThickness"]) / 1000,
                echo_time_s=float(yaps["alTE"][0]) * 1e-6,
                nominal_b_value_s_mm2=float(doubles[0]),
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                "Incomplete Hybrid SPEN-Diff2 acquisition parameters"
            ) from exc


# Public compatibility name for earlier SPENPy releases.
SPENProtocol = SPEN180Protocol


def _validate(p):
    if len(p.acquisition_shape) != 2 or any(
        not isinstance(v, int) or isinstance(v, bool) or v < 1
        for v in p.acquisition_shape
    ):
        raise ValueError("acquisition_shape must contain two positive integers")
    if len(p.fov_m) != 2 or any(not math.isfinite(v) or v <= 0 for v in p.fov_m):
        raise ValueError("fov_m must contain two positive finite lengths")
    if not math.isfinite(p.r_value) or p.r_value <= 0:
        raise ValueError("r_value must be positive")
    if not math.isfinite(p.echo_spacing_s) or p.echo_spacing_s <= 0:
        raise ValueError("echo_spacing_s must be positive")


def protocol_from_dict(values):
    values = dict(values)
    sequence = values.pop("sequence")
    for key in ("acquisition_shape", "fov_m"):
        if key in values:
            values[key] = tuple(values[key])
    cls = {
        "spen": SPEN180Protocol,
        "spen180": SPEN180Protocol,
        "xspen": XSPENProtocol,
        "hybrid_spen_diff2": HybridSPENDiff2Protocol,
    }.get(sequence)
    if cls is None:
        raise ValueError(f"Unknown sequence: {sequence}")
    return cls(**values)
