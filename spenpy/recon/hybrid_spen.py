"""Hybrid SPEN-Diff2 referenceless PhaseMap + Gaussian-windowed InvA.

Reference: xspen_recons/reconstruct_xspen.m and the historical Siemens
Reffless1ShotHybridSPENEvenOddFix.m. This scanner sequence uses quadratic
encoding; it is not automatically identified with the ideal crossed-chirp
XSPENProtocol. Matrix products run in Torch. Legacy phase fitting uses NumPy
and SciPy and is not differentiated.
"""

import math

import numpy as np
import torch

from ..core import Acquisition
from .phasemap import _frame, reconstruct_phasemap_inva, ro_fft


def _polynomial_unwrap_phase_fit(even, odd, order=2):
    from scipy.ndimage import median_filter

    even_np = even.detach().cpu().numpy().astype(np.complex128)
    odd_np = odd.detach().cpu().numpy().astype(np.complex128)
    me = np.sqrt(np.sum(np.abs(even_np) ** 2, axis=-1))
    mo = np.sqrt(np.sum(np.abs(odd_np) ** 2, axis=-1))
    cross = np.sum(even_np * odd_np.conj(), axis=-1)
    noise = np.std(me - mo, ddof=1) * 2
    mask = (me > noise) & (mo > noise)
    n, k = cross.shape
    y, x = np.meshgrid(np.arange(1, n + 1), np.arange(1, k + 1), indexing="ij")

    delta = np.angle(cross[:, 1:] * cross[:, :-1].conj())
    good = mask[:, 1:] & mask[:, :-1]
    gross = 0.0
    if good.any():
        a, b = delta[good], np.mod(delta[good], 2 * np.pi)
        gross = float(np.mean(b if np.var(b) < np.var(a) else a))
    modified = cross * np.exp(-1j * gross * x)
    unit = np.divide(
        modified,
        np.abs(modified),
        out=np.zeros_like(modified),
        where=np.abs(modified) > 0,
    )
    gy = np.gradient(unit, axis=0) if n > 1 else np.zeros_like(unit)
    gx = np.gradient(unit, axis=1) if k > 1 else np.zeros_like(unit)
    scores = median_filter(
        np.sqrt(np.abs(gx) ** 2 + np.abs(gy) ** 2), size=(7, 7), mode="reflect"
    )
    mask &= scores < 1

    def fine_linear(values, valid, axis):
        # MATLAB permutes the selected dimension first, then column-flattens.
        if axis == 1:
            values, valid = values.T, valid.T
        angles = np.angle(values.ravel(order="F")[valid.ravel(order="F")])
        if len(angles) < 2:
            return 0.0
        return float(np.polyfit(np.arange(1, len(angles) + 1), np.unwrap(angles), 1)[0])

    fine_x = fine_linear(modified, mask, 1)
    modified *= np.exp(-1j * fine_x * x)
    fine_y = fine_linear(modified, mask, 0)
    modified *= np.exp(-1j * fine_y * y)
    constant = float(np.angle(np.mean(modified[mask]))) if mask.any() else 0.0
    modified *= np.exp(-1j * constant)
    terms = [np.ones_like(x), x, y]
    if order == 2:
        terms += [x * x, x * y, y * y]
    design = np.stack(terms, axis=-1).reshape(-1, len(terms))
    weights = (mask * np.minimum(me, mo)).ravel()
    coeffs = np.zeros(len(terms))
    valid = np.count_nonzero(weights) >= len(terms)
    if valid:
        # The MATLAB function multiplies both V and z by w (not sqrt(w)).
        coeffs, _, rank, _ = np.linalg.lstsq(
            design * weights[:, None], np.angle(modified).ravel() * weights, rcond=None
        )
        valid = rank == len(terms)
    if not valid:
        # Explicit finite fallback for empty or rank-deficient masks.
        coeffs[:] = 0
    coeffs[0] += constant
    coeffs[1] += gross + fine_x
    coeffs[2] += fine_y
    phase = (design @ coeffs).reshape(n, k)
    return (
        torch.as_tensor(phase, device=even.device, dtype=even.real.dtype),
        torch.as_tensor(coeffs, device=even.device, dtype=even.real.dtype),
        valid,
    )


def calc_hybrid_spen_matrices(
    num_pe, fov_m, r_value, *, device="cpu", dtype=torch.complex128
):
    """Build the archived Siemens matrices, including seeded border perturbations.

    FOV input is metres; the historical matrix internally integrates in cm.
    Gaussian width 0.3, low-resolution factor 0.9, 100 perturbations of 10%
    pixel width, and MATLAB rng(0, 'twister') are preserved.
    """
    from .._legacy.core.matrix import calcSRMatrixApprox

    if not isinstance(num_pe, int) or num_pe < 4 or num_pe % 2:
        raise ValueError("num_pe must be an even integer >= 4")
    if not all(math.isfinite(v) and v > 0 for v in (fov_m, r_value)):
        raise ValueError("fov_m and r_value must be finite and positive")
    if dtype not in (torch.complex64, torch.complex128):
        raise ValueError("dtype must be complex64 or complex128")
    length = fov_m * 100
    maximum = 2 * math.pi * r_value
    a = maximum / length**2
    borders = torch.linspace(-length / 2, length / 2, num_pe + 1, dtype=torch.float64)
    ky = -2 * a * torch.arange(num_pe, dtype=torch.float64) * length / num_pe
    b = -ky[0] - 2 * a * borders[0]
    full = calcSRMatrixApprox(maximum, num_pe, ky, borders, b)[0]
    distance = torch.arange(num_pe)[:, None] - torch.arange(num_pe)[None, :]
    variance = (0.3 * math.pi * num_pe**2 / maximum) ** 2
    inv_a = (full * torch.exp(-distance.double().square() / (2 * variance))).mH
    low = int(0.9 * num_pe / 2 + 0.5)  # MATLAB round, including half ties.
    edges = torch.linspace(-length / 2, length / 2, low + 1, dtype=torch.float64)
    half = calcSRMatrixApprox(maximum, low, ky, edges, b)[0]
    # MATLAB seed zero maps to the MT19937 reference seed 5489.
    rng = np.random.RandomState(5489)
    inverses = []
    for parity in (0, 1):
        matrices = [half[parity::2]]
        for _ in range(100):
            perturbed = edges.clone()
            perturbed[1:-1] += torch.from_numpy(
                (2 * rng.rand(low - 1) - 1) * 0.1 * length / low
            )
            matrices.append(
                calcSRMatrixApprox(maximum, low, ky[parity::2], perturbed, b)[0]
            )
        column = torch.cat(matrices, dim=0)
        inverse = torch.linalg.pinv(
            column, rtol=max(column.shape) * torch.finfo(torch.float64).eps
        )
        inverses.append(inverse.reshape(low, 101, num_pe // 2).sum(dim=1))
    return {
        name: value.to(device=device, dtype=dtype)
        for name, value in {
            "a": full,
            "inv_a": inv_a,
            "a_half": half,
            "inv_odd": inverses[0],
            "inv_even": inverses[1],
        }.items()
    }


def estimate_hybrid_spen_shift(acquisition, *, max_frames=10):
    """Estimate the original global position shift from up to ten raw frames.

    Pass the complete acquisition before selecting a frame. Slice ordering
    from the reader is undone for this calibration, matching MATLAB's first
    ten acquisition-order frames. No reconstructed reference is used.
    """
    if not isinstance(max_frames, int) or max_frames < 1:
        raise ValueError("max_frames must be a positive integer")
    _validate_hybrid_spen(acquisition)
    axes = acquisition.axes
    if set(axes) - {"spen", "readout", "coil", "slice", "volume"}:
        raise ValueError("Select unsupported counters before shift calibration")
    names = [a for a in ("volume", "slice", "spen", "readout", "coil") if a in axes]
    array = acquisition.permute(*names).data
    if "slice" in names:
        order = acquisition.metadata.get("slice_order")
        if order is not None:
            undo = torch.argsort(torch.tensor(order, device=array.device))
            array = array.index_select(names.index("slice"), undo)
    m, k, c = array.shape[-3:]
    if m % 4:
        raise ValueError("Automatic shift requires num_pe divisible by four")
    frames = array.reshape(-1, m, k, c)[:max_frames]
    fov, r = acquisition.metadata["phase_fov_m"], acquisition.metadata["r_value"]
    matrices = calc_hybrid_spen_matrices(m // 2, fov, r, device=array.device)
    phases = []
    for frame in frames:
        result = reconstruct_phasemap_inva(
            ro_fft(frame[::2].to(torch.complex128)),
            matrices["inv_a"],
            matrices["a_half"],
            inv_odd=matrices["inv_odd"],
            inv_even=matrices["inv_even"],
            estimator="polynomial_unwrap",
            polynomial_order=1,
        )
        phases.append(float(result.coefficients[0]))
    phases = np.asarray(phases)
    positive = np.mod(phases, 2 * np.pi)
    use = phases if np.std(phases) < np.std(positive) else positive
    return float(np.mean(use) / (4 * math.pi * r / (m / 2) ** 2))


def _validate_hybrid_spen(acquisition):
    if not isinstance(acquisition, Acquisition):
        raise TypeError("Pass read_siemens output with acquisition metadata")
    info = acquisition.metadata
    if "esrs_hyb_spen_Diff2" not in info.get("sequence_name", ""):
        raise ValueError(
            "This scanner baseline is calibrated only for esrs_hyb_spen_Diff2"
        )
    simulated = (
        acquisition.stage == "uniform_kspace"
        and info.get("simulated") is True
        and info.get("protocol", {}).get("sequence") == "hybrid_spen_diff2"
    )
    if not simulated and (
        acquisition.stage != "regridded_kspace"
        or not info.get("remove_os")
        or not info.get("readout_reflections_corrected")
    ):
        raise ValueError(
            "Requires regridding, RO oversampling removal and reflected-line correction"
        )
    if not all(
        math.isfinite(info.get(key, 0)) and info.get(key, 0) > 0
        for key in ("phase_fov_m", "r_value")
    ):
        raise ValueError("Missing valid phase FOV / R metadata")


def reconstruct_hybrid_spen(
    acquisition, *, shift_pixels=None, matrices=None, correct_phase=True
):
    """Reconstruct one selected frame with the archived quadratic Siemens model.

    For a scan-wide calibration, first call estimate_hybrid_spen_shift on
    the full acquisition and reuse that value for every frame. If omitted,
    the shift is estimated from this frame only and reported as such.
    """
    _validate_hybrid_spen(acquisition)
    signal = _frame(acquisition, {"regridded_kspace", "uniform_kspace"}).to(
        torch.complex128
    )
    m = signal.shape[0]
    info = acquisition.metadata
    mode = "supplied" if shift_pixels is not None else "single_frame_estimate"
    if shift_pixels is None:
        shift_pixels = estimate_hybrid_spen_shift(acquisition)
    if not math.isfinite(shift_pixels):
        raise ValueError("shift_pixels must be finite")
    ramp = torch.exp(
        4j
        * math.pi
        * info["r_value"]
        / m**2
        * shift_pixels
        * torch.arange(1, m + 1, dtype=torch.float64, device=signal.device)
    )
    if matrices is None:
        matrices = calc_hybrid_spen_matrices(
            m, info["phase_fov_m"], info["r_value"], device=signal.device
        )
    result = reconstruct_phasemap_inva(
        ro_fft(signal * ramp[:, None, None]),
        matrices["inv_a"],
        matrices["a_half"],
        inv_odd=matrices["inv_odd"],
        inv_even=matrices["inv_even"],
        correct_phase=correct_phase,
        estimator="polynomial_unwrap",
    )
    result.metadata.update(
        sequence_name=info["sequence_name"],
        inv_a_kind="Gaussian-windowed quadratic encoding adjoint; historical Siemens width 0.3",
        automatic_shift_pixels=shift_pixels,
        shift_calibration=mode,
        encoding_model="legacy Siemens quadratic SPEN; not ideal crossed-chirp xSPEN",
        sequence="hybrid_spen_diff2",
    )
    return result
