"""1D regridding for EPI trajectory correction.

Ported from oneD_regriding_PV360.m and FGG_1d_type1.m.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator, splrep, splev


def _matlab_round(x: float) -> int:
    return int(np.floor(x + 0.5))


def _fft_xspace_to_kspace_np(x: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.fft.fftshift(
        np.fft.ifft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis
    )


def smooth_trajectory(epi_traj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Smooth EPI trajectory using pchip + spline approximation.

    Args:
        epi_traj: [1, N] trajectory array

    Returns:
        smoothed trajectory
    """
    epi_traj = np.asarray(epi_traj, dtype=np.float64)
    epi_traj = np.squeeze(epi_traj)
    x_traj = np.arange(1, epi_traj.size + 1)
    xq_traj = np.arange(-4, epi_traj.size + 6)

    # PCHIP interpolation
    pchip_func = PchipInterpolator(x_traj, epi_traj)
    epi_traj_pchip = pchip_func(xq_traj)

    # Spline smoothing (equivalent to MATLAB spaps with tolerance 0.1)
    tck = splrep(xq_traj, epi_traj_pchip, s=0.1)
    values = np.asarray(splev(xq_traj, tck), dtype=np.float64)

    # MATLAB: Values(1,6:end-5)
    return values[5:-5], values


def _place_regridded_block(
    kfield_regrid: np.ndarray,
    target_size: int,
    placement_size: int,
) -> np.ndarray:
    """Center a MATLAB-style regridded block in a zero-filled readout."""
    n_pe = kfield_regrid.shape[1]
    kfield_zf = np.zeros((target_size, n_pe), dtype=np.complex128)

    # MATLAB computes the insertion window from a rounded trajectory maximum.
    # SciPy's spline can move that by one sample, so use the actual block size
    # when needed to preserve a valid centered insertion.
    trunc = target_size - placement_size
    if target_size - trunc != kfield_regrid.shape[0]:
        trunc = target_size - kfield_regrid.shape[0]

    half = _matlab_round(trunc / 2)
    if trunc % 2 != 0:
        start = half - 1
        stop = target_size - half
    else:
        start = half
        stop = target_size - half
    kfield_zf[start:stop, :] = kfield_regrid
    return kfield_zf


def _one_d_regridding(
    kfield_slice: np.ndarray,
    epi_traj: np.ndarray,
    n_segments: int,
    matrix_size: list,
    *,
    reverse_offset: str,
    placement_offset: str,
) -> np.ndarray:
    """Shared implementation for PV5/PV6 and PV360 regridding variants."""
    epi_traj_s, values = smooth_trajectory(epi_traj)

    epi_traj_1 = epi_traj_s
    if reverse_offset == "values":
        epi_traj_2 = -np.flip(epi_traj_s) + np.max(values)
    elif reverse_offset == "smoothed":
        epi_traj_2 = -np.flip(epi_traj_s) + np.max(epi_traj_1)
    else:
        raise ValueError(f"Unknown reverse_offset: {reverse_offset}")

    max_traj = _matlab_round(float(np.max(epi_traj_1)))
    n_pe = kfield_slice.shape[1]
    kfield_regrid = np.zeros((max_traj, n_pe), dtype=np.complex128)

    if n_segments % 2 == 0:
        for row_ind in range(1, n_pe + 1):
            seg_idx = (row_ind - 1) // n_segments + 1
            if seg_idx % 2 == 1:
                traj = epi_traj_2
            else:
                traj = epi_traj_1
            kfield_regrid[:, row_ind - 1] = _fft_xspace_to_kspace_np(
                fgg_1d_type1(kfield_slice[:, row_ind - 1], traj, max_traj, accuracy=6),
                axis=0,
            )
    else:
        for row_ind in range(1, n_pe + 1):
            if row_ind % 2 == 1:
                traj = epi_traj_1
            else:
                traj = epi_traj_2
            kfield_regrid[:, row_ind - 1] = _fft_xspace_to_kspace_np(
                fgg_1d_type1(kfield_slice[:, row_ind - 1], traj, max_traj, accuracy=6),
                axis=0,
            )

    # Zero out edge samples (NUFFT artifacts)
    kfield_regrid[:3, :] = 0
    kfield_regrid[-3:, :] = 0

    target_size = matrix_size[0]
    if placement_offset == "values":
        placement_size = _matlab_round(float(np.max(values)))
    elif placement_offset == "smoothed":
        placement_size = max_traj
    else:
        raise ValueError(f"Unknown placement_offset: {placement_offset}")
    return _place_regridded_block(kfield_regrid, target_size, placement_size)


def one_d_regridding_pv360(
    kfield_slice: np.ndarray,
    epi_traj: np.ndarray,
    n_segments: int,
    matrix_size: list,
) -> np.ndarray:
    """1D regridding matching ``oneD_regriding_PV360.m``."""
    return _one_d_regridding(
        kfield_slice,
        epi_traj,
        n_segments,
        matrix_size,
        reverse_offset="values",
        placement_offset="values",
    )


def one_d_regridding_pv6(
    kfield_slice: np.ndarray,
    epi_traj: np.ndarray,
    n_segments: int,
    matrix_size: list,
) -> np.ndarray:
    """1D regridding matching the PV5/PV6 MATLAB path."""
    return _one_d_regridding(
        kfield_slice,
        epi_traj,
        n_segments,
        matrix_size,
        reverse_offset="smoothed",
        placement_offset="smoothed",
    )


def fgg_1d_type1(
    signal: np.ndarray,
    knots: np.ndarray,
    n_out: int,
    accuracy: int = 12,
) -> np.ndarray:
    """Python port of MATLAB ``FGG_1d_type1`` plus its MEX convolution."""
    f = np.asarray(signal, dtype=np.complex128).reshape(-1)
    knots = np.asarray(knots, dtype=np.float64).reshape(-1)
    m = f.size
    nx = int(n_out)
    r = 2
    m_sp = int(accuracy)
    tau = np.pi * m_sp / (nx * nx * r * (r - 0.5))
    m_r = r * nx

    kmin = float(np.min(knots))
    kmax = float(np.max(knots))
    scale = (nx - 1) / (kmax - kmin)
    shift = -nx / 2 - kmin * scale
    knots = scale * knots + shift
    knots = np.mod(2 * np.pi * knots / nx, 2 * np.pi)

    e3_pos = np.exp(-((np.pi * np.arange(1, m_sp + 1) / m_r) ** 2) / tau)
    e3 = np.concatenate([e3_pos[: m_sp - 1][::-1], np.ones(1), e3_pos])

    out = np.zeros(m_r, dtype=np.complex128)
    m_r_d2 = m_r / 2
    e2 = np.empty(2 * m_sp, dtype=np.float64)
    for datum, knotx in zip(f, knots):
        m1 = int(np.floor(m_r * knotx / (2 * np.pi)))
        x = knotx - m1 * np.pi / m_r_d2
        e1 = np.exp(-(x * x) / (4 * tau))
        e2_dummy = np.exp(x * np.pi / (m_r * tau))
        e2_dummy_inv = 1.0 / e2_dummy
        e2[m_sp - 1] = 1.0
        for j in range(m_sp, 2 * m_sp):
            e2[j] = e2_dummy * e2[j - 1]
        for j in range(m_sp - 2, -1, -1):
            e2[j] = e2[j + 1] * e2_dummy_inv

        v0 = datum * e1
        for l1 in range(1 - m_sp, m_sp + 1):
            lx = int((m1 + l1 + m_r_d2) >= 0)
            rx = int((m1 + l1) < m_r_d2)
            xind = int(m1 + l1 + (rx - lx) * m_r + m_r_d2)
            e23 = e2[m_sp + l1 - 1] * e3[m_sp + l1 - 1]
            out[xind] += v0 * e23

    f_tau = out
    f_tau_fft = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(f_tau)))
    chop = _matlab_round(0.5 * (r - 1) * nx)
    f_tau_fft = f_tau_fft[chop : chop + nx]
    kx_vec = np.arange(-nx / 2, nx / 2)
    e4 = np.sqrt(np.pi / tau) * np.exp(tau * (kx_vec**2))
    return f_tau_fft * e4 / (m * r)
