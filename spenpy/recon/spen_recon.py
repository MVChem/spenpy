"""Traditional PV360 SPEN reconstruction.

This module follows the data and shape path of
``Function_Process_NewPE_SPEN_OddNumWithMask_bruker_PV360.m`` for the
non-visual reconstruction core:

raw/regridded k-space -> [PE, RO, channel, image] -> RO FFT ->
SPEN weighted-adjoint SR matrix -> adaptive receiver combination.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from spenpy.bruker.param import read_pv_param
from spenpy.bruker.raw import read_bruker_kspace_pv360_fid_multichannel
from spenpy.core.matrix import calcInvA
from spenpy.fft.transform import fft_kspace_to_xspace, fft_xspace_to_kspace
from spenpy.recon.gridding import one_d_regridding_pv360, one_d_regridding_pv6
from spenpy.recon.phase import PhaseCorrectionDiagnostics, apply_pv360_one_shot_phase_correction
from spenpy.utils.coil_combine import coil_combine
from spenpy.utils.tensor import mult_mat_tensor
from spenpy.utils.zero_fill import rm_zero_filling_pv6, zero_filling_pv6


@dataclass
class SpenReconResult:
    """Container matching the MATLAB output variables."""

    images: torch.Tensor
    imag_origin: torch.Tensor
    imag_low: torch.Tensor
    spen_az: dict[str, torch.Tensor] = field(default_factory=dict)
    kfield: np.ndarray | None = None
    roffted_data_origin: torch.Tensor | None = None
    roffted_data_corrected: torch.Tensor | None = None
    sr_data: torch.Tensor | None = None
    phase_diagnostics: list[PhaseCorrectionDiagnostics] = field(default_factory=list)


def _as_list(value, default=None):
    if value is None:
        return [] if default is None else list(default)
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _as_int(value, default: int = 1) -> int:
    vals = _as_list(value, [default])
    return int(vals[0]) if vals else default


def _as_float(value, default: float = 0.0) -> float:
    vals = _as_list(value, [default])
    return float(vals[0]) if vals else default


def _ensure_5d_kfield(kfield: np.ndarray) -> np.ndarray:
    """Return k-space as [RO, PE, slice, array/receiver, echo]."""
    kfield = np.asarray(kfield)
    while kfield.ndim < 5:
        kfield = kfield[..., np.newaxis]
    if kfield.ndim > 5:
        # MATLAB often carries trailing singleton dimensions. Keep the first
        # echo-like axis and fold any remaining singleton axes away.
        trailing = int(np.prod(kfield.shape[4:]))
        kfield = np.reshape(kfield, (*kfield.shape[:4], trailing), order="F")
    return kfield


def _regrid_readout_if_needed(
    kfield: np.ndarray,
    fid_dir: str,
    n_segments: int,
    matrix_size: list[int],
    traj_dir: str | None = None,
    regrid_flavor: str = "pv360",
) -> np.ndarray:
    traj_param_dir = fid_dir if traj_dir is None else str(Path(traj_dir))
    traj = read_pv_param(traj_param_dir, "PVM_EpiTrajAdjkx")
    if traj is None:
        return kfield

    traj_arr = np.asarray(traj, dtype=np.float64).reshape(1, -1)
    if not np.any(traj_arr > 0):
        return kfield

    out = np.zeros(
        (matrix_size[0], matrix_size[1], kfield.shape[2], kfield.shape[3], kfield.shape[4]),
        dtype=np.complex128,
    )
    if regrid_flavor == "pv5":
        regrid_func = one_d_regridding_pv6
    elif regrid_flavor == "pv360":
        regrid_func = one_d_regridding_pv360
    else:
        raise ValueError(f"Unsupported regrid_flavor: {regrid_flavor}")

    for echo in range(kfield.shape[4]):
        for arr in range(kfield.shape[3]):
            for sl in range(kfield.shape[2]):
                out[:, :, sl, arr, echo] = regrid_func(
                    kfield[:, :, sl, arr, echo],
                    traj_arr,
                    n_segments,
                    matrix_size,
                )
    return out


def _zero_fill_readout(kfield: np.ndarray, zf: int) -> tuple[np.ndarray, int]:
    size_orig = kfield.shape[0]
    zf = max(zf, size_orig)
    if zf == size_orig:
        return kfield, size_orig

    filled = np.zeros((zf, *kfield.shape[1:]), dtype=kfield.dtype)
    for echo in range(kfield.shape[4]):
        for arr in range(kfield.shape[3]):
            for sl in range(kfield.shape[2]):
                filled[:, :, sl, arr, echo] = (
                    zero_filling_pv6(torch.from_numpy(kfield[:, :, sl, arr, echo]), zf)
                    .numpy()
                )
    return filled, size_orig


def _remove_readout_zero_fill(images: torch.Tensor, size_orig: int) -> torch.Tensor:
    if images.shape[0] == size_orig:
        return images

    kspace = fft_xspace_to_kspace(images, dim=0)
    cropped = rm_zero_filling_pv6(kspace, size_orig)
    return fft_kspace_to_xspace(cropped, dim=0)


def _build_cmplx_data(kfield_echo: np.ndarray) -> tuple[torch.Tensor, int, int, int, int]:
    """MATLAB ``CmplxData = permute(kField, [2 1 3])`` convention."""
    ro, pe, slice_num2, array_num2 = kfield_echo.shape
    matrix_slice_num = slice_num2

    # MATLAB reshape is column-major. With a singleton slice dimension this
    # folds receiver/array frames into the image index exactly as MATLAB does.
    kfield_3d = np.reshape(kfield_echo, (ro, pe, slice_num2 * array_num2), order="F")
    slice_num = kfield_3d.shape[2]
    array_num = 1

    cmplx_np = np.transpose(kfield_3d, (1, 0, 2))[:, :, np.newaxis, :]
    cmplx_data = torch.from_numpy(np.ascontiguousarray(cmplx_np.astype(np.complex128)))
    return cmplx_data, matrix_slice_num, slice_num2, array_num2, slice_num * array_num


def _coil_combine_origin(
    roffted_data: torch.Tensor,
    pe: int,
    ro: int,
    slice_num2: int,
    array_num2: int,
) -> torch.Tensor:
    imag = roffted_data.reshape(pe, ro, slice_num2, array_num2)
    return coil_combine(imag).squeeze()


def _coil_combine_sr(
    sr_data: torch.Tensor,
    pe: int,
    ro: int,
    num_receiver_images: int,
) -> torch.Tensor:
    # MATLAB: reshape(...,[PE,RO,SliceNum,ArrayNum]) with SliceNum holding
    # the receiver frames, then permute(...,[1,2,4,3]) before coilCombinebao.
    temp = sr_data[:, :, 0, :num_receiver_images].unsqueeze(3)
    return coil_combine(temp.permute(0, 1, 3, 2)).squeeze()


def reconstruct_odd_segments(
    fid_dir: str,
    traj_dir: str | None = None,
    kfield: np.ndarray | torch.Tensor | None = None,
    device: str = "cpu",
    debug_show: bool = False,
    process_with_pre_phase_corr: bool = True,
    smooth_motion_phase_between_shots: bool = True,
    b_amp_correct: bool = False,
    zf: int = 0,
    spen_gy: float | None = None,
    spat_enc_duration: float | None = None,
    regrid_flavor: str = "pv360",
    return_phase_diagnostics: bool = False,
) -> SpenReconResult:
    """Reconstruct an odd-segment PV360 SPEN scan.

    The default phase-correction path follows the one-shot odd-segment branch
    used by the PV360 MATLAB script. General multi-shot phase correction is not
    implemented here.
    """
    del debug_show, b_amp_correct

    fid_dir = str(Path(fid_dir))
    matrix_size = [int(v) for v in _as_list(read_pv_param(fid_dir, "PVM_Matrix"), [256, 256])]
    while len(matrix_size) < 2:
        matrix_size.append(1)

    n_segments = _as_int(read_pv_param(fid_dir, "NSegments"), 1)
    pvm_n_echo_images = _as_int(read_pv_param(fid_dir, "PVM_NEchoImages"), 1)

    fov = [float(v) for v in _as_list(read_pv_param(fid_dir, "PVM_Fov"), [40.0, 40.0])]
    while len(fov) < 2:
        fov.append(fov[-1])
    fov_cm = [v / 10 for v in fov]
    lpe = fov_cm[1]

    spen_gy_param = read_pv_param(fid_dir, "SpenGyGaussStren")
    tp_param = read_pv_param(fid_dir, "SpatEncDuration")
    if spen_gy is None:
        if spen_gy_param is None:
            raise ValueError(f"Missing SpenGyGaussStren in {fid_dir} and no manual override provided")
        spen_gy = _as_float(spen_gy_param)
    if spat_enc_duration is None:
        if tp_param is None:
            raise ValueError(f"Missing SpatEncDuration in {fid_dir} and no manual override provided")
        tp = _as_float(tp_param) / 1000
    else:
        tp = spat_enc_duration / 1000
    ppe = -_as_float(read_pv_param(fid_dir, "PVM_SPackArrPhase1Offset"), 0.0) / 10
    shift_pe = -ppe * 10

    gamma_hz = 4.2574e3
    a_sign = -1
    gauss_relative_width = 0.8
    a_rad2cmsqr_base = a_sign * 2 * math.pi * gamma_hz * spen_gy * tp / lpe

    if kfield is None:
        kfield_np = read_bruker_kspace_pv360_fid_multichannel(fid_dir)
    elif isinstance(kfield, torch.Tensor):
        kfield_np = kfield.detach().cpu().numpy()
    else:
        kfield_np = np.asarray(kfield)

    kfield_np = _ensure_5d_kfield(kfield_np)
    kfield_np = _regrid_readout_if_needed(
        kfield_np,
        fid_dir,
        n_segments,
        matrix_size,
        traj_dir=traj_dir,
        regrid_flavor=regrid_flavor,
    )
    kfield_np, size_orig = _zero_fill_readout(kfield_np, zf)
    kfield_np = kfield_np.astype(np.complex64, copy=False)

    last_images: torch.Tensor | None = None
    last_origin: torch.Tensor | None = None
    last_low: torch.Tensor | None = None
    last_roffted_origin: torch.Tensor | None = None
    last_roffted_corrected: torch.Tensor | None = None
    last_sr_data: torch.Tensor | None = None
    spen_az: dict[str, torch.Tensor] = {}
    phase_diagnostics: list[PhaseCorrectionDiagnostics] = []

    num_echoes = min(pvm_n_echo_images, kfield_np.shape[4])
    for echo_idx in range(num_echoes):
        kfield_echo = kfield_np[:, :, :, :, echo_idx]
        phase_factor = -1 if (echo_idx + 1) % 2 == 0 else 1
        if phase_factor == -1:
            kfield_echo = np.flip(kfield_echo, axis=1)
        a_rad2cmsqr = phase_factor * a_rad2cmsqr_base

        cmplx_data, _matrix_slice_num, slice_num2, array_num2, num_images = _build_cmplx_data(kfield_echo)
        cmplx_data = cmplx_data.to(device)
        pe, ro = cmplx_data.shape[0], cmplx_data.shape[1]

        roffted_data = fft_kspace_to_xspace(cmplx_data, dim=1)
        imag_origin = _coil_combine_origin(roffted_data, pe, ro, slice_num2, array_num2)
        roffted_data_origin = roffted_data.clone()

        inv_a, a_final = calcInvA(
            a_rad2cmsqr,
            lpe,
            pe,
            shift_pe,
            -a_sign,
            0,
            gauss_relative_width,
        )
        inv_a = inv_a.to(device)
        a_final = a_final.to(device)

        one_shot_odd_inv: torch.Tensor | None = None
        one_shot_even_inv: torch.Tensor | None = None
        if pe % 2 == 0:
            one_shot_odd_inv, _ = calcInvA(
                a_rad2cmsqr,
                lpe,
                pe // 2,
                shift_pe,
                -a_sign,
                0,
                gauss_relative_width,
            )
            one_shot_even_inv, _ = calcInvA(
                a_rad2cmsqr,
                lpe,
                pe // 2,
                shift_pe,
                -a_sign,
                0.5,
                gauss_relative_width,
            )
            one_shot_odd_inv = one_shot_odd_inv.to(device)
            one_shot_even_inv = one_shot_even_inv.to(device)

        if (
            process_with_pre_phase_corr
            and n_segments == 1
            and one_shot_odd_inv is not None
            and one_shot_even_inv is not None
        ):
            phase_corrected = apply_pv360_one_shot_phase_correction(
                roffted_data,
                inv_a,
                one_shot_odd_inv,
                one_shot_even_inv,
                optimize=True,
                smooth_motion_phase_between_shots=smooth_motion_phase_between_shots,
                return_diagnostics=return_phase_diagnostics,
            )
            if return_phase_diagnostics:
                roffted_data, diagnostics = phase_corrected
                phase_diagnostics.append(diagnostics)
            else:
                roffted_data = phase_corrected

        whole_fixed_signal_post_rofft = roffted_data
        sr_data = mult_mat_tensor(inv_a, whole_fixed_signal_post_rofft)
        images = _coil_combine_sr(sr_data, pe, ro, num_images)
        imag_low = _coil_combine_origin(whole_fixed_signal_post_rofft, pe, ro, slice_num2, array_num2)

        images = _remove_readout_zero_fill(images, size_orig)

        if one_shot_odd_inv is not None and one_shot_even_inv is not None:
            spen_az["OneShotOddInvAZ"] = one_shot_odd_inv
            spen_az["OneShotEvenInvAZ"] = one_shot_even_inv
        spen_az["tmpInvAZ"] = inv_a
        spen_az["tmpAFinal"] = a_final

        last_images = images
        last_origin = imag_origin
        last_low = imag_low
        last_roffted_origin = roffted_data_origin
        last_roffted_corrected = whole_fixed_signal_post_rofft.clone()
        last_sr_data = sr_data.clone()

    if last_images is None or last_origin is None or last_low is None:
        raise ValueError(f"No echoes were available for reconstruction in {fid_dir}")

    return SpenReconResult(
        images=last_images,
        imag_origin=last_origin,
        imag_low=last_low,
        spen_az=spen_az,
        kfield=kfield_np,
        roffted_data_origin=last_roffted_origin,
        roffted_data_corrected=last_roffted_corrected,
        sr_data=last_sr_data,
        phase_diagnostics=phase_diagnostics,
    )


def orient_pv360_spen_image(images: torch.Tensor | np.ndarray) -> np.ndarray:
    """Apply the final ``pv360.m`` orientation: ``flip(flip(images,1),2)``."""
    if isinstance(images, torch.Tensor):
        arr = images.detach().cpu().numpy()
    else:
        arr = np.asarray(images)
    return np.flip(np.flip(arr, axis=0), axis=1)
