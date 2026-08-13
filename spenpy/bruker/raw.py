"""Bruker raw data reader for SPEN FID data.

Ported from ReadBrukerkSpace_PV360_fid_multichannel.m.
Reads rawdata.job0 and reshapes into complex k-space arrays.

Key MATLAB conventions matched:
- fread(fid, Inf, 'int32') -> np.fromfile with '<i4' (little-endian int32)
- reshape(data, [a,b,c,d,e]) with column-major order -> order='F'
- permute(data, [1,2,4,3,5]) -> transpose with 0-indexed dims
- complex(data(1,:,:,:,:), data(2,:,:,:,:)) -> data[0] + 1j*data[1]
- flipud(A) on axis-0 slice -> A[::-1, ...]
"""

import os
import numpy as np
from spenpy.bruker.param import read_pv_param


def _matlab_round(x):
    """MATLAB-style round: round(x) with 0.5 rounding away from zero."""
    import math
    return int(math.floor(x + 0.5))


def read_bruker_kspace_pv360_fid_multichannel(fid_dir: str) -> np.ndarray:
    """Read Bruker PV360 raw FID data into complex k-space array.

    This is the Python equivalent of MATLAB's:
        ReadBrukerkSpace_PV360_fid_multichannel(fid_dir)

    Args:
        fid_dir: directory containing rawdata.job0 and method/acqp files.

    Returns:
        kField: complex array. Shape depends on acquisition:
            - FLASH/RARE: [xsize, ysize, zsize, echo_no]
            - EPI: [xsize, ysize, zsize, TempNumEcho, ChannelNum]
            - SPEN multi-echo: [xsize, ysize, zsize, TempNumEcho, ChannelNum, PVM_NEchoImage]
    """
    # Read acquisition parameters
    matrix_size = read_pv_param(fid_dir, "PVM_Matrix")
    if matrix_size is None:
        raise ValueError("PVM_Matrix not found")
    matrix_size = _to_list(matrix_size)
    while len(matrix_size) < 2:
        matrix_size.append(1)

    slice_per_pack = read_pv_param(fid_dir, "PVM_SPackArrNSlices")
    if slice_per_pack is None:
        slice_per_pack = [1]
    slice_per_pack = _to_list(slice_per_pack)
    zsize = int(sum(slice_per_pack))

    obj_order_list = read_pv_param(fid_dir, "PVM_ObjOrderList")
    if obj_order_list is None:
        obj_order_list = [[0]]
    obj_order_list = _to_2d_list(obj_order_list)

    enc_n_receivers = read_pv_param(fid_dir, "PVM_EncNReceivers")
    if enc_n_receivers is None:
        enc_n_receivers = 1

    diff_num = read_pv_param(fid_dir, "PVM_DwNDiffExp")
    if diff_num is None or (isinstance(diff_num, (int, float)) and diff_num < 1):
        diff_num = read_pv_param(fid_dir, "DwNDiffExp")
    if diff_num is None or (isinstance(diff_num, (int, float)) and diff_num < 1):
        diff_num = 1

    n_repetitions = read_pv_param(fid_dir, "PVM_NRepetitions")
    if n_repetitions is None:
        n_repetitions = 1
    diff_num = int(diff_num * n_repetitions)

    method_name = read_pv_param(fid_dir, "Method")
    if method_name is None:
        method_name = ""
    method_name = str(method_name).lower().strip("<>")

    n_segments = read_pv_param(fid_dir, "NSegments")
    if n_segments is None:
        n_segments = 1

    pvm_n_echo_images = read_pv_param(fid_dir, "PVM_NEchoImages")
    if pvm_n_echo_images is None:
        pvm_n_echo_images = 1

    xsize = int(matrix_size[0])
    ysize = int(matrix_size[1])
    matrix_size = [xsize, ysize, zsize]

    # Read raw binary data
    # PV360 uses rawdata.job0, older formats use 'fid'
    raw_path = os.path.join(fid_dir, "rawdata.job0")
    if not os.path.exists(raw_path):
        raw_path = os.path.join(fid_dir, "fid")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Neither rawdata.job0 nor fid found in: {fid_dir}")

    # MATLAB: fread(fid, Inf, 'int32') reads native little-endian signed int32
    data = np.fromfile(raw_path, dtype="<i4")
    data = data.astype(np.float64)

    # TempNumEcho and ChannelNum tracking (MATLAB conventions)
    temp_num_echo = diff_num
    channel_num = 1

    # Route to method-specific parser
    if method_name.endswith("flash"):
        kfield = _parse_flash(data, xsize, ysize, zsize, temp_num_echo)
    elif method_name.endswith("rare"):
        kfield = _parse_rare(data, xsize, ysize, zsize, temp_num_echo, fid_dir)
    elif method_name.endswith("epi") or method_name == "user:lucioepi":
        kfield = _parse_epi(
            data, xsize, ysize, zsize, n_segments,
            temp_num_echo, channel_num, matrix_size,
        )
    else:
        # SPEN or other methods
        if pvm_n_echo_images > 1:
            if n_segments % 2 == 0:
                kfield = _parse_spen_multi_echo_even(
                    data, xsize, ysize, zsize, n_segments,
                    temp_num_echo, channel_num, pvm_n_echo_images,
                    enc_n_receivers, matrix_size,
                )
            else:
                kfield = _parse_spen_multi_echo_odd(
                    data, xsize, ysize, zsize, n_segments,
                    temp_num_echo, channel_num, pvm_n_echo_images,
                    enc_n_receivers, matrix_size,
                )
        else:
            # Non-T2 EPI (PVM_NEchoImages == 1)
            if n_segments % 2 == 0:
                kfield = _parse_nont2_even(
                    data, xsize, ysize, zsize, n_segments,
                    temp_num_echo, channel_num,
                    enc_n_receivers, matrix_size,
                )
            else:
                kfield = _parse_nont2_odd(
                    data, xsize, ysize, zsize, n_segments,
                    temp_num_echo, channel_num,
                    enc_n_receivers, matrix_size,
                )

    # Slice reordering using PVM_ObjOrderList
    # MATLAB: kFieldOrdered(:,:,ObjOrderList(1,sl)+1,:,:,:) = kField(:,:,sl,:,:,:)
    if kfield.ndim >= 3 and kfield.shape[2] == len(obj_order_list[0]):
        kfield = _reorder_slices(kfield, obj_order_list, matrix_size)
    else:
        print("Warning: slice reordering skipped (dimension mismatch)")

    return kfield


def _to_list(val):
    """Convert parameter value to list."""
    if val is None:
        return []
    if isinstance(val, list):
        return val
    return [val]


def _to_2d_list(val):
    """Convert parameter value to 2D list."""
    if val is None:
        return [[0]]
    if isinstance(val, list):
        if len(val) > 0 and isinstance(val[0], list):
            return val
        return [val]
    return [[val]]


def _flip_along_axis0(arr, indices):
    """Flip specified indices along axis 0.

    MATLAB: A(:, 2:2:end, ...) = flipud(A(:, 2:2:end, ...))
    Python equivalent: arr[indices] = arr[::-1][indices]

    This reverses axis 0 only for the selected indices.
    """
    result = arr.copy()
    flipped = arr[::-1]  # reverse along axis 0
    result[indices] = flipped[indices]
    return result


def _zero_fill_1d(data, target_size, axis=0):
    """Zero-fill along an axis, matching MATLAB convention.

    MATLAB:
        dataZF = zeros([target_size, other_dims]);
        if round(trunc/2)*2 ~= trunc:
            dataZF(round(trunc/2):end-round(trunc/2), ...) = data;
        else:
            dataZF(round(trunc/2)+1:end-round(trunc/2), ...) = data;

    Python conversion (1-based -> 0-based):
        odd trunc: dataZF[round(trunc/2) : target_size - round(trunc/2)] = data
        even trunc: dataZF[round(trunc/2) : target_size - round(trunc/2)] = data
        But MATLAB's round(trunc/2)+1 for even becomes round(trunc/2) in 0-based.
    """
    orig_size = data.shape[axis]
    trunc = target_size - orig_size
    if trunc <= 0:
        return data

    half = _matlab_round(trunc / 2)
    new_shape = list(data.shape)
    new_shape[axis] = target_size
    data_zf = np.zeros(new_shape, dtype=data.dtype)

    if trunc % 2 != 0:
        # odd trunc: MATLAB round(trunc/2):end-round(trunc/2)
        # 0-based: half - 1 : target_size - half
        sl = [slice(None)] * data.ndim
        sl[axis] = slice(half - 1, target_size - half)
        data_zf[tuple(sl)] = data
    else:
        # even trunc: MATLAB round(trunc/2)+1:end-round(trunc/2)
        # 0-based: half : target_size - half
        sl = [slice(None)] * data.ndim
        sl[axis] = slice(half, target_size - half)
        data_zf[tuple(sl)] = data

    return data_zf


def _expand_trailing_singletons(src, target_ndim):
    """Add trailing singleton axes so MATLAB-style singleton dims broadcast."""
    while src.ndim < target_ndim:
        src = src[..., np.newaxis]
    return src


def _reorder_slices(kfield, obj_order_list, matrix_size):
    """Reorder slices using PVM_ObjOrderList."""
    kfield_ordered = np.zeros_like(kfield)
    zsize = matrix_size[2] if len(matrix_size) > 2 else 1

    for sl in range(zsize):
        dest_idx = int(obj_order_list[0][sl])  # ensure integer
        src_sl = [slice(None)] * kfield.ndim
        src_sl[2] = sl
        dest_sl = [slice(None)] * kfield.ndim
        dest_sl[2] = dest_idx
        kfield_ordered[tuple(dest_sl)] = kfield[tuple(src_sl)]

    return kfield_ordered


# ─── FLASH ───────────────────────────────────────────────────────────────────
# MATLAB: reshape(data, [2 xsize zsize ysize echo_no])
#         permute([1 2 4 3 5]) -> [2 xsize ysize zsize echo_no]
#         kField = complex(data(1,:,:,:,:), data(2,:,:,:,:))
#         result: [xsize, ysize, zsize, echo_no]
# NO flip applied.

def _parse_flash(data, xsize, ysize, zsize, echo_no):
    """Parse FLASH k-space data."""
    expected = 2 * xsize * zsize * ysize * echo_no
    data = data[:expected]
    # MATLAB reshape: [2, xsize, zsize, ysize, echo_no]
    d = data.reshape(2, xsize, zsize, ysize, echo_no, order="F")
    # MATLAB permute: [1,2,4,3,5] -> 0-indexed: [0,1,3,2,4]
    d = d.transpose(0, 1, 3, 2, 4)
    # Trim: data(:,1:xsize,:,:,:) -- identity
    d = d[:, :xsize, :, :, :]
    # Complex from dim 0
    kfield = d[0] + 1j * d[1]
    return kfield


# ─── RARE ────────────────────────────────────────────────────────────────────
# Same reshape/permute as FLASH, plus RARE step handling.

def _parse_rare(data, xsize, ysize, zsize, echo_no, fid_dir):
    """Parse RARE k-space data."""
    expected = 2 * xsize * zsize * ysize * echo_no
    data = data[:expected]
    d = data.reshape(2, xsize, zsize, ysize, echo_no, order="F")
    d = d.transpose(0, 1, 3, 2, 4)
    d = d[:, :xsize, :, :, :]
    kfield = d[0] + 1j * d[1]
    kfield = np.squeeze(kfield)

    # RARE step: kField(:,RARESTEP+max(RARESTEP)+2,:,:) = kField
    rare_step = read_pv_param(fid_dir, "PVM_EncSteps1")
    if rare_step is not None:
        rare_step = _to_list(rare_step)
        # MATLAB indexing: RARESTEP + max(RARESTEP) + 2
        # RARESTEP in MATLAB is 1-based, +2 is also 1-based
        # Convert to 0-based: RARESTEP[i] + max(RARESTEP) + 2 - 1
        offset = max(rare_step) + 1  # 0-based offset
        src = kfield.copy()
        for i, rs in enumerate(rare_step):
            idx = rs + offset  # MATLAB: RARESTEP(i) + max + 2, 0-based = rs + max + 1
            if idx < kfield.shape[1]:
                kfield[:, idx] = src[:, i]

    return kfield


# ─── EPI ─────────────────────────────────────────────────────────────────────

def _parse_epi(data, xsize, ysize, zsize, n_segments,
               temp_num_echo, channel_num, matrix_size):
    """Parse EPI k-space data."""
    if n_segments > 1:
        return _parse_epi_multiseg(
            data, xsize, ysize, zsize, n_segments,
            temp_num_echo, channel_num, matrix_size,
        )
    else:
        return _parse_epi_single(
            data, xsize, ysize, zsize,
            temp_num_echo, channel_num, matrix_size,
        )


def _parse_epi_multiseg(data, xsize, ysize, zsize, n_segments,
                        temp_num_echo, channel_num, matrix_size):
    """EPI with NSegments > 1."""
    if zsize == 1:
        expected = 2 * xsize * zsize * (ysize // n_segments) * n_segments * temp_num_echo
        data = data[:expected]
        d = data.reshape(2, xsize, zsize, ysize // n_segments, n_segments, temp_num_echo, order="F")
        # permute [1,2,4,3,5] -> [0,1,3,2,4] (dims 0..4, then 5 stays)
        d = d.transpose(0, 1, 3, 2, 4, 5)
    else:
        expected = 2 * xsize * (ysize // n_segments) * zsize * n_segments * temp_num_echo
        data = data[:expected]
        d = data.reshape(2, xsize, ysize // n_segments, zsize, n_segments, temp_num_echo, order="F")
        # No permute

    kfield = d[0] + 1j * d[1]
    # Reshape to working form
    kfield_temp = kfield.reshape(
        matrix_size[0], matrix_size[1] // n_segments, matrix_size[2],
        n_segments, temp_num_echo, channel_num,
    )

    # Flip alternating lines for ALL segments: 2:2:end (MATLAB 1-based)
    # In 0-based: indices 1, 3, 5, ... -> slice [1::2]
    for k in range(temp_num_echo):
        for g in range(n_segments):
            for i in range(matrix_size[2]):
                kfield_temp[:, 1::2, i, g, k] = \
                    kfield_temp[::-1, 1::2, i, g, k].copy()

    # Interleave segments
    full_shape = (matrix_size[0], matrix_size[1], matrix_size[2],
                  temp_num_echo, channel_num)
    kfield = np.zeros(full_shape, dtype=np.complex128)
    for i in range(n_segments):
        kfield[:, i:n_segments:matrix_size[1] - n_segments + i, :, :, :] = \
            np.squeeze(kfield_temp[:, :, :, i, :, :])

    return kfield


def _parse_epi_single(data, xsize, ysize, zsize,
                      temp_num_echo, channel_num, matrix_size):
    """EPI with NSegments == 1 (with zero-filling)."""
    # Zero-filling path
    # MATLAB determines actual xsize from data length
    actual_xsize = len(data) // (2 * ysize * zsize * temp_num_echo)

    # Zero-fill to target xsize
    trunc = xsize - actual_xsize
    if trunc > 0:
        d = np.zeros(2 * xsize * ysize * zsize * temp_num_echo, dtype=np.float64)
        # Reshape source data
        src = data[:2 * actual_xsize * ysize * zsize * temp_num_echo]
        src = src.reshape(2, actual_xsize, ysize, zsize, 1, temp_num_echo, order="F")

        half = _matlab_round(trunc / 2)
        if trunc % 2 != 0:
            d_reshaped = d.reshape(2, xsize, ysize, zsize, 1, temp_num_echo, order="F")
            d_reshaped[:, half - 1:xsize - half, :, :, :, :] = src
            d = d_reshaped.reshape(-1)
        else:
            d_reshaped = d.reshape(2, xsize, ysize, zsize, 1, temp_num_echo, order="F")
            d_reshaped[:, half:xsize - half, :, :, :, :] = src
            d = d_reshaped.reshape(-1)
    else:
        d = data[:2 * xsize * ysize * zsize * temp_num_echo]

    d = d.reshape(2, xsize, ysize, zsize, 1, temp_num_echo, order="F")
    kfield = d[0] + 1j * d[1]
    kfield = kfield.reshape(matrix_size[0], matrix_size[1], matrix_size[2],
                            temp_num_echo, channel_num)

    # Flip 2:2:end -> 0-based [1::2]
    for k in range(temp_num_echo):
        for i in range(matrix_size[2]):
            kfield[:, 1::2, i, k] = kfield[::-1, 1::2, i, k].copy()

    return kfield


# ─── SPEN Multi-Echo (PVM_NEchoImages > 1), Even NSegments ──────────────────

def _parse_spen_multi_echo_even(data, xsize, ysize, zsize, n_segments,
                                temp_num_echo, channel_num, pvm_n_echo_images,
                                enc_n_receivers, matrix_size):
    """SPEN multi-echo with even number of segments."""
    if n_segments > 1:
        if enc_n_receivers != 1:
            # Multicoil
            xsize = len(data) // (2 * zsize * ysize * temp_num_echo * pvm_n_echo_images) // enc_n_receivers
            d = data[:len(data)]
            d = d.reshape(
                2, xsize, ysize // n_segments, enc_n_receivers,
                pvm_n_echo_images, zsize, n_segments, temp_num_echo, order="F",
            )
            # permute [1,2,3,5,6,7,8,4] -> [0,1,2,4,5,6,7,3]
            d = d.transpose(0, 1, 2, 4, 5, 6, 7, 3)
            temp_num_echo = temp_num_echo * enc_n_receivers
            d = d.reshape(
                2, xsize, ysize // n_segments, pvm_n_echo_images,
                zsize, n_segments, temp_num_echo, order="F",
            )
        else:
            xsize = len(data) // (2 * zsize * ysize * temp_num_echo * pvm_n_echo_images)
            d = data.reshape(
                2, xsize, ysize // n_segments, pvm_n_echo_images,
                zsize, n_segments, temp_num_echo, order="F",
            )

        # Conditional permute based on ndims
        if d.ndim == 6:
            d = d.transpose(0, 1, 2, 4, 5, 3)
        elif d.ndim == 7:
            d = d.transpose(0, 1, 2, 4, 5, 6, 3)

        kfield = d[0] + 1j * d[1]

        # Zero-fill
        trunc = matrix_size[0] - xsize
        if trunc > 0:
            kfield_zf = np.zeros(
                (matrix_size[0], matrix_size[1] // n_segments, matrix_size[2],
                 n_segments, pvm_n_echo_images * temp_num_echo),
                dtype=np.complex128,
            )
            half = _matlab_round(trunc / 2)
            kfield_zf = _apply_zero_fill_3plus(kfield, kfield_zf, trunc, half)
            kfield = kfield_zf
            kfield = kfield.reshape(
                matrix_size[0], matrix_size[1] // n_segments, matrix_size[2],
                n_segments, temp_num_echo, channel_num, pvm_n_echo_images,
            )

        # Flip 1:2:end (MATLAB 1-based) -> 0-based [0::2]
        for m in range(kfield.shape[-1]):
            for l_idx in range(kfield.shape[-2]):
                for k in range(temp_num_echo):
                    for g in range(n_segments):
                        for i in range(matrix_size[2]):
                            kfield[:, 0::2, i, g, k, l_idx, m] = \
                                kfield[::-1, 0::2, i, g, k, l_idx, m].copy()

        # Interleave segments
        kfield_full = np.zeros(
            (matrix_size[0], matrix_size[1], matrix_size[2],
             temp_num_echo, channel_num, pvm_n_echo_images),
            dtype=np.complex128,
        )
        for i in range(n_segments):
            kfield_full[:, i:n_segments:matrix_size[1] - n_segments + i, :, :, :] = \
                np.squeeze(kfield[:, :, :, i, :, :, :])
        return kfield_full
    else:
        # NSegments == 1
        return _parse_spen_multi_echo_even_single(
            data, xsize, ysize, zsize, temp_num_echo, channel_num,
            pvm_n_echo_images, enc_n_receivers, matrix_size,
        )


def _parse_spen_multi_echo_even_single(data, xsize, ysize, zsize,
                                       temp_num_echo, channel_num,
                                       pvm_n_echo_images, enc_n_receivers,
                                       matrix_size):
    """SPEN multi-echo, even segments, NSegments == 1."""
    if enc_n_receivers != 1:
        xsize = len(data) // (2 * zsize * ysize * temp_num_echo * pvm_n_echo_images) // enc_n_receivers
        d = data.reshape(
            2, xsize, ysize, enc_n_receivers, pvm_n_echo_images, zsize, temp_num_echo, order="F",
        )
        d = d.transpose(0, 1, 2, 4, 5, 6, 3)
        temp_num_echo = temp_num_echo * enc_n_receivers
        d = d.reshape(
            2, xsize, ysize, pvm_n_echo_images, zsize, temp_num_echo, order="F",
        )
    else:
        xsize = len(data) // (2 * zsize * ysize * temp_num_echo * pvm_n_echo_images)
        d = data.reshape(
            2, xsize, ysize, pvm_n_echo_images, zsize, temp_num_echo, order="F",
        )

    d = d.transpose(0, 1, 2, 4, 5, 3)
    d = d[:, :xsize, :, :, :, :]
    kfield = d[0] + 1j * d[1]
    kfield = kfield.reshape(matrix_size[0], matrix_size[1], matrix_size[2],
                            temp_num_echo, channel_num)

    # Flip 2:2:end -> 0-based [1::2]
    for k in range(temp_num_echo):
        for i in range(matrix_size[2]):
            kfield[:, 1::2, i, k] = kfield[::-1, 1::2, i, k].copy()

    return kfield


# ─── SPEN Multi-Echo (PVM_NEchoImages > 1), Odd NSegments ───────────────────

def _parse_spen_multi_echo_odd(data, xsize, ysize, zsize, n_segments,
                               temp_num_echo, channel_num, pvm_n_echo_images,
                               enc_n_receivers, matrix_size):
    """SPEN multi-echo with odd number of segments."""
    if n_segments > 1:
        if enc_n_receivers != 1:
            xsize = len(data) // (2 * zsize * ysize * temp_num_echo * pvm_n_echo_images) // enc_n_receivers
            d = data.reshape(
                2, xsize, ysize // n_segments, enc_n_receivers,
                pvm_n_echo_images, zsize, n_segments, temp_num_echo, order="F",
            )
            d = d.transpose(0, 1, 2, 4, 5, 6, 7, 3)
            temp_num_echo = temp_num_echo * enc_n_receivers
            d = d.reshape(
                2, xsize, ysize // n_segments, pvm_n_echo_images,
                zsize, n_segments, temp_num_echo, order="F",
            )
        else:
            xsize = len(data) // (2 * zsize * ysize * temp_num_echo * pvm_n_echo_images)
            d = data.reshape(
                2, xsize, ysize // n_segments, pvm_n_echo_images,
                zsize, n_segments, temp_num_echo, order="F",
            )

        if d.ndim == 6:
            d = d.transpose(0, 1, 2, 4, 5, 3)
        elif d.ndim == 7:
            d = d.transpose(0, 1, 2, 4, 5, 6, 3)

        kfield = d[0] + 1j * d[1]

        # Zero-fill
        trunc = matrix_size[0] - xsize
        if trunc > 0:
            kfield_zf = np.zeros(
                (matrix_size[0], matrix_size[1] // n_segments, matrix_size[2],
                 n_segments, pvm_n_echo_images * temp_num_echo),
                dtype=np.complex128,
            )
            half = _matlab_round(trunc / 2)
            kfield_zf = _apply_zero_fill_3plus(kfield, kfield_zf, trunc, half)
            kfield = kfield_zf
            kfield = kfield.reshape(
                matrix_size[0], matrix_size[1] // n_segments, matrix_size[2],
                n_segments, temp_num_echo, channel_num, pvm_n_echo_images,
            )

        # Flip: odd segments -> 2:2:end [1::2], even segments -> 1:2:end [0::2]
        for m in range(kfield.shape[-1]):
            for l_idx in range(kfield.shape[-2]):
                for k in range(temp_num_echo):
                    for g in range(n_segments):
                        if (g + 1) % 2 == 1:  # odd segment (1-based)
                            for i in range(matrix_size[2]):
                                kfield[:, 1::2, i, g, k, l_idx, m] = \
                                    kfield[::-1, 1::2, i, g, k, l_idx, m].copy()
                        else:  # even segment (1-based)
                            for i in range(matrix_size[2]):
                                kfield[:, 0::2, i, g, k, l_idx, m] = \
                                    kfield[::-1, 0::2, i, g, k, l_idx, m].copy()

        # Interleave segments
        kfield_full = np.zeros(
            (matrix_size[0], matrix_size[1], matrix_size[2],
             temp_num_echo, channel_num, pvm_n_echo_images),
            dtype=np.complex128,
        )
        for i in range(n_segments):
            kfield_full[:, i:n_segments:matrix_size[1] - n_segments + i, :, :, :] = \
                np.squeeze(kfield[:, :, :, i, :, :, :])
        return kfield_full
    else:
        # NSegments == 1, odd
        return _parse_spen_multi_echo_odd_single(
            data, xsize, ysize, zsize, temp_num_echo, channel_num,
            pvm_n_echo_images, enc_n_receivers, matrix_size,
        )


def _parse_spen_multi_echo_odd_single(data, xsize, ysize, zsize,
                                      temp_num_echo, channel_num,
                                      pvm_n_echo_images, enc_n_receivers,
                                      matrix_size):
    """SPEN multi-echo, odd segments, NSegments == 1."""
    if enc_n_receivers != 1:
        xsize = len(data) // (2 * zsize * ysize * temp_num_echo * pvm_n_echo_images) // enc_n_receivers
        d = data.reshape(
            2, xsize, ysize, enc_n_receivers, pvm_n_echo_images, zsize, temp_num_echo, order="F",
        )
        d = d.transpose(0, 1, 2, 4, 5, 6, 3)
        temp_num_echo = temp_num_echo * enc_n_receivers
        d = d.reshape(
            2, xsize, ysize, pvm_n_echo_images, zsize, temp_num_echo, order="F",
        )
    else:
        xsize = len(data) // (2 * zsize * ysize * temp_num_echo * pvm_n_echo_images)
        d = data.reshape(
            2, xsize, ysize, pvm_n_echo_images, zsize, temp_num_echo, order="F",
        )

    if d.ndim == 5:
        d = d.transpose(0, 1, 2, 4, 3)
    elif d.ndim == 6:
        d = d.transpose(0, 1, 2, 4, 5, 3)

    kfield = d[0] + 1j * d[1]

    # Zero-fill
    trunc = matrix_size[0] - xsize
    if trunc >= 0:
        kfield_zf = np.zeros(
            (matrix_size[0], matrix_size[1], matrix_size[2],
             temp_num_echo, pvm_n_echo_images, channel_num),
            dtype=np.complex128,
        )
        half = _matlab_round(trunc / 2)
        if trunc % 2 != 0:
            target = kfield_zf[half - 1:matrix_size[0] - half, :, :, :, :]
            target[...] = _expand_trailing_singletons(kfield, target.ndim)
        else:
            target = kfield_zf[half:matrix_size[0] - half, :, :, :, :]
            target[...] = _expand_trailing_singletons(kfield, target.ndim)
        kfield = kfield_zf
    else:
        ms = list(d.shape[1:])
        matrix_size = ms[:]
        if len(matrix_size) < 3:
            matrix_size.append(1)

    kfield = kfield.reshape(matrix_size[0], matrix_size[1], matrix_size[2],
                            temp_num_echo, pvm_n_echo_images, channel_num)

    # Flip 2:2:end -> 0-based [1::2]
    for m in range(channel_num):
        for k in range(temp_num_echo):
            for i in range(matrix_size[2]):
                for l in range(pvm_n_echo_images):
                    kfield[:, 1::2, i, k, l, m] = \
                        kfield[::-1, 1::2, i, k, l, m].copy()

    return kfield


# ─── Non-T2 EPI (PVM_NEchoImages == 1), Even NSegments ──────────────────────

def _parse_nont2_even(data, xsize, ysize, zsize, n_segments,
                      temp_num_echo, channel_num,
                      enc_n_receivers, matrix_size):
    """Non-T2 EPI with even number of segments."""
    if n_segments > 1:
        if enc_n_receivers != 1:
            xsize = len(data) // (2 * zsize * ysize * temp_num_echo) // enc_n_receivers
            d = data.reshape(
                2, xsize, ysize // n_segments, enc_n_receivers,
                zsize, n_segments, temp_num_echo, order="F",
            )
            d = d.transpose(0, 1, 2, 4, 5, 6, 3)
            temp_num_echo = temp_num_echo * enc_n_receivers
            d = d.reshape(
                2, xsize, ysize // n_segments, zsize, n_segments, temp_num_echo, order="F",
            )
        else:
            xsize = len(data) // (2 * zsize * ysize * temp_num_echo)
            d = data.reshape(
                2, xsize, ysize // n_segments, zsize, n_segments, temp_num_echo, order="F",
            )

        d = d[:, :xsize, :, :, :, :]
        kfield = d[0] + 1j * d[1]

        # Zero-fill
        trunc = matrix_size[0] - xsize
        if trunc >= 0:
            kfield_zf = np.zeros(
                (matrix_size[0], matrix_size[1] // n_segments, matrix_size[2],
                 n_segments * temp_num_echo),
                dtype=np.complex128,
            )
            half = _matlab_round(trunc / 2)
            if kfield.ndim == 4:
                if trunc % 2 != 0:
                    kfield_zf[half - 1:matrix_size[0] - half, :, :, :] = kfield
                else:
                    kfield_zf[half:matrix_size[0] - half, :, :, :] = kfield
            kfield = kfield_zf
        else:
            ms = list(d.shape[1:])
            matrix_size = ms[:]

        kfield = kfield.reshape(
            matrix_size[0], matrix_size[1] // n_segments, matrix_size[2],
            n_segments, temp_num_echo, channel_num,
        )

        # Flip 1:2:end -> 0-based [0::2]
        for k in range(temp_num_echo):
            for g in range(n_segments):
                for i in range(matrix_size[2]):
                    kfield[:, 0::2, i, g, k] = \
                        kfield[::-1, 0::2, i, g, k].copy()

        # Interleave segments
        kfield_full = np.zeros(
            (matrix_size[0], matrix_size[1], matrix_size[2],
             temp_num_echo, channel_num),
            dtype=np.complex128,
        )
        for i in range(n_segments):
            kfield_full[:, i:n_segments:matrix_size[1] - n_segments + i, :, :, :] = \
                np.squeeze(kfield[:, :, :, i, :, :])
        return kfield_full
    else:
        # NSegments == 1
        if enc_n_receivers != 1:
            xsize = len(data) // (2 * zsize * ysize * temp_num_echo) // enc_n_receivers
            d = data.reshape(
                2, xsize, ysize, enc_n_receivers, zsize, temp_num_echo, order="F",
            )
            d = d.transpose(0, 1, 2, 4, 5, 3)
            temp_num_echo = temp_num_echo * enc_n_receivers
            d = d.reshape(
                2, xsize, ysize, zsize, temp_num_echo, order="F",
            )
        else:
            xsize = len(data) // (2 * zsize * ysize * temp_num_echo)
            d = data.reshape(
                2, xsize, zsize, ysize, temp_num_echo, order="F",
            )

        d = d.transpose(0, 1, 3, 2, 4)
        d = d[:, :xsize, :, :, :]
        kfield = d[0] + 1j * d[1]
        kfield = kfield.reshape(matrix_size[0], matrix_size[1], matrix_size[2],
                                temp_num_echo, channel_num)

        # Flip 1:2:end -> 0-based [0::2]
        for k in range(temp_num_echo):
            for i in range(matrix_size[2]):
                kfield[:, 0::2, i, k] = kfield[::-1, 0::2, i, k].copy()

        return kfield


# ─── Non-T2 EPI (PVM_NEchoImages == 1), Odd NSegments ───────────────────────

def _parse_nont2_odd(data, xsize, ysize, zsize, n_segments,
                     temp_num_echo, channel_num,
                     enc_n_receivers, matrix_size):
    """Non-T2 EPI with odd number of segments."""
    if n_segments > 1:
        if enc_n_receivers != 1:
            xsize = len(data) // (2 * zsize * ysize * temp_num_echo) // enc_n_receivers
            d = data.reshape(
                2, xsize, ysize // n_segments, enc_n_receivers,
                zsize, n_segments, temp_num_echo, order="F",
            )
            d = d.transpose(0, 1, 2, 4, 5, 6, 3)
            temp_num_echo = temp_num_echo * enc_n_receivers
            d = d.reshape(
                2, xsize, ysize // n_segments, zsize, n_segments, temp_num_echo, order="F",
            )
        else:
            xsize = len(data) // (2 * zsize * ysize * temp_num_echo)
            d = data.reshape(
                2, xsize, ysize // n_segments, zsize, n_segments, temp_num_echo, order="F",
            )

        kfield = d[0] + 1j * d[1]

        # Zero-fill
        trunc = matrix_size[0] - xsize
        if trunc >= 0:
            kfield_zf = np.zeros(
                (matrix_size[0], matrix_size[1] // n_segments, matrix_size[2],
                 n_segments * temp_num_echo),
                dtype=np.complex128,
            )
            half = _matlab_round(trunc / 2)
            if kfield.ndim == 4:
                if trunc % 2 != 0:
                    kfield_zf[half - 1:matrix_size[0] - half, :, :, :] = kfield
                else:
                    kfield_zf[half:matrix_size[0] - half, :, :, :] = kfield
            kfield = kfield_zf
        else:
            ms = list(d.shape[1:])
            matrix_size = ms[:]

        kfield = kfield.reshape(
            matrix_size[0], matrix_size[1] // n_segments, matrix_size[2],
            n_segments, temp_num_echo, channel_num,
        )

        # Flip: odd segments -> 2:2:end [1::2], even segments -> 1:2:end [0::2]
        for k in range(temp_num_echo):
            for g in range(n_segments):
                if (g + 1) % 2 == 1:  # odd segment
                    for i in range(matrix_size[2]):
                        kfield[:, 1::2, i, g, k] = \
                            kfield[::-1, 1::2, i, g, k].copy()
                else:  # even segment
                    for i in range(matrix_size[2]):
                        kfield[:, 0::2, i, g, k] = \
                            kfield[::-1, 0::2, i, g, k].copy()

        # Interleave segments
        kfield_full = np.zeros(
            (matrix_size[0], matrix_size[1], matrix_size[2],
             temp_num_echo, channel_num),
            dtype=np.complex128,
        )
        for i in range(n_segments):
            kfield_full[:, i:n_segments:matrix_size[1] - n_segments + i, :, :, :] = \
                np.squeeze(kfield[:, :, :, i, :, :])
        return kfield_full
    else:
        # NSegments == 1, odd
        return _parse_nont2_odd_single(
            data, xsize, ysize, zsize, temp_num_echo, channel_num,
            enc_n_receivers, matrix_size,
        )


def _parse_nont2_odd_single(data, xsize, ysize, zsize,
                            temp_num_echo, channel_num,
                            enc_n_receivers, matrix_size):
    """Non-T2 EPI, odd segments, NSegments == 1."""
    if enc_n_receivers != 1:
        xsize = len(data) // (2 * zsize * ysize * temp_num_echo) // enc_n_receivers
        d = data.reshape(
            2, xsize, ysize, enc_n_receivers, zsize, temp_num_echo, order="F",
        )
        d = d.transpose(0, 1, 2, 4, 5, 3)
        temp_num_echo = temp_num_echo * enc_n_receivers
        d = d.reshape(
            2, xsize, ysize, zsize, temp_num_echo, order="F",
        )
    else:
        xsize = len(data) // (2 * zsize * ysize * temp_num_echo)
        d = data.reshape(
            2, xsize, ysize, zsize, temp_num_echo, order="F",
        )

    d = d[:, :xsize, :, :, :]
    kfield = d[0] + 1j * d[1]

    # Zero-fill
    trunc = matrix_size[0] - xsize
    if trunc >= 0:
        kfield_zf = np.zeros(
            (matrix_size[0], matrix_size[1], matrix_size[2],
             temp_num_echo, channel_num),
            dtype=np.complex128,
        )
        half = _matlab_round(trunc / 2)
        if trunc % 2 != 0:
            target = kfield_zf[half - 1:matrix_size[0] - half, :, :, :]
            target[...] = _expand_trailing_singletons(kfield, target.ndim)
        else:
            target = kfield_zf[half:matrix_size[0] - half, :, :, :]
            target[...] = _expand_trailing_singletons(kfield, target.ndim)
        kfield = kfield_zf
    else:
        ms = list(d.shape[1:])
        matrix_size = ms[:]
        if len(matrix_size) < 3:
            matrix_size.append(1)

    kfield = kfield.reshape(matrix_size[0], matrix_size[1], matrix_size[2],
                            temp_num_echo, channel_num)

    # Flip 2:2:end -> 0-based [1::2]
    for k in range(temp_num_echo):
        for i in range(matrix_size[2]):
            kfield[:, 1::2, i, k] = kfield[::-1, 1::2, i, k].copy()

    return kfield


def _apply_zero_fill_3plus(kfield, kfield_zf, trunc, half):
    """Apply zero-filling for arrays with 3+ dimensions."""
    nd = kfield.ndim
    if trunc % 2 != 0:
        sl_src = [slice(None)] * nd
        sl_dest = [slice(None)] * nd
        sl_src[0] = slice(half - 1, kfield_zf.shape[0] - half)
        sl_dest[0] = slice(half - 1, kfield_zf.shape[0] - half)
        kfield_zf[tuple(sl_dest)] = kfield
    else:
        sl_dest = [slice(None)] * nd
        sl_dest[0] = slice(half, kfield_zf.shape[0] - half)
        kfield_zf[tuple(sl_dest)] = kfield
    return kfield_zf
