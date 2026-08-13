"""Bruker ParaVision image data reader.

Ported from ImageDataObject.m + readBruker2dseq.m.
Reads pdata/N/2dseq binary image files with visu_pars metadata.
"""

import os
import numpy as np
from spenpy.bruker.param import _extract_param_value


def read_bruker_2dseq(pdata_dir: str) -> np.ndarray:
    """Read Bruker 2dseq processed image data.

    Equivalent to MATLAB:
        imageObj = ImageDataObject(fullfile(rare_dir, 'pdata', '1'));
        ImageData = squeeze(imageObj.data);

    Args:
        pdata_dir: path to pdata/N/ directory (e.g. .../10/pdata/1/)

    Returns:
        data: numpy array with shape matching VisuCoreSize + frame count.
              Complex if VisuCoreFrameType == 'COMPLEX_IMAGE'.
    """
    path_to_2dseq = os.path.join(pdata_dir, "2dseq")
    path_to_visu = os.path.join(pdata_dir, "visu_pars")

    if not os.path.exists(path_to_2dseq):
        raise FileNotFoundError(f"2dseq not found: {path_to_2dseq}")
    if not os.path.exists(path_to_visu):
        raise FileNotFoundError(f"visu_pars not found: {path_to_visu}")

    visu = _read_visu_params(path_to_visu)
    data = _read_2dseq_binary(path_to_2dseq, visu)
    return data


def _read_visu_params(visu_path: str) -> dict:
    """Parse key parameters from visu_pars file."""
    with open(visu_path, "r") as f:
        content = f.read()

    params = {}
    for name in [
        "VisuCoreSize",
        "VisuCoreFrameCount",
        "VisuCoreDim",
        "VisuCoreWordType",
        "VisuCoreByteOrder",
        "VisuCoreFrameType",
        "VisuCoreDataSlope",
        "VisuCoreDataOffs",
        "VisuCoreDiskSliceOrder",
        "VisuCoreExtent",
    ]:
        value = _extract_param_value(content, name)
        if value is not None:
            params[name] = value
    return params


def _read_2dseq_binary(filepath: str, visu: dict) -> np.ndarray:
    """Read 2dseq binary file with proper dtype, endian, shape, and scaling."""

    # Determine numpy dtype from VisuCoreWordType
    word_type = str(visu.get("VisuCoreWordType", "_32BIT_SGN_INT")).upper()
    dtype_map = {
        "_32BIT_SGN_INT": "<i4",
        "_16BIT_SGN_INT": "<i2",
        "_32BIT_FLOAT": "<f4",
        "_8BIT_UNSGN_INT": "u1",
    }
    dtype_str = dtype_map.get(word_type, "<i4")

    byte_order = str(visu.get("VisuCoreByteOrder", "littleEndian")).lower()
    if byte_order == "bigendian":
        dtype_str = dtype_str.replace("<", ">")
    else:
        dtype_str = dtype_str.replace(">", "<")

    core_size = visu.get("VisuCoreSize", [256, 256])
    ndim = visu.get("VisuCoreDim", 2)
    frame_count = visu.get("VisuCoreFrameCount", 1)

    frame_type = str(visu.get("VisuCoreFrameType", "MAGNITUDE_IMAGE")).upper()
    is_complex = frame_type == "COMPLEX_IMAGE"

    # Read flat binary data
    data_flat = np.fromfile(filepath, dtype=dtype_str)

    # MATLAB: fread(FileID, [VisuCoreSize(1), inf], format)
    # Reads column-major: first column = all rows for frame 0
    ncols = len(data_flat) // core_size[0]
    data_2d = data_flat[:core_size[0] * ncols].reshape(core_size[0], ncols, order="F")

    if is_complex:
        half = data_2d.shape[1] // 2
        real_part = data_2d[:, :half]
        imag_part = data_2d[:, half:]
        data_2d = real_part.astype(np.complex128) + 1j * imag_part.astype(np.complex128)
        n_frames = frame_count
    else:
        data_2d = data_2d.astype(np.float64)
        n_frames = frame_count

    # Reshape to full dimensions
    # The remaining columns after core_size[0] = product(core_size[1:]) * n_frames
    remaining = data_2d.shape[1]

    if len(core_size) >= 4:
        expected = int(np.prod(core_size[1:4]) * n_frames)
        if remaining == expected:
            data_2d = data_2d.reshape(
                core_size[0], core_size[1], core_size[2], core_size[3], n_frames, order="F",
            )
        else:
            data_2d = data_2d.reshape(core_size[0], remaining, order="F")
    elif len(core_size) >= 3:
        expected = int(np.prod(core_size[1:3]) * n_frames)
        if remaining == expected:
            data_2d = data_2d.reshape(
                core_size[0], core_size[1], core_size[2], n_frames, order="F",
            )
        else:
            data_2d = data_2d.reshape(core_size[0], remaining, order="F")
    elif len(core_size) >= 2:
        expected = int(core_size[1] * n_frames)
        if remaining == expected:
            data_2d = data_2d.reshape(core_size[0], core_size[1], n_frames, order="F")
        else:
            data_2d = data_2d.reshape(core_size[0], remaining, order="F")
    else:
        # 1D data
        data_2d = data_2d.reshape(core_size[0], n_frames, order="F")
        if n_frames == 1:
            data_2d = data_2d.squeeze(axis=-1)

    # Apply slope/offset scaling
    slope = visu.get("VisuCoreDataSlope", None)
    offs = visu.get("VisuCoreDataOffs", None)

    if slope is not None:
        slope = np.atleast_1d(slope)
    if offs is not None:
        offs = np.atleast_1d(offs)

    if slope is not None and len(slope) == n_frames and n_frames > 1:
        slope_arr = np.array(slope, dtype=data_2d.dtype)
        if offs is not None and len(offs) == n_frames:
            offs_arr = np.array(offs, dtype=data_2d.dtype)
        else:
            offs_arr = np.zeros(n_frames, dtype=data_2d.dtype)
        nd = data_2d.ndim
        slope_shape = [1] * (nd - 1) + [n_frames]
        data_2d = data_2d * slope_arr.reshape(slope_shape) + offs_arr.reshape(slope_shape)
    elif slope is not None and len(slope) == 1:
        # Single slope applies to all data
        data_2d = data_2d * slope[0]
        if offs is not None and len(offs) == 1:
            data_2d = data_2d + offs[0]

    # Handle disk slice order reversal
    disk_order = str(visu.get("VisuCoreDiskSliceOrder", ""))
    if "reverse" in disk_order.lower() and ndim == 3:
        if data_2d.ndim >= 3:
            data_2d = np.flip(data_2d, axis=2)

    return data_2d
