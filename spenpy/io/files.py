"""Image inputs, MATLAB results and portable acquisition interchange."""

import json
from pathlib import Path

import numpy as np
import torch

from ..core import Acquisition


def load_mat_array(path, variable):
    """Load v5/v7.3 arrays in MATLAB logical axis order, preserving complex data."""
    import h5py

    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as handle:
            data = np.asarray(handle[variable])
        if data.dtype.names and {"real", "imag"}.issubset(data.dtype.names):
            data = data["real"] + 1j * data["imag"]
        data = data.transpose(tuple(reversed(range(data.ndim))))
    else:
        from scipy.io import loadmat

        data = loadmat(path, variable_names=[variable])[variable]
    return torch.from_numpy(np.ascontiguousarray(data))


def read_mat(
    path, *, variable="SignalFixedPostROFFTPostSR", axes=None, stage=None, device="cpu"
):
    """Read xSPEN reconstruction exports; known reconstructed variables are labelled.

    Custom variables require explicit axes and stage. A reconstructed coil
    image is never advertised as raw acquisition data.
    """
    data = load_mat_array(path, variable).to(device)
    if variable.startswith("SignalFixedPostROFFTPostSR"):
        defaults = ("spen", "readout", "coil", "slice", "volume", "set")
        if data.ndim not in (3, 4, 5, 6):
            raise ValueError(
                f"Unexpected reconstructed coil dimensions: {tuple(data.shape)}"
            )
        axes = axes or defaults[: data.ndim]
        stage = stage or "coil_image"
    elif variable in ("Smat", "SmatBeforePhaseMapInvA"):
        axes = axes or ("spen", "readout", "slice", "volume")[: data.ndim]
        stage = stage or "magnitude_image"
    if axes is None or stage is None:
        raise ValueError("Provide axes and stage for a custom MAT variable")
    return Acquisition(
        data,
        tuple(axes),
        stage,
        {"source": str(Path(path).resolve()), "variable": variable},
    )


def load_image(
    path,
    *,
    variable=None,
    slice_index=None,
    slice_axis=-1,
    normalize=False,
    device="cpu",
):
    """Read a 2D MRI magnitude/complex image without implicit spatial resizing.

    NIfTI slicing is explicit and does not reorient voxel axes. NPZ/MAT
    require a variable name. PNG/JPEG are converted to grayscale magnitude.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        data = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        if variable is None:
            raise ValueError("NPZ image input requires variable")
        with np.load(path, allow_pickle=False) as f:
            data = f[variable]
    elif suffix == ".mat":
        if variable is None:
            raise ValueError("MAT image input requires variable")
        data = load_mat_array(path, variable)
    elif suffix == ".nii" or path.name.lower().endswith(".nii.gz"):
        import nibabel as nib

        data = np.asarray(nib.load(path).dataobj)
    elif suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        from PIL import Image

        with Image.open(path) as im:
            data = np.asarray(im.convert("F")).copy()
    else:
        raise ValueError(f"Unsupported image type: {path.name}")
    image = torch.as_tensor(np.array(data, copy=True)).to(device)
    if slice_index is not None:
        image = image.select(slice_axis, slice_index)
    if image.ndim != 2:
        raise ValueError(f"Select a 2D image explicitly; got {tuple(image.shape)}")
    image = image.to(torch.complex64 if image.is_complex() else torch.float32)
    if not torch.isfinite(image).all():
        raise ValueError("Image contains nonfinite values")
    if normalize:
        image = image / image.abs().amax().clamp_min(torch.finfo(image.real.dtype).eps)
    return image


def _json_default(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().resolve_conj().numpy()
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {"real": value.real.tolist(), "imag": value.imag.tolist()}
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def save_acquisition(acquisition, path):
    """Save named axes/stage/metadata and complex values without pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "axes": acquisition.axes,
        "stage": acquisition.stage,
        "metadata": acquisition.metadata,
    }
    with path.open("wb") as f:
        np.savez_compressed(
            f,
            data=acquisition.data.detach().cpu().resolve_conj().numpy(),
            header=np.asarray(json.dumps(header, default=_json_default)),
        )
    return path


def read_npz(path, *, device="cpu"):
    with np.load(path, allow_pickle=False) as f:
        header = json.loads(str(f["header"].item()))
        data = torch.from_numpy(np.array(f["data"], copy=True)).to(device)
    return Acquisition(data, tuple(header["axes"]), header["stage"], header["metadata"])
