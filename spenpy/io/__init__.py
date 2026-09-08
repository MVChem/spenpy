from pathlib import Path

from .bruker import protocol_from_bruker, read_bruker
from .files import load_image, load_mat_array, read_mat, read_npz, save_acquisition
from .siemens import read_siemens
from .simulation import load_simulation, save_simulation


def read_acquisition(path, **kwargs):
    path = Path(path)
    if path.is_dir():
        return read_bruker(path, **kwargs)
    if path.suffix.lower() == ".dat":
        return read_siemens(path, **kwargs)
    if path.suffix.lower() == ".mat":
        return read_mat(path, **kwargs)
    if path.suffix.lower() == ".npz":
        return read_npz(path, **kwargs)
    raise ValueError(f"Unsupported acquisition path: {path}")


__all__ = [
    "load_image",
    "load_mat_array",
    "load_simulation",
    "protocol_from_bruker",
    "read_acquisition",
    "read_bruker",
    "read_mat",
    "read_npz",
    "read_siemens",
    "save_acquisition",
    "save_simulation",
]
