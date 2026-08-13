"""Bruker ParaVision parameter file reader.

Ported from ReadPVParam.m -- reads parameters from method/acqp/reco files.
"""

import os
import re
from pathlib import Path

# Cache for parsed files
_cache: dict[str, str] = {}


def _parse_bruker_file(filepath: str) -> str:
    """Read and concatenate all lines from a Bruker parameter file."""
    try:
        with open(filepath, "r") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def _param_dir(path: str) -> str:
    """Return the Bruker parameter directory for a scan dir or file path."""
    p = Path(path)
    if p.is_dir():
        return str(p)
    return str(p.parent) if str(p.parent) else "."


def _load_param_files(fid_path: str, two_dseq_path: str = "") -> str:
    """Load and concatenate method, acqp, and optionally reco files."""
    cache_key = fid_path + "|" + two_dseq_path
    if cache_key in _cache:
        return _cache[cache_key]

    dir_path = _param_dir(fid_path)

    content = ""
    for fname in ["method", "acqp"]:
        fpath = os.path.join(dir_path, fname)
        content += _parse_bruker_file(fpath) + "\n"

    if two_dseq_path:
        reco_dir = _param_dir(two_dseq_path)
        content += _parse_bruker_file(os.path.join(reco_dir, "reco"))

    _cache[cache_key] = content
    return content


def _expand_repeated_values(value: str) -> str:
    """Expand Bruker JCAMP-DX shorthand like @11*(0) into 11 values."""

    def repl(match: re.Match) -> str:
        count = int(match.group(1))
        repeated_value = match.group(2).strip()
        return " ".join([repeated_value] * count)

    previous = None
    expanded = value
    while previous != expanded:
        previous = expanded
        expanded = re.sub(r"@(\d+)\*\(([^()]*)\)", repl, expanded)
    return expanded


def _numeric_or_string(value: str, p_dim: list[int] | None):
    """Convert a Bruker value string to MATLAB-like numeric arrays or strings."""
    value = _expand_repeated_values(value).strip()
    if "$$" in value:
        value = value[: value.index("$$")].strip()

    try:
        values = [float(v) for v in value.split()]
    except (ValueError, TypeError):
        return value.lower()

    if not values:
        return None

    def maybe_int(x: float):
        return int(x) if x == int(x) else x

    if not p_dim:
        if len(values) == 1:
            return maybe_int(values[0])
        return [maybe_int(v) for v in values]

    import numpy as np

    if len(p_dim) == 1:
        arr = np.array([maybe_int(v) for v in values], dtype=object)
        arr = np.squeeze(arr)
    elif len(p_dim) == 2:
        arr = np.array(values)
        arr = arr.reshape((p_dim[0], p_dim[1]), order="C")
        if np.all(arr == arr.astype(int)):
            arr = arr.astype(int)
        arr = np.squeeze(arr)
    else:
        arr = np.array(values)
        arr = arr.reshape(tuple(p_dim), order="C")
        if np.all(arr == arr.astype(int)):
            arr = arr.astype(int)
        arr = np.squeeze(arr)

    if getattr(arr, "ndim", 0) == 0:
        return maybe_int(float(arr))
    if arr.ndim == 1:
        return arr.tolist()
    return arr


def _extract_param_value(content: str, param_name: str):
    """Extract and parse one parameter from Bruker method/acqp/visu text."""
    pattern = re.compile(r"##\$" + re.escape(param_name.strip()) + r"\s*=", re.IGNORECASE)
    match = pattern.search(content)
    if not match:
        return None

    rest = content[match.end():]

    dim_match = re.match(r"\s*\(\s*(\d+(?:\s*,\s*\d+)*)\s*\)", rest)
    p_dim = None
    val_start = 0
    if dim_match:
        p_dim = [int(d.strip()) for d in dim_match.group(1).split(",")]
        val_start = dim_match.end()

    end_candidates = []
    for marker in ("\n##$", "\n$$ @vis"):
        pos = rest.find(marker, val_start)
        if pos != -1:
            end_candidates.append(pos)
    end_marker = min(end_candidates) if end_candidates else len(rest)

    val_str = rest[val_start:end_marker].strip()
    return _numeric_or_string(val_str, p_dim)


def read_pv_param(fid_path: str, param_name: str, two_dseq_path: str = ""):
    """Read a parameter value from Bruker ParaVision files.

    Args:
        fid_path: path to the fid/rawdata directory
        param_name: name of the parameter (case-insensitive)
        two_dseq_path: optional path to 2dseq directory for reco params

    Returns:
        Numeric value/array, string, or None if not found.
    """
    return _extract_param_value(_load_param_files(fid_path, two_dseq_path), param_name)
