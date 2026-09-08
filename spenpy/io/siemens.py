"""Siemens Twix decoding using twixtools, with explicit preprocessing flags."""

from pathlib import Path

import numpy as np
import torch

from ..core import Acquisition


def read_siemens(
    path,
    *,
    regrid=True,
    remove_os=True,
    ignore_segments=True,
    sort_slices=True,
    device="cpu",
):
    """Read SPEN/xSPEN .dat without a processed-image calibration target.

    The usual output axes are [spen,readout,coil,slice,volume]. If both Set
    and Rep vary they remain separate named axes. Other non-singleton counters
    are also retained. Repetitions/averages are never implicitly combined.
    Slice sorting follows the existing MATLAB z-coordinate ordering.
    """
    try:
        import twixtools
    except ImportError as exc:
        raise ImportError("Install spenpy[io] to read Siemens .dat files") from exc
    source = Path(path).expanduser().resolve()
    scans = twixtools.read_twix(
        str(source), parse_geometry=False, parse_pmu=False, verbose=False
    )
    if not scans:
        raise ValueError("No Siemens measurements found")
    mapped = twixtools.map_twix(scans[-1], verbose=False)
    if "image" not in mapped:
        raise ValueError("No Siemens image acquisitions found")
    array = mapped["image"]
    for name in array.flags["average"]:
        array.flags["average"][name] = False
    if ignore_segments:
        # Segment collapse is valid for sparse line allocation, not repeated
        # acquisitions at the same counters. Avoid averaging such repeats away.
        seen = set()
        names = [n for n in array.dim_order[:-2] if n != "Seg"]
        for mdb in array.mdb_list:
            key = tuple(int(getattr(mdb.mdh.Counter, n)) for n in names)
            if key in seen:
                raise ValueError(
                    "Repeated counters across segments; use ignore_segments=False"
                )
            seen.add(key)
        array.flags["average"]["Seg"] = True
    array.flags["remove_os"] = remove_os
    array.flags["regrid"] = regrid
    array.flags["squeeze_singletons"] = False
    array.flags["squeeze_ave_dims"] = False
    original_shape = tuple(array.base_size[name].item() for name in array.dim_order)
    size = dict(zip(array.dim_order, array.shape))
    core = ["Lin", "Col", "Cha", "Sli"]
    varying_volume = [d for d in ("Rep", "Set") if size[d] > 1]
    volume = (
        varying_volume[0]
        if len(varying_volume) == 1
        else ("Rep" if not varying_volume else None)
    )
    keep = core + ([volume] if volume else ["Rep", "Set"])
    keep += [d for d in array.dim_order if size[d] > 1 and d not in keep]
    selection = tuple(slice(None) if d in keep else 0 for d in array.dim_order)
    data = array[selection]
    remaining = [d for d in array.dim_order if d in keep]
    data = data.transpose(*(remaining.index(d) for d in keep))
    names = {
        "Lin": "spen",
        "Col": "readout",
        "Cha": "coil",
        "Sli": "slice",
        "Rep": "repetition",
        "Set": "set",
        "Seg": "segment",
        "Ave": "average",
        "Eco": "echo",
        "Par": "partition",
        "Phs": "phase_cycle",
    }
    if volume:
        names[volume] = "volume"
    axes = tuple(names.get(d, d.lower()) for d in keep)
    positions = {}
    for mdb in array.mdb_list:
        sl = int(mdb.mdh.Counter.Sli)
        if sl not in positions:
            pos = mdb.mdh.SliceData.SlicePos
            positions[sl] = [float(pos.Sag), float(pos.Cor), float(pos.Tra)]
    order = list(range(data.shape[3]))
    if sort_slices:
        if len(positions) != data.shape[3]:
            raise ValueError("Cannot sort incomplete slice-position metadata")
        order = sorted(order, key=lambda s: positions[s][2])
        data = np.take(data, order, axis=3)
    yaps = mapped["hdr"].get("MeasYaps", {})
    sequence = str(yaps.get("tSequenceFileName", ""))
    metadata = {
        "source": str(source),
        "vendor": "siemens",
        "sequence_name": sequence,
        "regridded": bool(regrid and array.rs_traj is not None),
        "remove_os": remove_os,
        "ignore_segments": ignore_segments,
        "raw_shape": original_shape,
        "raw_axes": array.dim_order,
        "slice_order": order,
        "slice_positions_lps_mm": [positions[s] for s in order],
        "volume_counter": volume,
        "readout_reflections_corrected": True,
        "calibrated_to_simulator": False,
    }
    slices = yaps.get("sSliceArray", {}).get("asSlice", [])
    if slices:
        metadata["phase_fov_m"] = float(slices[0].get("dPhaseFOV", 0)) / 1000
        metadata["readout_fov_m"] = float(slices[0].get("dReadoutFOV", 0)) / 1000
    if "esrs_hyb_spen_Diff2" in sequence:
        free = yaps.get("sWiPMemBlock", {}).get("alFree", [])
        if len(free) > 12:
            metadata["r_value"] = float(free[12])
            metadata["r_value_source"] = "MeasYaps.sWiPMemBlock.alFree[12]"
        from ..core import HybridSPENDiff2Protocol

        try:
            metadata["protocol"] = HybridSPENDiff2Protocol.from_siemens_header(
                yaps
            ).to_dict()
            metadata["protocol_source"] = "MeasYaps and esrs_hyb_spen_Diff2 WIP mapping"
        except ValueError as exc:
            # Preserve readable raw data even for an incomplete header. Never
            # replace missing scanner parameters by the MID253 defaults.
            metadata["protocol_parse_error"] = str(exc)
    stage = "regridded_kspace" if metadata["regridded"] else "sorted_samples"
    return Acquisition(
        torch.from_numpy(np.ascontiguousarray(data.astype(np.complex64))).to(device),
        axes,
        stage,
        metadata,
    )
