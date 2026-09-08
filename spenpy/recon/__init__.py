from .even_odd import reconstruct_even_odd_inva
from .hybrid_spen import (
    calc_hybrid_spen_matrices,
    estimate_hybrid_spen_shift,
    reconstruct_hybrid_spen,
)
from .phasemap import (
    PhaseMapInvAResult,
    reconstruct_phasemap_inva,
)
from .solvers import (
    ReconstructionResult,
    conjugate_gradient,
    data_consistency,
    reconstruct,
    rss,
)
from .super_resolution import calc_inva_matrices, reconstruct_simulated_inva


def reconstruct_bruker(scan_dir, acquisition=None, **kwargs):
    """Existing single-shot PV5/PV360 baseline, isolated from differentiable CG."""
    from .._legacy.recon.spen_recon import reconstruct_odd_segments

    if acquisition is not None:
        if any(k in kwargs for k in ("kfield", "input_stage")):
            raise ValueError(
                "Use acquisition OR the low-level kfield/input_stage arguments"
            )
        a = acquisition
        if "volume" in a.axes:
            if a.data.shape[a.axes.index("volume")] != 1:
                raise ValueError(
                    "Select one volume before legacy Bruker reconstruction"
                )
            a = a.select(volume=0)
        if "slice" in a.axes:
            if a.data.shape[a.axes.index("slice")] != 1:
                raise ValueError("Select one slice before legacy Bruker reconstruction")
            a = a.select(slice=0)
        a = a.permute("readout", "spen", "coil")
        kwargs["kfield"] = a.data.unsqueeze(2).unsqueeze(-1)
        kwargs["input_stage"] = a.stage
    return reconstruct_odd_segments(str(scan_dir), **kwargs)


__all__ = [
    "PhaseMapInvAResult",
    "ReconstructionResult",
    "calc_hybrid_spen_matrices",
    "calc_inva_matrices",
    "conjugate_gradient",
    "data_consistency",
    "estimate_hybrid_spen_shift",
    "reconstruct",
    "reconstruct_bruker",
    "reconstruct_even_odd_inva",
    "reconstruct_hybrid_spen",
    "reconstruct_phasemap_inva",
    "reconstruct_simulated_inva",
    "rss",
]
