#!/usr/bin/env python3
"""Run a minimal ideal xSPEN acquisition and reconstruction.

Example:
    python demo/18_minimal_xspen_simulation.py --output /tmp/xspen_demo.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from spenpy.sim import XSPENParameters, XSPENSimulator


def make_phantom(simulator: XSPENSimulator) -> torch.Tensor:
    """Create a small two-ellipse phantom on the simulator grid."""
    y = simulator.xspen_positions_cm[:, None]
    x = simulator.readout_positions_cm[None, :]
    first = torch.exp(-((y + 0.65) / 0.38).square() - ((x + 0.45) / 0.60).square())
    second = 0.7 * torch.exp(
        -((y - 0.65) / 0.27).square() - ((x - 0.55) / 0.42).square()
    )
    return first + second


def normalized_error(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(estimate - reference)
    denominator = torch.linalg.vector_norm(reference).clamp_min(1e-12)
    return float((numerator / denominator).item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/xspen_minimal_demo.png"),
        help="PNG summary path (default: /tmp/xspen_minimal_demo.png).",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.002,
        help="Relative complex receiver-noise RMS (default: 0.002).",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=1e-3,
        help="Relative Tikhonov reconstruction strength (default: 1e-3).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parameters = XSPENParameters(n_xspen=64, n_readout=64, r_value=64.0)
    simulator = XSPENSimulator(parameters)
    phantom = make_phantom(simulator)

    acquisition = simulator.acquire(phantom, noise_std=args.noise_std, seed=7)
    localized_from_raw = simulator.raw_to_localized(acquisition.raw_kspace)
    reconstruction = simulator.reconstruct(
        acquisition,
        regularization=args.regularization,
    )

    nrmse = normalized_error(phantom, reconstruction.abs())
    print(f"input shape       : {tuple(phantom.shape)}")
    print(f"raw k-space shape : {tuple(acquisition.raw_kspace.shape)}")
    print(f"reconstruction    : {tuple(reconstruction.shape)}")
    print(f"magnitude NRMSE   : {nrmse:.6f}")
    print(f"chirp bandwidth   : {parameters.chirp_bandwidth_hz:.3f} Hz")
    print(f"Gy / Gz           : {parameters.gy_gauss_per_cm:.6f} / "
          f"{parameters.gz_gauss_per_cm:.6f} G/cm")

    panels = [
        (phantom, "Input object", "gray"),
        (localized_from_raw.abs(), "Localized signal after RO IFFT", "gray"),
        (torch.log1p(acquisition.raw_kspace.abs()), "log(1 + |raw k-space|)", "magma"),
        (reconstruction.abs(), f"Reconstruction\nNRMSE={nrmse:.4f}", "gray"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), constrained_layout=True)
    for axis, (image, title, cmap) in zip(axes, panels):
        axis.imshow(image.detach().cpu(), cmap=cmap, origin="lower")
        axis.set_title(title)
        axis.set_axis_off()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(f"saved figure      : {args.output}")


if __name__ == "__main__":
    main()
