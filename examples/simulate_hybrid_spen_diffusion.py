"""Hybrid SPEN-Diff2: four simulated diffusion volumes, RO-only versus SR.

Uses the MID253 nominal protocol and an explicitly synthetic diffusion tensor.
No claim of reproducing that participant's anatomy, diffusion or scanner errors.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from compare_reconstructions import reference_mri

from spenpy import HybridSPENDiff2Protocol
from spenpy.recon import reconstruct_simulated_inva
from spenpy.sim import simulate_hybrid_spen_diffusion


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs"
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    image, _ = reference_mri(args.device)
    protocol = HybridSPENDiff2Protocol()
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, image.shape[0], device=image.device),
        torch.linspace(-1, 1, image.shape[1], device=image.device),
        indexing="ij",
    )
    # A smooth positive diagonal tensor for a controlled example. Values and
    # orientations are prescribed, not estimated from the MRHead magnitude.
    diagonal = torch.stack(
        (
            0.6e-3 + 0.4e-3 * x.square(),
            0.9e-3 + 0.4e-3 * y.square(),
            1.2e-3 + 0.3e-3 * (x * y).square(),
        ),
        dim=-1,
    )
    sample = simulate_hybrid_spen_diffusion(
        image,
        protocol,
        diffusion_tensor_mm2_s=torch.diag_embed(diagonal),
        snr_db=35,
        seed=42,
        even_odd_constant_rad=0.7,
        even_odd_linear_rad=0.3,
    )
    before, after = [], []
    for volume in range(4):
        result = reconstruct_simulated_inva(
            sample.acquisition.select(volume=volume), sample.operator
        )
        before.append(result.ro_image.abs().square().sum(-1).sqrt().cpu().numpy())
        after.append(result.magnitude.cpu().numpy())
    labels = sample.acquisition.metadata["diffusion"]["volume_labels"]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.5), layout="constrained")
    for row, (frames, stage) in enumerate(
        [(before, "RO FFT + RSS"), (after, "PhaseMap + InvA")]
    ):
        values = np.stack(frames)
        upper = np.percentile(values[values > 0], 99.5)
        for i, frame in enumerate(frames):
            axes[row, i].imshow(
                frame, cmap="gray", vmin=0, vmax=upper, interpolation="nearest"
            )
            axes[row, i].set_title(f"{labels[i]}\n{stage}")
            axes[row, i].axis("off")
    fig.suptitle(
        "Hybrid SPEN-Diff2 simulation | 60 x 64, R=60, 32 coils, nominal b=600 s/mm2"
    )
    fig.supxlabel(
        "Synthetic spatial diffusion tensor; nominal PGSE b-matrices; known simulated geometry, estimated odd/even phase\nCommon display scale per row. MRI input is a phantom assigned the protocol FOV, not a calibrated MID253 reference.",
        fontsize=9,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "hybrid_spen_diffusion.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    info = {
        "protocol": protocol.to_dict(),
        "diffusion_model": "nominal_orthogonal_pgse",
        "tensor_model": "synthetic_positive_diagonal",
        "volume_labels": labels,
        "signal_shape": list(sample.measurements.shape),
        "calibrated_to_scanner": False,
    }
    (args.output / "hybrid_spen_diffusion.json").write_text(
        json.dumps(info, indent=2) + "\n"
    )
    print(path)


if __name__ == "__main__":
    main()
