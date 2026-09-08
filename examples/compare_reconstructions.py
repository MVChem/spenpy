"""Compare SPEN180, xSPEN and Hybrid SPEN-Diff2: GT, RO-only, CG and InvA."""

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from spenpy import HybridSPENDiff2Protocol, SPEN180Protocol, XSPENProtocol
from spenpy.core import EncodingOperator
from spenpy.io import load_image
from spenpy.recon import reconstruct, reconstruct_simulated_inva
from spenpy.sim import simulate

PROJECT = Path(__file__).resolve().parents[1]
MRI_SHA256 = "cc211f0dfd9a05ca3841ce1141b292898b2dd2d3f08286affadf823a7e58df93"


def reference_mri(device):
    """Native sagittal slice 65 of 3D Slicer's unrestricted-use MRHead sample.

    Provenance: https://github.com/Slicer/Slicer/blob/main/Modules/Scripted/SampleData/SampleData.py
    No image resizing or denoising is applied before simulation.
    """
    import nrrd

    path = PROJECT / "data/mri/MR-head.nrrd"
    if not path.exists():
        url = (
            "https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/"
            + MRI_SHA256
        )
        with urlopen(url, timeout=45) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != MRI_SHA256:
            raise ValueError("MRI download checksum mismatch")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    if hashlib.sha256(path.read_bytes()).hexdigest() != MRI_SHA256:
        raise ValueError("Cached MRI checksum mismatch")
    volume, header = nrrd.read(str(path), index_order="F")
    if volume.shape != (256, 256, 130) or not np.allclose(
        header["space directions"],
        [[0, 1, 0], [0, 0, -1], [-1.2999954223632812, 0, 0]],
    ):
        raise ValueError("Unexpected MRHead geometry")
    image = torch.from_numpy(
        np.ascontiguousarray(volume[:, :, 65].T, dtype=np.float32)
    ).to(device)
    return image / image.max(), (0.256, 0.256)


def robust_abs01(value, percentile=99.5):
    """Display normalization from notebook 19, independent for each panel."""
    image = np.abs(value.detach().cpu().numpy()).astype(np.float32)
    positive = image[image > 0]
    scale = np.percentile(positive, percentile) if positive.size else 1.0
    return np.clip(image / max(float(scale), 1e-8), 0, 1)


@torch.no_grad()
def normal_norm(operator):
    generator = torch.Generator(device=operator.coil_maps.device).manual_seed(7)
    vector = torch.randn(
        operator.image_shape,
        generator=generator,
        device=operator.coil_maps.device,
        dtype=operator.coil_maps.dtype,
    )
    vector /= vector.norm()
    for _ in range(40):
        vector = operator.normal(vector)
        vector /= vector.norm()
    return (vector.conj() * operator.normal(vector)).sum().real.item()


def phase_for_cg(phase):
    """Use the same estimated phase on the common native reconstruction grid."""
    result = phase.new_zeros((phase.shape[0] * 2, phase.shape[1]))
    result[1::2] = phase
    return result


def native_coil_maps(maps, shape):
    """Evaluate the known smooth complex maps on the CG reconstruction grid."""
    channels = maps.permute(2, 0, 1)[None]
    real = F.interpolate(
        channels.real, size=shape, mode="bilinear", align_corners=False
    )
    imag = F.interpolate(
        channels.imag, size=shape, mode="bilinear", align_corners=False
    )
    return torch.complex(real, imag)[0].permute(1, 2, 0)


def save_comparison(rows, filename, snr_db):
    fig, axes = plt.subplots(
        len(rows), 4, figsize=(17, 4.5 * len(rows)), layout="constrained"
    )
    for row, (name, coils, ground_truth, degraded, cg, inva) in enumerate(rows):
        titles = [
            f"{name}: Original GT",
            f"Degraded image: RO FFT + RSS\n{coils} coils",
            "CG reconstruction",
            "PhaseMap + InvA reconstruction",
        ]
        for axis, image, title in zip(
            axes[row], [ground_truth, degraded, cg, inva], titles
        ):
            axis.imshow(
                robust_abs01(image),
                cmap="gray",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            axis.set_title(f"{title}\n{image.shape[0]} x {image.shape[1]}", fontsize=12)
            axis.axis("off")
    fig.suptitle(
        "SPEN180, xSPEN and Hybrid SPEN-Diff2 | Ground truth and reconstruction",
        fontsize=16,
    )
    fig.supxlabel(
        f"Same simulated measurements and estimated odd/even phase | SNR {snr_db:g} dB\n"
        "Independent 99.5th-percentile display normalization; no post-reconstruction sharpening",
        fontsize=10,
    )
    fig.savefig(filename, dpi=160)
    plt.close(fig)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--matrix-size",
        type=int,
        default=256,
        help="Square acquisition and reconstruction grid for all three protocols (default: 256)",
    )
    parser.add_argument("--image", type=Path)
    parser.add_argument("--fov-mm", type=float, nargs=2)
    parser.add_argument("--snr-db", type=float, default=35)
    parser.add_argument("--phase-constant", type=float, default=0.7)
    parser.add_argument("--phase-linear", type=float, default=0.3)
    parser.add_argument(
        "--phase-estimator", choices=["quadratic", "tiny"], default="quadratic"
    )
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs")
    args = parser.parse_args()
    if args.matrix_size < 4 or args.matrix_size % 2:
        parser.error("--matrix-size must be an even integer >= 4")
    shape = (args.matrix_size, args.matrix_size)
    torch.set_num_threads(args.threads)
    if args.image is None:
        image, fov = reference_mri(args.device)
    else:
        if args.fov_mm is None or any(
            not np.isfinite(v) or v <= 0 for v in args.fov_mm
        ):
            parser.error("A custom image requires two positive --fov-mm values")
        image = load_image(args.image, device=args.device, normalize=True)
        fov = tuple(v / 1000 for v in args.fov_mm)
    if image.ndim != 2:
        parser.error("Select a single 2D MRI image")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for protocol, coils, label in [
        (
            SPEN180Protocol(
                acquisition_shape=shape,
                fov_m=fov,
                r_value=120,
                echo_spacing_s=0.0002304,
            ),
            4,
            "SPEN180",
        ),
        (XSPENProtocol(acquisition_shape=shape, fov_m=fov, r_value=60), 32, "xSPEN"),
        (
            HybridSPENDiff2Protocol(acquisition_shape=shape),
            32,
            "Hybrid SPEN-Diff2",
        ),
    ]:
        sample = simulate(
            image,
            protocol,
            num_coils=coils,
            snr_db=args.snr_db,
            seed=42,
            even_odd_constant_rad=args.phase_constant,
            even_odd_linear_rad=args.phase_linear,
        )
        inva = reconstruct_simulated_inva(
            sample.acquisition,
            sample.operator,
            estimator="quadratic"
            if isinstance(protocol, HybridSPENDiff2Protocol)
            else args.phase_estimator,
        )
        native_shape = protocol.acquisition_shape
        phase = inva.phase_map_rad
        if tuple(phase.shape) != (native_shape[0] // 2, native_shape[1]):
            # Hybrid uses a 0.9*(Npe/2) phase grid. Interpolate the circular
            # phase, so a +/-pi wrap does not become a spurious phase ramp.
            unit = torch.exp(1j * phase)[None, None]
            phase = torch.complex(
                F.interpolate(
                    unit.real,
                    size=(native_shape[0] // 2, native_shape[1]),
                    mode="bilinear",
                    align_corners=False,
                ),
                F.interpolate(
                    unit.imag,
                    size=(native_shape[0] // 2, native_shape[1]),
                    mode="bilinear",
                    align_corners=False,
                ),
            )[0, 0].angle()
        # Both methods use this measurement-derived estimate. The injected
        # phase is never passed to either reconstruction as a known answer.
        calibrated = EncodingOperator(
            protocol,
            native_shape,
            coil_maps=native_coil_maps(sample.operator.coil_maps, native_shape),
            phase_map_rad=phase_for_cg(phase),
        )
        cg = reconstruct(
            sample.acquisition,
            calibrated,
            regularization=1e-3 * normal_norm(calibrated),
            max_iter=250,
            rtol=1e-5,
        )
        if (
            not torch.isfinite(cg.image).all()
            or not torch.isfinite(inva.magnitude).all()
        ):
            raise RuntimeError("Nonfinite reconstruction")
        print(
            f"{label}: CG converged={bool(cg.converged)}, iterations={cg.iterations}; "
            f"PhaseMap+InvA grid={tuple(inva.magnitude.shape)}, estimator={args.phase_estimator}"
        )
        # The degradation view is the uncorrected complex RO image, combined
        # by RSS for display. No phase correction or PE reconstruction is used.
        degraded = inva.ro_image.abs().square().sum(-1).sqrt()
        if not all(
            tuple(value.shape) == native_shape
            for value in (degraded, cg.image, inva.magnitude)
        ):
            raise RuntimeError(
                "Degradation and both reconstructions must share the requested grid"
            )
        rows.append(
            (
                label,
                coils,
                sample.target.abs(),
                degraded,
                cg.image.abs(),
                inva.magnitude,
            )
        )
    save_comparison(rows, args.output / "comparison.png", args.snr_db)


if __name__ == "__main__":
    main()
