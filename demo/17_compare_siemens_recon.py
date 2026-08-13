#!/usr/bin/env python3
"""Build aligned contact sheets for all three Siemens reconstruction routes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = PROJECT_DIR / "recon_spenpy"
DEFAULT_CALIBRATED_DIR = PROJECT_DIR / "recon_siemens_raw"
DEFAULT_PHYSICS_DIR = PROJECT_DIR / "recon_siemens_raw_physics"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "recon_comparison"

METHODS = (
    ("SPENPY", "__spenpy_recon.npz"),
    ("CALIBRATED", "__raw_calibrated_global_pe_operator.npz"),
    ("RAW-ONLY", "__raw_raw_only_spenpy_weighted_adjoint.npz"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--calibrated-dir", type=Path, default=DEFAULT_CALIBRATED_DIR)
    parser.add_argument("--physics-dir", type=Path, default=DEFAULT_PHYSICS_DIR)
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _load_volume(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        volume = np.asarray(payload["adaptive_magnitude"])
    if volume.ndim != 4:
        raise ValueError(f"Expected [phase, readout, slice, set], got {volume.shape}")
    return volume


def _save_scan_comparison(
    base: str,
    volumes: list[tuple[str, np.ndarray]],
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patheffects
    from matplotlib.font_manager import FontProperties

    reference_shape = volumes[0][1].shape
    if any(volume.shape != reference_shape for _, volume in volumes):
        shapes = {name: volume.shape for name, volume in volumes}
        raise ValueError(f"Reconstruction shapes do not match for {base}: {shapes}")

    phase, readout, slices, sets = reference_shape
    slice_indices = np.unique(
        np.linspace(0, slices - 1, min(7, slices), dtype=int)
    )
    method_count = len(volumes)
    rows = sets * method_count
    panel_width = 1.15
    panel_height = panel_width * readout / phase
    fig, axes = plt.subplots(
        rows,
        len(slice_indices),
        figsize=(panel_width * len(slice_indices), panel_height * rows),
        squeeze=False,
        facecolor="black",
        gridspec_kw={"wspace": 0.01, "hspace": 0.01},
    )
    fig.subplots_adjust(
        left=0.002,
        right=0.998,
        bottom=0.002,
        top=0.998,
        wspace=0.01,
        hspace=0.01,
    )
    label_font = FontProperties(
        family="Times New Roman",
        weight="bold",
        size=7.5,
    )
    label_effect = [
        patheffects.withStroke(linewidth=1.25, foreground="black")
    ]

    for set_index in range(sets):
        for method_index, (method_name, volume) in enumerate(volumes):
            row = set_index * method_count + method_index
            scale = float(np.percentile(volume[..., set_index], 99.5))
            scale = scale if scale > 0 else 1.0
            for column, slice_index in enumerate(slice_indices):
                axis = axes[row, column]
                axis.set_facecolor("black")
                axis.imshow(
                    np.rot90(volume[:, :, slice_index, set_index]),
                    cmap="gray",
                    vmin=0,
                    vmax=scale,
                    aspect="equal",
                    interpolation="nearest",
                )
                axis.text(
                    0.025,
                    0.98,
                    f"{method_name}\nSET {set_index}  SLICE {slice_index}",
                    transform=axis.transAxes,
                    color="white",
                    fontproperties=label_font,
                    horizontalalignment="left",
                    verticalalignment="top",
                    linespacing=0.9,
                    path_effects=label_effect,
                )
                axis.set_axis_off()

    fig.savefig(
        path,
        dpi=180,
        facecolor="black",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)


def _save_master(paths: list[tuple[str, Path]], output: Path) -> None:
    images = [(label, Image.open(path).convert("RGB")) for label, path in paths]
    width = max(image.width for _, image in images)
    bar_height = 58
    height = sum(image.height + bar_height for _, image in images)
    master = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(master)
    font_path = Path(
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf"
    )
    font = ImageFont.truetype(str(font_path), 30)
    y = 0
    for label, image in images:
        draw.text((12, y + 11), label, fill="white", font=font)
        y += bar_height
        master.paste(image, (0, y))
        y += image.height
    master.save(output, optimize=True)
    for _, image in images:
        image.close()


def main() -> None:
    args = parse_args()
    processed_dir = args.processed_dir.expanduser().resolve()
    calibrated_dir = args.calibrated_dir.expanduser().resolve()
    physics_dir = args.physics_dir.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_suffix = METHODS[0][1]
    processed_paths = sorted(processed_dir.glob(f"*{processed_suffix}"))
    if not processed_paths:
        raise SystemExit(f"No processed NPZ files found in {processed_dir}")

    combined: list[tuple[str, Path]] = []
    for processed_path in processed_paths:
        base = processed_path.name.removesuffix(processed_suffix)
        paths = (
            processed_path,
            calibrated_dir / f"{base}{METHODS[1][1]}",
            physics_dir / f"{base}{METHODS[2][1]}",
        )
        missing = [path for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing comparison inputs for {base}: {missing}")
        volumes = [
            (method_name, _load_volume(path))
            for (method_name, _), path in zip(METHODS, paths, strict=True)
        ]
        output_path = output_dir / f"{base}__all_three_methods.png"
        _save_scan_comparison(base, volumes, output_path)
        combined.append((base, output_path))
        print(f"{output_path}: {[volume.shape for _, volume in volumes]}")

    master_path = output_dir / "all_scans__all_three_methods.png"
    _save_master(combined, master_path)
    print(master_path)


if __name__ == "__main__":
    main()
