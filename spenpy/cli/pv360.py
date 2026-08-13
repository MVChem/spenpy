"""pv360 driver script -- Python equivalent of pv360.m.

Processes RARE, EPI, and SPEN data from a Bruker experiment directory.
Only data reading and saving are implemented -- no visualization.
"""

import os
import sys
import numpy as np
from pathlib import Path


def read_datalist(file_dir: str):
    """Load datalist.txt and split into RARE, EPI, and SPEN scan IDs.

    MATLAB:
        datalist = textread(datalist_path, '%u');
        RARE_datalist = datalist(1);
        EPI_datalist  = datalist(2);
        SPEN_datalist = datalist(3:end);
    """
    datalist_path = os.path.join(file_dir, "datalist.txt")
    if not os.path.exists(datalist_path):
        raise FileNotFoundError(f"Datalist file not found at: {datalist_path}")

    with open(datalist_path) as f:
        datalist = [int(token) for token in f.read().split()]

    if len(datalist) < 2:
        raise ValueError(f"Datalist must contain at least RARE and EPI scan IDs: {datalist_path}")

    rare_id = datalist[0]
    epi_id = datalist[1]
    spen_ids = datalist[2:]

    return rare_id, epi_id, spen_ids


def process_rare_epi(file_dir: str, scan_id: int, export_dir: str, save_name: str, var_name: str):
    """Process RARE or EPI data using ImageDataObject equivalent.

    MATLAB:
        rare_dir = fullfile(file_dir, num2str(RARE_datalist(1)), filesep);
        rare_pdata = fullfile(rare_dir, 'pdata', '1', filesep);
        imageObj = ImageDataObject(rare_pdata);
        ImageData = squeeze(imageObj.data);
        ImageData = permute(ImageData, [2, 1, 3, 4]);
        Image_RARE = ImageData / max(ImageData(:));

    Args:
        file_dir: base experiment directory
        scan_id: numeric scan ID from datalist
        export_dir: output directory
        save_name: filename for saved .mat (e.g. 'ratbrain_RARE.mat')
        var_name: variable name in the .mat file (e.g. 'Image_RARE')
    """
    from spenpy.bruker.image import read_bruker_2dseq

    scan_dir = os.path.join(file_dir, str(scan_id), "")
    pdata_dir = os.path.join(scan_dir, "pdata", "1", "")

    if not os.path.isdir(pdata_dir):
        raise FileNotFoundError(f"pdata directory not found: {pdata_dir}")

    print(f"  Reading from: {pdata_dir}")

    # ImageDataObject equivalent: read 2dseq + visu_pars
    image_data = read_bruker_2dseq(pdata_dir)

    # MATLAB: squeeze removes singleton dimensions
    image_data = np.squeeze(image_data)

    # MATLAB: permute(ImageData, [2, 1, 3, 4]) -- swap dims 1 and 2
    # Python 0-indexed: permute axes (1, 0, 2, 3)
    # Need to handle variable number of dimensions
    if image_data.ndim >= 4:
        axes = list(range(image_data.ndim))
        axes[0], axes[1] = 1, 0
        image_data = np.transpose(image_data, axes)
    elif image_data.ndim == 3:
        # [dim1, dim2, dim3] -> [dim2, dim1, dim3]
        image_data = np.transpose(image_data, (1, 0, 2))
    elif image_data.ndim == 2:
        image_data = np.transpose(image_data, (1, 0))

    # Normalize: divide by max absolute value
    max_val = np.max(np.abs(image_data))
    if max_val > 0:
        image_data = image_data / max_val

    print(f"  Shape after permute: {image_data.shape}")
    print(f"  Max value after normalization: {np.max(np.abs(image_data)):.6f}")

    # Save to .mat file
    os.makedirs(export_dir, exist_ok=True)
    save_path = os.path.join(export_dir, save_name)

    import scipy.io
    scipy.io.savemat(save_path, {var_name: image_data})
    print(f"  Saved: {save_path}")

    return image_data


def process_spen(
    file_dir: str,
    scan_id: int,
    export_dir: str,
    spen_index: int,
    traj_scan_id: int | None = None,
    recon_flavor: str = "pv360",
):
    """Process a single SPEN scan.

    MATLAB:
        spen_dir = fullfile(file_dir, num2str(SPEN_datalist(ispen)), filesep);
        NSegments = ReadPVParam(spen_dir, 'NSegments');
        EpiTrajAdjkx = ReadPVParam(spen_dir, 'PVM_EpiTrajAdjkx');
        if mod(NSegments, 2) == 1
            [images, Imag_origin, Imag_low, SPEN_AZ] = ...
                Function_Process_NewPE_SPEN_OddNumWithMask_bruker_PV360(spen_dir);
            images = permute(images, [1, 2, 5, 3, 4]);
        else
            [images] = Function_Process_NormalmultiSPEN_bruker_PV6(fid_dir);
        end
        Image_SPEN = flip(flip(images, 1), 2);

    This reads the raw k-space and parameters, then calls the reconstruction
    pipeline if available.
    """
    from spenpy.bruker.param import read_pv_param

    spen_dir = os.path.join(file_dir, str(scan_id), "")
    print(f"  Processing SPEN scan {spen_index}: {spen_dir}")
    traj_dir = None
    if traj_scan_id is not None:
        traj_dir = os.path.join(file_dir, str(traj_scan_id), "")
        print(f"    Trajectory scan: {traj_dir}")

    # Read parameters (equivalent to MATLAB ReadPVParam)
    n_segments = read_pv_param(spen_dir, "NSegments")
    if n_segments is None:
        n_segments = 1
    if isinstance(n_segments, list):
        n_segments = n_segments[0]
    n_segments = int(n_segments)

    epi_traj = read_pv_param(traj_dir or spen_dir, "PVM_EpiTrajAdjkx")
    print(f"    NSegments: {n_segments}")
    print(f"    EpiTrajAdjkx: {epi_traj is not None}")

    os.makedirs(export_dir, exist_ok=True)
    save_name = f"ratbrain_SPEN_96_{spen_index}.mat"
    save_path = os.path.join(export_dir, save_name)

    if n_segments % 2 != 1:
        raise NotImplementedError(
            "PV360 even-segment SPEN reconstruction is not implemented yet; "
            "the MATLAB path calls Function_Process_NormalmultiSPEN_bruker_PV6."
        )

    from spenpy.recon.spen_recon import orient_pv360_spen_image, reconstruct_odd_segments

    if recon_flavor not in {"pv360", "pv5"}:
        raise ValueError(f"Unsupported recon_flavor: {recon_flavor}")

    recon = reconstruct_odd_segments(
        spen_dir,
        traj_dir=traj_dir,
        regrid_flavor="pv5" if recon_flavor == "pv5" else "pv360",
        smooth_motion_phase_between_shots=recon_flavor != "pv5",
    )
    image_spen = orient_pv360_spen_image(recon.images)

    def to_numpy(value):
        if hasattr(value, "detach"):
            return value.detach().cpu().resolve_conj().numpy()
        return np.asarray(value)

    # The MATLAB pv360.m only applies ``flip(flip(images,1),2)`` to Image_SPEN
    # but leaves Imag_low/Imag_origin in the raw (unflipped) orientation, so
    # the three panels appear rotated 180 deg relative to each other. Apply
    # the same orientation correction to all three so they line up with the
    # canonical Image_SPEN view.
    imag_low_oriented = orient_pv360_spen_image(to_numpy(recon.imag_low))
    imag_origin_oriented = orient_pv360_spen_image(to_numpy(recon.imag_origin))

    spen_az = {key: to_numpy(value) for key, value in recon.spen_az.items()}

    import scipy.io
    scipy.io.savemat(
        save_path,
        {
            "Imag_low": imag_low_oriented,
            "Imag_origin": imag_origin_oriented,
            "Image_SPEN": image_spen,
            "SPEN_AZ": spen_az,
            "NSegments": n_segments,
        },
    )
    print(f"    Saved: {save_path}")

    return image_spen


def run_pv360(file_dir: str, export_dir: str = None):
    """Main pv360 driver.

    Args:
        file_dir: path to Bruker experiment directory (contains datalist.txt)
        export_dir: output directory for .mat files (default: export_data/pv360)
    """
    if not os.path.isdir(file_dir):
        raise NotADirectoryError(f"Directory not found: {file_dir}")

    if export_dir is None:
        export_dir = os.path.join(os.getcwd(), "export_data", "pv360")

    print(f"Input directory: {file_dir}")
    print(f"Export directory: {export_dir}")

    # Load datalist
    rare_id, epi_id, spen_ids = read_datalist(file_dir)
    print(f"RARE scan ID: {rare_id}")
    print(f"EPI scan ID: {epi_id}")
    print(f"SPEN scan IDs: {spen_ids} ({len(spen_ids)} scans)")

    # Process RARE data
    print("\nProcessing RARE data...")
    process_rare_epi(file_dir, rare_id, export_dir, "ratbrain_RARE.mat", "Image_RARE")

    # Process EPI data
    print("\nProcessing EPI data...")
    process_rare_epi(file_dir, epi_id, export_dir, "ratbrain_EPI.mat", "image_EPI")

    # Process SPEN data series
    print("\nProcessing SPEN data series...")
    for i, spen_id in enumerate(spen_ids, start=1):
        process_spen(file_dir, spen_id, export_dir, i)

    print(f"\nDone. All files exported to {export_dir}")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="PV360 Bruker data processor")
    parser.add_argument("file_dir", help="Path to Bruker experiment directory")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory (default: export_data/pv360)")
    args = parser.parse_args()

    run_pv360(args.file_dir, args.output)


if __name__ == "__main__":
    main()
