"""Read-only checks on the surrounding workspace; skip when not available."""

import os
from pathlib import Path

import pytest
import torch

from spenpy.io import load_mat_array, read_bruker, read_mat, read_siemens
from spenpy.recon import reconstruct_bruker

pytestmark = pytest.mark.integration
ROOT = Path(
    os.environ.get("SPENPY_REFERENCE_ROOT", Path(__file__).resolve().parents[2])
)
BRUKER = (
    ROOT / "spen_recons/data/scanner/20240321_204022_lxj_spen_mouse_240321_1_1_1/15"
)
SIEMENS = next(
    ROOT.glob(
        "*/08_Eddy_Human_xSPEN_DTI/Trio_brain/full brain/meas_MID253_esrs_40slices_Q60_Ortho900_FID48808.dat"
    ),
    ROOT / "missing_siemens_reference.dat",
)
MAT = ROOT / "xspen_recons/outputs/xspen_raw_reconstruction.mat"


@pytest.fixture(scope="module")
def bruker():
    if not BRUKER.exists():
        pytest.skip("Local Bruker reference not present")
    return read_bruker(BRUKER)


def test_real_bruker_dimensions_and_protocol(bruker):
    assert bruker.select(slice=0, volume=0).data.shape == (96, 96, 4)
    assert bruker.metadata["protocol"]["r_value"] == pytest.approx(120)
    assert bruker.metadata["protocol"]["echo_spacing_s"] == pytest.approx(0.0002304)
    assert bruker.stage == "sorted_samples"


def test_legacy_bruker_accepts_explicit_regridded_stage_without_double_regridding(
    bruker,
):
    raw_recon = reconstruct_bruker(BRUKER, bruker, process_with_pre_phase_corr=False)
    prepared = read_bruker(BRUKER, regrid=True)
    prepared_recon = reconstruct_bruker(
        BRUKER, prepared, process_with_pre_phase_corr=False
    )
    torch.testing.assert_close(
        raw_recon.images, prepared_recon.images, rtol=1e-5, atol=1e-3
    )
    assert raw_recon.images.shape == (96, 96)
    assert torch.isfinite(raw_recon.images).all()


def test_hybrid_spen_mat_preserves_complex_coils_and_selection():
    if not MAT.exists():
        pytest.skip("Local MATLAB reference not present")
    a = read_mat(MAT)
    assert a.data.shape == (60, 64, 32, 40, 4)
    assert a.select(slice=19, volume=0).data.shape == (60, 64, 32)
    assert a.stage == "coil_image" and a.data.is_complex()


def test_siemens_raw_against_matlab_rofft_magnitude_baseline():
    if not SIEMENS.exists() or not MAT.exists():
        pytest.skip("Local Siemens reference not present")
    a = read_siemens(SIEMENS)
    assert a.data.shape == (60, 64, 32, 40, 4)
    assert a.metadata["r_value"] == 60
    assert a.metadata["protocol"]["sequence"] == "hybrid_spen_diff2"
    assert a.metadata["protocol"]["chirp_duration_s"] == pytest.approx(0.01408)
    assert a.metadata["protocol"]["nominal_b_value_s_mm2"] == 600
    assert a.stage == "regridded_kspace"
    ro = torch.fft.fftshift(
        torch.fft.fft(torch.fft.ifftshift(a.data, dim=1), dim=1), dim=1
    )
    actual = ro.abs().square().sum(2).sqrt().double().flatten()
    expected = load_mat_array(MAT, "SmatBeforePhaseMapInvA").double().flatten()
    scale = (actual @ expected) / (actual @ actual)
    nrmse = (scale * actual - expected).norm() / expected.norm()
    # twixtools and mapVBVD regrid interpolation are not pointwise identical.
    assert nrmse.item() < 0.02
