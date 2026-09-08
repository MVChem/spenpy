"""Regression against the user's original phase-map notebooks and MAT input."""

import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from spenpy import SPEN180Protocol
from spenpy.recon import calc_inva_matrices, reconstruct_even_odd_inva


def test_direct_correction_keeps_original_even_line_information():
    generator = torch.Generator().manual_seed(53)
    data = torch.randn(
        16, 10, 2, dtype=torch.complex128, generator=generator, requires_grad=True
    )
    full = torch.randn(16, 16, dtype=torch.complex128, generator=generator)
    half = torch.eye(8, dtype=torch.complex128)
    phase = torch.randn(
        8, 10, dtype=torch.float64, generator=generator, requires_grad=True
    )
    result = reconstruct_even_odd_inva(
        data, full, half, half, phase_map_rad=phase, coil_combination="rss"
    )
    expected = data.clone()
    expected[1::2] *= torch.exp(-1j * phase)[..., None]
    torch.testing.assert_close(result.corrected_ro_image, expected)
    torch.testing.assert_close(result.corrected_ro_image.abs(), data.abs())
    torch.testing.assert_close(
        result.coil_images, torch.einsum("ij,jxc->ixc", full, expected)
    )
    result.coil_images.abs().square().sum().backward()
    assert torch.isfinite(data.grad).all() and data.grad.norm() > 0
    assert torch.isfinite(phase.grad).all() and phase.grad.norm() > 0


def test_spen_full_and_half_matrices_match_notebook_parameters():
    from spenpy._legacy.core.matrix import calcInvA

    protocol = SPEN180Protocol(
        acquisition_shape=(96, 96),
        fov_m=(0.035, 0.035),
        r_value=120,
        focus_offset_voxels=0.2,
    )
    actual = calc_inva_matrices(protocol, dtype=torch.complex128)
    a = protocol.phase_coefficient_rad_m2 / 10000
    for matrix, size, offset in zip(actual, [96, 48, 48], [0.2, 0.1, 0.6]):
        expected = calcInvA(a, 3.5, size, 0, 1, offset, 0.8)[0]
        torch.testing.assert_close(matrix, expected, rtol=1e-10, atol=1e-10)


def reference_data():
    from scipy.io import loadmat

    from spenpy._legacy.core.matrix import calcInvA

    root = Path(__file__).resolve().parents[2] / "spen_recons"
    path = root / "data/mat/20231011_lxj_SPEN_data_231011_3_1_3/slice_19.mat"
    if not path.exists():
        pytest.skip("Original notebook MAT file unavailable")
    source = loadmat(path, squeeze_me=True, struct_as_record=False)
    data = torch.from_numpy(source["spen_original_signal_rofft"]).to(torch.complex64)
    matrices = []
    for key in ("full_args", "odd_args", "even_args"):
        args = getattr(source["calc_inva_params"], key)
        matrices.append(
            calcInvA(
                args[0], args[1], int(args[2]), args[3], int(args[4]), args[5], args[6]
            )[0].to(torch.complex64)
        )
    return root / "spenpy/demo", data, matrices


def notebook_functions(path, indices, scope):
    cells = json.loads(path.read_text())["cells"]
    for index in indices:
        tree = ast.parse("".join(cells[index]["source"]))
        # Only the inspected computational definitions are executed, never
        # notebook path setup, file writes, plotting or output cells.
        definitions = ast.Module(
            body=[
                node
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            ],
            type_ignores=[],
        )
        exec(compile(definitions, str(path), "exec"), scope)  # noqa: S102 -- inspected reference definitions
    return cells


@pytest.mark.integration
def test_notebook09_phase_and_signal_match_reference():
    folder, data, matrices = reference_data()
    scope = {
        "torch": torch,
        "F": torch.nn.functional,
        "device": torch.device("cpu"),
        "rofft": data[:, :, None, :],
        "odd_inv": matrices[1],
        "even_inv": matrices[2],
    }
    cells = notebook_functions(
        folder / "09_hand_by_hand_phase_map.ipynb", [3, 7], scope
    )
    for index in [5, 6, 8]:
        exec("".join(cells[index]["source"]), scope)  # noqa: S102 -- original numerical reference cells
    expected_phase = scope["phase_map"]
    result = reconstruct_even_odd_inva(
        data, *matrices, estimator="quadratic", coil_combination="rss"
    )
    torch.testing.assert_close(
        result.phase_map_rad, expected_phase, atol=2e-6, rtol=2e-6
    )
    expected = data.clone()
    expected[1::2] *= torch.exp(-1j * expected_phase)[..., None]
    torch.testing.assert_close(
        result.corrected_ro_image, expected, atol=2e-6, rtol=2e-6
    )


@pytest.mark.integration
def test_notebook19_tiny_phase_matches_reference_without_rng_side_effects():
    folder, data, matrices = reference_data()
    scope = {
        "torch": torch,
        "nn": torch.nn,
        "device": torch.device("cpu"),
        "OddInvA": matrices[1],
        "EvenInvA": matrices[2],
    }
    notebook_functions(folder / "19_spen_original_tinyphase_inva.ipynb", [5], scope)
    with torch.random.fork_rng(devices=[]):
        expected_phase = scope["fit_tiny_phase"](data[:, :, None, :], steps=30)[0]
    state = torch.get_rng_state().clone()
    result = reconstruct_even_odd_inva(
        data, *matrices, estimator="tiny", tiny_steps=30, coil_combination="rss"
    )
    assert torch.equal(state, torch.get_rng_state())
    torch.testing.assert_close(
        result.phase_map_rad, expected_phase, atol=2e-5, rtol=2e-5
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_notebook_phase_cpu_cuda_agree():
    # Identical odd/even images with a known smooth RO-dependent phase.
    generator = torch.Generator().manual_seed(27)
    image = torch.randn(8, 20, 2, generator=generator, dtype=torch.complex64)
    phase = 0.7 + 0.3 * torch.linspace(-1, 1, 20)
    data = torch.stack(
        [image, image * torch.exp(1j * phase)[None, :, None]], dim=1
    ).reshape(16, 20, 2)
    full, half = (
        torch.eye(16, dtype=torch.complex64),
        torch.eye(8, dtype=torch.complex64),
    )
    cpu = reconstruct_even_odd_inva(data, full, half, half, coil_combination="rss")
    gpu = reconstruct_even_odd_inva(
        data.cuda(), full.cuda(), half.cuda(), half.cuda(), coil_combination="rss"
    )
    torch.testing.assert_close(
        cpu.corrected_ro_image, gpu.corrected_ro_image.cpu(), atol=2e-5, rtol=2e-5
    )
    error = torch.angle(torch.exp(1j * (cpu.phase_map_rad - phase)))
    assert error.abs().max().item() < 2e-4


def test_stationary_point_matrix_integrals_are_stable():
    from spenpy._legacy.core.matrix import calcSRMatrixApprox

    edges = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    a = 0.7
    ky = torch.tensor([a + 1e-13, -a - 1e-13], dtype=torch.float64)
    matrix = calcSRMatrixApprox(4 * a, 2, ky, edges, b=0.0, stable_integrals=True)[0]
    nodes, weights = np.polynomial.legendre.leggauss(64)
    expected = np.empty((2, 2), dtype=np.complex128)
    for row, freq in enumerate(ky.numpy()):
        for col, centre in enumerate([-0.5, 0.5]):
            u = 0.5 * nodes
            # Independent integration of the approximation used by CalcSRMatrixApprox.
            expected[row, col] = np.exp(1j * (a * centre**2 + freq * centre)) * np.sum(
                0.5
                * weights
                * np.exp(1j * (2 * a * centre + freq) * u)
                * (1 + 1j * a * u**2)
            )
    torch.testing.assert_close(
        matrix, torch.from_numpy(expected), atol=1e-13, rtol=1e-13
    )
