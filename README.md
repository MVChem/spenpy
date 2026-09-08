# SPENPy 1.0.0

PyTorch tools for complex-valued MRI simulation, reconstruction, and data loading with **SPEN180, cross-term xSPEN, and Hybrid SPEN-Diff2**.

`main` contains the 1.0 implementation. The previous implementation is preserved at [v0.2.0](https://github.com/MVChem/spenpy/tree/v0.2.0); [legacy-pre-v1.0.0](https://github.com/MVChem/spenpy/tree/legacy-pre-v1.0.0) also includes the final legacy TinyPhase notebook and demo documentation. Existing applications using the old module layout should pin a legacy tag until migrated. See [release notes](CHANGELOG.md) for the supported scope and compatibility changes.

## Three protocols, two encoding families

| Protocol / Python class | Encoding and readout | Current example | Model scope |
|---|---|---|---|
| **SPEN180** / `SPEN180Protocol` | Quadratic phase from a 180° chirp; PE pixel integration and Fourier RO sampling | R=120, 256×256, 4 coils | Ideal single-shot SPEN, using the main encoding parameters of the Bruker example |
| **Cross-term xSPEN** / `XSPENProtocol` | Two crossed chirps generate a yz cross-term; integration over a uniform slice profile gives a sinc kernel | R=60, 256×256, 32 coils | Ideal cross-term signal equation from the original `xSPEN1D.m` |
| **Hybrid SPEN-Diff2** / `HybridSPENDiff2Protocol` | Quadratic-phase matrix from the original `CalcSRMatrixApprox`; Siemens centered-IFFT RO convention | R=60, 32 coils; comparison: 256×256; diffusion example: MID253 native 60×64 | Reduced signal model matching the existing `esrs_hyb_spen_Diff2` reconstruction matrix, with multiple diffusion-weighted volumes |

**Hybrid SPEN-Diff2 and SPEN180 share quadratic-phase encoding physics, with separately defined protocol parameters, discretization, readout conventions, and reconstruction workflows.** These are two encoding families represented by three protocols. Cross-term xSPEN uses its own encoding matrix.

The data shown in `../xspen_recons/outputs/xspen_raw_all_volumes_middle_slice.png` come from MID253, whose raw header identifies the sequence as `esrs_hyb_spen_Diff2`. Protocol classification follows the acquisition metadata and encoding model; the xSPEN figure title and directory name are insufficient to establish the encoding physics. The local source for the sequence of the same name contains separate PE and RO RF parameters. This library records those parameters; a complete Bloch simulation of the RF waveforms remains outside the implemented model. See [Protocols and physical models](docs/physics.md).

## Installation and examples

Install the tagged release directly from GitHub:

```bash
python -m pip install "spenpy[io,recon] @ git+https://github.com/MVChem/spenpy.git@v1.0.0"
```

To run the repository examples and tests:

```bash
git clone https://github.com/MVChem/spenpy.git
cd spenpy
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e ".[io,recon,dev]"
.venv/bin/python examples/compare_reconstructions.py --device cpu
.venv/bin/python examples/simulate_hybrid_spen_diffusion.py --device cpu
```

Use `--device cuda` when CUDA is available.

- [comparison.png](outputs/comparison.png): rows show SPEN180, xSPEN, and Hybrid SPEN-Diff2; columns show the input MRI, RO FFT + RSS, CG, and PhaseMap + InvA.
- [hybrid_spen_diffusion.png](outputs/hybrid_spen_diffusion.png): RO-only and reconstructed images for Hybrid SPEN-Diff2 b0 / DWI-RO / DWI-PE / DWI-SS volumes. Simulation parameters are also saved as JSON.

Both examples use a public MRHead 256×256 magnitude slice as the input phantom. The image is encoded directly onto the acquisition grid without first downsampling it. [MRI data provenance](data/mri/README.md) includes verification details. The Hybrid example assigns this array to the protocol FOV; it does not reproduce the MID253 subject's anatomy, contrast, or diffusion tensor.

In the comparison example, `--matrix-size` sets the acquisition and reconstruction grids for all three protocols, with a default of 256×256. The default input MRI also has 256×256 pixels. Measurements and reconstructions are computed on the requested grids, without display upsampling. The Hybrid comparison overrides the MID253 acquisition dimensions; `HybridSPENDiff2Protocol()` and the separate diffusion example retain the native 60×64 grid. FOV and encoding R values remain unchanged, so the grid size alone does not specify the effective spatial resolution.

Defaults include 35 dB complex noise and an even-line phase of `0.7 + 0.3*x` rad. Reconstruction estimates the odd/even phase from the signal. CG uses the known synthetic coil sensitivities and protocol. Hybrid simulation reconstruction also uses the known simulated geometric shift; for measured data, `estimate_hybrid_spen_shift` can estimate this shift.

The comparison figure scales each panel independently using its 99.5th intensity percentile. The diffusion figure uses a common display scale across all four volumes within each row. Both figures omit post-reconstruction sharpening. These display settings support visual inspection; they do not provide a quantitative comparison of absolute intensities or reconstruction performance.

## Unified forward simulation

```python
from spenpy import SPEN180Protocol, XSPENProtocol, HybridSPENDiff2Protocol
from spenpy.io import load_image
from spenpy.sim import simulate
from spenpy.recon import reconstruct, reconstruct_simulated_inva

image = load_image("brain.npy", normalize=True)
protocol = HybridSPENDiff2Protocol()  # MID253 nominal parameters
sample = simulate(image, protocol, num_coils=32, snr_db=35,
                  even_odd_constant_rad=0.7, even_odd_linear_rad=0.3)
sr = reconstruct_simulated_inva(sample.acquisition, sample.operator)
cg = reconstruct(sample.acquisition, sample.operator, regularization=1e-3)
```

The low-level CG call above uses `sample.operator`, which contains all configured simulation conditions. The comparison example constructs a CG operator with the phase estimated from the measurements. Input images have shape `[..., image_y, image_x]`; measurements have shape `[..., spen, readout, coil]`. Inputs may be complex images or magnitude images with an explicitly supplied object phase.

`EncodingOperator` provides `forward`, `adjoint`, `normal`, and `local_psf`. Image encoding, CG iterations, and data-consistency steps support differentiation with respect to network outputs. Phase fitting is excluded from backpropagation. Synthetic coil sensitivities are model inputs and do not constitute calibration of measured sensitivities.

## Hybrid SPEN-Diff2 diffusion simulation

```python
from spenpy.sim import simulate_hybrid_spen_diffusion

sample = simulate_hybrid_spen_diffusion(
    image, HybridSPENDiff2Protocol(),
    diffusivity_mm2_s=0.8e-3, num_coils=32, snr_db=35,
)
frame = sample.acquisition.select(volume=0)
recon = reconstruct_simulated_inva(frame, sample.operator)
```

By default, the simulator generates four volumes: b0 and diffusion weighting along RO, PE, and SS, with nominal b=600 s/mm² for the three diffusion-weighted volumes. This value comes from the MID253 header; it cannot be inferred from `Ortho900` in the filename. Each volume undergoes `exp(-B:D)` attenuation before complex forward encoding.

Supply `diffusion_tensor_mm2_s` with shape 3×3 or y×x×3×3, and optionally `b_matrices_s_mm2` with shape volume×3×3 or volume×y×x×3×3. Matrix axes are ordered **RO, PE, SS**. Spatial variation and off-diagonal terms are supported, with gradients preserved for the image and diffusion tensor.

The default b-matrices include nominal orthogonal PGSE weighting only. Intrinsic diffusion weighting from the encoding gradients and their cross-terms are not computed automatically. Spatial b-matrices calculated with the original `calcSPEN180_SIEMENSdata_v2BrainMS.m` can be supplied explicitly. The output target contains the diffusion-weighted object for each volume; all four volumes share one encoding operator.

## Reconstruction modules and naming

Modules are named after their MRI methods. The original notebooks serve as source and regression references.

| File | Purpose / main interface |
|---|---|
| [core/protocol.py](spenpy/core/protocol.py) | `SPEN180Protocol`, `XSPENProtocol`, `HybridSPENDiff2Protocol` |
| [core/operators.py](spenpy/core/operators.py) | Multicoil forward/adjoint operators, PE encoding, and RO Fourier sampling |
| [core/hybrid_spen.py](spenpy/core/hybrid_spen.py) | `calc_hybrid_spen_encoding`: PE forward matrix from the original Siemens reconstruction |
| [sim/simulator.py](spenpy/sim/simulator.py) | `simulate`: object, coil, phase, and noise simulation |
| [sim/hybrid_spen.py](spenpy/sim/hybrid_spen.py) | `simulate_hybrid_spen_diffusion`: encoding of diffusion-weighted volumes |
| [recon/even_odd.py](spenpy/recon/even_odd.py) | `reconstruct_even_odd_inva`: odd/even phase estimation and direct correction of acquired even lines |
| [recon/super_resolution.py](spenpy/recon/super_resolution.py) | `calc_inva_matrices`, `reconstruct_simulated_inva`: matrix construction and protocol-specific simulation reconstruction |
| [recon/hybrid_spen.py](spenpy/recon/hybrid_spen.py) | `calc_hybrid_spen_matrices`, `reconstruct_hybrid_spen`, `estimate_hybrid_spen_shift` |
| [recon/phasemap.py](spenpy/recon/phasemap.py) | PhaseMap result structure and low-resolution decode/correct/re-encode workflow |
| [recon/solvers.py](spenpy/recon/solvers.py) | CG, regularized solvers, and data consistency |

`notebook.py` has been split into `even_odd.py` and `super_resolution.py`. The former `recon/siemens.py` is now `recon/hybrid_spen.py`, identifying the sequence family. `io/siemens.py` retains the vendor name because it reads the Siemens file format.

API names used during pre-1.0 development:

| Previous name | Current name |
|---|---|
| `SPENProtocol` / `SPENOperator` | `SPEN180Protocol` / `SPEN180Operator`; previous names remain as compatibility aliases |
| `reconstruct_notebook_phasemap_inva` | `reconstruct_even_odd_inva` |
| `simulation_inva_matrices` | `calc_inva_matrices` |
| `reconstruct_simulation_inva` | `reconstruct_simulated_inva` |
| `siemens_reconstruction_matrices` | `calc_hybrid_spen_matrices` |
| `estimate_siemens_spen_shift` | `estimate_hybrid_spen_shift` |
| `reconstruct_siemens` | `reconstruct_hybrid_spen` |

The old reconstruction function names and `recon.notebook` module are no longer exported. Update callers using the table above. Original reference notebooks under `../spen_recons/spenpy/demo` retain their filenames for traceability and validation. [Reconstruction details](docs/phasemap_inva.md) explains the two odd/even phase-correction workflows.

## Measured data

```python
from spenpy.io import read_siemens, read_bruker
from spenpy.core import protocol_from_dict
from spenpy.recon import reconstruct_hybrid_spen, estimate_hybrid_spen_shift

scan = read_siemens("measurement.dat")
protocol = protocol_from_dict(scan.metadata["protocol"])
shift = estimate_hybrid_spen_shift(scan)
result = reconstruct_hybrid_spen(scan.select(slice=19, volume=0), shift_pixels=shift)
```

The Siemens reader performs ramp regridding, RO oversampling removal, reversed-line correction, and slice ordering. For `esrs_hyb_spen_Diff2`, it extracts protocol parameters from MeasYaps/WIP. Missing fields produce a recorded parsing error; defaults are not substituted to create an apparently calibrated protocol. The Bruker raw-data reconstruction interface is `reconstruct_bruker`. Reconstructed images loaded from MAT files are labeled `coil_image` and cannot be treated as raw measurements for another encoding/decoding pass.

`read_siemens` still reports `calibrated_to_simulator=False`. Current validation covers agreement with the MATLAB matrices and selected raw-readout comparisons. Independent readout validation, phase/shift calibration across the full scan, complete RF/ADC waveforms, and agreement with measured acquisition physics require separate validation.

## Tests and sources

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check spenpy tests examples
```

Tests cover forward/adjoint consistency, independent complex-exponential integration, complex gradients, diffusion tensors and spatial b-matrices, CG, I/O replay, and agreement with the original MATLAB InvA and reference phase algorithms. See [validation.md](docs/validation.md) for the scope of these checks.

Physical model references are retained under `../xSPEN_项目`: `xSPEN1D.m`, `CalcSRMatrixApprox.m`, `Reffless1ShotHybridSPENEvenOddFix.m`, `CalcbMat_NewBrainMS.m`, and the `esrs_hyb_spen_Diff2` sequence source. Their names and equations are used with explicit distinctions between cross-term xSPEN, quadratic-phase SPEN, and the actual sequence configurations used by each vendor.
