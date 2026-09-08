# Validation

Environment: Python 3.12, PyTorch 2.11.0+cu130, CPU and NVIDIA RTX 4090 CUDA. Installed dependencies are recorded in `requirements-tested.txt`.

Verification on 2026-09-08 after the MRI naming and Hybrid SPEN-Diff2 changes: **74 tests passed**, including local scanner/MATLAB references and CPU/CUDA numerical cases; Ruff passed. Both examples completed on CPU.

After changing the comparison to 256×256 for all three protocols, the comparison example was rerun on CPU and Ruff passed again. All degraded and reconstructed arrays passed the grid-shape and finite-value checks. All three CG reconstructions converged (SPEN180: 60 iterations, xSPEN: 41, Hybrid SPEN-Diff2: 28). The full suite was rerun for the 1.0.0 release after this example-grid change: **74 passed in 28.10 s**; Ruff also passed.

Run:

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check spenpy tests examples
.venv/bin/python examples/compare_reconstructions.py --device cpu
.venv/bin/python examples/simulate_hybrid_spen_diffusion.py --device cpu
```

The suite covers complex forward/adjoint identities, independent physical integrals, gradients, correlated noise, masks, CG versus dense solves, acquisition stages, snapshot replay and scanner files. Additional reconstruction regressions compare the original notebooks against the new implementation using their actual four-coil MAT input.

## Notebook parity

- Notebook 09: the original quadratic-plus-residual phase code and the new routine produce the same PhaseMap and corrected complex signal on the supplied MAT frame. Direct CUDA verification gave zero maximum phase/signal difference.
- Notebook 19: the original TinyPhase code and the new routine produce the same phase map after the default 501 optimization steps on CUDA. Maximum phase difference was zero. A shorter CPU regression also checks that the new implementation preserves caller RNG state.
- Tests verify direct even-line multiplication, unchanged pre-InvA signal magnitudes, supplied-phase gradients, full/odd/even sampling origins and CPU/CUDA agreement.
- Small-argument matrix moments are checked against independent Gaussian quadrature to avoid cancellation at stationary points.

The notebook integration tests load their original source and MAT input from the sibling `spen_recons` repository. They skip if those external reference files are absent. No generated plot or fitted traditional reconstruction is used as a calibration target.

## Scanner references

- Bruker scan 15 reads as complex `[96,96,4,1,1]`; the preserved scanner pipeline returns finite 96-by-96 results without double regridding.
- The Siemens file reads as `[60,64,32,40,4]`. Its InvA and seeded half-resolution inverses match the archived MATLAB matrices at approximately 1e-14 relative error.
- Prior same-input comparison with the unmodified MATLAB Siemens phase routine gave phase-map error 6.4e-16 and final complex coil-image error 2.3e-15. Independent raw readers still yield different position shifts, so this does not establish raw-path equivalence or calibration of the ideal crossed-chirp simulator.

The 1.0.0 release check also read MID253 from `.dat`, estimated one shift from the full scan, and reconstructed slice 19 (zero-based) in all four volumes using `reconstruct_hybrid_spen`. All four complex coil-image arrays were finite and each RSS image had shape 60×64. The Python shift was 3.864508 pixels versus 4.200540 in the archived MATLAB reconstruction. After fitting one intensity scale per frame, magnitude NRMSE against the corresponding saved MATLAB `Smat` frames was 8.537%, 7.472%, 7.246% and 6.615% for volumes 0–3. These measurements confirm an executable raw-to-image path and quantify the remaining disagreement; they are not a full-scan equivalence test or an estimate of error against anatomical ground truth.

## Hybrid SPEN-Diff2 verification

- The native PE forward matrix equals the existing historical matrix at the MID253 parameters. Applying its width-0.3 window reproduces the saved original MATLAB InvA, tested at atol=1e-11 / rtol=1e-10.
- Independent centred IFFT construction agrees with the simulated native-grid raw signal; RO FFT reversal had relative error about 1.26e-15 in a complex128 check.
- HR PE pixels are checked against independent 128-point Gauss-Legendre integration of the continuous quadratic phase. This bounds the historical moment approximation in the tested HR setting; it is not a guarantee for arbitrary coarse/high-R protocols.
- Non-square independent HR/acquisition grids, multi-coil adjoint identities, image gradients, position-phase convention, CG recovery and deterministic snapshot replay pass.
- Diffusion tests cover isotropic/anisotropic tensors, volume selection, spatial b-matrices, off-diagonal contractions, gradient propagation and rejection of invalid tensors.
- The raw Siemens integration check now also verifies extraction of `HybridSPENDiff2Protocol`, including the 14.08 ms PE pulse and nominal b=600 from the actual header.
- Simulated geometry is supplied as known protocol calibration; odd/even phase is estimated. The legacy automatic position estimator is not claimed to identify the origin of arbitrary synthetic objects.

These checks establish reduced-model implementation and reference-matrix agreement. They do not establish complete waveform/Bloch accuracy, intrinsic diffusion calibration, or equality of independent raw-reader reconstructions.

## Output scope

`examples/compare_reconstructions.py` writes `outputs/comparison.png` with three protocol rows and GT, RO-only degradation, CG and PhaseMap + InvA columns. All acquisition and reconstruction grids default to 256×256, matching the default input MRI dimensions. `examples/simulate_hybrid_spen_diffusion.py` retains the native Hybrid SPEN-Diff2 60×64 grid and writes `outputs/hybrid_spen_diffusion.png` plus explicit nominal/synthetic parameter metadata in JSON. Optional I/O helpers write snapshots only when called.

The original demo notebooks remain unmodified reference sources. The public implementation is named by method: `even_odd.py`, `super_resolution.py`, `hybrid_spen.py`; old reconstruction API names are listed in the README migration table.
