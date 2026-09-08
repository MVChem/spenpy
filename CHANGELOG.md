# SPENPy releases

## 1.0.0 — 2026-09-08

SPENPy now provides a shared PyTorch simulation and reconstruction interface for SPEN180, ideal crossed-chirp xSPEN, and the reduced Hybrid SPEN-Diff2 model used by the MID253 reconstruction. The new implementation replaces the legacy layout on `main` while retaining its Git history.

### Included

- Complex multicoil forward/adjoint operators, independent image/acquisition grids, noise and phase simulation, differentiable CG and data consistency.
- PhaseMap + InvA reconstruction, including the historical quadratic Hybrid SPEN-Diff2 matrices and automatic position-shift estimation.
- Siemens Twix loading with explicit preprocessing stages and `esrs_hyb_spen_Diff2` header extraction; the preserved Bruker single-shot reconstruction interface and MAT/image/simulation I/O.
- Hybrid diffusion-volume simulation with nominal orthogonal b-matrices or supplied spatial b-matrices and diffusion tensors.
- Three-protocol comparison and four-volume diffusion examples, with sample figures and MRI provenance.

### Compatibility

The old package's module layout, CLI entry points and notebook-oriented APIs are not preserved as a complete drop-in interface. Use the method-specific interfaces documented in the [README](README.md). `SPENProtocol` and `SPENOperator` remain aliases for `SPEN180Protocol` and `SPEN180Operator`.

- [v0.2.0](https://github.com/MVChem/spenpy/tree/v0.2.0) preserves the previous published commit.
- [legacy-pre-v1.0.0](https://github.com/MVChem/spenpy/tree/legacy-pre-v1.0.0) preserves the full legacy code snapshot, including the final demo README changes and `19_spen_original_tinyphase_inva.ipynb`.
- [v1.0.0](https://github.com/MVChem/spenpy/tree/v1.0.0) identifies this release.

### Validation and scope

The local suite passed all 74 tests, including available scanner/MATLAB references and CPU/CUDA cases; Ruff passed. Checks cover operator identities, independent physical integrals, gradients, reconstruction matrices, raw-readout comparisons and diffusion simulation. Reference-dependent tests skip when the external data and original notebooks are absent; see [validation details](docs/validation.md).

A raw-to-image check reconstructed the MID253 middle slice in all four volumes with finite results. Scale-aligned magnitude NRMSE against the archived MATLAB reconstruction was 6.6–8.5%; automatic shift estimates were 3.8645 pixels in Python and 4.2005 in MATLAB.

MID253 uses `esrs_hyb_spen_Diff2` quadratic-phase encoding. It does not validate the separate crossed-chirp xSPEN model. Independent Siemens readers still differ in ramp interpolation and position-shift estimation; full raw-to-image equivalence remains unestablished. The release does not add calibrated support for every sequence in the original project, SMS/multiband reconstruction, full 3D spin dynamics, complete RF/gradient/ADC waveform simulation, or automatic intrinsic diffusion weighting. The 1.0.0 version describes this software release within those boundaries.
