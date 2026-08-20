# SPEN PV360 reconstruction — Python demos

This folder contains illustrative SPEN/xSPEN scripts. Demos 01--17 focus on
scanner reconstruction and diagnostics. Demo 18 is a standalone ideal xSPEN
forward simulation and matched reconstruction.

The goal of these demos is to:

1. Show **the simplest one-liner** that reproduces the full pv360 export
   (RARE + EPI + every SPEN scan + figures).
2. Show **how to reconstruct a single SPEN scan** programmatically and
   inspect / save the intermediate variables.
3. **Open up the pipeline** so you can see the raw k-space, the readout
   FFT image, the SR encoding matrix, and the final SR-corrected image.
4. **Compare Python vs MATLAB** outputs numerically and visually.

> The simulation API in [`../demo.ipynb`](../demo.ipynb) is separate from
> this Bruker reconstruction pipeline. It now supports YAML-driven synthetic
> artifact profiles for training data, while this folder remains focused on
> reconstructing real PV360 scanner data.

---

## TL;DR — visualize every scan in `/tmp`

If you just want **the same outputs as `pv360.m`** (RARE, EPI and every
SPEN scan reconstructed, saved as `.mat`, plus PNG figures) for the demo
dataset under `spen_matlab/data/20240321_204022_lxj_spen_mouse_240321_1_1_1`,
exported into `/tmp/pv360_python`:

```bash
cd /home/data1/musong/workspace/python/spen_recons

.venv/bin/python -m spenpy.cli.pv360_full \
    --file-dir spen_matlab/data/20240321_204022_lxj_spen_mouse_240321_1_1_1 \
    --output   /tmp/pv360_python
```

After the run you'll have:

```text
/tmp/pv360_python/
├── ratbrain_RARE.mat
├── ratbrain_EPI.mat
├── ratbrain_SPEN_96_1.mat ... ratbrain_SPEN_96_43.mat
├── pv360_full_summary.json
└── figures/
    ├── ratbrain_RARE.png
    ├── ratbrain_EPI.png
    └── ratbrain_SPEN_96_1.png ... ratbrain_SPEN_96_43.png
```

Or, equivalently, run [`01_run_pv360_pipeline.py`](01_run_pv360_pipeline.py)
which is a thin wrapper around the same entry point with annotated
`pv360.m` ↔ Python mapping.

```bash
.venv/bin/python spenpy/demo/01_run_pv360_pipeline.py --output /tmp/pv360_python
```

---

## What each demo does

| Script | Role | MATLAB analogue |
| --- | --- | --- |
| [`01_run_pv360_pipeline.py`](01_run_pv360_pipeline.py) | Full RARE + EPI + SPEN export, just like `pv360.m`. | `pv360.m` (entire script) |
| [`02_single_spen_reconstruction.py`](02_single_spen_reconstruction.py) | Reconstruct one SPEN scan, inspect `Imag_low` / `Imag_origin` / `Image_SPEN`, save `.mat` + PNG. | The body of the `for ispen = …` loop in `pv360.m` |
| [`03_inspect_intermediate_steps.py`](03_inspect_intermediate_steps.py) | Walk through the reconstruction stage by stage: raw k-space → 1D regridding → readout FFT → SR matrix → final image. | The pipeline inside `Function_Process_NewPE_SPEN_OddNumWithMask_bruker_PV360.m` |
| [`04_compare_with_matlab.py`](04_compare_with_matlab.py) | Compare a Python `.mat` against an existing MATLAB `.mat` (correlation / NRMSE / max-abs-diff + difference image). | What `imshow3Dfull` / a side-by-side `imagesc` would tell you visually. |
| [`05_visualize_phase_maps.py`](05_visualize_phase_maps.py) | Visualize real PV360 odd/even phase-correction diagnostics, masks, and fitted `SmoothPhase` maps. | MATLAB phase-debugging figures around `EvenOddFix*` and `SmoothPhase`. |
| [`06_traditional_reconstruction_step_by_step.ipynb`](06_traditional_reconstruction_step_by_step.ipynb) | Notebook storyboard: synthetic phase truth, estimated phase, k-space, ROFFT image, `AFinal`, `InvA`, error maps, metrics, and optional real PV360 diagnostics. | A teaching version of the traditional reconstruction pipeline with intermediate images exposed. |
| [`07_real_scanner_reconstruction_step_by_step.ipynb`](07_real_scanner_reconstruction_step_by_step.ipynb) | Notebook walkthrough on the real PV360 scanner dataset: RARE/EPI context, raw k-space, regridded k-space, `Imag_origin`, `Imag_low`, fitted phase maps, masks, `Image_SPEN`, and matrix diagnostics. | A real-data teaching version of the traditional PV360 reconstruction pipeline. |
| [`09_compare_phase_estimators.py`](09_compare_phase_estimators.py) | Compare simple odd/even phase-map estimators on a saved `slice_*.mat`: raw wrapped phase, circular Gaussian smoothing, unwrap smoothing, linear polynomial, and quadratic smooth polynomial. | A compact experiment for deciding whether a simpler phase estimator is enough before trying DL. |
| [`10_test_time_phase_network.ipynb`](10_test_time_phase_network.ipynb) | Compare raw phase, the masked traditional quadratic/smoothed-residual method, and a tiny masked test-time coordinate network on one saved scanner slice, with phase maps, reconstructions, residuals, and runtimes. | Test-time replacement for a fixed quadratic phase surface. |
| [`18_minimal_xspen_simulation.py`](18_minimal_xspen_simulation.py) | Generate a two-ellipse phantom, ideal xSPEN raw k-space, and a regularized reconstruction. | The localization equation in the archived `xSPEN1D.m`, followed by a conventional readout FFT. |

---

## Pipeline mapping: `pv360.m` ↔ `spenpy`

| MATLAB step | Python equivalent |
| --- | --- |
| `addpath(genpath('spen')); addpath(genpath('pvtools'));` | `import spenpy` (after `pip install -e .`) |
| `datalist = textread('datalist.txt', '%u');` | `spenpy.cli.pv360.read_datalist(file_dir)` |
| `imageObj = ImageDataObject(rare_pdata); ImageData = squeeze(imageObj.data); ImageData = permute(ImageData, [2,1,3,4])` | `spenpy.bruker.image.read_bruker_2dseq` + `np.squeeze` + axis swap inside `spenpy.cli.pv360.process_rare_epi` |
| `[kField] = ReadBrukerkSpace_PV360_fid_multichannel(spen_dir);` | `spenpy.bruker.raw.read_bruker_kspace_pv360_fid_multichannel` |
| `[kField] = oneD_regriding_PV360(kField, EpiTraj, NSegments, MatrixSize)` | `spenpy.recon.gridding.one_d_regridding_pv360` (uses pure-Python NUFFT loop; see “What does `FGG_Convolution1D.c` become?” below) |
| `kField = Zero_filling_PV6(kField, ZF)` | `spenpy.utils.zero_fill.zero_filling_pv6` |
| `Imag = FFTKSpace2XSpace(CmplxData, 2)` | `spenpy.fft.transform.fft_kspace_to_xspace(..., dim=1)` |
| `[InvA, AFinal] = calcInvA(...)` | `spenpy.core.matrix.calcInvA` |
| `MultMatTensor(InvA, ROFFTedData)` | `spenpy.utils.tensor.mult_mat_tensor` |
| `coilCombinebao(...)` | `spenpy.utils.coil_combine.coil_combine` |
| `Image_SPEN = flip(flip(images, 1), 2)` | `spenpy.recon.spen_recon.orient_pv360_spen_image` (also applied to `Imag_low`/`Imag_origin` so all panels share the same anatomical orientation — see the "Orientation note" below). |
| `imshow3Dfull(...)`, `imagesc(...)`, `figure(...)` | `matplotlib` figures generated by `spenpy.cli.pv360_full.save_volume_figure` and `save_spen_figure` |
| `save(fullfile(export_dir, save_name), 'Imag_low', ...)` | `scipy.io.savemat(save_path, {...})` inside `process_spen` |

### What does `FGG_Convolution1D.c` become?

The MATLAB code uses a compiled MEX file
([`FGG_Convolution1D.c`](../../spen_matlab/spen/FGG_Convolution1D.c) →
`FGG_Convolution1D.mexa64` / `mexw64`) for the inner NUFFT convolution
loop in `FGG_1d_type1.m`.

The Python port re-implements the **same loop in pure Python with NumPy**
in [`spenpy/recon/gridding.py::fgg_1d_type1`](../spenpy/recon/gridding.py).
There is no compiled C/Cython extension. The output is numerically
faithful to the MATLAB MEX (parity checks pass with
`abs_corr > 0.999` against the reference `.mat` files), but it is
considerably slower; expect the regridding step to be the bottleneck on
multi-shot SPEN datasets.

### Orientation note

`pv360.m` only applies `flip(flip(images,1),2)` to `Image_SPEN`, leaving
`Imag_low`/`Imag_origin` in the raw orientation. The Python port
(`spenpy.cli.pv360.process_spen`) applies the same flip to **all three**
images so that side-by-side viewing in
`figures/ratbrain_SPEN_96_<i>.png` shows them in a consistent anatomical
orientation. If you need the unflipped low-resolution arrays for some
analysis (e.g. matching MATLAB byte-for-byte), call
`spenpy.recon.reconstruct_odd_segments(...)` directly and use
`recon.imag_low` / `recon.imag_origin`, which are the pre-flip tensors.

---

## How to run the demos

All scripts assume:

* the project venv at `/home/data1/musong/workspace/python/spen_recons/.venv`,
* the demo Bruker dataset at
  `/home/data1/musong/workspace/python/spen_recons/spen_matlab/data/20240321_204022_lxj_spen_mouse_240321_1_1_1`.

Both can be overridden with CLI flags. Examples:

```bash
# 1) Mimic pv360.m end-to-end into /tmp/pv360_python
.venv/bin/python spenpy/demo/01_run_pv360_pipeline.py \
    --output /tmp/pv360_python

# 2) Reconstruct only the first SPEN scan
.venv/bin/python spenpy/demo/02_single_spen_reconstruction.py \
    --spen-index 1 \
    --output /tmp/pv360_python_single

# 3) Visualise intermediate stages of one SPEN scan
.venv/bin/python spenpy/demo/03_inspect_intermediate_steps.py \
    --spen-index 1 \
    --output /tmp/pv360_python_steps

# 4) Compare a Python .mat vs the existing MATLAB .mat
.venv/bin/python spenpy/demo/04_compare_with_matlab.py \
    --spen-index 1 \
    --output /tmp/pv360_python_compare

# 5) Compare simple phase-map estimators on a saved slice
.venv/bin/python spenpy/demo/09_compare_phase_estimators.py \
    --output /tmp/spen_phase_methods

# 6) Open the notebook walkthrough
jupyter lab spenpy/demo/06_traditional_reconstruction_step_by_step.ipynb

# 7) Open the real scanner data notebook walkthrough
jupyter lab spenpy/demo/07_real_scanner_reconstruction_step_by_step.ipynb
```

Each script prints where the `.mat`, `.png` and `.json` files were
written.

---

## Output anatomy

### `ratbrain_SPEN_96_<i>.mat`

| Variable | Shape | Meaning |
| --- | --- | --- |
| `Imag_low` | `(PE, RO)` | Low-resolution image: complex, just `coilCombine(FFT_along_RO(kfield))`. Pre-SR-matrix view. |
| `Imag_origin` | `(PE, RO)` | Same as `Imag_low` before phase correction (kept for diffing). |
| `Image_SPEN` | `(PE, RO)` | High-resolution SR-corrected image, oriented to match anatomy (`flip(flip(.,1),2)`). |
| `SPEN_AZ` | dict | The encoding matrices used: `OneShotOddInvAZ`, `OneShotEvenInvAZ`, `tmpInvAZ`, `tmpAFinal`. |
| `NSegments` | scalar | Echo-train length (number of shots) read from `method`. Always odd in this demo. |

### `figures/ratbrain_SPEN_96_<i>.png`

3-panel layout: `Imag_low | Imag_origin | Image_SPEN` (all same
orientation), with magnitudes normalised to the 99.5th percentile.

### `pv360_full_summary.json`

Run summary: input/output paths, per-SPEN scan id and figure path, and
any errors collected in `--continue-on-error` mode.
