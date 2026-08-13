# SPENPy

SPENPy is a Python toolkit for SPEN (Spatiotemporally Encoded) MRI work. It
covers two related workflows:

1. Simulation: generate SPEN-encoded and phase-corrupted images from a clean
   reference image, build `AFinal` and `InvA`, and test reconstruction or
   ghost-correction ideas.
2. Bruker reconstruction: read ParaVision PV5 or PV360 studies, reconstruct
   SPEN scans, and export MATLAB-compatible `.mat` files plus optional PNG
   summaries.

The reconstruction code follows the MATLAB reference scripts in
[`spen_matlab/pv5.m`](../spen_matlab/pv5.m) and
[`spen_matlab/pv360.m`](../spen_matlab/pv360.m). The MATLAB MEX kernel used by
`FGG_1d_type1` is replaced by a pure-Python implementation in
[`spenpy/recon/gridding.py`](spenpy/recon/gridding.py), so the package does not
need compiled MATLAB extensions.

## Installation

From this package directory:

```bash
python -m venv .venv
.venv/bin/pip install -e .[dev]
.venv/bin/pip install matplotlib
```

If you are working from the parent workspace where this package is available as
`./spenpy`, install it with:

```bash
python -m venv .venv
.venv/bin/pip install -e ./spenpy[dev]
.venv/bin/pip install matplotlib
```

Core runtime dependencies are declared in [`pyproject.toml`](pyproject.toml).
`matplotlib` is optional, but needed for PNG figures and most demos.

| Package | Purpose |
| --- | --- |
| `numpy` | Array operations and Bruker binary reshaping |
| `torch` | Encoding matrices, FFT path, and tensor reconstruction |
| `scipy` | `.mat` I/O and trajectory smoothing |
| `pillow` | Demo image loading |
| `PyYAML` | Simulation configuration files |
| `matplotlib` | Optional demo and reconstruction figures |

Console entry points:

```bash
spenpy             # python -m spenpy.cli.pv360
spenpy-pv360-full  # python -m spenpy.cli.pv360_full
```

In the `spen_recons` workspace, the existing `uv.lock` environment can also be
used directly:

```bash
uv run python spenpy/demo/01_run_pv360_pipeline.py --help
```

## Quick Start

### Reconstruct Bruker PV5 or PV360 Data

Use the full pipeline when you want the same top-level outputs as the MATLAB
scripts: RARE, EPI, every selected SPEN scan, a JSON summary, and optional PNG
figures.

```bash
python -m spenpy.cli.pv360_full \
    --file-dir <bruker_study_dir> \
    --output /tmp/spen_recon
```

From the `spen_recons` workspace:

```bash
uv run python spenpy/demo/01_run_pv360_pipeline.py \
    --file-dir <bruker_study_dir> \
    --output /tmp/spen_recon
```

SPENPy auto-detects the supported datalist layouts:

| Mode | Datalist convention | Reconstruction behavior |
| --- | --- | --- |
| PV360 | `RARE, EPI, SPEN, SPEN, ...` | Each SPEN scan supplies its own trajectory and uses PV360 regridding. |
| PV5 | `RARE, trajectory, SPEN, trajectory, SPEN, ...` | Each SPEN scan is paired with the preceding trajectory scan and uses the PV5/PV6 regridding variant. |

You can force the interpretation if auto-detection is not what you want:

```bash
python -m spenpy.cli.pv360_full \
    --file-dir <bruker_study_dir> \
    --output /tmp/spen_recon \
    --pv-version pv5
```

Useful run controls:

```bash
# Reconstruct only the first three selected SPEN scans.
python -m spenpy.cli.pv360_full --file-dir <study> --output /tmp/spen --max-spen 3

# Reconstruct specific 1-based SPEN indices.
python -m spenpy.cli.pv360_full --file-dir <study> --output /tmp/spen --spen-index 1,5,10

# Continue after a failed scan and record errors in the summary JSON.
python -m spenpy.cli.pv360_full --file-dir <study> --output /tmp/spen --continue-on-error

# Skip PNG generation.
python -m spenpy.cli.pv360_full --file-dir <study> --output /tmp/spen --no-figures
```

Output layout:

```text
/tmp/spen_recon/
+-- ratbrain_RARE.mat
+-- ratbrain_EPI.mat
+-- ratbrain_SPEN_96_1.mat
+-- ratbrain_SPEN_96_2.mat
+-- ...
+-- pv360_full_summary.json
+-- figures/
    +-- ratbrain_RARE.png
    +-- ratbrain_EPI.png
    +-- ratbrain_SPEN_96_1.png
    +-- ...
```

Programmatic API:

```python
from spenpy.cli.pv360_full import run_pv360_full

summary = run_pv360_full(
    file_dir="<bruker_study_dir>",
    export_dir="/tmp/spen_recon",
    save_figures=True,
    pv_version="auto",  # "auto", "pv5", or "pv360"
)
```

### Simulate SPEN Data

```python
import numpy as np
import torch
from PIL import Image
from spenpy.spen import spen

img_pil = Image.open("spenpy/data/brain.png").convert("L")
img = torch.from_numpy(np.asarray(img_pil, dtype=np.float32) / 255.0)

InvA, AFinal = spen(acq_point=list(img.shape)).get_InvA()

corrupted, phase_map, good_lr = spen(noise_level=0.01).sim(
    img.unsqueeze(0),
    return_phase_map=True,
    return_good_lr_image=True,
)

# Phase correction must happen before the InvA reconstruction.
corr = corrupted.clone()
corr[:, 1::2, :] = corrupted[:, 1::2, :] * torch.exp(-1j * phase_map)
recon = torch.matmul(InvA, corr)
```

Simulation profiles are stored in [`spenpy/configs/`](spenpy/configs/):

| Profile | Purpose |
| --- | --- |
| `legacy_like.yaml` | Keep behavior close to the original notebook demo. |
| `scanner_like.yaml` | Use a scanner-like 96 x 96 Bruker regime for synthetic training. |
| `aggressive_training.yaml` | Expand artifact ranges for robustness testing. |

```python
from spenpy.spen import spen

sim = spen.from_yaml("spenpy/configs/scanner_like.yaml", seed=123)
corrupted, phase_map, good_lr, meta = sim.sim(
    img.unsqueeze(0),
    return_phase_map=True,
    return_good_lr_image=True,
    return_metadata=True,
)
```

See [`demo.ipynb`](demo.ipynb) for the simulation walkthrough,
[`docs/simulation_yaml.md`](docs/simulation_yaml.md) for the YAML schema, and
[`docs/scanner_parameter_notes.md`](docs/scanner_parameter_notes.md) for scanner
parameter notes.

## Repository Layout

```text
spenpy/
+-- README.md
+-- demo.ipynb
+-- demo/
|   +-- README.md
|   +-- 01_run_pv360_pipeline.py
|   +-- 02_single_spen_reconstruction.py
|   +-- 03_inspect_intermediate_steps.py
|   +-- 04_compare_with_matlab.py
|   +-- 05_visualize_phase_maps.py
+-- docs/
+-- spenpy/
|   +-- bruker/        # Bruker parameter, 2dseq, and raw FID readers
|   +-- cli/           # End-to-end reconstruction entry points
|   +-- core/          # SPEN encoding matrix builders
|   +-- fft/           # MATLAB-style FFT wrappers
|   +-- recon/         # Regridding, phase correction, and reconstruction
|   +-- sim/           # Simulation helpers
|   +-- utils/         # Coil combine, zero filling, smoothing, tensor helpers
|   +-- spen.py        # Main simulation-facing class
+-- tests/
+-- pyproject.toml
+-- setup.py
+-- LICENSE
```

## Module Map

| Area | Module | Main entry points |
| --- | --- | --- |
| Bruker I/O | `spenpy.bruker.param` | `read_pv_param` |
| Bruker I/O | `spenpy.bruker.image` | `read_bruker_2dseq` |
| Bruker I/O | `spenpy.bruker.raw` | `read_bruker_kspace_pv360_fid_multichannel` |
| Reconstruction | `spenpy.recon.gridding` | `one_d_regridding_pv360`, `one_d_regridding_pv6`, `fgg_1d_type1` |
| Reconstruction | `spenpy.recon.phase` | `apply_pv360_one_shot_phase_correction` |
| Reconstruction | `spenpy.recon.spen_recon` | `reconstruct_odd_segments`, `orient_pv360_spen_image` |
| Encoding | `spenpy.core.matrix` | `calcSRMatrixApprox`, `calcInvA` |
| Utilities | `spenpy.utils.tensor` | `mult_mat_tensor` |
| Utilities | `spenpy.utils.coil_combine` | `coil_combine` |
| Utilities | `spenpy.utils.zero_fill` | `zero_filling_pv6`, `rm_zero_filling_pv6` |
| Simulation | `spenpy.spen` | `spen.sim`, `spen.get_InvA`, `spen.from_yaml` |
| CLI | `spenpy.cli.pv360` | `run_pv360`, `process_spen`, `process_rare_epi` |
| CLI | `spenpy.cli.pv360_full` | `run_pv360_full`, figure writers, PV5/PV360 datalist handling |

## MATLAB to Python Mapping

| MATLAB reference | Python equivalent |
| --- | --- |
| `textread('datalist.txt', '%u')` | `spenpy.cli.pv360.read_datalist` |
| `ImageDataObject(...).data` | `spenpy.bruker.image.read_bruker_2dseq` |
| `ReadPVParam(...)` | `spenpy.bruker.param.read_pv_param` |
| `ReadBrukerkSpace_PV360_fid_multichannel(...)` | `spenpy.bruker.raw.read_bruker_kspace_pv360_fid_multichannel` |
| `oneD_regriding_PV360(...)` | `spenpy.recon.gridding.one_d_regridding_pv360` |
| `oneD_regriding_PV6(...)` in the PV5 path | `spenpy.recon.gridding.one_d_regridding_pv6` |
| `FGG_1d_type1(...)` plus MATLAB MEX convolution | `spenpy.recon.gridding.fgg_1d_type1` |
| `FFTKSpace2XSpace(...)` | `spenpy.fft.transform.fft_kspace_to_xspace` |
| `calcInvA(...)` | `spenpy.core.matrix.calcInvA` |
| `MultMatTensor(...)` | `spenpy.utils.tensor.mult_mat_tensor` |
| `coilCombinebao(...)` | `spenpy.utils.coil_combine.coil_combine` |
| `flip(flip(images,1),2)` | `spenpy.recon.spen_recon.orient_pv360_spen_image` |
| `save(..., 'Imag_low', 'Imag_origin', 'Image_SPEN', 'SPEN_AZ')` | `scipy.io.savemat` inside `process_spen` |

### Orientation Note

The MATLAB PV360 script flips only `Image_SPEN`, while `Imag_low` and
`Imag_origin` remain in the raw orientation. SPENPy applies the same final
orientation to all three exported images so the generated figures line up
anatomically. If you need the pre-flip tensors, call
`reconstruct_odd_segments(...)` directly and read `recon.imag_low` and
`recon.imag_origin`.

## Demos

| File | Purpose |
| --- | --- |
| [`demo.ipynb`](demo.ipynb) | Simulation walkthrough on a 2D image. |
| [`demo/01_run_pv360_pipeline.py`](demo/01_run_pv360_pipeline.py) | Full RARE, EPI, and SPEN export for PV5/PV360 studies. |
| [`demo/02_single_spen_reconstruction.py`](demo/02_single_spen_reconstruction.py) | Reconstruct one SPEN scan and inspect the main outputs. |
| [`demo/03_inspect_intermediate_steps.py`](demo/03_inspect_intermediate_steps.py) | Inspect raw k-space, regridding, readout FFT, SR matrix, and final image. |
| [`demo/04_compare_with_matlab.py`](demo/04_compare_with_matlab.py) | Compare Python `.mat` output against MATLAB output. |
| [`demo/05_visualize_phase_maps.py`](demo/05_visualize_phase_maps.py) | Experimental phase-correction diagnostics and visualization. |

## Testing and Validation

Install test dependencies first:

```bash
.venv/bin/pip install -e .[dev]
```

Pure Python unit tests:

```bash
.venv/bin/python -m pytest \
    tests/test_core.py \
    tests/test_fft.py \
    tests/test_tensor.py \
    tests/test_sim_config.py
```

Or, with `uv`:

```bash
uv run --extra dev python -m pytest \
    tests/test_core.py \
    tests/test_fft.py \
    tests/test_tensor.py \
    tests/test_sim_config.py
```

Bruker and MATLAB-oriented checks are environment dependent:

| File | How to use it |
| --- | --- |
| `tests/test_pv360.py` | Script-style smoke test. Run it with an explicit Bruker study path instead of collecting it directly with pytest. |
| `tests/test_pv360_matlab_parity.py` | Parity checks against local MATLAB exports; requires local data and MATLAB outputs. |
| `tests/verify_pv360_reconstruction.py` | Visual and numeric verification harness that writes comparison figures and metrics. |
| `tests/test_phase_diagnostics.py` | Optional diagnostics tests for phase-correction internals. |

Example smoke test:

```bash
python tests/test_pv360.py <bruker_study_dir> --component full --output /tmp/spen_smoke
```

## Current Scope

The primary reconstruction path covers odd-segment SPEN scans, which are the
datasets used by the included PV5 and PV360 workflows. Even-segment SPEN scans
still route to the MATLAB `Function_Process_NormalmultiSPEN_bruker_PV6` path and
are not implemented in Python yet.

### Processed Siemens SPEN data

The copied workspace version also includes an adapter for Siemens MATLAB
exports containing `SignalFixedPostROFFTPostSR_*`. These arrays have already
passed readout FFT and SPEN super-resolution in the Siemens ICE pipeline, so
the adapter performs only the remaining Walsh adaptive receiver combination
and magnitude averaging across repetitions.

```bash
python demo/15_reconstruct_siemens_mat.py \
    /home/data2/chk/workspace/2026/07/31/data/*.mat \
    --output /home/data2/chk/workspace/2026/07/31/recon_spenpy
```

Outputs for each scan are:

- a compressed NPZ with adaptive and RSS magnitude volumes;
- an index-space NIfTI (voxel spacing is set, scanner-space orientation is
  intentionally left unvalidated);
- a multi-slice, multi-set montage;
- JSON metadata recording axes, protocol, sequence, b-value, and processing.

The adapter is in `spenpy/siemens/processed.py`. The vectorized Walsh
implementation is `spenpy.utils.coil_combine.coil_combine_batched`.

## License

MIT. See [`LICENSE`](LICENSE).

## References

* Liu et al., *Unsupervised deep learning model for correcting Nyquist ghosts
  of single-shot spatiotemporal encoding*, included in this folder as
  the Liu et al. PDF.
* MATLAB reference guide:
  [`spen_matlab/SPEN_RECONSTRUCTION_GUIDE.md`](../spen_matlab/SPEN_RECONSTRUCTION_GUIDE.md).
* MATLAB entry points:
  [`spen_matlab/pv5.m`](../spen_matlab/pv5.m) and
  [`spen_matlab/pv360.m`](../spen_matlab/pv360.m).
