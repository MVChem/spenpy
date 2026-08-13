# YAML-driven SPEN simulation

This document describes the configurable simulator used by
`spenpy.sim.SpenSim` and the backward-compatible `spenpy.spen.spen` alias.
The goal is to create synthetic data that is more diverse than the original
single-image demo while still exposing scanner-relevant controls.

## Quick start

```python
from spenpy.spen import spen

sim = spen.from_yaml("spenpy/configs/scanner_like.yaml", seed=123)
corrupted, phase_map, good_lr, meta = sim.sim(
    image_batch,
    return_phase_map=True,
    return_good_lr_image=True,
    return_metadata=True,
)
```

To generate scanner-like raw data with receiver channels and a Bruker-shaped
`kfield`:

```python
from spenpy.sim import SpenSim

sim = SpenSim.from_yaml("spenpy/configs/scanner_like.yaml", seed=123)
sample = sim.sim_scanner_raw(image_batch, num_coils=4)

# Training-friendly tensors:
y_raw = sample.raw_rofft          # [B, PE, RO] coil-combined complex signal
y_coils = sample.raw_rofft_coils  # [B, PE, RO, receiver]
kfield = sample.kfield            # [B, RO, PE, slice, receiver, echo]

# Scanner-reconstruction path for one sample:
single_kfield = sample.kfield_for_recon(0)
params = sample.metadata["bruker_params"]
```

The old API still works:

```python
from spenpy.spen import spen

corrupted, phase_map, good_lr = spen(noise_level=0.01).sim(
    image_batch,
    return_phase_map=True,
    return_good_lr_image=True,
)
```

## Return values

`sim.sim(...)` always returns the corrupted readout-FFT image first.
Optional flags append more values in this order:

| Flag | Output | Shape | Meaning |
| --- | --- | --- | --- |
| always | `corrupted` | `(B, PE, RO)` complex64 | Simulated SPEN data after phase, trajectory, and noise artifacts. |
| `return_phase_map=True` | `phase_map` | `(B, PE/2, RO)` float32 | Estimated even-line phase map. It intentionally may differ from truth if `estimate_error_std_rad` is nonzero. |
| `return_good_lr_image=True` | `good_lr` | `(B, PE, RO)` complex64 | Clean low-resolution SPEN image before artifact corruption. |
| `return_metadata=True` | `meta` | dict | Effective config, sampled random values, true phase map, shot phase map, and noise/dropout events. |

The phase-correction convention is unchanged:

```python
corr = corrupted.clone()
corr[:, 1::2, :] *= torch.exp(-1j * phase_map)
recon = torch.matmul(InvA, corr)
```

`sim.sim_scanner_raw(...)` returns a `SimulatedScannerRaw` dataclass:

| Attribute | Shape | Meaning |
| --- | --- | --- |
| `object_gt` | `(B, H, W)` | Input object used as supervised ground truth. |
| `kfield` | `(B, RO, PE, slice, receiver, echo)` complex64 | Scanner/reconstruction-shaped raw k-space. |
| `raw_rofft_coils` | `(B, PE, RO, receiver)` complex64 | Per-receiver signal after RO FFT, matching the pre-combine scanner intermediate. |
| `raw_rofft` | `(B, PE, RO)` complex64 | Adaptive coil-combined raw signal, analogous to `Imag_origin`. |
| `good_lr_coils` | `(B, PE, RO, receiver)` complex64 | Clean per-receiver low-resolution signal before even/odd and noise artifacts. |
| `good_lr` | `(B, PE, RO)` complex64 | Coil-combined clean low-resolution signal. |
| `phase_map_true` | `(B, PE/2, RO)` float32 | True even-line phase applied to the raw signal. |
| `phase_map_estimate` | `(B, PE/2, RO)` float32 | Estimated phase map with optional configured error. |
| `inv_a`, `a_final` | `(PE, PE)` complex64 | Scanner-convention reconstruction and forward matrices. |
| `coil_sensitivities` | `(B, receiver, RO, PE * oversample_pe)` complex64 | Synthetic receiver sensitivity maps used before encoding. |

Use `sample.kfield_for_recon(i)` or `sample.kfield_numpy(i)` when passing one
synthetic scan into `reconstruct_odd_segments(..., kfield=...)`.

## Packaged profiles

| Profile | Purpose |
| --- | --- |
| `spenpy/configs/legacy_like.yaml` | A readable version of the historical single-shot behavior. |
| `spenpy/configs/scanner_like.yaml` | Default recommendation for synthetic training data. It uses the 96x96, 1.6 cm Bruker PV360 regime and moderate random artifacts. |
| `spenpy/configs/aggressive_training.yaml` | Wider stress-test domain. Mix this with `scanner_like.yaml`; do not train only on this profile unless the model is expected to handle severe outliers. |

## Top-level YAML structure

```yaml
version: 1

scanner:
  L: [1.6, 1.6]
  acq_point: [96, 96]
  nseg: 1
  chirp_rvalue: 150.0
  tblip: 0.0002304
  gamma_hz: 4257.4
  sw_hz: 416666.6667
  oversample_pe: 16
  a_sign: -1
  gauss_relative_width: 0.8

randomization:
  seed: 123

artifacts:
  b0: ...
  shot_phase: ...
  even_odd: ...
  trajectory: ...
  intensity: ...
  noise: ...

scanner_raw:
  num_coils: 4
  coil_combine: adaptive
  coil_sensitivity: ...
  receiver: ...
```

YAML values are merged over the built-in defaults. You can specify only the
fields you want to change.

## Scanner block

| Key | Meaning |
| --- | --- |
| `L` | Field of view in centimeters for the two simulator axes. |
| `acq_point` | Output matrix `[RO, PE]`. Existing demos use square images, so the output is `(B, PE, RO)`. |
| `nseg` | SPEN segment count. Current real demo scans are single-shot (`1`), but the simulator accepts any divisor of `PE`. |
| `chirp_rvalue` | Effective chirp R value used to build the SPEN quadratic phase. |
| `tblip` | Blip/readout timing term in seconds. Scanner data shows EPI echo spacing from 0.2304 to 0.384 ms. |
| `sw_hz` | Sampling bandwidth in Hz. |
| `oversample_pe` | Internal PE oversampling used before low-resolution acquisition. |
| `gauss_relative_width` | Weighted-adjoint reconstruction width used by `get_InvA()`. |

Use `SpenSim.from_bruker_scan(scan_dir, config=...)` to initialize matrix,
FOV, segment count, timing, and chirp strength from a Bruker `method` file
while keeping the YAML artifact settings.

## Scanner Raw Block

```yaml
scanner_raw:
  num_coils: 4
  coil_combine: adaptive
  coil_sensitivity:
    enabled: true
    radius: 0.65
    width: 0.85
    floor: 0.05
    phase_winding_range_rad: [-0.45, 0.45]
  receiver:
    gain_range: [0.90, 1.10]
    phase_range_rad: [-3.14159265, 3.14159265]
```

This block is used only by `sim_scanner_raw()` / `sim_raw()`.  It creates
smooth complex receiver sensitivities, receiver gain/phase offsets, per-coil
raw signals, and a Bruker-axis `kfield`.  The default `adaptive` combine uses
the same `coil_combine` implementation as the PV360 reconstruction path.

## Artifact blocks

### B0

```yaml
b0:
  enabled: true
  coef_ranges_cm:
    - [-0.010, 0.010]
    - [-0.025, 0.025]
    - [-0.008, 0.008]
    - [-0.002, 0.002]
```

These four coefficients perturb the PE coordinate with a cubic polynomial.
They create local warping and phase deviations before SPEN encoding.

### Shot phase

```yaml
shot_phase:
  enabled: true
  poly_coeff_ranges_rad:
    - [-0.25, 0.25]   # constant
    - [-0.60, 0.60]   # RO
    - [-0.60, 0.60]   # PE
    - [-0.20, 0.20]   # RO^2
    - [-0.16, 0.16]   # RO*PE
    - [-0.20, 0.20]   # PE^2
  smooth_std_range_rad: [0.0, 0.35]
  smooth_grid: 6
```

This models motion and slowly varying phase during the SPEN acquisition. The
true shot phase is returned in `meta["shot_phase_map"]`.

### Even/odd phase

```yaml
even_odd:
  enabled: true
  constant_range_rad: [-1.2, 1.2]
  linear_range_rad_per_cm: [-4.0, 4.0]
  quadratic_range_rad_per_cm2: [-1.0, 1.0]
  object_phase_scale_range_rad: [0.0, 3.2]
  smooth_std_range_rad: [0.0, 0.35]
  estimate_error_std_rad: 0.06
```

This is the most important block for the current reconstruction workflow.
Only even PE rows are multiplied by `exp(+i * phase)`. The returned
`phase_map` is the estimated correction map, while the truth is available as
`meta["phase_map_true"]`.

### Trajectory

```yaml
trajectory:
  segment_shift_range_cm: [-0.006, 0.006]
  readout_shift_range_px: [-0.8, 0.8]
  phase_shift_range_px: [-0.6, 0.6]
  line_dropout_probability: 0.08
  line_dropout_width: 1
```

This block introduces small acquisition-position errors, Fourier-domain image
shifts, and occasional missing PE lines.

### Intensity

```yaml
intensity:
  gain_range: [0.75, 1.35]
  bias_field_std_range: [0.0, 0.22]
  bias_grid: 5
  gamma_range: [0.85, 1.20]
```

This creates coil-loading-like gain changes, smooth bias fields, and contrast
changes before encoding.

### Noise

```yaml
noise:
  complex_std: [0.002, 0.025]
  relative_to_signal: true
  kspace_spike_probability: 0.03
  kspace_spike_scale: [0.10, 0.55]
```

Complex Gaussian noise is added in PE k-space. If `relative_to_signal` is
true, `complex_std` is scaled by the k-space standard deviation of each
sample. Rare k-space spikes emulate transient acquisition outliers.

## Training recommendation

Start with `scanner_like.yaml` and randomize the input anatomical images.
For robustness, mix in a smaller fraction of `aggressive_training.yaml`
examples. Keep validation separated by profile so you can see whether the
network is learning the nominal scanner regime or only surviving extreme
artifact augmentation.
