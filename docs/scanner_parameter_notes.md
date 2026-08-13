# Scanner parameters used to shape the simulator

This note records which real-data parameters were used when choosing the new
simulation controls. It is not a full Bruker method-file reference; it is a
practical bridge between the included scanner data and the YAML simulator.

The reference data was scanned from:

```text
spen_matlab/data/lxj_spen_230907.lJ2
spen_matlab/data/20240321_204022_lxj_spen_mouse_240321_1_1_1
```

Across 63 SPEN `method` files in those folders, the important ranges were:

| Bruker key | Observed values/range | Simulator relevance |
| --- | --- | --- |
| `PVM_Matrix` | `96x96`, `128x96`, `96x80` | Sets `scanner.acq_point`. The packaged profile uses `96x96` because it dominates the demo data and matches the traditional reconstruction examples. |
| `PVM_FovCm` | `1.6x1.6`, `3.0x3.0` cm | Sets `scanner.L`. The scanner-like profile uses `1.6 cm`; train separate profiles if the target data mixes FOV regimes heavily. |
| `NSegments` | Always `1` in the included SPEN scans | The real-data baseline is single-shot. Multi-segment simulation is available but should be validated against future real scans before relying on it. |
| `SpenGyGaussStren` | `0.400978` to `2.389375` | Drives the effective chirp strength. `from_bruker_scan(...)` converts this with `SpatEncDuration` into `chirp_rvalue`. |
| `SpatEncDuration` | `9.216` to `19.526` ms | Chirp duration. Shorter durations and stronger gradients imply higher SPEN phase curvature. |
| `PVM_EpiEchoSpacing` / `PVM_EchoTime` | `0.2304`, `0.3072`, `0.384` ms | Motivates `scanner.tblip` and readout/ghost variation. |
| `PVM_EpiEffBandwidth` | `2604.17` to `4340.28` Hz | Affects sensitivity to off-resonance and trajectory errors. |
| `PVM_EpiBandwidth` | `416666.67` Hz when present | Used as the scanner-like `sw_hz`. |
| `PVM_EpiRampTime` | `0.07`, `0.112`, `0.134` ms | Motivates trajectory-shift and line-error augmentation. |
| `PVM_EpiBlipTime` | `0.0585` to `0.1` ms | Motivates PE-direction timing variation. |
| `PVM_EpiReadOddGrad` / `PVM_EpiReadEvenGrad` | approximately `-0.968/0.968` or `-0.528/0.528` | Indicates odd/even readout polarity. This is represented by the even/odd phase block. |
| `PVM_EpiBlipOddGrad` | `0.0879` to `0.2447` | Motivates PE line shift and occasional line dropout simulation. |
| `PVM_EpiReadCenter` | `22`, `23`, `34`, `46` | Suggests readout center changes, represented by small Fourier readout shifts. |
| `PVM_EpiNEchoes` | `80` or `96` | Matches PE matrix and echo train length. |
| `PVM_EpiPhaseCorrection` | 4 receiver rows with small linear/constant terms | Supports making the returned `phase_map` an estimate rather than exact truth. |
| `PVM_EncNReceivers` | Always `4` | The reconstruction path combines receiver channels. The current simulator returns a combined image-like tensor; explicit multi-coil raw simulation is a future extension. |
| `PVM_NAverages` | `8`, `24`, `48`, `60` | Motivates a noise range rather than a fixed noise scalar. |

## Why these parameters matter for DL training

The traditional reconstruction path in `spenpy/demo/README.md` works from
real Bruker raw k-space, applies EPI trajectory handling, corrects phase, and
then applies the SPEN super-resolution matrix. A network trained only on the
old simulator sees a narrow corruption family: one object-dependent even/odd
phase pattern plus fixed-size noise. That can overfit to synthetic artifacts
and fail on scanner data where timing, phase-correction estimates, gain,
trajectory, and off-resonance vary scan to scan.

The new simulator therefore separates the important axes:

| YAML block | Real-data analogue |
| --- | --- |
| `scanner` | Matrix, FOV, chirp strength, echo spacing, bandwidth. |
| `b0` | Off-resonance and field inhomogeneity that warp the SPEN encoding. |
| `shot_phase` | Smooth motion or system phase accumulated during one shot. |
| `even_odd` | EPI Nyquist/odd-even ghost phase that must be corrected before `InvA`. |
| `trajectory` | Ramp timing, readout-center shifts, blip imperfections, and rare missing PE lines. |
| `intensity` | Receiver loading, coil-combination amplitude bias, and contrast variation. |
| `noise` | Average-dependent thermal noise and rare acquisition outliers. |

## Suggested simulation curriculum

1. Train an initial model on `scanner_like.yaml` only.
2. Validate on held-out synthetic data generated with different random seeds.
3. Add 10-25 percent `aggressive_training.yaml` samples once the baseline is
   stable.
4. Keep a validation split using only `scanner_like.yaml` so severe artifacts
   do not hide regressions in normal scanner-like reconstruction quality.
5. When new real scanner datasets are added, run `SpenSim.from_bruker_scan`
   on representative scan directories and save a new YAML profile that matches
   those dimensions and timing values.

## Current limitation

The scanner data is multi-receiver raw k-space, while the simulator currently
emits a combined image-like SPEN tensor matching the old training demo API.
This is intentional to keep existing notebooks and DL code working. A future
raw-coil simulator should add coil sensitivity maps and return
`(B, PE, RO, coil)` data before coil combination.
