---
name: spenpy-library-summary
description: Quick orientation for AI tools reviewing the SPENPy repository, its demo workflow, imports, and reconstruction concepts.
---

# SPENPy quick look

## What this repository is
SPENPy is a small Python simulation library for SPEN reconstruction experiments. It focuses on:
- simulating SPEN acquisition from a grayscale image,
- producing corrupted / phase-affected data,
- reconstructing with an approximate inverse system matrix `InvA`,
- offering a faster degradation path through the forward matrix `AFinal`.

The core implementation is in `spenpy/spen.py`.

## Quick start
**Context:** Given a grayscale image tensor `img` of shape `(B, W, H)`.
Very detailed example can be found in `demo.ipynb`.

**1. YAML-configured scanner-like simulation**
Use the scanner-like profile when generating synthetic data for DL training.
It adds B0, shot phase, even/odd phase, trajectory, intensity, and noise
variation based on the included Bruker scans.
```python
from spenpy.spen import spen

sim = spen.from_yaml("spenpy/configs/scanner_like.yaml", seed=123)
corrupted_data, phase_map, good_lr_image, meta = sim.sim(
    img.unsqueeze(0),
    return_phase_map=True,
    return_good_lr_image=True,
    return_metadata=True,
)
```
Details are in `docs/simulation_yaml.md` and
`docs/scanner_parameter_notes.md`.

**2. Full Simulation (Includes Phase Map Degradation)**
Simulate the full Single Point Encoding (SPEN) sequence to generate corrupted data and its corresponding phase map.
```python
from spenpy.spen import spen
import torch

# Get corrupted k-space data, phase map and good_lr_image
corrupted_data, phase_map, good_lr_image = spen(noise_level=0.01).sim(img.unsqueeze(0), return_phase_map=True, return_good_lr_image=True)

# corrupted data (torch.complex64, torch.Size([B, W, H])) is good_lr_image (torch.complex64, torch.Size([B, W, H])) with phase problem (phase map is torch.float32, torch.Size([B, W/2, H])), good_lr_image can be reconstructed with InvA (torch.complex64, and AFinal is torch.complex64 as well, both of shape torch.Size([B, W, H]))
```

**3. Fast Forward Degradation (No Phase Map)**
For a faster simulation that bypasses phase artifacts, apply the forward matrix `AFinal` directly to the complex image. (This is an approximation, that means the degraded data is similar to good_lr_image)
```python
InvA, AFinal = spen().get_InvA()

# Fast degradation
degraded_data = torch.matmul(AFinal, img.to(torch.complex64))
```

**4. Direct Matrix Reconstruction**
Reconstruct the image from the data using the inverse matrix `InvA`. *(Note: `InvA * AFinal` is not a perfect identity matrix, so minor reconstruction artifacts are expected).*
```python
# Simple reconstruction
reconstructed_img = torch.matmul(InvA, degraded_data)
```

**5. Full Reconstruction with Phase Correction**
To reconstruct the fully simulated `corrupted_data`, apply the `phase_map` to correct the even lines before matrix multiplication.
```python
# Correct the phase of the even lines
even_lines = corrupted_data[:, 1::2, :].clone()
even_lines *= torch.exp(-1j * phase_map)

# Replace even lines and reconstruct
corrected_data = corrupted_data.clone()
corrected_data[:, 1::2, :] = even_lines

# Final reconstruction
final_reconstruction = torch.matmul(InvA, corrected_data)
```

## Main files to read first
- `readme.md` — demo examples and installation instructions.
- `demo.ipynb` — best practical overview of intended usage.(!!!The most important file!!!)
- `spenpy/spen.py` — main simulation and reconstruction logic.
