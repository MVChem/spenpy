# Minimal xSPEN simulation

SPENPy 0.2 adds a deliberately small xSPEN forward model. It is the first
step toward a larger artifact and training-data simulator, not a claim of
scanner-complete xSPEN physics.

## Quick start

```python
import torch

from spenpy.sim import XSPENParameters, XSPENSimulator

params = XSPENParameters(n_xspen=64, n_readout=64, r_value=64)
sim = XSPENSimulator(params)

image = torch.rand(64, 64)
acquisition = sim.acquire(image, noise_std=0.002, seed=7)
reconstruction = sim.reconstruct(acquisition, regularization=1e-3)

print(acquisition.raw_kspace.shape)  # (64, 64), complex64
print(reconstruction.shape)         # (64, 64), complex64
```

Leading dimensions are treated as batch dimensions, so an input shaped
`(B, 64, 64)` produces raw data and a reconstruction with the same shape.

Run the complete visual example with:

```bash
python demo/18_minimal_xspen_simulation.py --output /tmp/xspen_demo.png
```

## Signal model

The implementation follows the compact acquisition equation in the archived
`xSPEN_项目/00_xSPEN_仿真与训练准备/前向物理模型与旧理论仿真/xSPEN1D.m`
source from the local research workspace. The two crossed chirps leave a
bilinear phase between the xSPEN coordinate `y` and an auxiliary coordinate
`z`. During sample `n`, the ideal phase in cycles is

```text
phi(n, y, z) = [C y + gamma Gz t(n)] z
C = -4 (Tp / BW) (gamma Gy) (gamma Gz)
```

For a uniform, on-resonance slice of thickness `Lz`, integrating `z` gives
the localization operator

```text
A[n, j] = sinc([C y(j) + gamma Gz t(n)] Lz)
```

where `sinc(u) = sin(pi u)/(pi u)`. `XSPENSimulator.encoding_matrix`
contains this operator. The acquisition applies `A` along `y`, followed by a
centered orthonormal FFT along the conventional readout direction.

`r_value` controls the nominal resolving power. The sample times preserve the
MATLAB expression `n * deltaT - Ta/2` for `n = 1, ..., N`; therefore the focus
sweeps from `-FOV/2 + FOV/N` through `+FOV/2`. Image values are represented at
voxel centers, and the matrix accounts for the resulting half-voxel offset.
Lower `r_value` values broaden the sinc point-spread function.

## Reconstruction

`XSPENSimulator.reconstruct()` performs two operations:

1. A centered inverse FFT along the regular readout dimension.
2. A Tikhonov-regularized SVD inverse of the xSPEN localization matrix.

The dimensionless `regularization` value is relative to the square of the
largest singular value. A value near `1e-3` is a conservative starting point;
noise-free experiments can use a smaller value. Setting it to zero uses a
truncated pseudoinverse.

## Current assumptions and limits

This first model includes:

- ideal crossed-chirp xSPEN localization;
- a uniform auxiliary slice profile;
- conventional readout FFT/IFFT;
- optional complex Gaussian receiver noise;
- batched real or complex images;
- a matched regularized reconstruction.

It does not yet include:

- off-resonance (`deltaw`) or spatially varying auxiliary profiles;
- relaxation, diffusion, motion, or eddy-current effects;
- multiple coils and sensitivity maps;
- EPI zigzag storage, odd/even phase mismatch, or multishot ordering;
- calibration against a specific Bruker or Siemens xSPEN protocol.

These exclusions are intentional: they keep the first acquisition method
small enough that its forward matrix, focus locations, raw-data convention,
and inverse can all be tested directly. The next physics extension should be
validated against the archived explicit `y-z` summation before it is used for
training data.
