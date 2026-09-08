# Physical models and numerical conventions

SPENPy implements three protocols with two underlying encoding families: quadratic-phase SPEN (SPEN180 and Hybrid SPEN-Diff2) and crossed-chirp xSPEN. Sequence labels are explicit; a directory name or a reconstruction figure title does not identify the acquisition physics.

## Shared linear signal operator

```text
coil_object[y,x,c] = S[y,x,c] * image[y,x]
localized[m,x,c]   = sum_y K[m,y,x] * coil_object[y,x,c]
signal[m,k,c]      = mask[m,k] * sum_x F[k,x] * exp(i phase[m,x]) * localized[m,x,c]
```

The adjoint reverses and conjugates these operations, including sensitivities, phase and sampling weights. Noise is added after deterministic encoding. Each image value uses a voxel-centred spatial coordinate; leading batches share the fixed operator. Object/coil maps are constant at the within-pixel PE quadrature points.

## SPEN180

`SPEN180Protocol` describes ideal single-shot encoding with a 180-degree chirp:

```text
a       = encoding_sign * 2*pi*R / FOV_y^2
focus_m = ((m + focus_offset_voxels) / N_spen - 1/2) * FOV_y
K_m(y)  = exp(i*a*(y^2 - 2*focus_m*y))
```

PE uses Gauss-Legendre integration over each image pixel, scaled by sqrt(N_spen)/image_y. The default node count increases with R/image_y and is at least 16. RO uses the positive-sign Fourier integral of rectangular pixels, including its sinc factor and sqrt(N_ro)/image_x scaling. Acquisition and image grids are independent.

`SPENProtocol` and `SPENOperator` remain compatibility aliases of `SPEN180Protocol` and `SPEN180Operator`. The serialized sequence `spen` remains readable; the selector also accepts `spen180`.

## Cross-term xSPEN

`XSPENProtocol` follows the archived `xSPEN1D.m` signal equation:

```text
Ta = N_spen * echo_spacing
Tp = Ta / (4*beta)
BW = R / Tp
Gy = beta*BW / (gamma*FOV_y)
Gz = (1-beta)*BW / (gamma*slice_thickness)
C  = -4*Tp/BW * (gamma*Gy)*(gamma*Gz)
t_m = (m+1)*echo_spacing - Ta/2
```

For a uniform auxiliary slice at zero off-resonance, normalized z integration yields:

```text
K_m(y) = sinc((C*y + gamma*Gz*t_m)*slice_thickness)
sinc(u) = sin(pi*u)/(pi*u)
```

PE voxel integration and scaling N_spen/image_y follow this kernel. RO uses the same Fourier pixel integral as SPEN180. The focus moves across the FOV, so RO-only data can already resemble a spatial image. This is still a band-limited response; it does not imply lossless high-resolution imaging.

A supplied nonnegative `slice_profile[n_z]` selects explicit normalized z integration. With `b0_hz`, the phase in cycles is `(t_m - 4*Tp/BW*gamma*Gy*y)*(gamma*Gz*z + delta_f)`, with the archived rectangular RF selection masks. Object and B0 are constant through z; RF amplitudes, adiabaticity and Bloch evolution are not integrated. RF-mask/n_z convergence needs checking for the chosen field.

## Hybrid SPEN-Diff2

`HybridSPENDiff2Protocol` is a reduced model associated with the historical Siemens `esrs_hyb_spen_Diff2` reconstruction. The MID253 defaults are:

| Parameter | Value |
|---|---:|
| Acquisition PE × RO | 60 × 64 |
| FOV PE × RO | 180 × 192 mm |
| PE R / RF duration | 60 / 14.080 ms |
| RO refocusing R / duration | 20 / 4.096 ms |
| Echo spacing / TE | 470 us / 52 ms |
| Slice thickness | 3 mm |
| Nominal PGSE b-value | 600 s/mm² |

The WIP mapping is `alFree[12:16] = [PE R, PE duration, RO R, RO duration]`, following the local sequence enum; nominal b is `adFree[0]`. The file/protocol name containing Ortho900 is not used to infer b. `from_siemens_header` validates this sequence and requires the fields; the reader never silently supplies defaults for missing metadata.

The PE kernel uses the same second-order pixel-moment expansion as `CalcSRMatrixApprox.m`, with stable small-argument limits:

```text
L = FOV_y in cm
a = 2*pi*R/L^2
k_m = -2*a*m*L/N_spen
b = a*L
A[m,j] ≈ integral_pixel_j exp(i*(a*y^2 + (b+k_m)*y)) dy
```

`calc_hybrid_spen_encoding` builds this matrix on the object's own PE grid. Values have the historical cm-integral gain. No Gaussian window, phase fit or reconstructed target enters the forward model. The pixel approximation is not an exact oscillatory integral at arbitrary coarse grids/high R; independent HR quadrature and native-grid MATLAB parity are tested. The constructor's generic `quadrature` setting controls the ideal SPEN/xSPEN integrals, not this historical moment expansion.

RO uses a centred positive-sign discrete Fourier basis with 1/image_x scaling. On the native grid it is exactly `fftshift(ifft(ifftshift(...)))`; `FFTKSpace2XSpace.m` reverses it. On an independent HR grid it evaluates/truncates the Fourier quadrature at the acquisition frequencies, with a phase origin aligned to the native grid. This discrete RO convention has no extra rectangular-pixel sinc factor. It is distinct from the ideal models' unitary Fourier-pixel convention.

`spen_shift_pixels=d` multiplies PE sample m by `exp(-4*pi*i*R*d*(m+1)/N_spen^2)`, the inverse of the historical position-phase correction. It does not alter raw RO magnitudes. Simulated reconstruction uses the configured geometry d; the odd/even phase is still estimated from data. A real scan may use scan-wide `estimate_hybrid_spen_shift`. That estimator has phase-origin ambiguities and is not guaranteed to identify the origin of an arbitrary synthetic phantom.

The current forward equation is calibrated to the historical matrix convention. RF duration, RO refocusing R/duration, TE and slice thickness are stored acquisition parameters; there is no assertion that changing them alone models full RF/gradient dynamics. The PE phase law depends on the chirp time-bandwidth product; RO refocusing phase and intra-readout evolution remain outside this reduced model. Scanner waveform/binary version equivalence has not been established.

## Diffusion volumes

`simulate_hybrid_spen_diffusion` computes a diffusion-weighted object before encoding:

```text
rho_v(y,x) = rho_0(y,x) * exp(-sum_ij B_v,ij(y,x)*D_ij(y,x))
```

B is in s/mm², D in mm²/s, with axes (RO, PE, SS). Symmetric positive-semidefinite tensors are required; spatially varying B and D and both off-diagonal contributions are retained. The output acquisition axes are `[volume, spen, readout, coil]`; target axes are `[volume, image_y, image_x]`. Gradients propagate to image and D. All volumes share the encoding operator.

Default B consists of one zero matrix plus three nominal orthogonal PGSE matrices. It omits imaging-gradient diffusion and cross terms. Supply a calibrated spatial B from the historical b-matrix computation when required. This is an effective Gaussian diffusion-tensor model, not a propagator/Bloch-Torrey simulation. No intrinsic b-map is invented from R or TE. If the input already contains intrinsic encoding/relaxation contrast, the caller must avoid counting it again.

## Additional effects and limits

SPEN180 and Hybrid SPEN-Diff2 require explicit `effective_times_s` whenever B0 is supplied; ordinary readout times do not establish off-resonance refocusing through RF pulses. T2 requires explicit nonnegative `decay_times_s`. These options apply fixed line-level factors. The InvA adapters reject non-nominal, masked or nonseparable kernels; use the matching forward/adjoint CG solver for those cases.

The models produce uniform/regridded samples after bipolar ordering. They do not generate vendor binary streams, ramp ADC points, gradient nonlinearities, motion, multiband, full 3D spin dynamics, dedicated QxSPEN or a waveform-driven Bloch calculation. Generating/reconstructing with a shared A checks solver consistency, not scanner physics.

The Siemens reader uses twixtools; its ramp interpolation differs from MATLAB mapVBVD. Prior RO-only magnitude disagreement on MID253 was about 1.47% after scale alignment. Same prepared-input phase/matrix parity has been checked, while independent raw-path shift/reconstruction equivalence remains unestablished. `calibrated_to_scanner` / `calibrated_to_simulator` remain false.

Source references: `../xSPEN_项目/.../xSPEN1D.m`, `CalcSRMatrixApprox.m`, `Reffless1ShotHybridSPENEvenOddFix.m`, `Trio_brain/CalcbMat_NewBrainMS.m`, `Trio_brain/calcSPEN180_SIEMENSdata_v2BrainMS.m`, and the `esrs_hyb_spen_Diff2` Siemens source. The older `SuperResolution_VariableSize_Sim_V3.m` ideal-flip phase evolution is a separate reference and has not been relabelled as this forward implementation.
