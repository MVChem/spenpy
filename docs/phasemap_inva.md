# Referenceless PhaseMap and spatial super-resolution

Public names describe MRI processing. `recon/even_odd.py` and `recon/super_resolution.py` replace the old `notebook.py`; `recon/hybrid_spen.py` identifies the actual sequence handled by the former `recon/siemens.py`.

## SPEN180 / xSPEN: direct correction of even acquired lines

`reconstruct_even_odd_inva(ro_image, inv_a, odd_inv_a, even_inv_a)` accepts one complex `[PE, RO, coil]` frame. Separate odd/even matrices first reconstruct partial images. Their multi-coil cross-product provides the even-minus-odd phase. The fitted phase then multiplies the original even ROFFT lines by `exp(-i*phase)` before the full Gaussian-windowed InvA. The original line magnitudes remain unchanged by the correction.

`estimator="quadratic"` performs circular six-coefficient polynomial fitting plus a masked 11×11 smoothed wrapped residual. `estimator="tiny"` uses a 2-8-8-1 Tanh phase network with the historical initialization and optimization. Both fit each acquisition; neither is a trained reusable image prior. Phase fitting is detached, while the corrected signal and matrix products preserve signal gradients. An explicit phase map can also be differentiated. Adaptive coil combination is available; derivatives at degenerate covariance eigenvalues are not guaranteed.

Provenance is the sibling `spen_recons/spenpy/demo/09_hand_by_hand_phase_map.ipynb` and `19_spen_original_tinyphase_inva.ipynb`. Their filenames remain in regression tests and source notes, not in public method names.

## Encoding-specific InvA matrices

`calc_inva_matrices(protocol)` returns full, odd and even native matrices for SPEN180 or xSPEN. SPEN180 uses the historical `calcInvA`, cm units and Gaussian width 0.8. For focus offset d, full/odd/even origins are d, d/2 and (d+1)/2. The simulator's SPEN signal is rescaled by FOV_cm/sqrt(Npe) to this convention.

xSPEN uses its actual crossed-chirp sinc kernel at full/half-grid focus positions, multiplies by a Gaussian window and takes the conjugate transpose. Its window sigma is `width*N_pixels^2/(2*R)` pixels. This is a simulation reconstruction extension requiring independent crossed-chirp scanner validation.

`calc_hybrid_spen_matrices(Npe, FOV_m, R)` returns a dictionary containing full A, Gaussian-windowed InvA, a rectangular half-resolution A, and distinct odd/even inverses. It follows the original Siemens width 0.3, `round(0.9*Npe/2)` phase grid, and seeded border perturbations. These matrices are not interchangeable with the SPEN180/xSPEN tuple.

An InvA here is a windowed adjoint, not a strict matrix inverse. Physical data-consistency gradients must use `EncodingOperator.adjoint`.

## Hybrid SPEN-Diff2: decode / correct / re-encode

`reconstruct_hybrid_spen(acquisition, shift_pixels=...)` implements the historical `Reffless1ShotHybridSPENEvenOddFix` pipeline. It applies the position-phase ramp, centred RO FFT, low-resolution odd/even reconstruction, polynomial/unwrap phase estimation, correction of the low-resolution even image, and re-encoding of that image to the acquired even rows. The full width-0.3 InvA and RSS follow.

This changes the corrected signal through its low-resolution projection; it is intentionally different from direct multiplication on the acquired lines. The low-level `reconstruct_phasemap_inva` exposes this decode/correct/re-encode construction with explicit matrices. `estimator="polynomial_unwrap"` identifies the historical phase estimator (formerly named `siemens`); `circular` is also available in the low-level routine.

For real acquisitions, pass the scan-wide result of `estimate_hybrid_spen_shift(scan)` into each selected frame. Without an explicit shift, the routine reports that it estimated from that single frame. The real path requires prepared/regridded data with oversampling and reflections handled. Simulated `uniform_kspace` is accepted only with explicit Hybrid SPEN-Diff2 provenance metadata.

## Simulation adapter and comparison

`reconstruct_simulated_inva(acquisition, operator)` checks that the operator is nominal and fully sampled. It applies the ideal models' voxel-centred Fourier inverse and direct even-line correction for SPEN180/xSPEN. Hybrid SPEN-Diff2 dispatches to the historical hybrid pipeline, using the protocol's known simulation geometry and estimating the odd/even phase. It records `shift_calibration="simulation_protocol"`; it does not present the configured shift as an estimate.

The three-row comparison uses 256×256 acquisition and reconstruction grids for all three protocols by default; `--matrix-size` controls all rows. This overrides the Hybrid SPEN-Diff2 native 60×64 acquisition dimensions for the comparison while preserving its other protocol parameters. CG uses the same estimated odd/even phase (circularly interpolated from the hybrid half grid when needed), known synthetic sensitivities and known protocol geometry. It does not use the injected odd/even phase as a supplied solution.

The separate four-volume example uses nominal PGSE b-matrices and a prescribed synthetic diffusion tensor. It demonstrates the acquisition/reconstruction interface, not calibrated real diffusion anatomy. See [physics.md](physics.md) for units and model scope, and [validation.md](validation.md) for checks.
