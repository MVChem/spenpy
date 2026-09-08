# MRI reference provenance

The example uses **MRHead**, an actual head MRI supplied by the 3D Slicer project. The [official SampleData source](https://github.com/Slicer/Slicer/blob/main/Modules/Scripted/SampleData/SampleData.py) identifies this sample, its checksum, and the donor's unrestricted-use permission.

- Download: [MRHead, SHA256-addressed dataset](https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/cc211f0dfd9a05ca3841ce1141b292898b2dd2d3f08286affadf823a7e58df93).
- SHA256: `cc211f0dfd9a05ca3841ce1141b292898b2dd2d3f08286affadf823a7e58df93`.
- Downloaded size: 6,607,313 bytes.
- Native NRRD dimensions: `256×256×130`, signed 16-bit integers.
- Native voxel direction vectors in LPS millimetres: `(0,1,0)`, `(0,0,-1)`, `(-1.2999954223632812,0,0)`.
- Default reference: `volume[:, :, 65].T`, using zero-based indexing and NRRD Fortran index order. This is a native `256×256` sagittal image with 1 mm in-plane spacing.
- Transformation: convert to float32 and divide by the slice maximum, 173. No interpolation, spatial resizing, denoising, skull stripping, or contrast enhancement is applied to the numerical reference.

Run `examples/compare_reconstructions.py` to download/check the source and select the slice in memory. The original volume and any previously cached slice files are data caches; the example writes only its 3-by-4 `comparison.png` to `outputs/`.

The MRI supplies magnitude anatomy only. The example assumes zero object phase, synthesizes coil sensitivities, and adds new complex acquisition noise. It does not claim that this volunteer was scanned using SPEN or xSPEN.
