# Trained RAD levels

`encode_trained_rad(directory, output_dir, levels, moment_factor, resolution_factor)`
encodes `level-1/point_cloud.ply` through `level-N/point_cloud.ply`.
`parents-L.bin` is a little-endian u32 array mapping each row of level L to a row
of level L+1. These are actual merge memberships, not nearest-center links.
The encoder retains all trained nodes and creates semantic spatial ancestors
above the final trained level. `moment_factor` converts projected kernel shape
to second moments (1 for k=1, 1/sqrt(2*pi) for k=2).

The output is `bay-lod.rad` and its `bay-lod-*.radc` chunks, including class and
instance properties. PLY label numbering is retained; instance IDs become ID+1
for the frontend, with zero representing unknown.

The optional `lod_size` f32 RAD property separates selection bounds from rendered
scales. Bounds contain descendants and maintain distinct trained-level intervals.
Both packed and extended WASM decoders use these sizes for traversal; older RADs
without this property retain their geometry-derived bounds. Deploy this decoder
with the new encoder. The rendered scales and opacity are not inflated.

The CropVision pipeline's `test_trained_rad.py` covers exported level retention,
f16 selection ordering and class/instance ID round trips through the encoder.
