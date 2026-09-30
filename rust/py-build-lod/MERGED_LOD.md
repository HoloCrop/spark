# GPU Bhatt RAD encoding

`encode_merged_archive(leaves, parents, children, output_file, lod_base)` accepts
a complete GPU-built binary hierarchy and writes its RAD header and chunks
directly into a stored ZIP. There are no intermediate RAD files.

Leaves use the `TrainedLevel` buffer layout. Parents use `MergedLevel(position,
rotation, scales, opacity, labels, right_weight)`: float32 positions (N,3), xyzw
quaternions (N,4), scales (N,3), raw opacities (N,), right-child mass weights (N,),
and int32 class/instance labels (N,2). Labels use -1 for unknown and zero for a
real ID. Parent opacity can exceed one and retains the native LOD opacity curve.

`children` contains N uint32 pairs. Leaves precede parents, and each parent
references two earlier nodes. Parent RGB and SH coefficients are propagated
using the supplied mass weights. Parent geometry is preserved, so the GPU
builder controls generalized-Gaussian moment matching.

The encoder applies native Bhatt level spacing and chunks the hierarchy. The
optional final `compression` argument accepts `Compression.Gz` (the default
level-1 DEFLATE encoding) or `Compression.Zstd` (zstd level 3). Geometry, SH,
labels and property quantization are identical for both. The outer ZIP stores
those already compressed chunks.
`encode_merged_arrays` remains available to write the same output to a directory.

`encode_merged_archive` returns immutable `EncodeTimings`: input conversion,
parent assembly, pruning, chunking, quantization-range resolution, property
encoding/compression, and archive writing times in seconds.
