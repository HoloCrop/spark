# Trained RAD levels

`TrainedLevel(position, rotation, log_scaling, alpha_logit, sh_feature, labels)`
accepts Python buffers directly: float32 positions (N,3), xyzw quaternions (N,4),
log scales (N,3), opacity logits (N,1), RGB-major SH3 coefficients (N,3,16), and
int32 class/instance labels (N,2). Labels use -1 for unknown; zero is a real ID.

`encode_trained_arrays(levels, parents, output_dir, moment_factor, resolution_factor)`
encodes levels in finest-to-coarsest order. Each uint32 parent buffer maps a row
of one level to its parent in the next level. Inputs remain in memory; the encoder
writes only `bay-lod.rad` and its `bay-lod-*.radc` chunks. There are no temporary
PLYs, binary mappings, or intermediate serialized representations.

`encode_trained_archive(levels, parents, output_file, moment_factor, resolution_factor)`
uses the same encoder and writes the chunks directly into a stored ZIP. The pipeline
uses this entry point so only the final upload archive reaches the filesystem.

The pipeline crops finished levels on the GPU before copying them to CPU memory.
It retains actual ancestors and remaps their child ownership, preserving labels.
The encoder retains every trained node and creates semantic spatial ancestors
above the final trained level. `moment_factor` converts projected kernel shape
to second moments (1 for k=1, 1/sqrt(2*pi) for k=2).

Class and instance IDs become ID+1 for the frontend, with zero representing unknown.
The independent `lod_size` property preserves all trained level transitions,
without inflating rendered geometry.

The native encoder uses zstd level 3 with bounded parallel property compression.
The Rust/WASM decoder supports both zstd and existing DEFLATE properties. Deploy
the matching viewer build before publishing zstd RADs. Upload ZIPs should store
RAD chunks without compressing them again.

`encode_trained_rad` remains available for file-based tools. Both interfaces share
the same hierarchy assembly and encoder. Pipeline tests compare all encoded
properties and protect ancestry, SH layout, k=2 selection bounds and label IDs.
