# HoloCrop Spark fork

This is the preservation and integration map for merging upstream Spark into
HoloCrop's fork. Keep upstream synchronization, custom implementation changes,
and regenerated distribution files in separate commits. Maintain custom
algorithms in dedicated modules with small hooks in upstream classes.

## Baselines

Audit date: 2026-11-27.

| Reference | Revision | Meaning |
| --- | --- | --- |
| HoloCrop `main` | `479781cecc68b181b913d865b17e2b5c9fa90717` | Fork before this maintenance work; package version 2.1.0 |
| Browser dependency | `a85405665da3a617bec7c97b810a714c2f74d567` | Cropvision frontend's exact Git dependency; included in fork `main` |
| Upstream `main` | `f94964a712e217b7121ae642299fdac6041e5a12` | `sparkjsdev/spark`, package version 2.4.0; audited upstream target |
| Common ancestor | `155af7bb1ceba0bd3ecf2f48be6749e37196e6ab` | Baseline for identifying our changes |
| Native encoder source | `6f71bd31f3805ddc82d21e3f64f59c92103d6156` | Workspace `build-cuda/build_native_wheels.py` pin, with additional workspace patches |

The upstream repository is <https://github.com/sparkjsdev/spark>.
`sparkjs-dev/spark` in older package metadata is obsolete. The fork is
<https://github.com/HoloCrop/spark>. Browser releases use committed `dist` files
from an exact Git revision, not the upstream npm release or a moving branch.
The Python wheel has a separate source pin and release process; changing the
browser dependency does not update that wheel.
In particular, the workspace applies `build-cuda/patches/py-build-lod.patch`
on top of its encoder pin. Reconcile that patch explicitly before moving the
native source pin to the synchronized fork.

## Changes that must survive

### Generalized Gaussian rendering

Origin: `cb3f07b`. `SparkRenderer.gaussianK` selects
`exp(-0.5 * (r*r)^k)`: k=1 is upstream's ordinary Gaussian, k=2 is Cropvision's
quartic profile. Cropvision sets `gaussianK: 2.0`, `lodInflate: false`, and uses
extended paged splats. Keep the renderer option, uniform initialization and
propagation, fragment response, and vertex support radius together.

The support radius is `adjustedMaxStdDev^(1/k)`. LoD alpha in 1..2 is remapped
to 1..5 before its support is expanded. Generalizing only the fragment exponent
would clip the profile incorrectly. Ordinary alpha and the LoD power-composited
alpha branch must both use the generalized kernel.

Implementation: `src/shaders/gaussianKernel.glsl`, registered with the shared
shader definitions by `src/shaders.ts`. Required upstream hooks are in
`SparkRenderer.ts`, `splatVertex.glsl`, and `splatFragment.glsl`. The editor's
Gaussian k control and both portal fragment shaders also use this kernel.
Portal code has its own alpha path; keep these entry points consistent.

### Measurement surface picking and identity

Origins: `63eadae`, `5dbd2c3`, `1be58d1`, `06f003e`, `daa278d`, `6d5abea`.
Picking returns one combined-transmittance 0.5 surface crossing, rather than
independent ellipsoid intersections or the first Gaussian center. The hit's
`index` identifies a contributing source splat for fruit inspection.

The TypeScript implementation lives in `src/SplatSurfaceRaycast.ts`, called by
one delegation in `SplatMesh.raycast`. Native geometry sampling remains in
`rust/spark-rs/src/raycast.rs`; depth and contributor selection live in
`surface_depth.rs`. Native tests live in their own files.

Preserve these numerical contracts:

- Picking is specifically k=2; changing the renderer's k does not change picking.
- Collect depth/alpha/sigma profiles across every source chunk before solving
  the combined crossing. Preserve source IDs across chunk boundaries and LoD
  index remapping.
- Perspective picking uses projected Gaussian coverage and camera clipping.
  Transforms, including nonuniform scale, retain world-ray distance units.
- Hidden classes, zero opacity, and profiles wholly behind the origin do not
  occlude a hit. Near/far bounds apply to the crossing, not just mean depths.
- Preserve true thin scales, including zero-scale planes; do not introduce a
  minimum raycast scale.
- Preserve the profile alpha threshold 1/255 and cap 0.99, the minimum response
  option, and the LoD opacity remapping.
- Cropvision sets `lodRaycast: 0` to pick the displayed selection. Picking does
  not sample final rendered pixels or reproduce screen-space antialiasing.

The former `SplatMesh` static scratch arrays were internal implementation
details, with no callers in the fork or Cropvision frontend. Their storage now
belongs to the dedicated synchronous picking module.

### Streamed picking bounds

Origins: `6d5abea`, `8b5f580`. `raycast_blocks.rs` and
`src/SplatRaycastBlocks.ts` own conservative page/block bounds independently of
the camera and current LoD selection. The broad phase filters the displayed
indices; the existing narrow phase still decides the hit.

`PagedSplatDecode.ts` prepares native bounds in decode workers.
`PagedSplatResult.ts` requires geometry, a LoD tree, and bounds together.
`worker.ts` exposes paged decode RPCs; page admission installs the returned
bounds with that page's geometry. Shorter replacement pages clear old blocks
without disturbing neighboring pages. Acquire native scratch views after an
index query, because the query can grow WASM memory.

### Semantic classes and fruit instances

Origins: `1246adb`, `3d03c5e`, `a1c634e`, `794954a`, `cf5251d`,
`9eaaa52`, `152c02d`. This is a cross-layer feature, not just a shader patch.

`Gsplat`, the receiver/getter traits, and the LoD splat traits carry class and
instance IDs. PLY and SOG decoding read labels; RAD properties are named `label`
and `instance_label` and the encoder resolves them to U32. The extended WASM
result exposes `labels` and `instances`; worker result types and transfers must
retain them. Legacy PLY and SOG imports have their own label conventions: do not
apply the trained-array ID offset indiscriminately to every import path.

`SplatPager` owns per-page label and instance textures plus 256-entry visibility
and highlight lookup textures. Its index reader and label modifier connect
these to `PagedSplats` and `SplatMesh`. `updateLabelLookup` and
`updateLabelHighlight` remain callable on meshes. The exported
`ExtSplats.randomColourFromID` shader helper preserves input alpha.

Picking uses the same class visibility lookup. Cropvision's own object modifier
also reads the pager's label and instance textures for class styling and fruit
highlighting, so their names and index alignment are consumer interfaces.

LoD's generic merged parent currently inherits labels from its first child.
Trained hierarchy assembly groups semantic ancestors; retain the distinction
when changing merging behavior.

### Trained hierarchy export and k=2 moments

Origins: `55164eb`, `21d01c9`, `3aa8291`, `59eaed2`, `72cdffa`.
`rust/py-build-lod` is our Python binding; `trained_lod.rs` implements explicit
ancestry and RAD assembly. See [the binding contract](rust/py-build-lod/TRAINED_LOD.md).

Retain file-based export and direct Python buffer/array export to directories
and stored ZIP archives. Preserve every trained node, parent ownership,
selection bounds, SH layout, and class/instance IDs. Trained-array IDs use -1
for unknown and are encoded as ID+1, making zero the browser's unknown ID.

`GsplatArray.kernel`, `GaussianKernel`, and the moment-aware similarity/merging
hooks in `gsplat.rs` and `tsplat.rs` preserve projected second moments. The k=2
factor is `1/sqrt(2*pi)`. It must affect both parent covariance and overlap
distance; an unchanged file format does not prove geometric compatibility.

RAD property compression is centralized and optionally parallel on native
builds. Keep DEFLATE output as the compatibility default and retain zstd decode
support in WASM (`ruzstd`). Native zstd encoding is feature-gated. Stored outer
ZIPs avoid compressing already-compressed chunks again. Do not replace the
deployed RAD v1 layout while adopting upstream codec improvements.

### Navigation performance and resource ownership

Origins: `dc317b5`, `0f3ae34`, `7be2f61`, `a854056`.

- Label/instance texture storage is initialized once and populated layers are
  uploaded individually, rather than repeatedly uploading the full cache.
- `SplatDepthReadback.ts` enqueues every layer into one pixel buffer before
  yielding, waits for one fence, and performs one complete CPU transfer.
  `SparkRenderer` snapshots mapping metadata and rejects stale ordering after
  asynchronous work. Cancellation, disposal, context loss and binding restoration
  are resource lifecycle requirements. One transfer can still stall the driver.
- `SplatProgramRegistry.ts` owns programs per renderer and generator identity,
  with separate packed/extended and covariance modes. `SplatAccumulator`
  borrowers share ownership without releasing one another's programs. Target
  growth preserves programs; graph replacement and removed residents release
  them. Invisible residents remain retained. `dyno/program.ts` supplies disposal
  and restores the shared fullscreen material after drawing.

The paged four-texture SH3 correction (`4a6c517`) and debug-level chunk partition
logging (`f1e7a7e`) are independent correctness/tooling fixes.

## Minimal integration and upstream overlap

The picking algorithm and shader kernel now live outside upstream's core
implementation files. Existing dedicated modules already isolate native surface
solving, bounds, page-decode preparation, readback, program ownership and trained
export. Keep these modules rather than inlining them during conflict resolution.

Some core edits remain necessary: renderer option/uniform wiring; the shader
call sites; picking's mesh delegation and WASM bindings; decoded label fields
through native traits and result objects; page admission/eviction hooks; and
kernel-aware covariance/similarity calls. Moving these declarations to another
file would not eliminate the interfaces upstream must call.

Upstream has 149 commits beyond the common history. The dry-run merge at the
audited baselines conflicts in rendering, paging, worker/result types, native
decoding, RAD and SOG codecs, picking bindings, package metadata and generated
distribution files. A clean auto-merge elsewhere does not establish compatibility.

| Upstream change | Integration decision |
| --- | --- |
| `f2e7324`, `9f541e0`: split `PagedSplats` and add custom fetchers | Use the new upstream structure; move our decode/bounds/label hooks to the new page fetch path. Do not bring back the monolithic old pager. |
| `d4c8dcd`, `f035eda`, `57180f9`: page races, root eviction, on-demand streaming | Preserve their lifecycle fixes while coupling our bounds and semantic textures to the same page generation. |
| `23c4c36`, `8edb5d0`: four SH textures and complete SH arrays | Use upstream's equivalent fix; drop our duplicate `4a6c517` implementation after verifying SH3. |
| `77eb33d`, `85c3c8a`, `16cd289`, `cf54df3`, `48ad0a8`: disposal and cancellation | Combine lifecycle requirements once. These do not replace our coherent depth snapshot or owned generated-program registry. |
| `10e05e9`, `929d77b`: sort scheduling and ordering dimensions | Keep upstream scheduling/sizing improvements together with our immutable mapping snapshot and single depth transfer. |
| `8a665d1`: packed covariance generator fix | Keep the corrected generator selection inside our per-output-mode registry integration. |
| `e0db664`, `2d1f297`, `9ab9086`: typed RPCs, extras and encoding | Extend upstream's concrete types with our required page bounds and optional label payloads; remove duplicate type scaffolding where upstream supplies it. |
| `a966086`: flate2/zlib-rs DEFLATE | Adopt upstream compression implementation without losing optional zstd decoding, native compression choices, or RAD property metadata. |
| New decoder `SplatPropsArray`/`SplatPropsMut` batch path | Carry labels through the new batch structures; do not paste back our old `copy_getter_to_receiver` implementation. |
| Three.js dependency | The fork uses 0.185; audited upstream still declares 0.180. Preserve compatibility with Cropvision's 0.185 frontend when resolving package files. |

Do not carry generated `dist` conflict resolutions by hand. Resolve source,
build matching WASM, then regenerate both JS variants, declarations and maps.
Delete obsolete worker assets through the normal clean production build.

## Audit findings that are not preservation requirements

- `.swp` is an accidentally committed editor swap file and is removed by this
  maintenance work.
- `lefthook.yml.old` disables the upstream hook configuration. It is tooling
  history, not a renderer feature; avoid silently re-enabling formatting hooks
  while synchronizing algorithms.
- `SplatGetter.has_labels` is never called; `ExtSplatsData.label_info` is always
  `None`. Neither has a demonstrated consumer in the audited trees. They are
  candidates to omit when adapting the upstream batch interfaces.
- Label additions are incomplete outside the deployed extended-paged path:
  `Csplat` and packed native label getters contain `todo!()`, and the extended
  native label getters ignore their `base` offset. These are defects, not behavior
  to preserve; add meaningful round-trip/offset regressions when repairing them.
- The old copy helper slices empty label buffers and omits `get_quat`. The
  `GsplatArray` batch receiver also indexes label arrays unconditionally. On fork
  `main`, `antisplat_roundtrip_basic` and `ksplat_roundtrip_basic` fail because
  unlabeled batches have empty labels. Preserve labeled and unlabeled decoding
  contracts when integrating upstream, rather than propagating these defects.

This audit identifies these paths by inspection and the existing failures; it
does not certify the undeployed import/export combinations as working.

## Validation and release

From a Spark checkout, using the workspace's Pixi environment, run:

```sh
pixi run cargo test --locked --manifest-path rust/Cargo.toml -p spark-rs --lib
pixi run cargo test --locked --manifest-path rust/Cargo.toml -p spark-lib --test kernel_moments
pixi run npm run test:raycast
pixi run npm run test:depth-readback
pixi run npm run test:generator-programs
```

Set `--manifest-path /path/to/cropvision-workspace/pixi.toml` on `pixi run` when
running outside that workspace. Build the matching native WASM package before
JavaScript tests. In this source-only maintenance change, tests reuse the exact
published WASM bytes and matching wrapper extracted from the committed bundle
and source map; no native implementation changed.

The maintenance extraction passes the existing four JS picking, seven readback,
six program-ownership, three kernel-moment, and 30 native picking tests.
No configuration-value tests are needed. The broader native library failures
above reproduce on unchanged `479781c` and are recorded separately from passing
picking regressions.
Standalone TypeScript reports the same 22 diagnostics on unchanged fork `main`
and after extraction; the maintenance work adds none. The production and
development Vite builds emit bundles and declarations, while their declaration
plugin also reports those existing diagnostics. A successful bundling exit
code is therefore not a clean type check.

After the upstream merge, also run upstream's unit/browser/native suites and
check real labeled RAD/SOG scenes: class hiding, instance coloring, fruit hit
identity, thin surfaces, camera clipping, SH3, page eviction/replacement,
continuous navigation, context restoration and resource disposal. Compare k=1
against upstream and k=2 against the published Cropvision renderer. Recheck
trained-RAD geometry and ancestry in the workspace pipeline's existing tests.

Use the workspace native-wheel policy for any encoder rebuild. Keep its pinned
source and extra patches explicit; never assume merging browser code updates
the deployed Python encoder. Update the frontend's exact Git pin only after the
new distribution and staging browser checks pass. Publish releases from fork
`main` so the deployed revision is always reachable from the default branch.

To reproduce the source inventory independently of generated artifacts:

```sh
git diff --stat 155af7bb1ceba0bd3ecf2f48be6749e37196e6ab origin/main -- . ':!dist' ':!package-lock.json' ':!rust/Cargo.lock'
git log --reverse --oneline upstream/main..origin/main
```
