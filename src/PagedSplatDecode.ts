import { build_ext_raycast_bounds, build_packed_raycast_bounds } from "spark-rs";
import type { PagedExtResult, PagedPackedResult } from "./PagedSplatResult";
import { LN_SCALE_MAX, LN_SCALE_MIN, type ExtResult, type PackedResult } from "./defines";

// Runs in the decode worker, before its geometry and compact bounds are transferred.
export function preparePagedPackedResult(result: PackedResult): PagedPackedResult {
  const { lodTree } = result.extra;
  if (!lodTree) throw new Error("Paged splats require LoD tree data");
  return {
    ...result,
    extra: { ...result.extra, lodTree },
    raycastBounds: build_packed_raycast_bounds(
      result.packedArray.subarray(0, result.numSplats * 4),
      result.splatEncoding.lnScaleMin ?? LN_SCALE_MIN,
      result.splatEncoding.lnScaleMax ?? LN_SCALE_MAX,
    ),
  };
}

export function preparePagedExtResult(result: ExtResult): PagedExtResult {
  const { lodTree } = result.extra;
  if (!lodTree) throw new Error("Paged splats require LoD tree data");
  return {
    ...result,
    extra: { ...result.extra, lodTree },
    raycastBounds: build_ext_raycast_bounds(
      result.extArrays[0].subarray(0, result.numSplats * 4),
      result.extArrays[1].subarray(0, result.numSplats * 4),
    ),
  };
}
