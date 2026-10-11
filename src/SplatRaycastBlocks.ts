import { RaycastBlockIndex } from "spark-rs";
import type { SplatPager } from "./SplatPager";
import { LN_SCALE_MIN, LN_SCALE_MAX } from "./defines";
import * as wasm from "./wasm";

// Bounds belong to streamed geometry, independently of the current camera/LoD selection.
export class SplatRaycastBlocks {
  private index?: RaycastBlockIndex;
  private dirtyPages = new Set<number>();

  update(pager: SplatPager, page: number) {
    this.dirtyPages.add(page);
    if (wasm.isInitialized()) this.refresh(pager);
  }

  remove(page: number) {
    this.dirtyPages.delete(page);
  }

  dispose() {
    this.index?.free();
    this.index = undefined;
    this.dirtyPages.clear();
  }

  private refresh(pager: SplatPager) {
    const index = this.index ??= new RaycastBlockIndex(pager.maxSplats);
    const packed = pager.packedTexture.value.image.data as Uint32Array;
    const geometry = pager.extTexture.value.image.data as Uint32Array;
    for (const page of this.dirtyPages) {
      const start = page * pager.pageSplats;
      const data = packed.subarray(start * 4, (start + pager.pageSplats) * 4);
      if (pager.extSplats) {
        index.update_ext(start, data, geometry.subarray(start * 4, (start + pager.pageSplats) * 4));
      } else {
        const encoding = pager.pageToSplatsChunk[page]!.splats.splatEncoding;
        index.update_packed(start, data, encoding?.lnScaleMin ?? LN_SCALE_MIN, encoding?.lnScaleMax ?? LN_SCALE_MAX);
      }
    }
    this.dirtyPages.clear();
    return index;
  }

  query(pager: SplatPager, indices: Uint32Array, projection: Float32Array, origin: number[], direction: number[]) {
    return this.refresh(pager).query(indices, projection, new Float32Array(origin), new Float32Array(direction));
  }
}
