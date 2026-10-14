import { RaycastBlockIndex } from "spark-rs";
import type { SplatPager } from "./SplatPager";
import * as wasm from "./wasm";

// Bounds belong to streamed geometry, independently of the current camera/LoD selection.
export class SplatRaycastBlocks {
  private index?: RaycastBlockIndex;
  private pendingPages = new Map<number, Float32Array>();

  install(pager: SplatPager, page: number, bounds: Float32Array) {
    this.pendingPages.set(page, bounds);
    if (wasm.isInitialized()) this.refresh(pager);
  }

  remove(page: number) {
    this.pendingPages.delete(page);
  }

  dispose() {
    this.index?.free();
    this.index = undefined;
    this.pendingPages.clear();
  }

  private refresh(pager: SplatPager) {
    const index = this.index ??= new RaycastBlockIndex(pager.maxSplats);
    for (const [page, bounds] of this.pendingPages) {
      index.install_page(page, bounds);
    }
    this.pendingPages.clear();
    return index;
  }

  query(pager: SplatPager, indices: Uint32Array, projection: Float32Array, origin: number[], direction: number[]) {
    return this.refresh(pager).query(indices, projection, new Float32Array(origin), new Float32Array(direction));
  }
}
