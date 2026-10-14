import type { ExtResult, PackedResult } from "./defines";

// Every decoded page carries the metadata needed to publish geometry and picking together.
export type PagedPackedResult = PackedResult & {
  extra: PackedResult["extra"] & { lodTree: Uint32Array };
  raycastBounds: Float32Array;
};

export type PagedExtResult = ExtResult & {
  extra: ExtResult["extra"] & { lodTree: Uint32Array };
  raycastBounds: Float32Array;
};
