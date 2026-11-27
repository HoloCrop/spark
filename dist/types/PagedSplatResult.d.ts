import { ExtResult, PackedResult } from './defines';
export type PagedPackedResult = PackedResult & {
    extra: PackedResult["extra"] & {
        lodTree: Uint32Array;
    };
    raycastBounds: Float32Array;
};
export type PagedExtResult = ExtResult & {
    extra: ExtResult["extra"] & {
        lodTree: Uint32Array;
    };
    raycastBounds: Float32Array;
};
