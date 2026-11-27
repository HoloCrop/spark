import { SplatPager } from './SplatPager';
export declare class SplatRaycastBlocks {
    private index?;
    private pendingPages;
    install(pager: SplatPager, page: number, bounds: Float32Array): void;
    remove(page: number): void;
    dispose(): void;
    private refresh;
    query(pager: SplatPager, indices: Uint32Array, projection: Float32Array, origin: number[], direction: number[]): Uint32Array<ArrayBufferLike>;
}
