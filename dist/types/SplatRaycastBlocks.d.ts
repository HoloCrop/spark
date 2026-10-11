import { SplatPager } from './SplatPager';
export declare class SplatRaycastBlocks {
    private index?;
    private dirtyPages;
    update(pager: SplatPager, page: number): void;
    remove(page: number): void;
    dispose(): void;
    private refresh;
    query(pager: SplatPager, indices: Uint32Array, projection: Float32Array, origin: number[], direction: number[]): Uint32Array<ArrayBufferLike>;
}
