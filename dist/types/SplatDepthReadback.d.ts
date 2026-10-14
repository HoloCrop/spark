export declare class ReadbackCancelledError extends Error {
    constructor();
}
export interface ReadbackScheduler {
    waitForGPU(signal: AbortSignal): Promise<void>;
}
/** Owns one immutable GPU snapshot until its complete CPU copy is available. */
export declare class SplatDepthReadback {
    private gl;
    private scheduler;
    private buffer;
    private capacity;
    private active?;
    private disposed;
    constructor(gl: WebGL2RenderingContext, scheduler?: ReadbackScheduler);
    private contextLost;
    private releaseBuffer;
    dispose(): void;
    read(destination: Uint8Array, snapshot: () => void): Promise<void>;
}
