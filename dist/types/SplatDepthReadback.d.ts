export declare class ReadbackCancelledError extends Error {
    constructor();
}
export interface ReadbackScheduler {
    now(): number;
    waitForGPU(signal: AbortSignal): Promise<void>;
    yieldToMain(signal: AbortSignal): Promise<void>;
    dispose(): void;
}
/** Owns one immutable GPU snapshot until its complete CPU copy is available. */
export declare class SplatDepthReadback {
    private gl;
    private scheduler;
    private buffer;
    private capacity;
    private active?;
    private disposed;
    private readonly copyBytes;
    private readonly copyBudgetMs;
    constructor(gl: WebGL2RenderingContext, scheduler?: ReadbackScheduler);
    private contextLost;
    private releaseBuffer;
    dispose(): void;
    read(destination: Uint8Array, snapshot: () => void): Promise<void>;
}
