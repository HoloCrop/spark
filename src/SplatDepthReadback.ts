export class ReadbackCancelledError extends Error {
  constructor() {
    super("Depth readback cancelled");
  }
}

export interface ReadbackScheduler {
  waitForGPU(signal: AbortSignal): Promise<void>;
}

function waitForGPU(signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const finish = () => {
      signal.removeEventListener("abort", cancel);
      resolve();
    };
    const cancel = () => {
      clearTimeout(timer);
      reject(new ReadbackCancelledError());
    };
    const timer = setTimeout(finish, 4);
    signal.addEventListener("abort", cancel, { once: true });
    if (signal.aborted) cancel();
  });
}

/** Owns one immutable GPU snapshot until its complete CPU copy is available. */
export class SplatDepthReadback {
  private buffer: WebGLBuffer | null = null;
  private capacity = 0;
  private active?: AbortController;
  private disposed = false;

  constructor(
    private gl: WebGL2RenderingContext,
    private scheduler: ReadbackScheduler = { waitForGPU },
  ) {
    gl.canvas.addEventListener("webglcontextlost", this.contextLost);
  }

  private contextLost = () => {
    this.active?.abort();
    this.releaseBuffer();
  };

  private releaseBuffer() {
    if (this.buffer) this.gl.deleteBuffer(this.buffer);
    this.buffer = null;
    this.capacity = 0;
  }

  dispose() {
    this.disposed = true;
    this.active?.abort();
    this.gl.canvas.removeEventListener("webglcontextlost", this.contextLost);
    this.releaseBuffer();
  }

  async read(destination: Uint8Array, snapshot: () => void): Promise<void> {
    if (this.disposed || this.gl.isContextLost()) throw new ReadbackCancelledError();
    if (this.active) throw new Error("Depth readback is already in progress");
    const controller = new AbortController();
    this.active = controller;
    const { signal } = controller;
    const gl = this.gl;
    let fence: WebGLSync | null = null;

    try {
      if (!this.buffer) this.buffer = gl.createBuffer();
      const previous = gl.getParameter(gl.PIXEL_PACK_BUFFER_BINDING) as WebGLBuffer | null;
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.buffer);
      try {
        if (destination.byteLength > this.capacity) {
          this.capacity = Math.max(destination.byteLength, this.capacity * 2);
          gl.bufferData(gl.PIXEL_PACK_BUFFER, this.capacity, gl.STREAM_READ);
        }
        // Every layer is enqueued before yielding: accumulators can be recycled later.
        snapshot();
        fence = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
        gl.flush();
      } finally {
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, previous);
      }

      while (true) {
        await this.scheduler.waitForGPU(signal);
        if (signal.aborted) throw new ReadbackCancelledError();
        const status = gl.clientWaitSync(fence!, 0, 0);
        if (status === gl.WAIT_FAILED) throw new Error("Depth readback fence failed");
        if (status !== gl.TIMEOUT_EXPIRED) break;
      }

      if (destination.byteLength === 0) return;
      const previousCopyBuffer = gl.getParameter(gl.PIXEL_PACK_BUFFER_BINDING) as WebGLBuffer | null;
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.buffer);
      try {
        // One transfer per snapshot: each call can incur a synchronous driver round trip.
        gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, destination);
      } finally {
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, previousCopyBuffer);
      }
    } finally {
      if (fence) gl.deleteSync(fence);
      this.active = undefined;
    }
  }
}
