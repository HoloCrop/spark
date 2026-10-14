export class ReadbackCancelledError extends Error {
  constructor() {
    super("Depth readback cancelled");
  }
}

export interface ReadbackScheduler {
  now(): number;
  waitForGPU(signal: AbortSignal): Promise<void>;
  yieldToMain(signal: AbortSignal): Promise<void>;
  dispose(): void;
}

// MessageChannel yields a real task without adding a timer clamp to every copy.
class ReadbackTasks implements ReadbackScheduler {
  private channel = new MessageChannel();
  private nextId = 0;
  private pending?: { id: number; resume: () => void };

  constructor() {
    this.channel.port1.onmessage = ({ data }) => {
      const pending = this.pending;
      if (pending && pending.id === data) pending.resume();
    };
  }

  now() { return performance.now(); }

  waitForGPU(signal: AbortSignal) { return this.wait(signal, true); }
  yieldToMain(signal: AbortSignal) { return this.wait(signal, false); }

  private wait(signal: AbortSignal, poll: boolean): Promise<void> {
    return new Promise((resolve, reject) => {
      let timer: ReturnType<typeof setTimeout> | undefined;
      const finish = () => {
        signal.removeEventListener("abort", cancel);
        this.pending = undefined;
        resolve();
      };
      const cancel = () => {
        clearTimeout(timer);
        this.pending = undefined;
        reject(new ReadbackCancelledError());
      };
      signal.addEventListener("abort", cancel, { once: true });
      if (signal.aborted) cancel();
      else if (poll) timer = setTimeout(finish, 4);
      else {
        const id = ++this.nextId;
        this.pending = { id, resume: finish };
        this.channel.port2.postMessage(id);
      }
    });
  }

  dispose() {
    this.channel.port1.close();
    this.channel.port2.close();
  }
}

/** Owns one immutable GPU snapshot until its complete CPU copy is available. */
export class SplatDepthReadback {
  private buffer: WebGLBuffer | null = null;
  private capacity = 0;
  private active?: AbortController;
  private disposed = false;
  private readonly copyBytes = 1024 * 1024;
  private readonly copyBudgetMs = 2;

  constructor(
    private gl: WebGL2RenderingContext,
    private scheduler: ReadbackScheduler = new ReadbackTasks(),
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
    this.scheduler.dispose();
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

      let turnStarted = this.scheduler.now();
      for (let offset = 0; offset < destination.byteLength;) {
        const bytes = Math.min(this.copyBytes, destination.byteLength - offset);
        const previous = gl.getParameter(gl.PIXEL_PACK_BUFFER_BINDING) as WebGLBuffer | null;
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.buffer);
        try {
          gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, offset, destination, offset, bytes);
        } finally {
          gl.bindBuffer(gl.PIXEL_PACK_BUFFER, previous);
        }
        offset += bytes;
        // Bound each GL copy and each task's work without limiting scene quality.
        if (offset < destination.byteLength && this.scheduler.now() - turnStarted >= this.copyBudgetMs) {
          await this.scheduler.yieldToMain(signal);
          if (signal.aborted) throw new ReadbackCancelledError();
          turnStarted = this.scheduler.now();
        }
      }
    } finally {
      if (fence) gl.deleteSync(fence);
      this.active = undefined;
    }
  }
}
