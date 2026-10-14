import assert from "node:assert/strict";
import { test } from "node:test";
import { ReadbackCancelledError, SplatDepthReadback, type ReadbackScheduler } from "../src/SplatDepthReadback";

type BufferStorage = { bytes: Uint8Array; deleted: boolean };
type Fence = { deleted: boolean };
type PendingTask = { kind: "gpu" | "copy"; resume: () => void };

class ControlledScheduler implements ReadbackScheduler {
  time = 0;
  pending: PendingTask[] = [];
  disposed = false;
  events: string[] = [];

  now() { return this.time; }
  waitForGPU(signal: AbortSignal) { return this.wait(signal, "gpu"); }
  yieldToMain(signal: AbortSignal) { return this.wait(signal, "copy"); }
  dispose() { this.disposed = true; }

  wait(signal: AbortSignal, kind: PendingTask["kind"]) {
    this.events.push(kind);
    return new Promise<void>((resolve, reject) => {
      const remove = () => {
        signal.removeEventListener("abort", abort);
        this.pending.splice(this.pending.indexOf(task), 1);
      };
      const task = { kind, resume: () => { remove(); resolve(); } };
      const abort = () => { remove(); reject(new ReadbackCancelledError()); };
      this.pending.push(task);
      signal.addEventListener("abort", abort, { once: true });
      if (signal.aborted) abort();
    });
  }

  async resume() {
    assert.equal(this.pending.length, 1, "the readback owns only one pending task");
    this.pending[0].resume();
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
  }
}

class ReadbackContext {
  readonly PIXEL_PACK_BUFFER = 0x88eb;
  readonly PIXEL_PACK_BUFFER_BINDING = 0x88ed;
  readonly STREAM_READ = 0x88e1;
  readonly SYNC_GPU_COMMANDS_COMPLETE = 0x9117;
  readonly CONDITION_SATISFIED = 0x911c;
  readonly TIMEOUT_EXPIRED = 0x911b;
  readonly WAIT_FAILED = 0x911d;
  canvas = new EventTarget();
  buffers: BufferStorage[] = [];
  fences: Fence[] = [];
  bound: BufferStorage | null = null;
  lost = false;
  status = this.CONDITION_SATISFIED;
  copies: { offset: number; length: number }[] = [];
  events: string[];

  constructor(readonly scheduler: ControlledScheduler) {
    this.events = scheduler.events;
  }

  createBuffer() {
    const buffer = { bytes: new Uint8Array(), deleted: false };
    this.buffers.push(buffer);
    return buffer;
  }
  deleteBuffer(buffer: BufferStorage) { buffer.deleted = true; }
  bindBuffer(target: number, buffer: BufferStorage | null) {
    assert.equal(target, this.PIXEL_PACK_BUFFER);
    this.bound = buffer;
  }
  getParameter(parameter: number) {
    assert.equal(parameter, this.PIXEL_PACK_BUFFER_BINDING);
    return this.bound;
  }
  bufferData(target: number, length: number) {
    assert.equal(target, this.PIXEL_PACK_BUFFER);
    this.bound!.bytes = new Uint8Array(length);
  }
  snapshotLayer(offset: number, source: Uint8Array) {
    this.events.push("layer");
    this.bound!.bytes.set(source, offset);
  }
  fenceSync() {
    this.events.push("fence");
    const fence = { deleted: false };
    this.fences.push(fence);
    return fence;
  }
  flush() { this.events.push("flush"); }
  clientWaitSync() { return this.status; }
  deleteSync(fence: Fence) { fence.deleted = true; }
  isContextLost() { return this.lost; }
  getBufferSubData(target: number, offset: number, destination: Uint8Array, destinationOffset: number, length: number) {
    assert.equal(target, this.PIXEL_PACK_BUFFER);
    assert.equal(this.bound!.deleted, false);
    this.copies.push({ offset, length });
    destination.set(this.bound!.bytes.subarray(offset, offset + length), destinationOffset);
    this.scheduler.time += length / (128 * 1024);
  }
}

function fixture() {
  const scheduler = new ControlledScheduler();
  const context = new ReadbackContext(scheduler);
  const previous = context.createBuffer();
  context.bound = previous;
  const readback = new SplatDepthReadback(context as unknown as WebGL2RenderingContext, scheduler);
  return { scheduler, context, previous, readback };
}

async function finish(promise: Promise<void>, scheduler: ControlledScheduler) {
  let completed = false;
  const settled = promise.finally(() => { completed = true; });
  await Promise.resolve();
  while (!completed) {
    await scheduler.resume();
  }
  await settled;
}

test("depth readback preserves one complete layered snapshot across yields and reuses its buffer", async () => {
  const { scheduler, context, previous, readback } = fixture();
  const firstLayer = new Uint8Array(2 * 1024 * 1024).fill(23);
  const secondLayer = new Uint8Array(2 * 1024 * 1024).fill(71);
  const expected = new Uint8Array(firstLayer.length + secondLayer.length);
  expected.set(firstLayer);
  expected.set(secondLayer, firstLayer.length);
  const storage = new Uint8Array(expected.length + 2).fill(199);
  const destination = storage.subarray(1, storage.length - 1);
  let completed = false;
  const pending = readback.read(destination, () => {
    assert.notEqual(context.bound, previous);
    context.snapshotLayer(0, firstLayer);
    context.snapshotLayer(firstLayer.length, secondLayer);
  }).then(() => { completed = true; });

  assert.deepEqual(context.events, ["layer", "layer", "fence", "flush", "gpu"]);
  assert.equal(context.bound, previous);
  assert.equal(context.copies.length, 0);
  firstLayer.fill(0);
  secondLayer.fill(0);
  await scheduler.resume();
  assert.ok(context.copies.length > 0);
  assert.equal(scheduler.pending[0].kind, "copy");
  assert.equal(completed, false, "a partial CPU copy must not publish a sort input");
  assert.equal(context.bound, previous);

  // Another renderer operation changes the binding while this readback is paused.
  const intervening = context.createBuffer();
  context.bound = intervening;
  await finish(pending, scheduler);
  assert.equal(context.bound, intervening);
  assert.deepEqual(destination, expected);
  assert.equal(storage[0], 199);
  assert.equal(storage[storage.length - 1], 199);
  assert.ok(context.fences.every((fence) => fence.deleted));

  const allocated = context.buffers.length;
  const replacement = new Uint8Array(17).fill(83);
  const next = new Uint8Array(replacement.length);
  await finish(readback.read(next, () => context.snapshotLayer(0, replacement)), scheduler);
  assert.deepEqual(next, replacement);
  assert.equal(context.buffers.length, allocated);
  readback.dispose();
});

test("an empty read completes without copying and leaves the reader reusable", async () => {
  const { scheduler, context, previous, readback } = fixture();
  await finish(readback.read(new Uint8Array(), () => {}), scheduler);
  assert.equal(context.copies.length, 0);
  assert.equal(context.bound, previous);

  const source = new Uint8Array(19).fill(113);
  const destination = new Uint8Array(source.length);
  await finish(readback.read(destination, () => context.snapshotLayer(0, source)), scheduler);
  assert.deepEqual(destination, source);
  readback.dispose();
});

for (const phase of ["gpu", "copy"] as const) {
  test(`disposing during ${phase} wait cancels the readback without further copies`, async () => {
    const { scheduler, context, previous, readback } = fixture();
    const destination = new Uint8Array(4 * 1024 * 1024);
    const pending = readback.read(destination, () => context.snapshotLayer(0, destination));
    if (phase === "copy") await scheduler.resume();
    assert.equal(scheduler.pending[0].kind, phase);
    const copied = context.copies.length;
    const rejected = assert.rejects(pending, ReadbackCancelledError);
    readback.dispose();
    await rejected;
    assert.equal(context.copies.length, copied);
    assert.equal(context.bound, previous);
    assert.equal(scheduler.pending.length, 0);
    assert.equal(scheduler.disposed, true);
    assert.ok(context.fences.every((fence) => fence.deleted));
    assert.equal(context.buffers[1].deleted, true);
    let snapshots = 0;
    await assert.rejects(readback.read(destination, () => { snapshots++; }), ReadbackCancelledError);
    assert.equal(snapshots, 0);
  });
}

test("context loss cancels the snapshot and restoration permits a fresh readback", async () => {
  const { scheduler, context, readback } = fixture();
  const destination = new Uint8Array(32);
  const pending = readback.read(destination, () => {});
  const rejected = assert.rejects(pending, ReadbackCancelledError);
  context.lost = true;
  context.canvas.dispatchEvent(new Event("webglcontextlost"));
  await rejected;
  assert.equal(context.buffers[1].deleted, true);
  assert.equal(scheduler.disposed, false);
  await assert.rejects(readback.read(destination, () => {}), ReadbackCancelledError);

  context.lost = false;
  context.canvas.dispatchEvent(new Event("webglcontextrestored"));
  const source = new Uint8Array(destination.length).fill(97);
  await finish(readback.read(destination, () => context.snapshotLayer(0, source)), scheduler);
  assert.deepEqual(destination, source);
  assert.notEqual(context.buffers[1], context.buffers[2]);
  readback.dispose();
});

test("concurrent reads cannot replace a pending snapshot", async () => {
  const { scheduler, context, readback } = fixture();
  const source = new Uint8Array(32).fill(101);
  const destination = new Uint8Array(source.length);
  const pending = readback.read(destination, () => context.snapshotLayer(0, source));
  let replacementSnapshots = 0;
  await assert.rejects(readback.read(destination, () => { replacementSnapshots++; }), /already in progress/);
  assert.equal(replacementSnapshots, 0);
  await finish(pending, scheduler);
  assert.deepEqual(destination, source);
  readback.dispose();
});

test("snapshot and fence failures release readback ownership and restore the caller binding", async () => {
  const { scheduler, context, previous, readback } = fixture();
  const destination = new Uint8Array(32);
  await assert.rejects(readback.read(destination, () => { throw new Error("snapshot failed"); }), /snapshot failed/);
  assert.equal(context.bound, previous);
  context.status = context.WAIT_FAILED;
  const pending = readback.read(destination, () => {});
  const rejected = assert.rejects(pending, /fence failed/);
  await scheduler.resume();
  await rejected;
  assert.equal(context.bound, previous);
  assert.ok(context.fences.every((fence) => fence.deleted));

  context.status = context.CONDITION_SATISFIED;
  const source = new Uint8Array(destination.length).fill(107);
  await finish(readback.read(destination, () => context.snapshotLayer(0, source)), scheduler);
  assert.deepEqual(destination, source);
  readback.dispose();
});
