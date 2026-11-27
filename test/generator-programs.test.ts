import assert from "node:assert/strict";
import { test } from "node:test";
import * as THREE from "three";
import { SplatAccumulator, SplatGenerator, SplatProgramRegistry, dyno } from "../src/index";

const { DynoVec4, Gsplat, combineGsplat, dynoBlock, dynoConst, dynoLiteral } = dyno;

function source() {
  const color = new DynoVec4({ value: new THREE.Vector4(0.2, 0.4, 0.6, 0.8) });
  const generator = dynoBlock({ index: "int" }, { gsplat: Gsplat }, ({ index }) => ({
    gsplat: combineGsplat({
      index,
      flags: dynoLiteral("uint", "GSPLAT_FLAG_ACTIVE"),
      center: dynoConst("vec3", [0, 0, 5]),
      scales: dynoConst("vec3", [0.1, 0.1, 0.1]),
      quaternion: dynoConst("vec4", [0, 0, 0, 1]),
      rgba: color,
    }),
  }));
  return { generator, color, node: new SplatGenerator({ generator, numSplats: 1 }) };
}

test("accumulator borrowers share programs while graph replacement preserves other users and uniform bindings", () => {
  const programs = new SplatProgramRegistry();
  const first = new SplatAccumulator({ programs });
  const second = new SplatAccumulator({ programs });
  const original = source();
  const otherUser = new SplatGenerator({ generator: original.generator, numSplats: 1 });
  const prepared = first.prepareProgramMaterial(original.generator);
  const shared = second.prepareProgramMaterial(original.generator);
  assert.equal(shared.program, prepared.program);
  assert.equal(shared.material, prepared.material);
  const released = new Set<THREE.Material>();
  prepared.material.addEventListener("dispose", () => released.add(prepared.material));

  const replacement = source();
  original.node.generator = replacement.generator;
  programs.retain([original.node, otherUser]);
  const changed = first.prepareProgramMaterial(replacement.generator);
  assert.notEqual(changed.program, prepared.program);
  assert.equal(second.prepareProgramMaterial(otherUser.generator).material, prepared.material);
  assert.equal(released.has(prepared.material), false);
  replacement.color.value = new THREE.Vector4(0.9, 0.7, 0.5, 0.3);
  changed.program.update();
  prepared.program.update();
  assert.ok(Object.values(changed.material.uniforms).includes(replacement.color.uniform));
  assert.ok(Object.values(prepared.material.uniforms).includes(original.color.uniform));
  assert.deepEqual(replacement.color.uniform.value.toArray(), [0.9, 0.7, 0.5, 0.3]);
  assert.deepEqual(original.color.uniform.value.toArray(), [0.2, 0.4, 0.6, 0.8]);

  first.dispose();
  assert.equal(second.prepareProgramMaterial(original.generator).material, prepared.material);
  assert.equal(released.has(prepared.material), false, "disposing a borrower cannot release another accumulator's program");
  programs.retain([original.node]);
  assert.equal(released.has(prepared.material), true);
  assert.equal(second.prepareProgramMaterial(replacement.generator).material, changed.material);
  const reacquired = second.prepareProgramMaterial(original.generator);
  assert.notEqual(reacquired.program, prepared.program);
  assert.notEqual(reacquired.material, prepared.material);
  reacquired.program.update();
  assert.ok(Object.values(reacquired.material.uniforms).includes(original.color.uniform));
  second.dispose();
  programs.dispose();
});

test("standalone owners can release the same generator independently", () => {
  const shared = source();
  const first = new SplatAccumulator();
  const second = new SplatAccumulator();
  const left = first.prepareProgramMaterial(shared.generator);
  const right = second.prepareProgramMaterial(shared.generator);
  assert.equal(left.program.shader, right.program.shader);
  assert.notEqual(left.program, right.program);
  assert.notEqual(left.material, right.material);
  let released = false;
  right.material.addEventListener("dispose", () => { released = true; });
  first.dispose();
  assert.equal(second.prepareProgramMaterial(shared.generator).material, right.material);
  assert.equal(released, false);
  shared.color.value = new THREE.Vector4(0.1, 0.3, 0.5, 0.7);
  right.program.update();
  assert.ok(Object.values(right.material.uniforms).includes(shared.color.uniform));
  assert.deepEqual(shared.color.uniform.value.toArray(), [0.1, 0.3, 0.5, 0.7]);
  second.dispose();
});

test("a generator compiled for extended output cannot contaminate packed or covariance output", () => {
  const programs = new SplatProgramRegistry();
  const shared = source();
  const extended = new SplatAccumulator({ extSplats: true, programs });
  const extendedShader = extended.prepareProgramMaterial(shared.generator).program.shader;
  for (const mode of [{ extSplats: false }, { extSplats: true, covSplats: true }, { extSplats: false, covSplats: true }]) {
    const actual = new SplatAccumulator({ ...mode, programs });
    const control = new SplatAccumulator(mode);
    const prepared = actual.prepareProgramMaterial(shared.generator);
    const independent = control.prepareProgramMaterial(source().generator);
    assert.equal(prepared.program.shader, independent.program.shader);
    assert.notEqual(prepared.program.shader, extendedShader);
    actual.dispose();
    control.dispose();
  }
  extended.dispose();
  programs.dispose();
});

test("growing an accumulator target preserves its compiled program and bindings", () => {
  const accumulator = new SplatAccumulator();
  const generated = source();
  const prepared = accumulator.prepareProgramMaterial(generated.generator);
  accumulator.ensureGenerate({ maxSplats: 1 });
  const previousTarget = accumulator.target;
  accumulator.ensureGenerate({ maxSplats: accumulator.maxSplats + 1 });
  assert.notEqual(accumulator.target, previousTarget);
  const afterGrowth = accumulator.prepareProgramMaterial(generated.generator);
  assert.equal(afterGrowth.program, prepared.program);
  assert.equal(afterGrowth.material, prepared.material);
  generated.color.value = new THREE.Vector4(0.4, 0.5, 0.6, 0.7);
  afterGrowth.program.update();
  assert.ok(Object.values(afterGrowth.material.uniforms).includes(generated.color.uniform));
  assert.deepEqual(generated.color.uniform.value.toArray(), [0.4, 0.5, 0.6, 0.7]);
  accumulator.dispose();
});

test("scene preparation keeps hidden and layer-excluded residents but retires removed or replaced generators", () => {
  const programs = new SplatProgramRegistry();
  const accumulator = new SplatAccumulator({ programs });
  const previous = new SplatAccumulator({ programs });
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera();
  const hidden = source();
  const excluded = source();
  hidden.node.visible = false;
  excluded.node.layers.set(3);
  scene.add(hidden.node, excluded.node);
  const hiddenProgram = accumulator.prepareProgramMaterial(hidden.generator);
  const excludedProgram = accumulator.prepareProgramMaterial(excluded.generator);
  const prepare = () => accumulator.prepareGenerate({
    renderer: {} as THREE.WebGLRenderer,
    scene, camera, previous, time: 0, sortRadial: false, renderSize: new THREE.Vector2(1, 1),
  });
  prepare();
  assert.equal(accumulator.prepareProgramMaterial(hidden.generator).program, hiddenProgram.program);
  assert.equal(accumulator.prepareProgramMaterial(excluded.generator).program, excludedProgram.program);

  const replacement = source();
  hidden.node.visible = true;
  hidden.node.frameUpdate = () => { hidden.node.generator = replacement.generator; };
  prepare();
  const replacementProgram = accumulator.prepareProgramMaterial(replacement.generator);
  assert.notEqual(accumulator.prepareProgramMaterial(hidden.generator).program, hiddenProgram.program);
  assert.equal(accumulator.prepareProgramMaterial(excluded.generator).program, excludedProgram.program);
  scene.remove(excluded.node);
  prepare();
  assert.notEqual(accumulator.prepareProgramMaterial(excluded.generator).program, excludedProgram.program);
  assert.equal(accumulator.prepareProgramMaterial(replacement.generator).program, replacementProgram.program);
  accumulator.dispose();
  previous.dispose();
  programs.dispose();
});

test("a failed generation does not leave its material retained by the shared fullscreen quad", () => {
  const accumulator = new SplatAccumulator();
  const generated = source();
  const previousMaterial = SplatAccumulator.fullScreenQuad.material;
  const prepared = accumulator.prepareProgramMaterial(generated.generator);
  assert.equal(SplatAccumulator.fullScreenQuad.material, previousMaterial);
  accumulator.ensureGenerate({ maxSplats: 1 });
  let target: THREE.WebGLRenderTarget | null = null;
  const renderer = {
    xr: { enabled: true },
    autoClear: true,
    getRenderTarget: () => target,
    getActiveCubeFace: () => 0,
    getActiveMipmapLevel: () => 0,
    setRenderTarget: (next: THREE.WebGLRenderTarget | null) => { target = next; },
    render: (mesh: THREE.Mesh) => {
      assert.equal(mesh.material, prepared.material);
      throw new Error("draw failed");
    },
  };
  assert.throws(() => accumulator.generate({
    generator: generated.generator, base: 0, count: 1,
    renderer: renderer as unknown as THREE.WebGLRenderer,
  }), /draw failed/);
  assert.equal(SplatAccumulator.fullScreenQuad.material, previousMaterial);
  assert.equal(target, null);
  assert.equal(renderer.xr.enabled, true);
  assert.equal(renderer.autoClear, true);
  accumulator.dispose();
});
