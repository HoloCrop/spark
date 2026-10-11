import assert from "node:assert/strict";
import { test } from "node:test";
import * as THREE from "three";
import { ExtSplats, PackedSplats, SplatMesh } from "../src/index";
import { initialization } from "../src/wasm";

await initialization;

for (const Source of [PackedSplats, ExtSplats]) {
  test(`${Source.name} keeps hit identity through filtering, chunks, and LoD remapping`, () => {
    const splats = new Source({ maxSplats: 65538 });
    const center = new THREE.Vector3(0, 0, 5);
    const scale = new THREE.Vector3(0.1, 0.1, 0.1);
    const rotation = new THREE.Quaternion();
    const color = new THREE.Color(1, 1, 1);
    splats.setSplat(0, center, scale, rotation, 0, color);
    splats.setSplat(65536, center, scale, rotation, 0.8, color);
    splats.setSplat(65537, new THREE.Vector3(10, 0, 5), scale, rotation, 0.8, color);
    const mesh = new SplatMesh(splats instanceof PackedSplats ? { packedSplats: splats } : { extSplats: splats });
    // Supply the splat count normally populated by the renderer frame update.
    mesh.context.numSplats.value = splats.numSplats;
    const ray = new THREE.Raycaster(new THREE.Vector3(), new THREE.Vector3(0, 0, 1));
    const [direct] = ray.intersectObject(mesh);
    assert.equal(direct.index, 65536);
    assert.ok(direct.distance > 4 && direct.distance < 5);

    if (splats instanceof PackedSplats) {
      splats.lodSplats = new PackedSplats({ packedArray: splats.packedArray, numSplats: splats.numSplats });
    } else {
      splats.lodSplats = new ExtSplats({ extArrays: splats.extArrays, numSplats: splats.numSplats });
    }
    const indices = new Uint32Array(65538);
    indices[0] = 65537; // A miss must not shift subsequent sample identities.
    indices[65537] = 65536; // The only surface contributor is in the second chunk.
    mesh.raycastIndices = { numSplats: indices.length, indices };
    const [remapped] = ray.intersectObject(mesh);
    assert.equal(remapped.index, 65536);
    assert.equal(remapped.distance, direct.distance);
    assert.deepEqual(remapped.point, direct.point);

    indices[65537] = 0; // A hidden/zero-opacity source cannot retain the previous hit.
    assert.deepEqual(ray.intersectObject(mesh), []);
    mesh.dispose();
  });
}
