import assert from "node:assert/strict";
import { test } from "node:test";
import * as THREE from "three";
import { ExtSplats, PackedSplats, SplatMesh, SplatPager, PagedSplats, SplatFileType } from "../src/index";
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

for (const ext of [false, true]) {
  test(`paged ${ext ? "extended" : "packed"} block cache follows uploads, visibility and display indices`, () => {
    // GPU uploads are irrelevant to this CPU picking integration test.
    const renderer = { initTexture() {} } as THREE.WebGLRenderer;
    const pager = new SplatPager({renderer, extSplats: ext, maxSplats: 65536, autoDrive: false});
    const paged = new PagedSplats({pager, fileType: SplatFileType.PLY});
    pager.pageToSplatsChunk[0] = {splats: paged, chunk: 0, time: 0};
    const mesh = new SplatMesh({paged});
    const splats = ext ? new ExtSplats({maxSplats: 128}) : new PackedSplats({maxSplats: 128});
    const scale = new THREE.Vector3(0.05, 0.05, 0.05);
    const rotation = new THREE.Quaternion();
    const color = new THREE.Color(1,1,1);
    for (let index = 0; index < 128; index++) {
      splats.setSplat(index, new THREE.Vector3(0,0,index < 64 ? -3 : -5), scale, rotation, 0.9, color);
    }
    const upload = () => {
      pager.readyUploads.push({page: 0, numSplats: 128,
        packedArray: splats instanceof ExtSplats ? splats.extArrays[0] : splats.packedArray,
        extArray: splats instanceof ExtSplats ? splats.extArrays[1] : undefined,
        shArrays: [], labels: new Uint32Array(128).fill(1), instances: new Uint32Array(128).fill(1)});
      pager.processUploads();
    };
    upload();
    const displayed = new Uint32Array(16384);
    displayed[0] = 64;
    displayed[1] = 0;
    paged.update(2, displayed);
    const ray = new THREE.Raycaster();
    ray.setFromCamera(new THREE.Vector2(), new THREE.PerspectiveCamera(60,1,0.01,100));
    const [indexed] = ray.intersectObject(mesh);
    mesh.raycastIndices = {numSplats: 2, indices: displayed};
    const [exhaustive] = ray.intersectObject(mesh);
    assert.equal(indexed.index, 0);
    assert.equal(indexed.index, exhaustive.index);
    assert.equal(indexed.distance, exhaustive.distance);
    mesh.raycastIndices = undefined;

    pager.updateLabelLookup(new Set());
    assert.deepEqual(ray.intersectObject(mesh), []);
    pager.updateLabelLookup(new Set([1]));
    assert.equal(ray.intersectObject(mesh)[0].index, 0);
    for (let index = 0; index < 64; index++) {
      splats.setSplat(index, new THREE.Vector3(10,0,-3), scale, rotation, 0.9, color);
    }
    upload();
    assert.equal(ray.intersectObject(mesh)[0].index, 64);
    paged.numSplats = 1;
    displayed[0] = 0;
    assert.deepEqual(ray.intersectObject(mesh), []);
    mesh.dispose();
    pager.dispose();
    splats.dispose();
  });
}
