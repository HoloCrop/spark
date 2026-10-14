// Synthetic overlapping layers: measure picker cost, not browser frame rate.
import * as THREE from 'three';
import { ExtSplats, SplatMesh, SplatPager, PagedSplats, SplatFileType } from '../src/index';
import { initialization } from '../src/wasm';
import { preparePagedExtResult } from '../src/PagedSplatDecode';
await initialization;
const count = 1250000;
const splats = new ExtSplats({maxSplats: count});
const center = new THREE.Vector3();
const scale = new THREE.Vector3(0.012, 0.007, 0.002);
const rotation = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0,1,0), 0.7);
const color = new THREE.Color(1,1,1);
for (let index = 0; index < count; index++) {
  center.set((index % 1000 - 500) * 0.006, (Math.floor(index / 1000) % 625 - 312) * 0.006, -5 - Math.floor(index / 625000));
  splats.pushSplat(center, scale, rotation, 0.8, color);
}
// Headless renderer: this benchmark measures CPU picking, not GPU upload or rendering.
const renderer = {initTexture() {}} as THREE.WebGLRenderer;
const pager = new SplatPager({renderer, maxSplats: count, extSplats: true, autoDrive: false});
const paged = new PagedSplats({pager, fileType: SplatFileType.PLY});
const uploads = [];
const boundsBuilds = [];
for (let start = 0; start < count; start += pager.pageSplats) {
  const page = start / pager.pageSplats;
  pager.pageToSplatsChunk[page] = {splats: paged, chunk: page, time: 0};
  const startBounds = performance.now();
  const decoded = preparePagedExtResult({numSplats: Math.min(pager.pageSplats,count-start),
    extArrays: [splats.extArrays[0].subarray(start*4,(start+pager.pageSplats)*4),
      splats.extArrays[1].subarray(start*4,(start+pager.pageSplats)*4)],
    extra: {lodTree: new Uint32Array()}});
  boundsBuilds.push(performance.now()-startBounds);
  pager.readyUploads.push({page, numSplats: decoded.numSplats,
    packedArray: decoded.extArrays[0], extArray: decoded.extArrays[1],
    raycastBounds: decoded.raycastBounds,
    labels: new Uint32Array(pager.pageSplats).fill(1), shArrays: []});
  const startUpload = performance.now();
  pager.processUploads();
  uploads.push(performance.now()-startUpload);
}
const displayed = new Uint32Array(Math.ceil(count/16384)*16384);
for (let index=0;index<count;index++) displayed[index]=index;
paged.update(count,displayed);
const mesh = new SplatMesh({paged});
const camera = new THREE.PerspectiveCamera(60, 1.5, 0.01, 100);
const ray = new THREE.Raycaster();
const results = [];
for (let index = 0; index < 12; index++) {
  ray.setFromCamera(new THREE.Vector2((index - 6)*0.025, 0.07), camera);
  const start = performance.now();
  const [hit] = ray.intersectObject(mesh);
  results.push({ms:performance.now()-start, index:hit?.index, distance:hit?.distance});
}
console.log(JSON.stringify({count, boundsBuilds, uploads, results}));
mesh.dispose();
pager.dispose();
splats.dispose();
