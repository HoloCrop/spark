// Synthetic overlapping layers: measure picker cost, not browser frame rate.
import * as THREE from 'three';
import { ExtSplats, SplatMesh } from '../src/index';
import { initialization } from '../src/wasm';
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
const mesh = new SplatMesh({extSplats:splats});
mesh.context.numSplats.value = count;
const camera = new THREE.PerspectiveCamera(60, 1.5, 0.01, 100);
const ray = new THREE.Raycaster();
const results = [];
for (let index = 0; index < 12; index++) {
  ray.setFromCamera(new THREE.Vector2((index - 6)*0.025, 0.07), camera);
  const start = performance.now();
  const [hit] = ray.intersectObject(mesh);
  results.push({ms:performance.now()-start, index:hit?.index, distance:hit?.distance});
}
console.log(JSON.stringify({count, results}));
mesh.dispose();
