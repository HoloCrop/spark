import * as THREE from "three";

import {
  get_raycast_buffer,
  get_raycast_buffer2,
  get_raycast_indices,
  raycast_ext_buffers,
  raycast_packed_buffer,
  raycast_surface_depth,
  raycast_surface_index,
} from "spark-rs";
import type { SplatMesh } from "./SplatMesh";
import { LN_SCALE_MAX, LN_SCALE_MIN } from "./defines";
import * as wasm from "./wasm";

export interface SplatSurfaceIntersection {
  distance: number;
  point: THREE.Vector3;
  object: THREE.Object3D;
  index: number;
}

// Shared synchronous scratch storage, independent of mesh and renderer lifetime.
let raycastBuffer = new Float32Array(1024);
let raycastSplatIndices = new Uint32Array(342);

/** Append the combined k=2 transmittance's 0.5 surface crossing. */
export function raycastSplatMesh(
  mesh: SplatMesh,
  raycaster: THREE.Raycaster,
  intersects: SplatSurfaceIntersection[],
) {
  if (
    !wasm.isInitialized() ||
    !mesh.raycastable ||
    mesh.opacity === 0 ||
    (!mesh.packedSplats && !mesh.extSplats && !mesh.paged)
  ) {
    return;
  }
  const paged = mesh.paged != null;
  const ext = paged
    ? (mesh.paged?.pager?.extSplats ?? false)
    : mesh.extSplats != null;

  const { ray } = raycaster;
  let { near, far } = raycaster;
  const worldToMesh = mesh.matrixWorld.clone().invert();
  const worldToMeshRot = new THREE.Matrix3().setFromMatrix4(worldToMesh);
  const origin = ray.origin.clone().applyMatrix4(worldToMesh);
  const direction = ray.direction.clone().applyMatrix3(worldToMeshRot);
  let projection: Float32Array | undefined;
  if (raycaster.camera instanceof THREE.PerspectiveCamera) {
    const meshToView = new THREE.Matrix4().multiplyMatrices(
      raycaster.camera.matrixWorldInverse,
      mesh.matrixWorld,
    );
    projection = new Float32Array([
      ...new THREE.Matrix3().setFromMatrix4(meshToView).elements,
      raycaster.camera.near,
      raycaster.camera.far,
      raycaster.camera.projectionMatrix.elements[0],
      raycaster.camera.projectionMatrix.elements[5],
      raycaster.camera.projectionMatrix.elements[8],
      raycaster.camera.projectionMatrix.elements[9],
    ]);
    const forward = -ray.direction
      .clone()
      .transformDirection(raycaster.camera.matrixWorldInverse).z;
    near = Math.max(near, raycaster.camera.near / forward);
    far = Math.min(far, raycaster.camera.far / forward);
  }

  let numSplats =
    mesh.raycastIndices?.numSplats ??
    (paged ? mesh.paged?.numSplats : mesh.context.numSplats.value) ??
    0;
  let indices =
    mesh.raycastIndices?.indices ??
    (paged
      ? (mesh.paged?.dynoIndices.value.image.data as Uint32Array)
      : mesh.context.enableLod.value
        ? (mesh.context.lodIndices.value.image.data as Uint32Array)
        : null) ??
    null;
  if (mesh.paged?.pager && projection && !mesh.raycastIndices) {
    const displayedIndices = mesh.paged.dynoIndices.value.image
      .data as Uint32Array;
    indices = mesh.paged.pager.raycastBlocks.query(
      mesh.paged.pager,
      displayedIndices.subarray(0, numSplats),
      projection,
      origin.toArray(),
      direction.toArray(),
    );
    numSplats = indices.length;
  }
  // Querying the index may grow Wasm memory: acquire its scratch views afterwards.
  const buffer = get_raycast_buffer();
  const bufferSize = buffer.length / 4;
  let samples = 0;
  const labels = mesh.paged?.pager?.labelTexture.value.image.data as
    | Uint32Array
    | undefined;
  const labelsLookUp = mesh.paged?.pager?.lookUpTexture.value.image.data as
    | Uint32Array
    | undefined;

  if (!ext) {
    const packed = paged
      ? (mesh.paged?.pager?.packedTexture.value.image.data as Uint32Array)
      : indices
        ? mesh.packedSplats?.lodSplats?.packedArray
        : mesh.packedSplats?.packedArray;
    if (!packed) {
      return;
    }
    const splatEncoding = paged
      ? mesh.paged?.splatEncoding
      : mesh.packedSplats?.splatEncoding;
    for (let base = 0; base < numSplats; base += bufferSize) {
      const count = Math.min(bufferSize, numSplats - base);
      if (!indices) {
        buffer.set(packed.subarray(base * 4, (base + count) * 4));
      } else {
        for (let i = 0; i < count; ++i) {
          const index = indices[base + i];
          const i4 = i * 4;
          const index4 = index * 4;
          buffer[i4] = packed[index4];
          buffer[i4 + 1] = packed[index4 + 1];
          buffer[i4 + 2] = packed[index4 + 2];
          buffer[i4 + 3] = packed[index4 + 3];
        }
      }

      if (labels !== undefined && labelsLookUp !== undefined) {
        for (let i = 0; i < count; ++i) {
          const index = indices ? indices[base + i] : base + i;
          if (labelsLookUp[labels[index]] === 0) {
            buffer[i * 4] &= 0x00ffffff;
          }
        }
      }

      const profiles = raycast_packed_buffer(
        origin.x,
        origin.y,
        origin.z,
        direction.x,
        direction.y,
        direction.z,
        mesh.minRaycastOpacity,
        count,
        splatEncoding?.lnScaleMin ?? LN_SCALE_MIN,
        splatEncoding?.lnScaleMax ?? LN_SCALE_MAX,
        splatEncoding?.lodOpacity ?? false,
        projection,
      );
      samples = appendRaycastBuffer(
        samples,
        profiles,
        get_raycast_indices(),
        base,
        indices,
      );
    }
  } else {
    const buffer2 = get_raycast_buffer2();
    const ext1 = paged
      ? (mesh.paged?.pager?.packedTexture.value.image.data as Uint32Array)
      : indices
        ? mesh.extSplats?.lodSplats?.extArrays[0]
        : mesh.extSplats?.extArrays[0];
    const ext2 = paged
      ? (mesh.paged?.pager?.extTexture.value.image.data as Uint32Array)
      : indices
        ? mesh.extSplats?.lodSplats?.extArrays[1]
        : mesh.extSplats?.extArrays[1];

    if (!ext1 || !ext2) {
      return;
    }
    for (let base = 0; base < numSplats; base += bufferSize) {
      const count = Math.min(bufferSize, numSplats - base);
      if (!indices) {
        buffer.set(ext1.subarray(base * 4, (base + count) * 4));
        buffer2.set(ext2.subarray(base * 4, (base + count) * 4));
      } else {
        for (let i = 0; i < count; ++i) {
          const index = indices[base + i];
          const i4 = i * 4;
          const index4 = index * 4;
          buffer[i4] = ext1[index4];
          buffer[i4 + 1] = ext1[index4 + 1];
          buffer[i4 + 2] = ext1[index4 + 2];
          buffer[i4 + 3] = ext1[index4 + 3];

          buffer2[i4] = ext2[index4];
          buffer2[i4 + 1] = ext2[index4 + 1];
          buffer2[i4 + 2] = ext2[index4 + 2];
          buffer2[i4 + 3] = ext2[index4 + 3];
        }
      }

      if (labels !== undefined && labelsLookUp !== undefined) {
        for (let i = 0; i < count; ++i) {
          const index = indices ? indices[base + i] : base + i;
          if (labelsLookUp[labels[index]] === 0) {
            buffer[i * 4 + 3] = 0;
          }
        }
      }

      const profiles = raycast_ext_buffers(
        origin.x,
        origin.y,
        origin.z,
        direction.x,
        direction.y,
        direction.z,
        mesh.minRaycastOpacity,
        count,
        projection,
      );
      samples = appendRaycastBuffer(
        samples,
        profiles,
        get_raycast_indices(),
        base,
        indices,
      );
    }
  }

  const distance = raycast_surface_depth(
    raycastBuffer.subarray(0, samples),
    near,
    far,
  );
  if (distance !== undefined) {
    const point = ray.direction
      .clone()
      .multiplyScalar(distance)
      .add(ray.origin);
    intersects.push({
      distance,
      point,
      object: mesh,
      index:
        raycastSplatIndices[
          raycast_surface_index(raycastBuffer.subarray(0, samples), distance)
        ],
    });
  }
}

function appendRaycastBuffer(
  count: number,
  additional: Float32Array,
  sourceIndices: Uint32Array,
  base: number,
  indices: Uint32Array | null,
) {
  const total = count + additional.length;
  let capacity = raycastBuffer.length;

  if (total > capacity) {
    while (capacity < total) {
      capacity *= 2;
    }
    const newBuffer = new Float32Array(capacity);
    newBuffer.set(raycastBuffer.subarray(0, count));
    raycastBuffer = newBuffer;
    const newIndices = new Uint32Array(Math.ceil(capacity / 3));
    newIndices.set(raycastSplatIndices.subarray(0, count / 3));
    raycastSplatIndices = newIndices;
  }

  for (let index = 0; index < sourceIndices.length; index++) {
    const source = base + sourceIndices[index];
    raycastSplatIndices[count / 3 + index] = indices ? indices[source] : source;
  }
  raycastBuffer.set(additional, count);
  return count + additional.length;
}
