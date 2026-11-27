import { SplatMesh } from './SplatMesh';
import * as THREE from "three";
export interface SplatSurfaceIntersection {
    distance: number;
    point: THREE.Vector3;
    object: THREE.Object3D;
    index: number;
}
/** Append the combined k=2 transmittance's 0.5 surface crossing. */
export declare function raycastSplatMesh(mesh: SplatMesh, raycaster: THREE.Raycaster, intersects: SplatSurfaceIntersection[]): void;
