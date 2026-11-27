use glam::DVec3;
use js_sys::Float32Array;
use spark_lib::{decoder::SplatEncoding, splat_encode::{decode_ext_splat_center, decode_ext_splat_scale, decode_packed_splat_center, decode_packed_splat_scale}};
use wasm_bindgen::prelude::*;
use crate::raycast::RaycastProjection;

const BLOCK_SIZE: usize = 64;
const BLOCKS_PER_PAGE: usize = 65536 / BLOCK_SIZE;
const VALUES_PER_BOUND: usize = 7;

#[derive(Clone, Copy)]
struct Bounds {
    minimum: [f32; 3],
    maximum: [f32; 3],
    scale: f32,
}

impl Default for Bounds {
    fn default() -> Self {
        Self { minimum: [f32::INFINITY; 3], maximum: [f32::NEG_INFINITY; 3], scale: 0.0 }
    }
}

impl Bounds {
    fn include(&mut self, other: &Self) {
        for axis in 0..3 {
            self.minimum[axis] = self.minimum[axis].min(other.minimum[axis]);
            self.maximum[axis] = self.maximum[axis].max(other.maximum[axis]);
        }
        self.scale = self.scale.max(other.scale);
    }

    fn intersects(&self, projection: &RaycastProjection, origin: [f32; 3], slope: DVec3, row_lengths: DVec3) -> bool {
        if self.minimum[0] > self.maximum[0] { return false; }
        // Subtract in f32, as the narrow phase does, before enclosing all centers.
        let minimum = DVec3::from_array(std::array::from_fn(|axis| f64::from(self.minimum[axis] - origin[axis])));
        let maximum = DVec3::from_array(std::array::from_fn(|axis| f64::from(self.maximum[axis] - origin[axis])));
        let matrix = projection.camera_from_mesh;
        let center = matrix * ((minimum + maximum) * 0.5);
        let extent = (maximum - minimum) * 0.5;
        let extent = matrix.x_axis.abs() * extent.x + matrix.y_axis.abs() * extent.y + matrix.z_axis.abs() * extent.z;
        let lower = center - extent;
        let upper = center + extent;
        if upper.z <= projection.near || lower.z >= projection.far { return false; }
        if lower.z <= projection.near { return true; }
        let inverse = lower.z.recip();
        // J * rotation * scale is bounded by max(scale) * |J|. The triangle
        // bound below encloses every projected footprint in this block.
        // Six exceeds the maximum opacity-dependent squared support radius.
        let radius = f64::from(self.scale) * 6.0_f64.sqrt();
        for axis in 0..2 {
            let padding = radius * (row_lengths[axis] * inverse
                + lower[axis].abs().max(upper[axis].abs()) * row_lengths.z * inverse * inverse);
            let low = (lower[axis] / lower.z).min(lower[axis] / upper.z) - padding;
            let high = (upper[axis] / lower.z).max(upper[axis] / upper.z) + padding;
            if slope[axis] < low || slope[axis] > high { return false; }
        }
        true
    }
}

fn build_bounds(count: usize, decode: impl Fn(usize) -> ([f32; 3], [f32; 3])) -> impl Iterator<Item = Bounds> {
    (0..count).step_by(BLOCK_SIZE).map(move |first| {
        let mut bounds = Bounds::default();
        for index in first..(first + BLOCK_SIZE).min(count) {
            let (center, scale) = decode(index);
            for axis in 0..3 {
                bounds.minimum[axis] = bounds.minimum[axis].min(center[axis]);
                bounds.maximum[axis] = bounds.maximum[axis].max(center[axis]);
                bounds.scale = bounds.scale.max(scale[axis]);
            }
        }
        bounds
    })
}

fn ext_bounds<'a>(packed: &'a [u32], geometry: &'a [u32]) -> impl Iterator<Item = Bounds> + 'a {
    build_bounds(packed.len() / 4, move |index| {
        (decode_ext_splat_center(&packed[index*4..index*4+4]), decode_ext_splat_scale(&geometry[index*4..index*4+4]))
    })
}

fn packed_bounds(packed: &[u32], ln_scale_min: f32, ln_scale_max: f32) -> impl Iterator<Item = Bounds> + '_ {
    let encoding = SplatEncoding { ln_scale_min, ln_scale_max, ..Default::default() };
    build_bounds(packed.len() / 4, move |index| {
        (decode_packed_splat_center(&packed[index*4..index*4+4]), decode_packed_splat_scale(&packed[index*4..index*4+4], &encoding))
    })
}

fn encode_bounds(bounds: impl Iterator<Item = Bounds>) -> Vec<f32> {
    bounds.flat_map(|bounds| [bounds.minimum[0], bounds.minimum[1], bounds.minimum[2],
        bounds.maximum[0], bounds.maximum[1], bounds.maximum[2], bounds.scale]).collect()
}

/// Worker-side page construction. Each block transfers min/max center and maximum scale.
#[wasm_bindgen]
pub fn build_ext_raycast_bounds(packed: &[u32], geometry: &[u32]) -> Vec<f32> {
    encode_bounds(ext_bounds(packed, geometry))
}

#[wasm_bindgen]
pub fn build_packed_raycast_bounds(packed: &[u32], ln_scale_min: f32, ln_scale_max: f32) -> Vec<f32> {
    encode_bounds(packed_bounds(packed, ln_scale_min, ln_scale_max))
}

/// Fixed blocks reuse the stream's ordering; this index never substitutes LoD splats.
#[wasm_bindgen]
pub struct RaycastBlockIndex {
    bounds: Vec<Bounds>,
    visited: Vec<u8>,
    pages: Vec<Bounds>,
    visited_pages: Vec<u8>,
}

impl RaycastBlockIndex {
    fn update(&mut self, start: usize, count: usize, bounds: impl Iterator<Item = Bounds>) {
        for (block, bounds) in bounds.enumerate() {
            self.bounds[start / BLOCK_SIZE + block] = bounds;
        }
        for page in start / 65536..(start + count).div_ceil(65536) {
            let mut bounds = Bounds::default();
            for block in &self.bounds[page * BLOCKS_PER_PAGE..((page + 1) * BLOCKS_PER_PAGE).min(self.bounds.len())] {
                bounds.include(block);
            }
            self.pages[page] = bounds;
        }
    }

    pub(crate) fn candidates(&mut self, indices: &[u32], projection: &RaycastProjection, origin: [f32; 3], direction: [f32; 3]) -> Vec<u32> {
        self.visited.fill(0);
        self.visited_pages.fill(0);
        let direction = projection.camera_from_mesh * DVec3::from_array(direction.map(f64::from));
        let slope = direction / direction.z;
        let rows = projection.camera_from_mesh.transpose();
        let lengths = DVec3::new(rows.x_axis.length(), rows.y_axis.length(), rows.z_axis.length());
        let mut result = Vec::new();
        for &index in indices {
            let block = index as usize / BLOCK_SIZE;
            let page = block / BLOCKS_PER_PAGE;
            if self.visited_pages[page] == 0 {
                self.visited_pages[page] = if self.pages[page].intersects(projection, origin, slope, lengths) { 2 } else { 1 };
            }
            if self.visited_pages[page] == 1 { continue; }
            if self.visited[block] == 0 {
                self.visited[block] = if self.bounds[block].intersects(projection, origin, slope, lengths) { 2 } else { 1 };
            }
            if self.visited[block] == 2 { result.push(index); }
        }
        result
    }
}

#[wasm_bindgen]
impl RaycastBlockIndex {
    #[wasm_bindgen(constructor)]
    pub fn new(max_splats: usize) -> Self {
        let blocks = max_splats.div_ceil(BLOCK_SIZE);
        let pages = blocks.div_ceil(BLOCKS_PER_PAGE);
        Self { bounds: vec![Bounds::default(); blocks], visited: vec![0; blocks],
            pages: vec![Bounds::default(); pages], visited_pages: vec![0; pages] }
    }

    pub fn update_ext(&mut self, start: usize, packed: &[u32], geometry: &[u32]) {
        self.update(start, packed.len() / 4, ext_bounds(packed, geometry));
    }

    pub fn update_packed(&mut self, start: usize, packed: &[u32], ln_scale_min: f32, ln_scale_max: f32) {
        self.update(start, packed.len() / 4, packed_bounds(packed, ln_scale_min, ln_scale_max));
    }

    /// Replace a resident page using worker-built bounds, including partial and empty pages.
    pub fn install_page(&mut self, page: usize, bounds: &[f32]) {
        let first = page * BLOCKS_PER_PAGE;
        let end = (first + BLOCKS_PER_PAGE).min(self.bounds.len());
        let resident = &mut self.bounds[first..end];
        resident.fill(Bounds::default());
        let mut page_bounds = Bounds::default();
        for (block, values) in bounds.chunks_exact(VALUES_PER_BOUND).enumerate() {
            let bounds = Bounds { minimum: [values[0], values[1], values[2]],
                maximum: [values[3], values[4], values[5]], scale: values[6] };
            resident[block] = bounds;
            page_bounds.include(&bounds);
        }
        self.pages[page] = page_bounds;
    }

    pub fn query(&mut self, indices: &[u32], projection: Float32Array, origin: &[f32], direction: &[f32]) -> Vec<u32> {
        let projection = crate::raycast_projection(Some(projection)).unwrap();
        self.candidates(indices, &projection, [origin[0], origin[1], origin[2]], [direction[0], direction[1], direction[2]])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DMat3;
    use spark_lib::splat_encode::{encode_ext_splat, encode_packed_splat};

    #[test]
    fn worker_bounds_preserve_displayed_candidates_across_transforms() {
        let encoding = SplatEncoding::default();
        let origin = [3.0, -5.0, 4.0];
        let displayed = [65536 + 144, 65536 + 64, 65536 + 1, 65536 + 128, 65536 + 80, 65536];
        for matrix in [
            DMat3::IDENTITY,
            DMat3::from_diagonal(DVec3::new(0.3, 2.0, 1.5)),
            DMat3::from_cols(DVec3::new(1.0, 0.2, 0.1), DVec3::new(-0.4, 1.0, 0.3), DVec3::new(0.1, -0.2, 1.0)),
        ] {
            let projection = RaycastProjection { camera_from_mesh: matrix, near: 0.01, far: 100.0, image_from_camera: [1.0, 1.0, 0.0, 0.0] };
            let mut packed = vec![0; 145 * 4];
            let mut extended = vec![0; 145 * 4];
            let mut geometry = vec![0; 145 * 4];
            for index in 0..145 {
                let center = matrix.inverse() * DVec3::new((index / BLOCK_SIZE) as f64 * 10.0, 0.0, 5.0)
                    + DVec3::from_array(origin.map(f64::from));
                let center = center.as_vec3().to_array();
                encode_packed_splat(&mut packed[index*4..index*4+4], center, 0.8, [1.0; 3], [0.01; 3], [0.0, 0.0, 0.0, 1.0], &encoding);
                encode_ext_splat(&mut extended[index*4..index*4+4], &mut geometry[index*4..index*4+4], center, 0.8, [1.0; 3], [0.01; 3], [0.0, 0.0, 0.0, 1.0]);
            }
            for ext in [false, true] {
                let mut direct = RaycastBlockIndex::new(65536 + 145);
                let bounds = if ext {
                    direct.update_ext(65536, &extended, &geometry);
                    build_ext_raycast_bounds(&extended, &geometry)
                } else {
                    direct.update_packed(65536, &packed, encoding.ln_scale_min, encoding.ln_scale_max);
                    build_packed_raycast_bounds(&packed, encoding.ln_scale_min, encoding.ln_scale_max)
                };
                let mut transferred = RaycastBlockIndex::new(65536 + 145);
                transferred.install_page(1, &bounds);
                for block in 0..3 {
                    let direction = (matrix.inverse() * DVec3::new(block as f64 * 2.0, 0.0, 1.0)).normalize().as_vec3().to_array();
                    let expected: Vec<u32> = displayed.iter().copied().filter(|index| (index - 65536) / BLOCK_SIZE as u32 == block).collect();
                    assert_eq!(direct.candidates(&displayed, &projection, origin, direction), expected);
                    assert_eq!(transferred.candidates(&displayed, &projection, origin, direction), expected);
                }
            }
        }
    }

    #[test]
    fn worker_page_replacement_clears_old_blocks_without_changing_neighbors() {
        let projection = RaycastProjection { camera_from_mesh: DMat3::IDENTITY, near: 0.01, far: 100.0, image_from_camera: [1.0, 1.0, 0.0, 0.0] };
        let mut packed = vec![0; 128 * 4];
        let mut geometry = vec![0; 128 * 4];
        for index in 0..128 {
            encode_ext_splat(&mut packed[index*4..index*4+4], &mut geometry[index*4..index*4+4],
                [0.0, 0.0, 5.0], 0.8, [1.0; 3], [0.01; 3], [0.0, 0.0, 0.0, 1.0]);
        }
        let mut blocks = RaycastBlockIndex::new(65536 + 128);
        let bounds = build_ext_raycast_bounds(&packed, &geometry);
        blocks.install_page(0, &bounds);
        blocks.install_page(1, &bounds);
        let displayed = [65536, 64, 65536 + 64, 0];
        assert_eq!(blocks.candidates(&displayed, &projection, [0.0; 3], [0.0, 0.0, 1.0]), displayed);

        let partial = build_ext_raycast_bounds(&packed[..4], &geometry[..4]);
        blocks.install_page(0, &partial);
        assert_eq!(blocks.candidates(&displayed, &projection, [0.0; 3], [0.0, 0.0, 1.0]), [65536, 65536 + 64, 0]);
        blocks.install_page(1, &partial);
        assert_eq!(blocks.candidates(&displayed, &projection, [0.0; 3], [0.0, 0.0, 1.0]), [65536, 0]);
        blocks.install_page(0, &[]);
        assert_eq!(blocks.candidates(&displayed, &projection, [0.0; 3], [0.0, 0.0, 1.0]), [65536]);
        blocks.install_page(1, &[]);
        assert!(blocks.candidates(&displayed, &projection, [0.0; 3], [0.0, 0.0, 1.0]).is_empty());
    }

    #[test]
    fn page_replacement_and_displayed_subset_preserve_identity_and_order() {
        let projection = RaycastProjection { camera_from_mesh: DMat3::IDENTITY, near: 0.01, far: 100.0, image_from_camera: [1.0, 1.0, 0.0, 0.0] };
        let mut packed = vec![0; 128*4];
        let mut geometry = vec![0; 128*4];
        for index in 0..128 {
            encode_ext_splat(&mut packed[index*4..index*4+4], &mut geometry[index*4..index*4+4],
                [if index < 64 {0.0} else {10.0}, 0.0, 5.0], 0.8, [1.0;3], [0.01;3], [0.0, 0.0, 0.0, 1.0]);
        }
        let mut blocks = RaycastBlockIndex::new(65536 + 128);
        blocks.update_ext(0, &packed, &geometry);
        let displayed = [65, 5, 64, 0];
        assert_eq!(blocks.candidates(&displayed, &projection, [0.0;3], [0.0,0.0,1.0]), [5,0]);
        blocks.update_ext(0, &packed[64*4..], &geometry[64*4..]);
        assert!(blocks.candidates(&displayed, &projection, [0.0;3], [0.0,0.0,1.0]).is_empty());
        blocks.update_ext(64, &packed[..64*4], &geometry[..64*4]);
        assert_eq!(blocks.candidates(&displayed, &projection, [0.0;3], [0.0,0.0,1.0]), [65,64]);
        blocks.update_ext(65536, &packed[..64*4], &geometry[..64*4]);
        let displayed = [65536, 5, 65];
        assert_eq!(blocks.candidates(&displayed, &projection, [0.0;3], [0.0,0.0,1.0]), [65536,65]);
        blocks.update_ext(65536, &packed[64*4..], &geometry[64*4..]);
        assert_eq!(blocks.candidates(&displayed, &projection, [0.0;3], [0.0,0.0,1.0]), [65]);
    }
}
