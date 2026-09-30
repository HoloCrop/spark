//! Encode a GPU-produced binary tree without rebuilding its geometry or serializing an intermediate file.
use pyo3::{buffer::PyBuffer, exceptions::PyValueError, prelude::*};
use spark_lib::{bhatt_lod, chunk_tree, gsplat::{Gsplat, GsplatSH1, GsplatSH2, GsplatSH3},
    rad::{RadEncoder, RadChunkPropertyCompression}, tsplat::{Tsplat, TsplatArray}};
use glam::{Quat, Vec3A};
use half::f16;
use std::{array, path::Path, time::Instant};
use crate::trained::TrainedLevel;
use spark_lib::trained_lod::RadOutput;

#[pyclass(eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq)]
pub enum Compression {
    Gz,
    Zstd,
}

impl From<Compression> for RadChunkPropertyCompression {
    fn from(compression: Compression) -> Self {
        match compression {
            Compression::Gz => Self::Gz,
            Compression::Zstd => Self::Zstd,
        }
    }
}

#[pyclass(frozen, get_all)]
pub struct EncodeTimings {
    input_seconds: f64,
    assemble_seconds: f64,
    prune_seconds: f64,
    chunk_seconds: f64,
    resolve_seconds: f64,
    encode_seconds: f64,
    write_seconds: f64,
}

#[pyclass]
pub struct MergedLevel {
    position: PyBuffer<f32>,
    rotation: PyBuffer<f32>,
    scales: PyBuffer<f32>,
    opacity: PyBuffer<f32>,
    labels: PyBuffer<i32>,
    right_weight: PyBuffer<f32>,
}

#[pymethods]
impl MergedLevel {
    #[new]
    fn new(position: PyBuffer<f32>, rotation: PyBuffer<f32>, scales: PyBuffer<f32>,
           opacity: PyBuffer<f32>, labels: PyBuffer<i32>, right_weight: PyBuffer<f32>) -> PyResult<Self> {
        let count = opacity.item_count();
        if position.item_count() != count * 3 || rotation.item_count() != count * 4
            || scales.item_count() != count * 3 || labels.item_count() != count * 2
            || right_weight.item_count() != count {
            return Err(PyValueError::new_err("expected N parent positions, xyzw rotations, scales, opacities, labels and right-child weights"));
        }
        Ok(Self { position, rotation, scales, opacity, labels, right_weight })
    }
}

fn mix<const N: usize>(left: [[f16; 3]; N], right: [[f16; 3]; N], weight: f32) -> [[f16; 3]; N] {
    array::from_fn(|band| array::from_fn(|axis| {
        let a = left[band][axis].to_f32();
        f16::from_f32(a + weight * (right[band][axis].to_f32() - a))
    }))
}

#[pyfunction]
#[pyo3(signature = (leaves, parents, children, output_dir, lod_base, compression=Compression::Gz))]
pub fn encode_merged_arrays(py: Python<'_>, leaves: &TrainedLevel, parents: &MergedLevel,
                            children: PyBuffer<u32>, output_dir: &str, lod_base: f32,
                            compression: Compression) -> PyResult<()> {
    encode(py, leaves, parents, children, RadOutput::Directory(Path::new(output_dir)), lod_base, compression)?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (leaves, parents, children, output_file, lod_base, compression=Compression::Gz))]
pub fn encode_merged_archive(py: Python<'_>, leaves: &TrainedLevel, parents: &MergedLevel,
                             children: PyBuffer<u32>, output_file: &str, lod_base: f32,
                             compression: Compression) -> PyResult<EncodeTimings> {
    encode(py, leaves, parents, children, RadOutput::Archive(Path::new(output_file)), lod_base, compression)
}

fn encode(py: Python<'_>, leaves: &TrainedLevel, parents: &MergedLevel,
          children: PyBuffer<u32>, output: RadOutput<'_>, lod_base: f32,
          compression: Compression) -> PyResult<EncodeTimings> {
    let start = Instant::now();
    let mut splats = leaves.splats(py)?;
    let leaf_count = splats.len();
    let position = parents.position.to_vec(py)?;
    let rotation = parents.rotation.to_vec(py)?;
    let scales = parents.scales.to_vec(py)?;
    let opacity = parents.opacity.to_vec(py)?;
    let labels = parents.labels.to_vec(py)?;
    let weights = parents.right_weight.to_vec(py)?;
    let children = children.to_vec(py)?;
    if opacity.len() + 1 != leaf_count || children.len() != opacity.len() * 2 {
        return Err(PyValueError::new_err("a full binary tree requires N-1 parents and N-1 child pairs"));
    }
    let mut used = vec![false; leaf_count + opacity.len()];
    for (index, pair) in children.chunks_exact(2).enumerate() {
        for &child in pair {
            let child = child as usize;
            if child >= leaf_count + index || used[child] {
                return Err(PyValueError::new_err("children must precede their parent and have exactly one owner"));
            }
            used[child] = true;
        }
    }
    let input_seconds = start.elapsed().as_secs_f64();
    py.detach(|| -> anyhow::Result<EncodeTimings> {
        let start = Instant::now();
        splats.splats.reserve(opacity.len());
        splats.sh1.reserve(opacity.len());
        splats.sh2.reserve(opacity.len());
        splats.sh3.reserve(opacity.len());
        for (index, pair) in children.chunks_exact(2).enumerate() {
            let (left, right) = (pair[0] as usize, pair[1] as usize);
            let weight = weights[index];
            let color = splats.get(left).rgb().lerp(splats.get(right).rgb(), weight);
            let splat = Gsplat::new(Vec3A::from_slice(&position[index*3..index*3+3]), opacity[index], color,
                Vec3A::from_slice(&scales[index*3..index*3+3]),
                Quat::from_slice(&rotation[index*4..index*4+4]).normalize(),
                (labels[index*2]+1) as u32, (labels[index*2+1]+1) as u32);
            let sh1 = GsplatSH1(mix(splats.sh1[left].0, splats.sh1[right].0, weight));
            let sh2 = GsplatSH2(mix(splats.sh2[left].0, splats.sh2[right].0, weight));
            let sh3 = GsplatSH3(mix(splats.sh3[left].0, splats.sh3[right].0, weight));
            splats.push_splat(splat, Some(sh1), Some(sh2), Some(sh3));
        }
        splats.prepare_children();
        for (index, pair) in children.chunks_exact(2).enumerate() {
            splats.set_children(leaf_count + index, &[pair[0] as usize, pair[1] as usize]);
        }
        let assemble_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        bhatt_lod::prune_lod_tree(&mut splats, leaf_count, lod_base, |_| {});
        let prune_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        chunk_tree::chunk_tree(&mut splats, 0, |_| {});
        splats.encode_lod_opacity();
        let chunk_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let mut encoder = RadEncoder::new(splats);
        encoder.compression = compression.into();
        encoder.resolve_encoding();
        let resolve_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let mut header = Vec::new();
        let chunks = encoder.encode_with_chunks(&mut header, "bay-lod-")?;
        let encode_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        output.write(header, chunks)?;
        let write_seconds = start.elapsed().as_secs_f64();
        Ok(EncodeTimings { input_seconds, assemble_seconds, prune_seconds, chunk_seconds,
            resolve_seconds, encode_seconds, write_seconds })
    }).map_err(|error| PyValueError::new_err(error.to_string()))
}
