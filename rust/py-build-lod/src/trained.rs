//! Direct array input: no intermediate file format or PLY decoding.
use pyo3::{buffer::PyBuffer, exceptions::PyValueError, prelude::*};
use spark_lib::{gsplat::{Gsplat, GsplatArray, GsplatSH1, GsplatSH2, GsplatSH3}, tsplat::TsplatArray};
use glam::{Quat, Vec3A};
use half::f16;
use std::{array, path::Path};
use spark_lib::trained_lod::RadOutput;

#[pyclass]
pub struct TrainedLevel {
    position: PyBuffer<f32>,
    rotation: PyBuffer<f32>,
    log_scaling: PyBuffer<f32>,
    alpha_logit: PyBuffer<f32>,
    sh_feature: PyBuffer<f32>,
    labels: PyBuffer<i32>,
    sh_degree: usize,
}

#[pymethods]
impl TrainedLevel {
    #[new]
    fn new(position: PyBuffer<f32>, rotation: PyBuffer<f32>, log_scaling: PyBuffer<f32>,
           alpha_logit: PyBuffer<f32>, sh_feature: PyBuffer<f32>, labels: PyBuffer<i32>) -> PyResult<Self> {
        let count = alpha_logit.item_count();
        let sh_degree = match sh_feature.shape() {
            [rows, 3, coefficients] if *rows == count => match coefficients {
                1 => 0,
                4 => 1,
                9 => 2,
                16 => 3,
                _ => return Err(PyValueError::new_err("expected 1, 4, 9 or 16 SH coefficients per RGB channel")),
            },
            _ => return Err(PyValueError::new_err("expected RGB-major SH coefficients shaped (N,3,C)")),
        };
        if position.item_count() != count * 3 || rotation.item_count() != count * 4
            || log_scaling.item_count() != count * 3
            || labels.item_count() != count * 2 {
            return Err(PyValueError::new_err("expected N positions, xyzw rotations, log scales, logits and class/instance labels"));
        }
        Ok(Self { position, rotation, log_scaling, alpha_logit, sh_feature, labels, sh_degree })
    }
}

impl TrainedLevel {
    pub(crate) fn splats(&self, py: Python<'_>) -> PyResult<GsplatArray> {
        let position = self.position.to_vec(py)?;
        let rotation = self.rotation.to_vec(py)?;
        let scales = self.log_scaling.to_vec(py)?;
        let opacity = self.alpha_logit.to_vec(py)?;
        let features = self.sh_feature.to_vec(py)?;
        let labels = self.labels.to_vec(py)?;
        let coefficients = (self.sh_degree + 1).pow(2);
        let mut result = GsplatArray::new_capacity(opacity.len(), self.sh_degree);
        for index in 0..opacity.len() {
            let shape = Vec3A::from_array(array::from_fn(|axis| scales[index * 3 + axis].exp()));
            let color = Vec3A::from_array(array::from_fn(|axis| 0.5 + features[(index * 3 + axis) * coefficients] * 0.28209479177387814));
            let quaternion = Quat::from_slice(&rotation[index * 4..index * 4 + 4]).normalize();
            let splat = Gsplat::new(Vec3A::from_slice(&position[index * 3..index * 3 + 3]),
                1.0 / (1.0 + (-opacity[index]).exp()), color, shape, quaternion,
                (labels[index * 2] + 1) as u32, (labels[index * 2 + 1] + 1) as u32);
            let sh = |band: usize, axis: usize| f16::from_f32(features[(index * 3 + axis) * coefficients + band]);
            result.push_splat(splat,
                (self.sh_degree >= 1).then(|| GsplatSH1(array::from_fn(|band| array::from_fn(|axis| sh(1 + band, axis))))),
                (self.sh_degree >= 2).then(|| GsplatSH2(array::from_fn(|band| array::from_fn(|axis| sh(4 + band, axis))))),
                (self.sh_degree >= 3).then(|| GsplatSH3(array::from_fn(|band| array::from_fn(|axis| sh(9 + band, axis))))));
        }
        Ok(result)
    }
}

#[pyfunction]
pub fn encode_trained_arrays(py: Python<'_>, levels: Vec<Py<TrainedLevel>>, parents: Vec<PyBuffer<u32>>,
                             output_dir: &str, moment_factor: f32) -> PyResult<()> {
    encode(py, levels, parents, RadOutput::Directory(Path::new(output_dir)), moment_factor)
}

#[pyfunction]
pub fn encode_trained_archive(py: Python<'_>, levels: Vec<Py<TrainedLevel>>, parents: Vec<PyBuffer<u32>>,
                              output_file: &str, moment_factor: f32) -> PyResult<()> {
    encode(py, levels, parents, RadOutput::Archive(Path::new(output_file)), moment_factor)
}

fn encode(py: Python<'_>, levels: Vec<Py<TrainedLevel>>, parents: Vec<PyBuffer<u32>>,
          output: RadOutput<'_>, moment_factor: f32) -> PyResult<()> {
    let clouds = levels.iter().map(|level| level.borrow(py).splats(py)).collect::<PyResult<Vec<_>>>()?;
    let parents = parents.iter().map(|parent| parent.to_vec(py)).collect::<PyResult<Vec<_>>>()?;
    py.detach(|| spark_lib::trained_lod::encode_arrays(clouds, parents, output, moment_factor))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}
