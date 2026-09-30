
use std::array;

use glam::{Mat3A, Quat, Vec3, Vec3A};
use half::f16;
use smallvec::SmallVec;

use crate::decoder::{SetSplatEncoding, SplatEncoding, SplatGetter, SplatInit, SplatProps, SplatReceiver};
use crate::splat_encode::{encode_packed_splat, encode_sh1, encode_sh2, encode_sh3, get_splat_tex_size};
use crate::symmat3::SymMat3;
use crate::tsplat::{Tsplat, TsplatArray, TsplatMut, apply_swaps, compute_swaps, similarity_metric_with_moments};

const INFLATE_SCALE: bool = false;

#[derive(Clone, Default)]
pub struct Gsplat {
    pub center: Vec3,
    pub opacity: f16,
    pub rgb: [f16; 3],
    pub ln_scales: [f16; 3],
    pub quaternion: [f16; 4],
    pub label: u32,
    pub instance_label: u32,
}

impl Gsplat {
    pub fn new(center: Vec3A, opacity: f32, rgb: Vec3A, scales: Vec3A, quaternion: Quat, label: u32, instance_label: u32) -> Self {
        Self {
            center: center.to_vec3(),
            opacity: f16::from_f32(opacity),
            rgb: rgb.to_array().map(|v| f16::from_f32(v)),
            ln_scales: scales.to_array().map(|v| f16::from_f32(v.ln())),
            quaternion: quaternion.to_array().map(|v| f16::from_f32(v)),
            label: label,
            instance_label: instance_label
        }
    }
}

impl std::fmt::Debug for Gsplat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Gsplat(center: {:?}, opacity: {:?}, rgb: {:?}, scales: {:?}, quaternion: {:?})", self.center(), self.opacity(), self.rgb(), self.scales(), self.quaternion())
    }
}

impl<'a> Tsplat for &'a Gsplat {
    fn center(&self) -> Vec3A {
        self.center.to_vec3a()
    }

    fn opacity(&self) -> f32 {
        self.opacity.to_f32()
    }

    fn rgb(&self) -> Vec3A {
        Vec3A::from_array(self.rgb.map(|x| x.to_f32()))
    }

    fn scales(&self) -> Vec3A {
        Vec3A::from_array(self.ln_scales.map(|x| x.to_f32().exp()))
    }

    fn quaternion(&self) -> Quat {
        Quat::from_array(self.quaternion.map(|x| x.to_f32()))
    }
    
    fn max_scale(&self) -> f32 {
        self.ln_scales[0].max(self.ln_scales[1]).max(self.ln_scales[2]).to_f32().exp()
    }

    fn label(&self) -> u32 {
        self.label
    }

    fn instance_label(&self) -> u32 {
        self.instance_label
    }

}

impl<'a> Tsplat for &'a mut Gsplat {
    fn center(&self) -> Vec3A {
        self.center.to_vec3a()
    }

    fn opacity(&self) -> f32 {
        self.opacity.to_f32()
    }

    fn rgb(&self) -> Vec3A {
        Vec3A::from_array(self.rgb.map(|x| x.to_f32()))
    }

    fn scales(&self) -> Vec3A {
        Vec3A::from_array(self.ln_scales.map(|x| x.to_f32().exp()))
    }

    fn quaternion(&self) -> Quat {
        Quat::from_array(self.quaternion.map(|x| x.to_f32()))
    }

    fn max_scale(&self) -> f32 {
        self.ln_scales[0].max(self.ln_scales[1]).max(self.ln_scales[2]).to_f32().exp()
    }

    fn label(&self) -> u32 {
        self.label
    }

    fn instance_label(&self) -> u32 {
        self.instance_label
    }
}

impl<'a> TsplatMut for &'a mut Gsplat {
    fn set_center(&mut self, center: Vec3A) {
        self.center = center.to_vec3();
    }

    fn set_opacity(&mut self, opacity: f32) {
        self.opacity = f16::from_f32(opacity);
    }


    fn set_rgb(&mut self, rgb: Vec3A) {
        self.rgb = rgb.to_array().map(|v| f16::from_f32(v));
    }


    fn set_scales(&mut self, scales: Vec3A) {
        self.ln_scales = scales.to_array().map(|v| f16::from_f32(v.ln()));
    }


    fn set_quaternion(&mut self, quaternion: Quat) {
        self.quaternion = quaternion.to_array().map(|v| f16::from_f32(v));
    }

    fn set_label(&mut self, label: u32) {
        self.label = label;
    }

    fn set_instance_label(&mut self, instance_label: u32) {
        self.instance_label = instance_label;
    }

}

#[derive(Debug, Clone, Default)]
pub struct GsplatSH1(pub [[f16; 3]; 3]);

impl GsplatSH1 {
    pub fn new(rgb3: [Vec3A; 3]) -> Self {
        Self(rgb3.map(|rgb| rgb.to_array().map(|v| f16::from_f32(v))))
    }

    pub fn set_from_array(&mut self, rgb3: &[f32]) {
        self.0 = array::from_fn(|k|
            array::from_fn(|d| f16::from_f32(rgb3[k * 3 + d]))
        );
    }

    pub fn to_array(&self) -> [f32; 9] {
        [
            self.0[0][0].to_f32(), self.0[0][1].to_f32(), self.0[0][2].to_f32(),
            self.0[1][0].to_f32(), self.0[1][1].to_f32(), self.0[1][2].to_f32(),
            self.0[2][0].to_f32(), self.0[2][1].to_f32(), self.0[2][2].to_f32(),
        ]
    }
}

#[derive(Debug, Clone, Default)]
pub struct GsplatSH2(pub [[f16; 3]; 5]);

impl GsplatSH2 {
    pub fn new(rgb5: [Vec3A; 5]) -> Self {
        Self(rgb5.map(|rgb| rgb.to_array().map(|v| f16::from_f32(v))))
    }

    pub fn set_from_array(&mut self, rgb5: &[f32]) {
        self.0 = array::from_fn(|k|
            array::from_fn(|d| f16::from_f32(rgb5[k * 3 + d]))
        );
    }

    pub fn to_array(&self) -> [f32; 15] {
        [
            self.0[0][0].to_f32(), self.0[0][1].to_f32(), self.0[0][2].to_f32(),
            self.0[1][0].to_f32(), self.0[1][1].to_f32(), self.0[1][2].to_f32(),
            self.0[2][0].to_f32(), self.0[2][1].to_f32(), self.0[2][2].to_f32(),
            self.0[3][0].to_f32(), self.0[3][1].to_f32(), self.0[3][2].to_f32(),
            self.0[4][0].to_f32(), self.0[4][1].to_f32(), self.0[4][2].to_f32(),
        ]
    }
}

#[derive(Debug, Clone, Default)]

pub struct GsplatSH3(pub [[f16; 3]; 7]);

impl GsplatSH3 {
    pub fn new(rgb7: [Vec3A; 7]) -> Self {
        Self(rgb7.map(|rgb| rgb.to_array().map(|v| f16::from_f32(v))))
    }

    pub fn set_from_array(&mut self, rgb7: &[f32]) {
        self.0 = array::from_fn(|k|
            array::from_fn(|d| f16::from_f32(rgb7[k * 3 + d]))
        );
    }

    pub fn to_array(&self) -> [f32; 21] {
        [
            self.0[0][0].to_f32(), self.0[0][1].to_f32(), self.0[0][2].to_f32(),
            self.0[1][0].to_f32(), self.0[1][1].to_f32(), self.0[1][2].to_f32(),
            self.0[2][0].to_f32(), self.0[2][1].to_f32(), self.0[2][2].to_f32(),
            self.0[3][0].to_f32(), self.0[3][1].to_f32(), self.0[3][2].to_f32(),
            self.0[4][0].to_f32(), self.0[4][1].to_f32(), self.0[4][2].to_f32(),
            self.0[5][0].to_f32(), self.0[5][1].to_f32(), self.0[5][2].to_f32(),
            self.0[6][0].to_f32(), self.0[6][1].to_f32(), self.0[6][2].to_f32(),
        ]
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub enum GaussianKernel {
    #[default]
    K1,
    K2,
}

impl GaussianKernel {
    /// Per-axis second moment of the projected exp(-r^(2k)/2) kernel.
    pub fn moment_factor(self) -> f32 {
        match self {
            Self::K1 => 1.0,
            Self::K2 => 1.0 / (2.0 * std::f32::consts::PI).sqrt(),
        }
    }
}

pub struct GsplatArray {
    pub kernel: GaussianKernel,
    pub max_sh_degree: usize,
    pub splats: Vec<Gsplat>,
    pub children: Vec<SmallVec<[usize; 4]>>,
    pub sh1: Vec<GsplatSH1>,
    pub sh2: Vec<GsplatSH2>,
    pub sh3: Vec<GsplatSH3>,
}

impl TsplatArray for GsplatArray {
    type Splat<'a> = &'a Gsplat;
    type SplatMut<'a> = &'a mut Gsplat;

    fn new_capacity(capacity: usize, max_sh_degree: usize) -> Self {
        assert!(max_sh_degree <= 3, "SH degrees must be between 0 and 3");
        Self {
            kernel: GaussianKernel::K1,
            max_sh_degree,
            splats: Vec::with_capacity(capacity),
            children: Vec::new(), //Vec::with_capacity(capacity),
            sh1: Vec::with_capacity(if max_sh_degree >= 1 { capacity } else { 0 }),
            sh2: Vec::with_capacity(if max_sh_degree >= 2 { capacity } else { 0 }),
            sh3: Vec::with_capacity(if max_sh_degree >= 3 { capacity } else { 0 }),
        }
    }

    fn max_sh_degree(&self) -> usize {
        self.max_sh_degree
    }

    fn clamp_sh_degree(&mut self, max_sh_degree: usize) {
        assert!(max_sh_degree <= 3, "SH degrees must be between 0 and 3");
        let max_sh_degree = max_sh_degree.min(self.max_sh_degree);

        if max_sh_degree < 3 {
            self.sh3.clear();
        }
        if max_sh_degree < 2 {
            self.sh2.clear();
        }
        if max_sh_degree < 1 {
            self.sh1.clear();
        }
        self.max_sh_degree = max_sh_degree;
    }

    fn len(&self) -> usize {
        self.splats.len()
    }

    fn get(&self, index: usize) -> &Gsplat {
        &self.splats[index]
    }

    fn get_mut(&mut self, index: usize) -> &mut Gsplat {
        &mut self.splats[index]
    }

    fn prepare_children(&mut self) {
        self.children.resize_with(self.len(), || SmallVec::new());
    }

    fn has_children(&self) -> bool {
        !self.children.is_empty()
    }

    fn new_merged(&mut self, indices: &[usize], step: f32) -> usize {
        let new_index = self.splats.len();

        let mut weights: SmallVec<[f32; 32]> = indices.iter().map(|&index| {
            let splat = self.get(index);
            splat.area() * splat.opacity()
        }).collect();
        let total_weight = weights.iter().sum::<f32>().max(1.0e-30);
        weights.iter_mut().for_each(|w| *w /= total_weight);

        // for &weight in weights.iter() {
        //     if !weight.is_finite() {
        //         println!("--- Weight is not finite: {}", weight);
        //         println!("Weights: {:?}", weights);
        //         println!("Total weight: {}", total_weight);
        //         println!("0 / total_weight: {}", 0.0 / total_weight);
        //         println!("Areas: {:?}", indices.iter().map(|&index| self.get(index).area()).collect::<Vec<f32>>());
        //         println!("Opacities: {:?}", indices.iter().map(|&index| self.get(index).opacity()).collect::<Vec<f32>>());
        //         for &index in indices.iter() {
        //             let splat = self.get(index);
        //             println!("Splat {}: {:?}", index, splat);
        //         }
        //         panic!("Weight is not finite");
        //     }
        // }

        let mut center = Vec3A::ZERO;
        let mut rgb = Vec3A::ZERO;

        for (i, &index) in indices.iter().enumerate() {
            let splat = self.get(index);
            let weight = weights[i];
            center = splat.center().mul_add(Vec3A::splat(weight), center);
            rgb = splat.rgb().mul_add(Vec3A::splat(weight), rgb);
        }

        let mut total_cov = SymMat3::new_zeros();
        let moment_factor = self.kernel.moment_factor();
        let filter2 = (0.5 * step).powi(2);
        // Average shape matrices, but convert center-spread covariance back
        // into kernel shape units. Equal co-located shapes remain unchanged.

        for (i, &index) in indices.iter().enumerate() {
            let splat = &self.splats[index as usize];
            let weight = weights[i];
            let delta = splat.center() - center;
            let cov = SymMat3::new_scale_quaternion(splat.scales(), splat.quaternion());
            let xx = delta.x * delta.x / moment_factor + cov.xx() + filter2;
            let yy = delta.y * delta.y / moment_factor + cov.yy() + filter2;
            let zz = delta.z * delta.z / moment_factor + cov.zz() + filter2;
            let xy = delta.x * delta.y / moment_factor + cov.xy();
            let xz = delta.x * delta.z / moment_factor + cov.xz();
            let yz = delta.y * delta.z / moment_factor + cov.yz();
            total_cov.add_weighted(&SymMat3::new([xx, yy, zz, xy, xz, yz]), weight);
        }

        // if !total_cov.xx().is_finite() || !total_cov.yy().is_finite() || !total_cov.zz().is_finite() || !total_cov.xy().is_finite() || !total_cov.xz().is_finite() || !total_cov.yz().is_finite() {
        //     println!("--- Total cov is not finite: {:?}", total_cov);
        //     println!("Weights: {:?}", weights);
        //     println!("Areas: {:?}", indices.iter().map(|&index| self.get(index).area()).collect::<Vec<f32>>());
        //     println!("Opacities: {:?}", indices.iter().map(|&index| self.get(index).opacity()).collect::<Vec<f32>>());
        //     for &index in indices.iter() {
        //         let splat = self.get(index);
        //         println!("Splat {}: {:?}", index, splat);
        //         println!("Splat {} cov: {:?}", index, SymMat3::new_scale_quaternion(splat.scales(), splat.quaternion()))
        //     }
        // }
        assert!(total_cov.xx().is_finite() && total_cov.yy().is_finite() && total_cov.zz().is_finite());
        assert!(total_cov.xy().is_finite() && total_cov.xz().is_finite() && total_cov.yz().is_finite());

        let (vals, vecs) = total_cov.positive_eigens();
        let scales = Vec3A::from_array(vals.map(|v| v.max(0.0).sqrt()));
        assert!(scales.x.is_finite() && scales.y.is_finite() && scales.z.is_finite());

        // if scales.x == 0.0 || scales.y == 0.0 || scales.z == 0.0 {
        //     println!("--- Scale is zero: {:?}", scales);
        //     println!("Weights: {:?}", weights);
        //     println!("Areas: {:?}", indices.iter().map(|&index| self.get(index).area()).collect::<Vec<f32>>());
        //     println!("Opacities: {:?}", indices.iter().map(|&index| self.get(index).opacity()).collect::<Vec<f32>>());
        //     for &index in indices.iter() {
        //         let splat = self.get(index);
        //         println!("Splat {}: {:?}", index, splat);
        //         println!("Splat {} cov: {:?}", index, SymMat3::new_scale_quaternion(splat.scales(), splat.quaternion()))
        //     }

        //     println!("Total cov: {:?}", total_cov);
        //     println!("vals: {:?}", vals);
        //     println!("vecs: {:?}", vecs);
        // }
        let scales = scales.max(Vec3A::splat(1.0e-30));

        let basis = Mat3A::from_cols(vecs[0], vecs[1], vecs[2]);
        let quaternion = Quat::from_mat3a(&basis);
        let opacity = total_weight / ellipsoid_area(scales);

        // if opacity <= 0.000001 {
        //     println!("--- Opacity is zero: {}", opacity);
        //     println!("Total weight: {}", total_weight);
        //     println!("Area: {}", ellipsoid_area(scales));
        //     println!("Scales: {:?}", scales);
        //     println!("Quaternion: {:?}", quaternion);
        //     println!("Center: {:?}", center);
        //     println!("RGB: {:?}", rgb);
        //     for &index in indices.iter() {
        //         let splat = self.get(index);
        //         println!("Splat {}: {:?}", index, splat);
        //         println!("Splat {} cov: {:?}", index, SymMat3::new_scale_quaternion(splat.scales(), splat.quaternion()))
        //     }
        //     // panic!("Opacity is zero!");
        // }
        let opacity = opacity.clamp(0.000001, 1000.0);

        let (scales, opacity) = if INFLATE_SCALE && opacity > 1.0 {
            let rescale = opacity.powf(1.0 / 3.0);
            (scales * rescale, 1.0)
        } else {
            (scales, opacity)
        };

        let mut label: u32 = 0;
        let mut instance_label: u32 = 0;
        if let Some(first_index) = indices.get(0).copied(){
            let first_splat = &self.splats[first_index];
            label = (*first_splat).label as u32;
            instance_label = (*first_splat).instance_label as u32;
        }
        
        
        self.splats.push(Gsplat::new(center, opacity, rgb, scales, quaternion, label, instance_label));
        self.children.push(indices.iter().copied().collect());

        if self.max_sh_degree >= 1 {
            let mut total = [Vec3A::ZERO; 3];
            for (i, &index) in indices.iter().enumerate() {
                let weight = weights[i];
                let sh1 = &self.sh1[index];
                total = std::array::from_fn(|i| {
                    let rgb = Vec3A::from_array(sh1.0[i].map(|v| v.to_f32()));
                    rgb.mul_add(Vec3A::splat(weight), total[i])
                })
            }
            self.sh1.push(GsplatSH1::new(total));
        }

        if self.max_sh_degree >= 2 {
            let mut total = [Vec3A::ZERO; 5];
            for (i, &index) in indices.iter().enumerate() {
                let weight = weights[i];
                let sh2 = &self.sh2[index];
                total = std::array::from_fn(|i| {
                    let rgb = Vec3A::from_array(sh2.0[i].map(|v| v.to_f32()));
                    rgb.mul_add(Vec3A::splat(weight), total[i])
                })
            }
            self.sh2.push(GsplatSH2::new(total));
        }

        if self.max_sh_degree >= 3 {
            let mut total = [Vec3A::ZERO; 7];
            for (i, &index) in indices.iter().enumerate() {
                let weight = weights[i];
                let sh3 = &self.sh3[index];
                total = std::array::from_fn(|i| {
                    let rgb = Vec3A::from_array(sh3.0[i].map(|v| v.to_f32()));
                    rgb.mul_add(Vec3A::splat(weight), total[i])
                })
            }
            self.sh3.push(GsplatSH3::new(total));
        }

        new_index
    }

    fn set_children(&mut self, parent: usize, children: &[usize]) {
        self.children[parent] = children.iter().copied().collect();
    }

    fn get_children(&self, parent: usize) -> SmallVec<[usize; 8]> {
        self.children[parent].iter().copied().collect()
    }

    fn get_child_count_start(&self, index: usize) -> (usize, usize) {
        (self.children[index].len(), self.children[index].first().copied().unwrap_or(0) as usize)
    }


    fn clear_children(&mut self) {
        self.children.clear();
    }

    fn get_sh1(&self, index: usize) -> [f32; 9] {
        self.sh1[index].to_array()
    }

    fn get_sh2(&self, index: usize) -> [f32; 15] {
        self.sh2[index].to_array()
    }

    fn get_sh3(&self, index: usize) -> [f32; 21] {
        self.sh3[index].to_array()
    }

    fn similarity(&self, a: usize, b: usize) -> f32 {
        similarity_metric_with_moments(&self.get(a), &self.get(b), self.kernel.moment_factor())
    }

    fn retain<F: (FnMut(&mut Gsplat) -> bool)>(&mut self, mut f: F) {
        let keep: Vec<bool> = self.splats.iter_mut().map(|splat| f(splat)).collect();
        let mut bits = keep.iter();
        self.splats.retain(|_splat| *bits.next().unwrap());
        if !self.children.is_empty() {
            let mut bits = keep.iter();
            self.children.retain(|_children| *bits.next().unwrap());
        }
        if !self.sh1.is_empty() {
            let mut bits = keep.iter();
            self.sh1.retain(|_sh1| *bits.next().unwrap());
        }
        if !self.sh2.is_empty() {
            let mut bits = keep.iter();
            self.sh2.retain(|_sh2| *bits.next().unwrap());
        }
        if !self.sh3.is_empty() {
            let mut bits = keep.iter();
            self.sh3.retain(|_sh3| *bits.next().unwrap());
        }
    }

    fn retain_children<F: (FnMut(&mut Gsplat, &[usize]) -> bool)>(&mut self, mut f: F) {
        let keep: Vec<bool> = self.splats.iter_mut().enumerate()
            .map(|(i, splat)| {
                if i < self.children.len() {
                    f(splat, &self.children[i])
                } else {
                    f(splat, &[])
                }
            })
            .collect();
        let mut bits = keep.iter();
        self.splats.retain(|_splat| *bits.next().unwrap());
        if !self.children.is_empty() {
            let mut bits = keep.iter();
            self.children.retain(|_children| *bits.next().unwrap());
        }
        if !self.sh1.is_empty() {
            let mut bits = keep.iter();
            self.sh1.retain(|_sh1| *bits.next().unwrap());
        }
        if !self.sh2.is_empty() {
            let mut bits = keep.iter();
            self.sh2.retain(|_sh2| *bits.next().unwrap());
        }
        if !self.sh3.is_empty() {
            let mut bits = keep.iter();
            self.sh3.retain(|_sh3| *bits.next().unwrap());
        }
    }

    fn permute(&mut self, index_map: &[usize]) {
        assert_eq!(index_map.len(), self.splats.len());
        let swaps = compute_swaps(index_map);
        #[cfg(feature = "parallel")]
        rayon::scope(|scope| {
            scope.spawn(|_| apply_swaps(&mut self.splats, &swaps));
            if !self.children.is_empty() {
                scope.spawn(|_| apply_swaps(&mut self.children, &swaps));
            }
            if !self.sh1.is_empty() {
                scope.spawn(|_| apply_swaps(&mut self.sh1, &swaps));
            }
            if !self.sh2.is_empty() {
                scope.spawn(|_| apply_swaps(&mut self.sh2, &swaps));
            }
            if !self.sh3.is_empty() {
                scope.spawn(|_| apply_swaps(&mut self.sh3, &swaps));
            }
        });
        #[cfg(not(feature = "parallel"))]
        {
            apply_swaps(&mut self.splats, &swaps);
            if !self.children.is_empty() {
                apply_swaps(&mut self.children, &swaps);
            }
            if !self.sh1.is_empty() {
                apply_swaps(&mut self.sh1, &swaps);
            }
            if !self.sh2.is_empty() {
                apply_swaps(&mut self.sh2, &swaps);
            }
            if !self.sh3.is_empty() {
                apply_swaps(&mut self.sh3, &swaps);
            }
        }
    }

    fn truncate(&mut self, count: usize) {
        self.splats.truncate(count);
        if !self.children.is_empty() {
            self.children.truncate(count);
        }
        if !self.sh1.is_empty() {
            self.sh1.truncate(count);
        }
        if !self.sh2.is_empty() {
            self.sh2.truncate(count);
        }
        if !self.sh3.is_empty() {
            self.sh3.truncate(count);
        }
    }

    fn new_from_index_map(&mut self, index_map: &[usize]) -> Self {
        Self {
            kernel: self.kernel,
            max_sh_degree: self.max_sh_degree,
            splats: index_map.iter().map(|&i| self.splats[i].clone()).collect(),
            children: if !self.children.is_empty() {
                index_map.iter().map(|&i| self.children[i].clone()).collect()
            } else {
                Vec::new()
            },
            sh1: if !self.sh1.is_empty() {
                index_map.iter().map(|&i| self.sh1[i].clone()).collect()
            } else {
                Vec::new()
            },
            sh2: if !self.sh2.is_empty() {
                index_map.iter().map(|&i| self.sh2[i].clone()).collect()
            } else {
                Vec::new()
            },
            sh3: if !self.sh3.is_empty() {
                index_map.iter().map(|&i| self.sh3[i].clone()).collect()
            } else {
                Vec::new()
            },
        }
    }

    fn clone_subset(&self, start: usize, count: usize) -> Self {
        Self {
            kernel: self.kernel,
            max_sh_degree: self.max_sh_degree,
            splats: self.splats[start..start + count].to_vec(),
            children: if self.children.is_empty() { Vec::new() } else { self.children[start..start + count].to_vec() },
            sh1: if self.sh1.is_empty() { Vec::new() } else { self.sh1[start..start + count].to_vec() },
            sh2: if self.sh2.is_empty() { Vec::new() } else { self.sh2[start..start + count].to_vec() },
            sh3: if self.sh3.is_empty() { Vec::new() } else { self.sh3[start..start + count].to_vec() },
        }
    }
}

impl GsplatArray {
    pub fn new() -> Self {
        Self::new_capacity(0, 0)
    }

    pub fn push_splat(
        &mut self,
        splat: Gsplat,
        sh1: Option<GsplatSH1>,
        sh2: Option<GsplatSH2>,
        sh3: Option<GsplatSH3>,
    ) -> usize {
        let index = self.splats.len();
        
        self.splats.push(splat);
        
        if self.max_sh_degree >= 1 {
            assert!(sh1.is_some(), "SH1 must be provided");
            self.sh1.push(sh1.unwrap());
        }

        if self.max_sh_degree >= 2 {
            assert!(sh2.is_some(), "SH2 must be provided");
            self.sh2.push(sh2.unwrap());
        }

        if self.max_sh_degree >= 3 {
            assert!(sh3.is_some(), "SH3 must be provided");
            self.sh3.push(sh3.unwrap());
        }

        index
    }

    pub fn to_packed_array(&self, encoding: &SplatEncoding) -> (usize, Vec<u32>) {
        let (_, _, _, max_splats) = get_splat_tex_size(self.splats.len());
        let mut packed = Vec::new();
        packed.resize(max_splats * 4, 0);

        for i in 0..self.splats.len() {
            let i4 = i * 4;
            let splat = &self.splats[i];
            encode_packed_splat(
                &mut packed[i4..i4 + 4],
                splat.center().to_array(),
                splat.opacity(),
                splat.rgb().to_array(),
                splat.scales().to_array(),
                splat.quaternion().to_array(),
                encoding
            );
        }

        (self.splats.len(), packed)
    }

    pub fn to_packed_sh1(&self, encoding: &SplatEncoding) -> Vec<u32> {
        if self.max_sh_degree < 1 {
            return Vec::new();
        }
        let (_, _, _, max_splats) = get_splat_tex_size(self.splats.len());
        let mut sh1 = Vec::new();
        sh1.resize(max_splats * 2, 0);
        let SplatEncoding { sh1_max, .. } = encoding;

        for i in 0..self.splats.len() {
            let i2 = i * 2;
            let encoded = encode_sh1(&self.sh1[i].to_array(), *sh1_max);
            for w in 0..2 {
                sh1[i2 + w] = encoded[w];
            }
        }
        sh1
    }

    pub fn to_packed_sh2(&self, encoding: &SplatEncoding) -> Vec<u32> {
        if self.max_sh_degree < 2 {
            return Vec::new();
        }
        let (_, _, _, max_splats) = get_splat_tex_size(self.splats.len());
        let mut sh2 = Vec::new();
        sh2.resize(max_splats * 4, 0);
        let SplatEncoding { sh2_max, .. } = encoding;
        
        for i in 0..self.splats.len() {
            let i4 = i * 4;
            let encoded = encode_sh2(&self.sh2[i].to_array(), *sh2_max);
            for w in 0..4 {
                sh2[i4 + w] = encoded[w];
            }
        }
        sh2
    }

    pub fn to_packed_sh3(&self, encoding: &SplatEncoding) -> Vec<u32> {
        if self.max_sh_degree < 3 {
            return Vec::new();
        }
        let (_, _, _, max_splats) = get_splat_tex_size(self.splats.len());
        let mut sh3 = Vec::new();
        sh3.resize(max_splats * 4, 0);
        let SplatEncoding { sh3_max, .. } = encoding;

        for i in 0..self.splats.len() {
            let i4 = i * 4;
            let encoded = encode_sh3(&self.sh3[i].to_array(), *sh3_max);
            for w in 0..4 {
                sh3[i4 + w] = encoded[w];
            }
        }
        sh3
    }
}

pub fn ellipsoid_area(scales: Vec3A) -> f32 {
    const P: f32 = 1.6075;
    let numerator = (scales.x * scales.y).powf(P) + (scales.x * scales.z).powf(P) + (scales.y * scales.z).powf(P);
    4.0 * std::f32::consts::PI * (numerator / 3.0).powf(1.0 / P)
}

impl SplatReceiver for GsplatArray {
    fn init_splats(&mut self, init: &SplatInit) -> anyhow::Result<()> {
        self.max_sh_degree = init.max_sh_degree;
        self.splats.resize_with(init.num_splats, Default::default);

        self.sh1.clear();
        if self.max_sh_degree >= 1 {
            self.sh1.resize_with(init.num_splats, Default::default);
        }

        self.sh2.clear();
        if self.max_sh_degree >= 2 {
            self.sh2.resize_with(init.num_splats, Default::default);
        }

        self.sh3.clear();
        if self.max_sh_degree >= 3 {
            self.sh3.resize_with(init.num_splats, Default::default);
        }

        self.children.clear();
        if init.lod_tree {
            self.children.resize_with(init.num_splats, Default::default);
        }

        Ok(())
    }

    fn finish(&mut self) -> anyhow::Result<()> {
        Ok(())
    }

    fn set_encoding(&mut self, _encoding: &SetSplatEncoding) -> anyhow::Result<()> {
        Ok(())
    }

    fn set_batch(&mut self, base: usize, count: usize, batch: &SplatProps) {
        // Validate that we have enough data in the batch arrays
        assert!(batch.center.len() >= count * 3, "center array too small: {} < {}", batch.center.len(), count * 3);
        assert!(batch.opacity.len() >= count, "opacity array too small: {} < {}", batch.opacity.len(), count);
        assert!(batch.rgb.len() >= count * 3, "rgb array too small: {} < {}", batch.rgb.len(), count * 3);
        assert!(batch.scale.len() >= count * 3, "scale array too small: {} < {}", batch.scale.len(), count * 3);
        assert!(batch.quat.len() >= count * 4, "quat array too small: {} < {}", batch.quat.len(), count * 4);
        assert!(base + count <= self.splats.len(), "base + count out of bounds: {} + {} > {}", base, count, self.splats.len());

        for i in 0..count {
            let [i3, i4] = [i * 3, i * 4];
            let mut splat = self.get_mut(base + i);
            splat.set_center(Vec3A::from_slice(&batch.center[i3..i3 + 3]));
            splat.set_opacity(batch.opacity[i]);
            splat.set_rgb(Vec3A::from_slice(&batch.rgb[i3..i3 + 3]));
            splat.set_scales(Vec3A::from_slice(&batch.scale[i3..i3 + 3]));
            splat.set_quaternion(Quat::from_slice(&batch.quat[i4..i4 + 4]));
            splat.set_label(batch.labels[i]);
            splat.set_instance_label(batch.instances[i]);
        }

        self.set_sh(base, count, batch.sh1, batch.sh2, batch.sh3);

        // if !batch.labels.is_empty() {
        //     for i in 0..count {
        //         let i4 = i * 4;
        //         self.labels[base + i] = [ 
        //             batch.labels[i4] as f32, 
        //             batch.labels[i4 + 1] as f32, 
        //             0.0, 
        //             255.0
        //         ];
        //     }
        // }

        if !batch.child_count.is_empty() && !batch.child_start.is_empty() {
            self.set_child_start(base, count, batch.child_start);
            self.set_child_count(base, count, batch.child_count);
        }
    }

    fn set_center(&mut self, base: usize, count: usize, center: &[f32]) {
        for i in 0..count {
            let i3 = i * 3;
            self.get_mut(base + i).set_center(Vec3A::from_slice(&center[i3..i3 + 3]));
        }
    }

    fn set_opacity(&mut self, base: usize, count: usize, opacity: &[f32]) {
        for i in 0..count {
            self.get_mut(base + i).set_opacity(opacity[i]);
        }
    }

    fn set_rgb(&mut self, base: usize, count: usize, rgb: &[f32]) {
        for i in 0..count {
            let i3 = i * 3;
            self.get_mut(base + i).set_rgb(Vec3A::from_slice(&rgb[i3..i3 + 3]));
        }
    }

    fn set_rgba(&mut self, base: usize, count: usize, rgba: &[f32]) {
        for i in 0..count {
            let i4 = 4 * i;
            let mut splat = self.get_mut(base + i);
            splat.set_rgb(Vec3A::from_slice(&rgba[i4..i4 + 3]));
            splat.set_opacity(rgba[i4 + 3]);
        }
    }

    fn set_scale(&mut self, base: usize, count: usize, scale: &[f32]) {
        for i in 0..count {
            let i3 = i * 3;
            self.get_mut(base + i).set_scales(Vec3A::from_slice(&scale[i3..i3 + 3]));
        }
    }

    fn set_quat(&mut self, base: usize, count: usize, quat: &[f32]) {
        for i in 0..count {
            let i4 = 4 * i;
            self.get_mut(base + i).set_quaternion(Quat::from_slice(&quat[i4..i4 + 4]));
        }
    }

    fn set_sh(&mut self, base: usize, count: usize, sh1: &[f32], sh2: &[f32], sh3: &[f32]) {
        if !sh1.is_empty() {
            self.set_sh1(base, count, sh1);
        }
        if !sh2.is_empty() {
            self.set_sh2(base, count, sh2);
        }
        if !sh3.is_empty() {
            self.set_sh3(base, count, sh3);
        }
    }

    fn set_label(&mut self, base: usize, count: usize, labels: &[u32]) {
        for i in 0..count {
            self.splats[base + i].label = labels[i];
        }
    }

    fn set_instance_label(&mut self, base: usize, count: usize, instances: &[u32]) {
        for i in 0..count {
            self.splats[base + i].instance_label = instances[i];
        }
    }

    fn set_sh1(&mut self, base: usize, count: usize, sh1: &[f32]) {
        if self.max_sh_degree >= 1 {
            for i in 0..count {
                let i9 = i * 9;
                self.sh1[base + i].set_from_array(&sh1[i9..i9 + 9]);
            }
        }
    }

    fn set_sh2(&mut self, base: usize, count: usize, sh2: &[f32]) {
        if self.max_sh_degree >= 2 {
            for i in 0..count {
                let i15 = i * 15;
                self.sh2[base + i].set_from_array(&sh2[i15..i15 + 15]);
            }
        }
    }

    fn set_sh3(&mut self, base: usize, count: usize, sh3: &[f32]) {
        if self.max_sh_degree >= 3 {
            for i in 0..count {
                let i21 = i * 21;
                self.sh3[base + i].set_from_array(&sh3[i21..i21 + 21]);
            }
        }
    }

    fn set_child_count(&mut self, base: usize, count: usize, child_count: &[u16]) {
        for i in 0..count {
            let mut child_index = *self.children[base + i].get(0).unwrap_or(&0);
            self.children[base + i].clear();
            self.children[base + i].resize_with(child_count[i] as usize, || {
                let child = child_index;
                child_index += 1;
                child
            });
        }
    }

    fn set_child_start(&mut self, base: usize, count: usize, child_start: &[usize]) {
        for i in 0..count {
            let mut child_index = child_start[i];
            if child_index == 0 {
                self.children[base + i].clear();
            } else {
                let count = self.children[base + i].len().max(1);
                self.children[base + i].clear();
                self.children[base + i].resize_with(count, || {
                    let child = child_index;
                    child_index += 1;
                    child
                });
            }
        }
    }
}

impl SplatGetter for GsplatArray {
    fn num_splats(&self) -> usize { self.len() }
    fn max_sh_degree(&self) -> usize { self.max_sh_degree }
    fn flag_antialias(&self) -> bool { true }
    fn has_lod_tree(&self) -> bool { !self.children.is_empty() }
    fn get_encoding(&mut self) -> Option<SplatEncoding> { None }

    fn get_center(&mut self, base: usize, count: usize, out: &mut [f32]) {
        for i in 0..count {
            let c = self.splats[base + i].center.to_array();
            out[i * 3 + 0] = c[0];
            out[i * 3 + 1] = c[1];
            out[i * 3 + 2] = c[2];
        }
    }

    fn get_opacity(&mut self, base: usize, count: usize, out: &mut [f32]) {
        for i in 0..count {
            out[i] = self.get(base + i).opacity();
        }
    }

    fn get_rgb(&mut self, base: usize, count: usize, out: &mut [f32]) {
        for i in 0..count {
            let r = self.get(base + i).rgb().to_array();
            out[i * 3 + 0] = r[0];
            out[i * 3 + 1] = r[1];
            out[i * 3 + 2] = r[2];
        }
    }

    fn get_scale(&mut self, base: usize, count: usize, out: &mut [f32]) {
        for i in 0..count {
            let s = self.get(base + i).scales().to_array();
            out[i * 3 + 0] = s[0];
            out[i * 3 + 1] = s[1];
            out[i * 3 + 2] = s[2];
        }
    }

    fn get_quat(&mut self, base: usize, count: usize, out: &mut [f32]) {
        for i in 0..count {
            let q = self.get(base + i).quaternion().to_array();
            out[i * 4 + 0] = q[0];
            out[i * 4 + 1] = q[1];
            out[i * 4 + 2] = q[2];
            out[i * 4 + 3] = q[3];
        }
    }

        
    fn get_label(&mut self, base: usize, count: usize, out: &mut [u32]) {
        for i in 0..count {
            out[i] = self.get(base + i).label() as u32;
        }
    }

    fn get_instance_label(&mut self, base: usize, count: usize, out: &mut [u32]) {
        for i in 0..count {
            out[i] = self.get(base + i).instance_label() as u32;
        }
    }

    fn get_sh1(&mut self, base: usize, count: usize, out: &mut [f32]) {
        if self.max_sh_degree >= 1 {
            for i in 0..count { out[i * 9..i * 9 + 9].copy_from_slice(&self.sh1[base + i].to_array()); }
        }
    }

    fn get_sh2(&mut self, base: usize, count: usize, out: &mut [f32]) {
        if self.max_sh_degree >= 2 {
            for i in 0..count { out[i * 15..i * 15 + 15].copy_from_slice(&self.sh2[base + i].to_array()); }
        }
    }

    fn get_sh3(&mut self, base: usize, count: usize, out: &mut [f32]) {
        if self.max_sh_degree >= 3 {
            for i in 0..count { out[i * 21..i * 21 + 21].copy_from_slice(&self.sh3[base + i].to_array()); }
        }
    }

    fn get_child_count(&mut self, base: usize, count: usize, out: &mut [u16]) {
        for i in 0..count {
            out[i] = self.children[base + i].len() as u16;
        }
    }

    fn get_child_start(&mut self, base: usize, count: usize, out: &mut [usize]) {
        for i in 0..count {
            let children = &self.children[base + i];
            out[i] = children.first().copied().unwrap_or(0) as usize;
        }
    }
}
