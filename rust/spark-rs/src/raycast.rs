use spark_lib::{
    decoder::SplatEncoding,
    splat_encode::{decode_ext_splat_center, decode_ext_splat_opacity, decode_ext_splat_quat, decode_ext_splat_scale, decode_packed_splat_center, decode_packed_splat_opacity, decode_packed_splat_quat, decode_packed_splat_scale},
};
use crate::surface_depth::{ALPHA_THRESHOLD, MAX_ALPHA, PROFILE_REACH, RayProfile};

pub fn raycast_packed_ellipsoids(
    buffer: &[u32], profiles: &mut Vec<f32>,
    origin: [f32; 3], dir: [f32; 3], min_opacity: f32,
    encoding: &SplatEncoding,
) {
    for packed in buffer.chunks(4) {
        let opacity = decode_packed_splat_opacity(packed, encoding);
        if opacity == 0.0 {
            continue;
        }
        let center = decode_packed_splat_center(packed);
        let scale = decode_packed_splat_scale(packed, encoding);
        let quat = decode_packed_splat_quat(packed);
        if let Some(profile) = raycast_gaussian(origin, dir, opacity, center, scale, quat, min_opacity) {
            profiles.extend_from_slice(&[profile.depth, profile.alpha, profile.sigma]);
        }
    }
}

pub fn raycast_ext_ellipsoids(
    buffer: &[u32], buffer2: &[u32], profiles: &mut Vec<f32>,
    origin: [f32; 3], dir: [f32; 3], min_opacity: f32,
) {
    assert_eq!(buffer.len(), buffer2.len());
    for (ext_a, ext_b) in buffer.chunks(4).zip(buffer2.chunks(4)) {
        let opacity = decode_ext_splat_opacity(ext_a);
        if opacity == 0.0 {
            continue;
        }
        let center = decode_ext_splat_center(ext_a);
        let scale = decode_ext_splat_scale(ext_b);
        let quat = decode_ext_splat_quat(ext_b);
        if let Some(profile) = raycast_gaussian(origin, dir, opacity, center, scale, quat, min_opacity) {
            profiles.extend_from_slice(&[profile.depth, profile.alpha, profile.sigma]);
        }
    }
}

fn raycast_gaussian(
    origin: [f32; 3], dir: [f32; 3],
    opacity: f32, center: [f32; 3], scale: [f32; 3], quat: [f32; 4],
    min_opacity: f32,
) -> Option<RayProfile> {
    if scale == [0.0; 3] {
        return None;
    }
    let origin = vec3_sub(origin, center);
    let inv_quat = [-quat[0], -quat[1], -quat[2], quat[3]];
    let local_origin = quat_vec(inv_quat, origin);
    let local_dir = quat_vec(inv_quat, dir);
    let mut scaled_origin = [0.0; 3];
    let mut scaled_dir = [0.0; 3];
    let mut plane_distance = None;
    for axis in 0..3 {
        if scale[axis] != 0.0 {
            scaled_origin[axis] = local_origin[axis] / scale[axis];
            scaled_dir[axis] = local_dir[axis] / scale[axis];
        } else if local_dir[axis] == 0.0 {
            if local_origin[axis] != 0.0 {
                return None;
            }
        } else {
            // A zero scale constrains the ray to this exact plane.
            let distance = -local_origin[axis] / local_dir[axis];
            if plane_distance.is_some_and(|previous| previous != distance) {
                return None;
            }
            plane_distance = Some(distance);
        }
    }

    let precision = vec3_dot(scaled_dir, scaled_dir);
    let distance = plane_distance.unwrap_or_else(|| -vec3_dot(scaled_origin, scaled_dir) / precision);
    let sigma = if plane_distance.is_some() { 0.0 } else { precision.sqrt().recip() };
    if f64::from(distance) + PROFILE_REACH * f64::from(sigma) < 0.0 {
        return None;
    }
    let closest = std::array::from_fn(|axis| scaled_origin[axis] + distance * scaled_dir[axis]);
    let radius_squared = vec3_dot(closest, closest);
    let alpha = if opacity > 1.0 { (4.0 * opacity - 3.0).min(5.0) } else { opacity };
    let support_squared = 8.0_f32.sqrt() + 0.7 * (alpha - 1.0).max(0.0);
    if radius_squared > support_squared {
        return None;
    }
    let kernel = (-0.5 * radius_squared * radius_squared).exp();
    let peak_alpha = (if alpha <= 1.0 {
        alpha * kernel
    } else {
        let power = ((alpha * alpha - 1.0) / std::f32::consts::E).exp();
        -(power * (-kernel).ln_1p()).exp_m1()
    }).min(MAX_ALPHA);
    (peak_alpha > ALPHA_THRESHOLD && peak_alpha >= min_opacity).then_some(RayProfile {
        depth: distance, alpha: peak_alpha, sigma,
    })
}

fn vec3_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn vec3_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn vec3_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn quat_vec(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let q_vec = [q[0], q[1], q[2]];
    let uv = vec3_cross(q_vec, v);
    let uuv = vec3_cross(q_vec, uv);
    [
        v[0] + 2.0 * (q[3] * uv[0] + uuv[0]),
        v[1] + 2.0 * (q[3] * uv[1] + uuv[1]),
        v[2] + 2.0 * (q[3] * uv[2] + uuv[2]),
    ]
}

#[cfg(test)]
#[path = "raycast_tests.rs"]
mod tests;
