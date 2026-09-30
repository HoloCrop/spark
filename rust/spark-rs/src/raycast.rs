use spark_lib::{
    decoder::SplatEncoding,
    splat_encode::{decode_ext_splat_center, decode_ext_splat_opacity, decode_ext_splat_quat, decode_ext_splat_scale, decode_packed_splat_center, decode_packed_splat_opacity, decode_packed_splat_quat, decode_packed_splat_scale},
};

pub fn raycast_packed_ellipsoids(
    buffer: &[u32], distances: &mut Vec<f32>, 
    origin: [f32; 3], dir: [f32; 3], min_opacity: f32, near: f32, far: f32,
    encoding: &SplatEncoding,
) {
    for packed in buffer.chunks(4) {
        let opacity = decode_packed_splat_opacity(packed, encoding);
        if opacity < min_opacity {
            continue;
        }
    
        let center = decode_packed_splat_center(packed);
        let scale = decode_packed_splat_scale(packed, encoding);
        let quat = decode_packed_splat_quat(packed);
        if let Some((t, alpha)) = raycast_ellipsoid(origin, dir, opacity, center, scale, quat) {
            if t >= near && t <= far {
                distances.push(t);
            }
        }
    }
}

pub fn raycast_ext_ellipsoids(
    buffer: &[u32], buffer2: &[u32], distances: &mut Vec<f32>, 
    origin: [f32; 3], dir: [f32; 3], min_opacity: f32, near: f32, far: f32,
) {
    assert_eq!(buffer.len(), buffer2.len());
    for (ext_a, ext_b) in buffer.chunks(4).zip(buffer2.chunks(4)) {
        let opacity = decode_ext_splat_opacity(ext_a);
        if opacity < min_opacity {
            continue;
        }
    
        let center = decode_ext_splat_center(ext_a);
        let scale = decode_ext_splat_scale(ext_b);
        let quat = decode_ext_splat_quat(ext_b);
        if let Some((t, alpha)) = raycast_ellipsoid(origin, dir, opacity, center, scale, quat) {
            if t >= near && t <= far {
                distances.extend_from_slice(&[t, alpha]);
            }
        }
    }
}

fn raycast_ellipsoid(
    origin: [f32; 3], dir: [f32; 3],
    opacity: f32, center: [f32; 3], scale: [f32; 3], quat: [f32; 4],
) -> Option<(f32, f32)> {
    let origin = vec3_sub(origin, center);
    let inv_quat = [-quat[0], -quat[1], -quat[2], quat[3]];

    // u and v: origin and direction in rotated, inverse-scaled splat space.
    let u = quat_vec(inv_quat, origin);
    let v = quat_vec(inv_quat, dir);
    let u = [u[0] / scale[0], u[1] / scale[1], u[2] / scale[2]];
    let v = [v[0] / scale[0], v[1] / scale[1], v[2] / scale[2]];

    let vv = vec3_dot(v, v);
    if vv <= 0.0 || !vv.is_finite() {
        return None;
    }
    
    let t = -vec3_dot(u, v) / vv;
    let closest = [
        u[0] + t * v[0],
        u[1] + t * v[1],
        u[2] + t * v[2],
    ];
    let q_perp = vec3_dot(closest, closest);
    let alpha = apply_kernel_alpha(opacity, q_perp, 2.0, 1.0);
    if alpha == 0.0 {
        return None;
    }

    let qn = (quat[0]*quat[0] + quat[1]*quat[1] + quat[2]*quat[2] + quat[3]*quat[3]).sqrt();
    web_sys::console::log_1(&format!(
        "scale={:?} qnorm={} q_perp={} t={} opacity={}",
        scale, qn, q_perp, t, opacity
    ).into());

    Some((t, alpha))
}


fn sqr(x: f32) -> f32 {
    x * x
}


fn vec3_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn vec3_mul(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2]]
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


#[inline]
fn mix(x: f32, y: f32, a: f32) -> f32 {
    x * (1.0 - a) + y * a
}

#[inline]
fn gaussian_kernel_power(z2: f32, k: f32) -> f32 {
    if k == 1.0 {
        z2
    } else if k == 2.0 {
        z2 * z2
    } else {
        z2.max(0.0).powf(k)
    }
}

#[inline]
fn gaussian_kernel_scale(max_std_dev: f32, k: f32) -> f32 {
    if k == 1.0 {
        max_std_dev
    } else if k == 2.0 {
        max_std_dev.sqrt()
    } else {
        max_std_dev.max(0.0).powf(1.0 / k)
    }
}

#[inline]
fn gaussian_kernel(z2: f32, k: f32) -> f32 {
    (-0.5 * gaussian_kernel_power(z2, k)).exp()
}

/// Applies the kernel falloff to an alpha value.
#[inline]
fn apply_kernel_alpha(a: f32, z2: f32, gaussian_k: f32, falloff: f32) -> f32 {
    let kernel = gaussian_kernel(z2, gaussian_k);
    if a <= 1.0 {
        mix(a, a * kernel, falloff)
    } else {
        let p = ((a * a - 1.0) / std::f32::consts::E).exp();
        let alpha = 1.0 - (1.0 - kernel).powf(p);
        mix(1.0, alpha, falloff)
    }
}