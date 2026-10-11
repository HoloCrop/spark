use super::*;
use crate::raycast_blocks::RaycastBlockIndex;
use spark_lib::splat_encode::{encode_ext_splat, encode_packed_splat};

#[test]
fn bounds_preserve_all_profiles_and_ids_from_the_unfiltered_narrow_phase() {
    let encoding = SplatEncoding { lod_opacity: true, ..Default::default() };
    let origin = [3.0, -5.0, 4.0];
    let mut hits = 0;
    let mut rejected = 0;
    for matrix in [
        DMat3::IDENTITY,
        DMat3::from_diagonal(DVec3::new(0.3, 2.0, 1.5)),
        DMat3::from_cols(DVec3::new(1.0, 0.2, 0.1), DVec3::new(-0.4, 1.0, 0.3), DVec3::new(0.1, -0.2, 1.0)),
    ] {
        let projection = RaycastProjection { camera_from_mesh: matrix, near: 0.01, far: 100.0, image_from_camera: [1.0, 1.0, 0.0, 0.0] };
        let mut packed = vec![0; 128 * 4];
        let mut extended = vec![0; 128 * 4];
        let mut geometry = vec![0; 128 * 4];
        for index in 0..128 {
            let center = matrix.inverse() * DVec3::new((index % 16) as f64 * 0.15 - 1.2, (index / 16) as f64 * 0.15 - 0.6, 3.0 + (index % 3) as f64);
            let center = (center + DVec3::from_array(origin.map(f64::from))).as_vec3().to_array();
            let scale = [[0.05, 0.02, 0.01], [0.5, 0.001, 0.01], [0.2, 0.1, 0.0], [0.15; 3]][index % 4];
            let quat = DQuat::from_rotation_y(index as f64 * 0.37).as_quat().to_array();
            let opacity = [0.0, 0.2, 0.8, 2.0][index % 4];
            encode_packed_splat(&mut packed[index*4..index*4+4], center, opacity, [1.0;3], scale, quat, &encoding);
            encode_ext_splat(&mut extended[index*4..index*4+4], &mut geometry[index*4..index*4+4], center, opacity, [1.0;3], scale, quat);
        }
        let mut packed_index = RaycastBlockIndex::new(128);
        let mut ext_index = RaycastBlockIndex::new(128);
        packed_index.update_packed(0, &packed, encoding.ln_scale_min, encoding.ln_scale_max);
        ext_index.update_ext(0, &extended, &geometry);
        let displayed: Vec<u32> = (0..128).collect();
        for x in -15..=15 {
            for y in -10..=10 {
                let direction = (matrix.inverse() * DVec3::new(x as f64 * 0.025, y as f64 * 0.025, 1.0)).normalize().as_vec3().to_array();
                for ext in [false, true] {
                    let mut expected = Vec::new();
                    let mut expected_indices = Vec::new();
                    for index in 0..128 {
                        let range = index*4..index*4+4;
                        let (center, scale, quat, opacity) = if ext {
                            (decode_ext_splat_center(&extended[range.clone()]), decode_ext_splat_scale(&geometry[range.clone()]), decode_ext_splat_quat(&geometry[range.clone()]), decode_ext_splat_opacity(&extended[range]))
                        } else {
                            (decode_packed_splat_center(&packed[range.clone()]), decode_packed_splat_scale(&packed[range.clone()], &encoding), decode_packed_splat_quat(&packed[range.clone()]), decode_packed_splat_opacity(&packed[range], &encoding))
                        };
                        if !projection.intersects_bounds(origin, direction, center, scale) { rejected += 1; }
                        if let Some(profile) = raycast_gaussian(origin, direction, opacity, center, scale, quat, 0.0, Some(&projection)) {
                            expected.extend_from_slice(&[profile.depth, profile.alpha, profile.sigma]);
                            expected_indices.push(index as u32);
                        }
                    }
                    let mut actual = Vec::new();
                    let mut indices = Vec::new();
                    if ext {
                        raycast_ext_ellipsoids(&extended, &geometry, &mut actual, &mut indices, origin, direction, 0.0, Some(&projection));
                    } else {
                        raycast_packed_ellipsoids(&packed, &mut actual, &mut indices, origin, direction, 0.0, &encoding, Some(&projection));
                    }
                    assert_eq!(actual, expected);
                    assert_eq!(indices, expected_indices);
                    let candidates = if ext { &mut ext_index } else { &mut packed_index }
                        .candidates(&displayed, &projection, origin, direction);
                    assert!(expected_indices.iter().all(|index| candidates.contains(index)));
                    hits += indices.len();
                }
            }
        }
    }
    assert!(hits > 1000);
    assert!(rejected > 100000);
}
