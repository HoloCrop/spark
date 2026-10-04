use crate::surface_depth::ALPHA_THRESHOLD;
use super::{RaycastProjection, raycast_ext_ellipsoids, raycast_gaussian, raycast_packed_ellipsoids};
use glam::{DMat3, DVec3};
use spark_lib::{decoder::SplatEncoding, splat_encode::{encode_ext_splat, encode_packed_splat}};

const IDENTITY: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

fn hit(origin: [f32; 3], direction: [f32; 3], scale: [f32; 3], opacity: f32, minimum: f32) -> Option<f32> {
    raycast_gaussian(origin, direction, opacity, [0.0; 3], scale, IDENTITY, minimum, None).map(|profile| profile.depth)
}

#[test]
fn quartic_edges_hit_at_density_peak_and_stop_at_support() {
    for offset in [1.2, 1.6] {
        assert_eq!(hit([offset, 0.0, -10.0], [0.0, 0.0, 1.0], [1.0; 3], 1.0, 0.0), Some(10.0));
    }
    assert_eq!(hit([1.7, 0.0, -10.0], [0.0, 0.0, 1.0], [1.0; 3], 1.0, 0.0), None);
    assert_eq!(hit([1.6, 0.0, -10.0], [0.0, 0.0, 1.0], [1.0; 3], 1.0, 0.03), Some(10.0));
    assert_eq!(hit([1.6, 0.0, -10.0], [0.0, 0.0, 1.0], [1.0; 3], 1.0, 0.04), None);
}

#[test]
fn positive_needle_parallel_to_thin_axis_is_not_flattened() {
    assert_eq!(hit([-10.0, 0.0, 0.002], [1.0, 0.0, 0.0], [1.0, 1.0, 0.005], 1.0, 0.0), Some(10.0));
}

#[test]
fn rotated_anisotropic_peak_keeps_world_ray_parameter() {
    let angle = std::f32::consts::FRAC_PI_8;
    let quaternion = [0.0, angle.sin(), 0.0, angle.cos()];
    let profile = raycast_gaussian(
        [0.0; 3], [0.0, 0.0, 1.0], 1.0, [1.0, 0.0, 5.0], [1.0, 0.1, 0.1], quaternion, 0.0, None,
    ).unwrap();
    assert!((profile.depth - 5.980198).abs() < 1e-5);
    assert!((profile.sigma - 1.0 / 50.5_f32.sqrt()).abs() < 1e-6);
    let rescaled = raycast_gaussian(
        [0.0; 3], [0.0, 0.0, 0.5], 1.0, [1.0, 0.0, 5.0], [1.0, 0.1, 0.1], quaternion, 0.0, None,
    ).unwrap();
    assert!((rescaled.depth - profile.depth * 2.0).abs() < 1e-5);
    assert!((rescaled.sigma - profile.sigma * 2.0).abs() < 1e-6);
}

#[test]
fn zero_scale_planes_handle_crossing_in_plane_and_parallel_rays() {
    assert_eq!(hit([1.2, 0.0, -10.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0], 1.0, 0.0), Some(10.0));
    assert_eq!(hit([-10.0, 0.2, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], 1.0, 0.0), Some(10.0));
    assert_eq!(hit([-10.0, 0.2, 0.001], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], 1.0, 0.0), None);
    assert_eq!(hit([-1.0, 0.2, -1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0], 1.0, 0.0), Some(1.0));
    assert_eq!(hit([-1.0, 0.2, -2.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0], 1.0, 0.0), None);
}

#[test]
fn lod_alpha_remapping_controls_support_and_peak_threshold() {
    assert_eq!(hit([1.9, 0.0, -10.0], [0.0, 0.0, 1.0], [1.0; 3], 1.5, 0.01), Some(10.0));
    assert_eq!(hit([1.9, 0.0, -10.0], [0.0, 0.0, 1.0], [1.0; 3], 1.5, 0.05), None);
    assert_eq!(hit([2.1, 0.0, -10.0], [0.0, 0.0, 1.0], [1.0; 3], 1.5, 0.0), None);
}

#[test]
fn profile_alpha_floor_excludes_faint_tails_and_caps_opaque_peaks() {
    assert_eq!(hit([2.35, 0.0, 10.0], [0.0, 0.0, -1.0], [1.0; 3], 2.0, 0.0), None);
    assert_eq!(hit([0.0, 0.0, -10.0], [0.0, 0.0, 1.0], [1.0; 3], ALPHA_THRESHOLD, 0.0), None);
    assert_eq!(hit([0.0, 0.0, -10.0], [0.0, 0.0, 1.0], [1.0; 3], 0.004, 0.0), Some(10.0));
    let profile = raycast_gaussian(
        [0.0, 0.0, -10.0], [0.0, 0.0, 1.0], 2.0, [0.0; 3], [1.0; 3], IDENTITY, 0.0, None,
    ).unwrap();
    assert_eq!(profile.alpha, 0.99);
}

#[test]
fn contributing_lod_edge_above_threshold_survives_cancellation() {
    assert_eq!(hit([2.3137, 0.0, 10.0], [0.0, 0.0, -1.0], [1.0; 3], 2.0, 0.004075), Some(10.0));
}

#[test]
fn faint_edge_layers_jointly_reach_the_surface_threshold() {
    let mut splat = [0; 4];
    let mut scales = [0; 4];
    encode_ext_splat(&mut splat, &mut scales, [0.0, 0.0, 5.0], 0.1, [1.0; 3], [1.0, 1.0, 0.1], IDENTITY);
    let mut profiles = Vec::new();
    raycast_ext_ellipsoids(
        &splat.repeat(250), &scales.repeat(250), &mut profiles,
        [1.6, 0.0, 0.0], [0.0, 0.0, 1.0], 0.0, None,
    );
    let depth = crate::sample_surface_depth(&profiles, 0.0, 10.0, 0.5).unwrap();
    assert!((depth - 5.1063).abs() < 0.001);
    assert_eq!(crate::sample_surface_depth(&profiles[..150 * 3], 0.0, 10.0, 0.5), None);
}

#[test]
fn perspective_tilted_edges_use_projected_coverage_and_triton_depth() {
    let projection = RaycastProjection { camera_from_mesh: DMat3::IDENTITY, near: 0.01, far: 100.0, image_from_camera: [1.0, 1.0, 0.0, 0.0] };
    let angle = std::f32::consts::FRAC_PI_8;
    let quaternion = [0.0, angle.sin(), 0.0, angle.cos()];
    for (sign, expected) in [(-1.0, 5.6834255), (1.0, 4.4003675)] {
        for radius in [0.9, 1.1] {
            let direction = DVec3::new(sign * radius / 2.0_f64.sqrt() / 5.0, 0.0, 1.0).normalize().as_vec3().to_array();
            let profile = raycast_gaussian(
                [0.0; 3], direction, 0.8, [0.0, 0.0, 5.0], [1.0, 0.1, 0.001], quaternion, 0.0, Some(&projection),
            ).unwrap();
            let depth = crate::sample_surface_depth(&[profile.depth, profile.alpha, profile.sigma], 0.0, 10.0, 0.5);
            if radius == 0.9 {
                assert!((depth.unwrap() - expected).abs() < 3e-5);
            } else {
                assert_eq!(depth, None);
            }
        }
    }
}

#[test]
fn perspective_nonuniform_mesh_transform_keeps_world_depth_and_sigma() {
    let basis = DMat3::from_diagonal(DVec3::new(2.0, 1.5, 0.75));
    let projection = RaycastProjection { camera_from_mesh: basis, near: 0.01, far: 100.0, image_from_camera: [1.0, 1.0, 0.0, 0.0] };
    let direction = basis.inverse() * DVec3::new(0.65, 0.01, 1.0).normalize();
    let angle = std::f32::consts::FRAC_PI_8;
    let profile = raycast_gaussian(
        [0.0; 3], direction.as_vec3().to_array(), 0.8, [1.0, 0.0, 5.0], [1.0, 0.1, 0.001],
        [0.0, angle.sin(), 0.0, angle.cos()], 0.0, Some(&projection),
    ).unwrap();
    let depth = crate::sample_surface_depth(&[profile.depth, profile.alpha, profile.sigma], 0.0, 10.0, 0.5).unwrap();
    assert!((depth - 4.5257233).abs() < 3e-5);
    assert!((profile.sigma - 0.0010542323).abs() < 1e-7);
}

#[test]
fn camera_clipped_sources_do_not_occlude_visible_surfaces() {
    let projection = RaycastProjection { camera_from_mesh: DMat3::IDENTITY, near: 0.01, far: 100.0, image_from_camera: [1.0, 1.0, 0.0, 0.0] };
    let mut splats = [0; 12];
    let mut scales = [0; 12];
    for (index, depth, width) in [(0, 0.0001, 0.00001), (1, 5.0, 0.25), (2, 100.1, 10.0)] {
        encode_ext_splat(
            &mut splats[index * 4..index * 4 + 4], &mut scales[index * 4..index * 4 + 4],
            [0.0, 0.0, depth], 0.8, [1.0; 3], [1.0, 1.0, width], IDENTITY,
        );
    }
    let mut profiles = Vec::new();
    raycast_ext_ellipsoids(
        &splats, &scales, &mut profiles, [0.0; 3], [0.0, 0.0, 1.0], 0.0, Some(&projection),
    );
    let depth = crate::sample_surface_depth(&profiles, 0.01, 100.0, 0.5).unwrap();
    assert!((depth - 4.850357).abs() < 3e-5);
    profiles.clear();
    raycast_ext_ellipsoids(
        &splats[..4], &scales[..4], &mut profiles, [0.0; 3], [0.0, 0.0, 1.0], 0.0, Some(&projection),
    );
    raycast_ext_ellipsoids(
        &splats[8..], &scales[8..], &mut profiles, [0.0; 3], [0.0, 0.0, 1.0], 0.0, Some(&projection),
    );
    assert_eq!(crate::sample_surface_depth(&profiles, 0.01, 100.0, 0.5), None);
}

#[test]
fn real_rad_camera_clipped_sources_cannot_create_viewer_ghost_points() {
    let projection = RaycastProjection {
        camera_from_mesh: DMat3::from_cols_array(&[0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        near: 0.01, far: 100.0,
        image_from_camera: [3.0_f64.sqrt(), 3.0_f64.sqrt(), 0.0, 0.0],
    };
    let camera = [74.0, 45.0, 1.5];
    // RAD 218360: one mean before near, another inside depth bounds but far outside the image.
    let sources = [
        ([1116995531, 1110297876, 1023686739, 15590], [945043271, 3292149469, 3300508815, 4041825187]),
        ([1116993444, 1110328262, 1017776664, 15507], [922826143, 3282122037, 3299132474, 2929726427]),
    ];
    for (source, geometry) in sources {
        let mut profiles = Vec::new();
        for offset in [0.0, -0.8] {
            let direction = DVec3::new(-1.0, offset / 3.0_f64.sqrt(), 0.0).normalize().as_vec3().to_array();
            raycast_ext_ellipsoids(&source, &geometry, &mut profiles, camera, direction, 0.0, Some(&projection));
        }
        assert_eq!(crate::sample_surface_depth(&profiles, 0.01, 100.0, 0.5), None);
        let mut visible = [0; 4];
        let mut shape = [0; 4];
        encode_ext_splat(&mut visible, &mut shape, [69.0, 45.0, 1.5], 0.8, [1.0; 3], [0.25, 1.0, 1.0], IDENTITY);
        raycast_ext_ellipsoids(&visible, &shape, &mut profiles, camera, [-1.0, 0.0, 0.0], 0.0, Some(&projection));
        let depth = crate::sample_surface_depth(&profiles, 0.01, 100.0, 0.5).unwrap();
        assert!((depth - 4.850357).abs() < 3e-5);
    }
}

#[test]
fn collectors_keep_profiles_before_origin_and_return_depth_alpha_sigma_triples() {
    let encoding = SplatEncoding::default();
    let mut packed = [0; 4];
    let mut extended = [0; 4];
    let mut extended_scale = [0; 4];
    encode_packed_splat(&mut packed, [0.0, 0.0, 5.0], 1.0, [1.0; 3], [1.0; 3], IDENTITY, &encoding);
    encode_ext_splat(&mut extended, &mut extended_scale, [0.0, 0.0, 5.0], 1.0, [1.0; 3], [1.0; 3], IDENTITY);
    for (origin, expected_depth) in [([0.0, 0.0, 4.5], 0.5), ([0.0, 0.0, 5.5], -0.5)] {
        let mut profiles = Vec::new();
        raycast_packed_ellipsoids(&packed, &mut profiles, origin, [0.0, 0.0, 1.0], 0.0, &encoding, None);
        assert_eq!(profiles.len(), 3);
        assert_eq!(profiles[0], expected_depth);
        assert_eq!(profiles[1], 0.99);
        assert!((profiles[2] - 1.0).abs() < 0.05);
        profiles.clear();
        raycast_ext_ellipsoids(&extended, &extended_scale, &mut profiles, origin, [0.0, 0.0, 1.0], 0.0, None);
        assert_eq!(profiles, vec![expected_depth, 0.99, 1.0]);
    }
}

#[test]
fn profiles_wholly_behind_origin_do_not_hide_foreground_surface() {
    let mut extended = [0; 8];
    let mut extended_scale = [0; 8];
    for (index, depth) in [-10.0, 10.0].into_iter().enumerate() {
        encode_ext_splat(
            &mut extended[index * 4..index * 4 + 4],
            &mut extended_scale[index * 4..index * 4 + 4],
            [0.0, 0.0, depth], 1.0, [1.0; 3], [0.1; 3], IDENTITY,
        );
    }
    let mut profiles = Vec::new();
    raycast_ext_ellipsoids(&extended, &extended_scale, &mut profiles, [0.0; 3], [0.0, 0.0, 1.0], 0.0, None);
    assert_eq!(profiles.len(), 3);
    assert_eq!(profiles[0], 10.0);
    let depth = crate::sample_surface_depth(&profiles, 0.0, 20.0, 0.5).unwrap();
    assert!((depth - 9.9136773).abs() < 2e-5);
}

#[test]
fn hidden_zero_alpha_never_hits_with_zero_threshold() {
    let encoding = SplatEncoding::default();
    let mut packed = [0; 4];
    let mut extended = [0; 4];
    let mut extended_scale = [0; 4];
    encode_packed_splat(&mut packed, [0.0, 0.0, 5.0], 0.0, [1.0; 3], [1.0; 3], IDENTITY, &encoding);
    encode_ext_splat(&mut extended, &mut extended_scale, [0.0, 0.0, 5.0], 0.0, [1.0; 3], [1.0; 3], IDENTITY);
    let mut distances = Vec::new();
    raycast_packed_ellipsoids(&packed, &mut distances, [0.0; 3], [0.0, 0.0, 1.0], 0.0, &encoding, None);
    assert!(distances.is_empty());
    raycast_ext_ellipsoids(&extended, &extended_scale, &mut distances, [0.0; 3], [0.0, 0.0, 1.0], 0.0, None);
    assert!(distances.is_empty());
}

#[test]
fn crossing_plane_has_zero_depth_width_but_in_plane_ray_keeps_width() {
    let crossing = raycast_gaussian(
        [0.2, 0.0, -10.0], [0.0, 0.0, 1.0], 1.0, [0.0; 3], [1.0, 1.0, 0.0], IDENTITY, 0.0, None,
    ).unwrap();
    assert_eq!(crossing.depth, 10.0);
    assert_eq!(crossing.sigma, 0.0);
    let tangent = raycast_gaussian(
        [-10.0, 0.2, 0.0], [0.5, 0.0, 0.0], 1.0, [0.0; 3], [1.0, 1.0, 0.0], IDENTITY, 0.0, None,
    ).unwrap();
    assert_eq!(tangent.depth, 20.0);
    assert_eq!(tangent.sigma, 2.0);
}
