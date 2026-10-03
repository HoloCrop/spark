use crate::surface_depth::ALPHA_THRESHOLD;
use super::{raycast_ext_ellipsoids, raycast_gaussian, raycast_packed_ellipsoids};
use spark_lib::{decoder::SplatEncoding, splat_encode::{encode_ext_splat, encode_packed_splat}};

const IDENTITY: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

fn hit(origin: [f32; 3], direction: [f32; 3], scale: [f32; 3], opacity: f32, minimum: f32) -> Option<f32> {
    raycast_gaussian(origin, direction, opacity, [0.0; 3], scale, IDENTITY, minimum).map(|profile| profile.depth)
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
        [0.0; 3], [0.0, 0.0, 1.0], 1.0, [1.0, 0.0, 5.0], [1.0, 0.1, 0.1], quaternion, 0.0,
    ).unwrap();
    assert!((profile.depth - 5.980198).abs() < 1e-5);
    assert!((profile.sigma - 1.0 / 50.5_f32.sqrt()).abs() < 1e-6);
    let rescaled = raycast_gaussian(
        [0.0; 3], [0.0, 0.0, 0.5], 1.0, [1.0, 0.0, 5.0], [1.0, 0.1, 0.1], quaternion, 0.0,
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
    assert_eq!(hit([2.341049, 0.0, 10.0], [0.0, 0.0, -1.0], [1.0; 3], 2.0, 0.0), None);
    assert_eq!(hit([0.0, 0.0, -10.0], [0.0, 0.0, 1.0], [1.0; 3], ALPHA_THRESHOLD, 0.0), None);
    assert_eq!(hit([0.0, 0.0, -10.0], [0.0, 0.0, 1.0], [1.0; 3], 0.004, 0.0), Some(10.0));
    let profile = raycast_gaussian(
        [0.0, 0.0, -10.0], [0.0, 0.0, 1.0], 2.0, [0.0; 3], [1.0; 3], IDENTITY, 0.0,
    ).unwrap();
    assert_eq!(profile.alpha, 0.99);
}

#[test]
fn contributing_lod_edge_above_threshold_survives_cancellation() {
    assert_eq!(hit([2.3137, 0.0, 10.0], [0.0, 0.0, -1.0], [1.0; 3], 2.0, 0.004075), Some(10.0));
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
        raycast_packed_ellipsoids(&packed, &mut profiles, origin, [0.0, 0.0, 1.0], 0.0, &encoding);
        assert_eq!(profiles.len(), 3);
        assert_eq!(profiles[0], expected_depth);
        assert_eq!(profiles[1], 0.99);
        assert!((profiles[2] - 1.0).abs() < 0.05);
        profiles.clear();
        raycast_ext_ellipsoids(&extended, &extended_scale, &mut profiles, origin, [0.0, 0.0, 1.0], 0.0);
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
    raycast_ext_ellipsoids(&extended, &extended_scale, &mut profiles, [0.0; 3], [0.0, 0.0, 1.0], 0.0);
    assert_eq!(profiles.len(), 3);
    assert_eq!(profiles[0], 10.0);
    let depth = crate::sample_surface_depth(&profiles, 0.0, 20.0).unwrap();
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
    raycast_packed_ellipsoids(&packed, &mut distances, [0.0; 3], [0.0, 0.0, 1.0], 0.0, &encoding);
    assert!(distances.is_empty());
    raycast_ext_ellipsoids(&extended, &extended_scale, &mut distances, [0.0; 3], [0.0, 0.0, 1.0], 0.0);
    assert!(distances.is_empty());
}

#[test]
fn crossing_plane_has_zero_depth_width_but_in_plane_ray_keeps_width() {
    let crossing = raycast_gaussian(
        [0.2, 0.0, -10.0], [0.0, 0.0, 1.0], 1.0, [0.0; 3], [1.0, 1.0, 0.0], IDENTITY, 0.0,
    ).unwrap();
    assert_eq!(crossing.depth, 10.0);
    assert_eq!(crossing.sigma, 0.0);
    let tangent = raycast_gaussian(
        [-10.0, 0.2, 0.0], [0.5, 0.0, 0.0], 1.0, [0.0; 3], [1.0, 1.0, 0.0], IDENTITY, 0.0,
    ).unwrap();
    assert_eq!(tangent.depth, 20.0);
    assert_eq!(tangent.sigma, 2.0);
}
