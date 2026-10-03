use super::{RayProfile, surface_depth};

fn profile(depth: f32, alpha: f32, sigma: f32) -> RayProfile {
    RayProfile { depth, alpha, sigma }
}

fn assert_depth(profiles: &[RayProfile], expected: f32) {
    let depth = surface_depth(profiles, 0.0, f32::INFINITY).unwrap();
    assert!((depth - expected).abs() < 2e-6, "{depth} != {expected}");
}

#[test]
fn single_front_and_back_crossings_match_triton_float64_reference() {
    assert_depth(&[profile(5.0, 0.8, 0.1)], 4.9400606);
    assert_depth(&[profile(5.0, 0.6, 0.1)], 5.100537);
}

#[test]
fn overlapping_profiles_cross_between_their_depth_profiles() {
    // Values from triton-splatting/tests/reference/bisection.py, not discrete alpha medians.
    assert_depth(&[profile(5.0, 0.3, 0.1); 2], 5.1525533);
    assert_depth(&[profile(5.0, 0.4, 0.1); 2], 5.0919023);
    assert_depth(&[profile(5.0, 0.6, 0.1); 2], 4.922292);
}

#[test]
fn asymmetric_means_and_sigmas_are_order_independent() {
    let profiles = [profile(3.0, 0.25, 0.05), profile(3.12, 0.65, 0.2), profile(3.3, 0.2, 0.1)];
    assert_depth(&profiles, 3.0418587);
    let reversed = [profiles[2], profiles[1], profiles[0]];
    assert_eq!(surface_depth(&profiles, 0.0, 10.0), surface_depth(&reversed, 0.0, 10.0));
}

#[test]
fn changing_ray_origin_or_parameter_preserves_the_surface() {
    assert_depth(&[profile(4.0, 0.8, 0.1)], 3.9400606);
    assert_depth(&[profile(2.5, 0.8, 0.05)], 2.4700303);
}

#[test]
fn near_and_far_bound_the_surface_crossing_instead_of_its_peak() {
    let profiles = [profile(1.01, 0.8, 0.1)];
    let depth = surface_depth(&profiles, 0.9, 1.0).unwrap();
    assert!((depth - 0.9500606).abs() < 2e-6);
    assert_eq!(surface_depth(&profiles, 0.96, 1.0), None);
    assert_eq!(surface_depth(&profiles, 0.0, 0.94), None);
}

#[test]
fn insufficient_combined_opacity_has_no_surface() {
    assert_eq!(surface_depth(&[], 0.0, 10.0), None);
    assert_eq!(surface_depth(&[profile(5.0, 0.4, 0.1)], 0.0, 10.0), None);
    assert_eq!(surface_depth(&[profile(5.0, 0.2, 0.1), profile(6.0, 0.3, 0.2)], 0.0, 10.0), None);
    assert_eq!(surface_depth(&[profile(5.0, 0.0, 0.1)], 0.0, 10.0), None);
}

#[test]
fn zero_width_profiles_cross_at_the_limiting_step() {
    assert_depth(&[profile(5.0, 0.8, 0.0)], 5.0);
    assert_depth(&[profile(5.0, 0.3, 0.0); 2], 5.0);
    assert_depth(&[profile(4.0, 0.3, 0.0), profile(5.0, 0.3, 0.0)], 5.0);
    assert_eq!(surface_depth(&[profile(5.0, 0.8, 0.0)], 5.0, 5.0), Some(5.0));
    assert_eq!(surface_depth(&[profile(5.0, 0.8, 0.0)], 5.1, 6.0), None);
}
