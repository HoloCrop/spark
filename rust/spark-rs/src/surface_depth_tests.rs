use super::{RayProfile, surface_depth};

fn profile(depth: f32, alpha: f32, sigma: f32) -> RayProfile {
    RayProfile { depth, alpha, sigma }
}

fn assert_depth(profiles: &[RayProfile], expected: f32) {
    let depth = surface_depth(profiles, 0.0, f32::INFINITY, 0.5).unwrap();
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
    assert_eq!(surface_depth(&profiles, 0.0, 10.0, 0.5), surface_depth(&reversed, 0.0, 10.0, 0.5));
}

#[test]
fn changing_ray_origin_or_parameter_preserves_the_surface() {
    assert_depth(&[profile(4.0, 0.8, 0.1)], 3.9400606);
    assert_depth(&[profile(2.5, 0.8, 0.05)], 2.4700303);
}

#[test]
fn near_and_far_bound_the_surface_crossing_instead_of_its_peak() {
    let profiles = [profile(1.01, 0.8, 0.1)];
    let depth = surface_depth(&profiles, 0.9, 1.0, 0.5).unwrap();
    assert!((depth - 0.9500606).abs() < 2e-6);
    assert_eq!(surface_depth(&profiles, 0.96, 1.0, 0.5), None);
    assert_eq!(surface_depth(&profiles, 0.0, 0.94, 0.5), None);
}

#[test]
fn insufficient_combined_opacity_has_no_surface() {
    assert_eq!(surface_depth(&[], 0.0, 10.0, 0.5), None);
    assert_eq!(surface_depth(&[profile(5.0, 0.4, 0.1)], 0.0, 10.0, 0.5), None);
    assert_eq!(surface_depth(&[profile(5.0, 0.2, 0.1), profile(6.0, 0.3, 0.2)], 0.0, 10.0, 0.5), None);
    assert_eq!(surface_depth(&[profile(5.0, 0.0, 0.1)], 0.0, 10.0, 0.5), None);
}

#[test]
fn zero_width_profiles_cross_at_the_limiting_step() {
    assert_depth(&[profile(5.0, 0.8, 0.0)], 5.0);
    assert_depth(&[profile(5.0, 0.3, 0.0); 2], 5.0);
    assert_depth(&[profile(4.0, 0.3, 0.0), profile(5.0, 0.3, 0.0)], 5.0);
    assert_eq!(surface_depth(&[profile(5.0, 0.8, 0.0)], 5.0, 5.0, 0.5), Some(5.0));
    assert_eq!(surface_depth(&[profile(5.0, 0.8, 0.0)], 5.1, 6.0, 0.5), None);
}

#[test]
fn occlusion_opacity_allowance_preserves_contact_through_faint_foreground() {
    let target = profile(5.0, 0.8, 0.015);
    let contact = surface_depth(&[target], 0.0, 10.0, 0.5).unwrap();
    assert_eq!(surface_depth(&[target], 0.0, contact, 0.55), None);
    for alpha in [0.01, 0.05, 0.1] {
        let profiles = [profile(3.0, alpha, 0.015), target];
        assert!(surface_depth(&profiles, 0.0, contact, 0.5).is_some());
        let hit = surface_depth(&profiles, 0.0, contact, 0.55);
        assert!(!hit.is_some_and(|depth| depth < contact));
    }
    let profiles = [profile(3.0, 0.2, 0.015), target];
    assert!(surface_depth(&profiles, 0.0, contact, 0.55).unwrap() < contact);
}

#[test]
fn occlusion_opacity_allowance_still_finds_a_surface_five_millimeters_ahead() {
    let profiles = [profile(5.0, 0.8, 0.015)];
    let contact = surface_depth(&profiles, 0.0, 10.0, 0.5).unwrap();
    let point = contact + 0.005;
    let hit = surface_depth(&profiles, 0.0, point, 0.55).unwrap();
    assert!(hit < point);
}
