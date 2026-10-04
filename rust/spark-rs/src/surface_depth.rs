pub const MAX_ALPHA: f32 = 0.99;
pub const ALPHA_THRESHOLD: f32 = 0.5 / 255.0;
pub const PROFILE_REACH: f64 = 3.851_285_106_843_081_3; // 220^(1/4), matching Triton's depth profile.

#[derive(Clone, Copy)]
pub struct RayProfile {
    pub depth: f32,
    pub alpha: f32,
    pub sigma: f32,
}

enum DepthSide {
    Front,
    Center,
    Back,
}

fn log_transmittance(profiles: &[RayProfile], depth: f64, side: DepthSide) -> f64 {
    profiles.iter().map(|profile| {
        let alpha = f64::from(profile.alpha);
        let remaining = (-alpha).ln_1p();
        let offset = depth - f64::from(profile.depth);
        if profile.sigma == 0.0 {
            if offset < 0.0 {
                0.0
            } else if offset > 0.0 {
                remaining
            } else {
                match side {
                    DepthSide::Front => 0.0,
                    DepthSide::Center => 0.5 * remaining,
                    DepthSide::Back => remaining,
                }
            }
        } else {
            let normalized = offset / f64::from(profile.sigma);
            let squared = normalized * normalized;
            let gate = (-0.5 * squared * squared).exp();
            let half = 0.5 * (-alpha * gate).ln_1p();
            if offset <= 0.0 { half } else { remaining - half }
        }
    }).sum()
}

pub fn surface_depth(profiles: &[RayProfile], near: f32, far: f32) -> Option<f32> {
    if profiles.is_empty() {
        return None;
    }
    let mut lower = profiles.iter().map(|profile| {
        f64::from(profile.depth) - PROFILE_REACH * f64::from(profile.sigma)
    }).fold(f64::INFINITY, f64::min).max(f64::from(near));
    let mut upper = profiles.iter().map(|profile| {
        f64::from(profile.depth) + PROFILE_REACH * f64::from(profile.sigma)
    }).fold(f64::NEG_INFINITY, f64::max).min(f64::from(far));
    let target = 0.5_f64.ln();
    // One-sided endpoints bracket exact zero-width jumps as well as smooth crossings.
    if lower > upper || log_transmittance(profiles, lower, DepthSide::Front) < target
        || log_transmittance(profiles, upper, DepthSide::Back) > target
    {
        return None;
    }
    for _ in 0..32 {
        let middle = lower + 0.5 * (upper - lower);
        if log_transmittance(profiles, middle, DepthSide::Center) > target {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    Some((lower + 0.5 * (upper - lower)) as f32)
}

#[cfg(test)]
#[path = "surface_depth_tests.rs"]
mod tests;
