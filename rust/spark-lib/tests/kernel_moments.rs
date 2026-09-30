use glam::{Quat, Vec3A};
use spark_lib::{gsplat::{GaussianKernel, Gsplat, GsplatArray}, symmat3::SymMat3,
    tsplat::{Tsplat, TsplatArray, moment_matched_distance}};

fn cloud(kernel: GaussianKernel, displacement: f32) -> GsplatArray {
    let mut cloud = GsplatArray::new();
    cloud.kernel = kernel;
    for x in [-displacement, displacement] {
        cloud.push_splat(Gsplat::new(Vec3A::new(x, 0.0, 0.0), 0.25,
            Vec3A::ONE, Vec3A::ONE, Quat::IDENTITY, 1, 1), None, None, None);
    }
    cloud.prepare_children();
    cloud
}

#[test]
fn separated_children_preserve_projected_second_moments() {
    for kernel in [GaussianKernel::K1, GaussianKernel::K2] {
        let mut splats = cloud(kernel, 1.0);
        let parent = splats.new_merged(&[0, 1], 0.0);
        let splat = splats.get(parent);
        let shape = SymMat3::new_scale_quaternion(splat.scales(), splat.quaternion());
        let factor = kernel.moment_factor();
        assert!((factor * shape.xx() - (factor + 1.0)).abs() < 0.004);
        assert!((factor * shape.yy() - factor).abs() < 0.004);
        assert!((factor * shape.zz() - factor).abs() < 0.004);
        assert!(splat.center().length() < 1e-6);
        assert_eq!(splats.get_children(parent).as_slice(), &[0, 1]);
        assert!((splat.area() * splat.opacity() - 2.0 * splats.get(0).area() * splats.get(0).opacity()).abs() < 0.02);
    }
}

#[test]
fn colocated_children_do_not_inflate_when_switching_kernel() {
    let mut splats = cloud(GaussianKernel::K2, 0.0);
    let parent = splats.new_merged(&[0, 1], 0.0);
    assert!((splats.get(parent).scales() - Vec3A::ONE).length() < 0.003);
    let subset = splats.clone_subset(0, 2);
    assert!((subset.similarity(0, 1) - splats.similarity(0, 1)).abs() < 1e-6);
}

#[test]
fn overlap_uses_narrower_k2_moments_for_separated_shapes() {
    let splats = cloud(GaussianKernel::K2, 1.0);
    let a = splats.get(0);
    let b = splats.get(1);
    let factor = splats.kernel.moment_factor();
    assert!((moment_matched_distance(&a, &b, factor) - 0.5 / factor).abs() < 1e-6);
    assert!((splats.similarity(0, 1) - (-0.5 / factor).exp()).abs() < 1e-6);
    let subset = splats.clone_subset(0, 2);
    assert!((subset.similarity(0, 1) - splats.similarity(0, 1)).abs() < 1e-6);
}
