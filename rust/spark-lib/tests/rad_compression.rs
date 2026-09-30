#![cfg(feature = "native-zstd")]

use glam::{Quat, Vec3A};
use half::f16;
use spark_lib::{decoder::ChunkReceiver,
    gsplat::{Gsplat, GsplatArray, GsplatSH1, GsplatSH2, GsplatSH3},
    rad::{RadChunkPropertyCompression, RadDecoder, RadEncoder},
    tsplat::{Tsplat, TsplatArray}};

fn hierarchy() -> GsplatArray {
    let mut splats = GsplatArray::new_capacity(3, 3);
    for (index, opacity) in [4.0, 0.7, 0.9].into_iter().enumerate() {
        let value = f16::from_f32(index as f32 * 0.125 - 0.125);
        splats.push_splat(Gsplat::new(Vec3A::new(index as f32 * 0.25, 0.1, -0.2),
            opacity, Vec3A::new(0.3, 0.5, 0.7), Vec3A::new(0.2, 0.1, 0.05),
            Quat::from_rotation_z(0.2), 6, [0, 102, 102][index]),
            Some(GsplatSH1([[value; 3]; 3])), Some(GsplatSH2([[value; 3]; 5])),
            Some(GsplatSH3([[value; 3]; 7])));
    }
    splats.prepare_children();
    splats.set_children(0, &[1, 2]);
    splats.encode_lod_opacity();
    splats
}

fn roundtrip(compression: RadChunkPropertyCompression) -> GsplatArray {
    let mut encoder = RadEncoder::new(hierarchy());
    encoder.compression = compression;
    encoder.resolve_encoding();
    let mut bytes = Vec::new();
    encoder.encode(&mut bytes).unwrap();
    let mut decoder = RadDecoder::new(GsplatArray::new());
    for chunk in bytes.chunks(137) {
        decoder.push(chunk).unwrap();
    }
    decoder.finish().unwrap();
    decoder.into_splats()
}

#[test]
fn zstd_preserves_quantized_hierarchy_geometry_appearance_and_labels() {
    let gz = roundtrip(RadChunkPropertyCompression::Gz);
    let zstd = roundtrip(RadChunkPropertyCompression::Zstd);
    assert_eq!(zstd.len(), 3);
    assert_eq!(zstd.get_children(0).as_slice(), &[1, 2]);
    assert!(zstd.get(0).opacity() > 1.0);
    assert_eq!(zstd.get(0).instance_label(), 0);
    assert_eq!(zstd.get(1).instance_label(), 102);
    for index in 0..zstd.len() {
        let (a, b) = (gz.get(index), zstd.get(index));
        assert_eq!(a.center(), b.center());
        assert_eq!(a.scales(), b.scales());
        assert_eq!(a.quaternion(), b.quaternion());
        assert_eq!(a.opacity(), b.opacity());
        assert_eq!(a.rgb(), b.rgb());
        assert_eq!(b.label(), 6);
        assert_eq!(a.instance_label(), b.instance_label());
        assert_eq!(gz.get_sh1(index), zstd.get_sh1(index));
        assert_eq!(gz.get_sh2(index), zstd.get_sh2(index));
        assert_eq!(gz.get_sh3(index), zstd.get_sh3(index));
        assert_eq!(gz.get_children(index), zstd.get_children(index));
    }
}
