use std::io::{Cursor, Write};

use glam::{Quat, Vec3A};
use image::{DynamicImage, RgbaImage};
use spark_lib::{
    csplat::CsplatArray,
    decoder::{ChunkReceiver, SplatGetter, SplatPropsArray, copy_getter_to_receiver},
    gsplat::{Gsplat, GsplatArray},
    rad::{RadDecoder, RadEncoder},
    sogs::SogsDecoder,
    tsplat::{Tsplat, TsplatArray},
};

#[test]
fn compact_transfer_preserves_labels_and_rotation_across_chunks() {
    let mut source = GsplatArray::new();
    let rotation = Quat::from_rotation_y(0.6);
    for index in 0..65539 {
        source.push_splat(Gsplat::new(Vec3A::new(index as f32, 0.0, 0.0), 0.8,
            Vec3A::ONE, Vec3A::ONE, rotation, (index % 17) as u32, index as u32), None, None, None);
    }
    let mut compact = CsplatArray::new();
    copy_getter_to_receiver(&mut source, &mut compact).unwrap();
    let mut result = GsplatArray::new();
    copy_getter_to_receiver(&mut compact, &mut result).unwrap();
    for index in [0, 16384, 65535, 65536, 65538] {
        let splat = result.get(index);
        assert_eq!(splat.label(), (index % 17) as u32);
        assert_eq!(splat.instance_label(), index as u32);
        assert!(splat.quaternion().dot(rotation).abs() > 0.999);
        assert_eq!(splat.center().x, index as f32);
    }
    let mut batch = SplatPropsArray::new_empty(65536, 3);
    batch.labels.resize(3, 0);
    batch.instances.resize(3, 0);
    result.get_batch(65536, 3, &mut batch.as_mut());
    assert_eq!(batch.labels, [1, 2, 3]);
    assert_eq!(batch.instances, [65536, 65537, 65538]);
}

#[test]
fn rad_roundtrip_keeps_semantic_groups_separate() {
    let mut source = GsplatArray::new();
    for (label, instance) in [(1, 7), (2, 7), (1, 8)] {
        source.push_splat(Gsplat::new(Vec3A::ZERO, 0.8, Vec3A::ONE, Vec3A::ONE,
            Quat::IDENTITY, label, instance), None, None, None);
    }
    assert_eq!(source.similarity(0, 1), 0.0);
    assert_eq!(source.similarity(0, 2), 0.0);
    let mut bytes = Vec::new();
    let mut encoder = RadEncoder::new(source);
    encoder.resolve_encoding();
    encoder.encode(&mut bytes).unwrap();
    let mut decoder = RadDecoder::new(GsplatArray::new());
    decoder.push(&bytes).unwrap();
    decoder.finish().unwrap();
    let result = decoder.into_splats();
    for (index, expected) in [(1, 7), (2, 7), (1, 8)].into_iter().enumerate() {
        assert_eq!((result.get(index).label(), result.get(index).instance_label()), expected);
    }
}

#[test]
fn sog_labels_survive_the_full_image_and_partial_final_chunk() {
    let count = 65539usize;
    let width = 256;
    let height = count.div_ceil(width);
    let geometry = [127, 127, 127, 255].repeat(width * height);
    let labels: Vec<u8> = (0..width * height).flat_map(|index| {
        let instance = (index as u16).to_le_bytes();
        [(index % 17) as u8, instance[0], instance[1], 255]
    }).collect();
    let metadata = serde_json::json!({
        "version": 2, "count": count,
        "means": {"files": ["geometry.png", "geometry.png"], "mins": [0.0, 0.0, 0.0], "maxs": [1.0, 1.0, 1.0]},
        "scales": {"files": ["geometry.png"], "codebook": vec![0.0; 256]},
        "quats": {"files": ["geometry.png"]},
        "sh0": {"files": ["geometry.png"], "codebook": vec![0.0; 256]},
        "labels": {"files": ["labels.png"]},
    });
    let mut archive = zip::ZipWriter::new(Cursor::new(Vec::new()));
    let options = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    archive.start_file("meta.json", options).unwrap();
    archive.write_all(metadata.to_string().as_bytes()).unwrap();
    for (name, pixels) in [("geometry.png", geometry), ("labels.png", labels)] {
        let image = RgbaImage::from_raw(width as u32, height as u32, pixels).unwrap();
        let mut png = Cursor::new(Vec::new());
        DynamicImage::ImageRgba8(image).write_to(&mut png, image::ImageFormat::Png).unwrap();
        archive.start_file(name, options).unwrap();
        archive.write_all(&png.into_inner()).unwrap();
    }
    let mut decoder = SogsDecoder::new(GsplatArray::new(), None);
    decoder.push(&archive.finish().unwrap().into_inner()).unwrap();
    decoder.finish().unwrap();
    let result = decoder.into_splats();
    for index in [0, 16384, 65535, 65536, 65538] {
        assert_eq!(result.get(index).label(), (index % 17) as u32);
        assert_eq!(result.get(index).instance_label(), index as u16 as u32);
    }
}
