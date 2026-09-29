//! Encode trained levels with explicit ancestry and independent selection bounds.
use crate::chunk_tree;
use crate::decoder::{ChunkReceiver, MultiDecoder};
use crate::gsplat::{GsplatArray, GsplatSH3};
use crate::rad::RadEncoder;
use crate::tsplat::{Tsplat, TsplatArray, TsplatMut};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

fn read(path: &Path) -> anyhow::Result<GsplatArray> {
    let name = path.to_str().unwrap();
    let mut decoder = MultiDecoder::new(GsplatArray::new(), None, Some(name));
    let mut reader = BufReader::new(File::open(path)?);
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        decoder.push(&buffer[..count])?;
    }
    decoder.finish()?;
    let mut splats = decoder.into_splats();
    if splats.max_sh_degree == 2 {
        splats
            .sh3
            .resize(splats.len(), GsplatSH3([[Default::default(); 3]; 7]));
        splats.max_sh_degree = 3;
    }
    anyhow::ensure!(splats.max_sh_degree == 3, "expected SH degree 2 or 3");
    Ok(splats)
}

fn verify(splats: &GsplatArray, leaves: usize) {
    let mut incoming = vec![0u8; splats.len()];
    for children in &splats.children {
        assert!(children.len() <= 65535);
        for &child in children {
            assert_eq!(incoming[child], 0);
            incoming[child] += 1;
        }
    }
    assert_eq!(incoming[0], 0);
    assert!(incoming[1..].iter().all(|&count| count == 1));
    assert_eq!(
        splats
            .children
            .iter()
            .filter(|children| children.is_empty())
            .count(),
        leaves
    );
}

// Untrained ancestors above the final optimized level: preserve instance groups
// first, then classes, and cross classes only after those groups reach one node.
fn semantic_ancestors(splats: &mut GsplatArray) {
    splats.prepare_children();
    let mut frontier: Vec<_> = (0..splats.len()).collect();
    let mut stage = 0;
    while frontier.len() > 1 {
        let groups = loop {
            let mut groups: BTreeMap<(u32, u32), Vec<usize>> = BTreeMap::new();
            for &index in &frontier {
                let splat = splats.get(index);
                let key = match stage {
                    0 => (splat.label(), splat.instance_label()),
                    1 => (splat.label(), 0),
                    _ => (0, 0),
                };
                groups.entry(key).or_default().push(index);
            }
            if groups.len() < frontier.len() {
                break groups;
            }
            stage += 1;
        };
        let mut next = Vec::new();
        for mut group in groups.into_values() {
            // Pair spatial neighbors along this group's widest center axis.
            let mut lower = splats.get(group[0]).center();
            let mut upper = lower;
            for &index in &group {
                let center = splats.get(index).center();
                lower = lower.min(center);
                upper = upper.max(center);
            }
            let extent = upper - lower;
            let axis = (0..3)
                .max_by(|&a, &b| extent[a].total_cmp(&extent[b]))
                .unwrap();
            group.sort_unstable_by(|&a, &b| {
                splats.get(a).center()[axis].total_cmp(&splats.get(b).center()[axis])
            });
            for pair in group.chunks(2) {
                if pair.len() == 1 {
                    next.push(pair[0]);
                    continue;
                }
                let left = splats.get(pair[0]);
                let right = splats.get(pair[1]);
                let label = if left.label() == right.label() {
                    left.label()
                } else {
                    0
                };
                let instance = if left.label() == right.label()
                    && left.instance_label() == right.instance_label()
                {
                    left.instance_label()
                } else {
                    0
                };
                let parent = splats.new_merged(pair, 0.0);
                splats.get_mut(parent).set_label(label);
                splats.get_mut(parent).set_instance_label(instance);
                next.push(parent);
            }
        }
        frontier = next;
    }
    assert_eq!(frontier[0], splats.len() - 1);
}

pub fn encode_levels(
    directory: &Path,
    output: &Path,
    levels: usize,
    moment_factor: f32,
    resolution_factor: f32,
) -> anyhow::Result<()> {
    anyhow::ensure!(levels > 0, "at least one trained level is required");
    let mut splats = read(&directory.join(format!("level-{levels}/point_cloud.ply")))?;
    let original_scales: Vec<_> = splats.splats.iter().map(|s| s.ln_scales).collect();
    let moment_scale = moment_factor.sqrt();
    for mut splat in &mut splats.splats {
        splat.set_scales(splat.scales() * moment_scale);
    }
    semantic_ancestors(&mut splats);
    for index in 0..splats.len() {
        if index < original_scales.len() {
            splats.splats[index].ln_scales = original_scales[index];
        } else {
            let scales = splats.get(index).scales() / moment_scale;
            splats.get_mut(index).set_scales(scales);
        }
    }
    // Parents are appended chronologically. Reverse once to put the root first.
    let length = splats.len();
    splats.permute(&(0..length).rev().collect::<Vec<_>>());
    for children in &mut splats.children {
        for child in children {
            *child = length - 1 - *child;
        }
    }
    let mut coarse_indices: Vec<_> = (0..original_scales.len())
        .map(|index| length - 1 - index)
        .collect();
    let mut leaf_count = original_scales.len();
    for level in (1..levels).rev() {
        let path = directory.join(format!("level-{level}/point_cloud.ply"));
        let mut fine = read(&path)?;
        leaf_count = fine.len();
        let start = splats.len();
        let bytes = fs::read(directory.join(format!("parents-{level}.bin")))?;
        anyhow::ensure!(
            bytes.len() == fine.len() * 4,
            "parent map length differs from input"
        );
        splats.splats.append(&mut fine.splats);
        splats.sh1.append(&mut fine.sh1);
        splats.sh2.append(&mut fine.sh2);
        splats.sh3.append(&mut fine.sh3);
        splats.prepare_children();
        for (child, word) in bytes.chunks_exact(4).enumerate() {
            let parent = u32::from_le_bytes(word.try_into().unwrap()) as usize;
            splats.children[coarse_indices[parent]].push(start + child);
        }
        assert!(coarse_indices
            .iter()
            .all(|&index| !splats.children[index].is_empty()));
        coarse_indices = (start..splats.len()).collect();
        println!("Attached level {level}: {leaf_count} splats");
    }
    verify(&splats, leaf_count);
    chunk_tree::chunk_tree(&mut splats, 0, |_| {});
    verify(&splats, leaf_count);
    println!(
        "Validated {} nodes, {} finest leaves",
        splats.len(),
        leaf_count
    );
    splats.encode_lod_opacity();
    // Selection bounds are independent of rendered geometry. Each intermediate
    // representative gets a distinct scale interval, even if training shrank it.
    let mut lod_sizes = vec![0.0f32; splats.len()];
    let mut heights = vec![0usize; splats.len()];
    for index in (0..splats.len()).rev() {
        let splat = splats.get(index);
        let mut size = 2.0 * splat.scales().element_sum() / 3.0;
        for &child in &splats.children[index] {
            assert!(child > index);
            let distance = (splat.center() - splats.get(child).center()).length();
            heights[index] = heights[index].max(heights[child] + 1);
            // Only trained level transitions use the training resolution ratio.
            // Ancestors above them need ordering, not another exponential schedule.
            let ratio = if heights[child] < levels - 1 {
                resolution_factor
            } else {
                1.001
            };
            size = size.max(ratio * lod_sizes[child] + 2.0 * distance);
        }
        assert!(
            size.is_finite() && size <= 65504.0,
            "LOD size exceeds decoder f16 range"
        );
        lod_sizes[index] = size;
    }
    let mut encoder = RadEncoder::new(splats);
    encoder.lod_sizes = lod_sizes;
    encoder.resolve_encoding();
    fs::create_dir_all(&output)?;
    let mut writer = BufWriter::new(File::create(output.join("bay-lod.rad"))?);
    for (name, bytes) in encoder.encode_with_chunks(&mut writer, "bay-lod-")? {
        fs::write(output.join(name), bytes)?;
    }
    writer.flush()?;
    Ok(())
}
