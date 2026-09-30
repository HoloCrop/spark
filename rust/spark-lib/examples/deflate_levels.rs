use std::{fs, path::Path, time::Instant};
use miniz_oxide::{deflate::compress_to_vec, inflate::decompress_to_vec};
use rayon::prelude::*;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let directory = Path::new(&args[1]);
    let mut properties = Vec::new();
    for index in [0, 15, 35, 60] {
        let data = fs::read(directory.join(format!("bay-lod-{index}.radc")))?;
        let size = u32::from_le_bytes(data[4..8].try_into()?) as usize;
        let metadata: serde_json::Value = serde_json::from_slice(&data[8..8+size])?;
        let payload = 16 + size.div_ceil(8)*8;
        for property in metadata["properties"].as_array().unwrap() {
            if property["compression"] == "gz" {
                let offset = payload + property["offset"].as_u64().unwrap() as usize;
                let length = property["bytes"].as_u64().unwrap() as usize;
                properties.push(decompress_to_vec(&data[offset..offset+length]).unwrap());
            }
        }
    }
    let pool = rayon::ThreadPoolBuilder::new().num_threads(16).build()?;
    for level in [1, 3, 6] {
        let started = Instant::now();
        let output: Vec<_> = pool.install(|| properties.par_iter().map(|data| compress_to_vec(data, level)).collect());
        let elapsed = started.elapsed();
        println!("level {level}: {:.4}s, {} compressed bytes", elapsed.as_secs_f64(), output.iter().map(|data| data.len()).sum::<usize>());
        for (raw, compressed) in properties.iter().zip(output) {
            assert_eq!(*raw, decompress_to_vec(&compressed).unwrap());
        }
    }
    Ok(())
}
