use spark_lib::{chunk_tree, tiny_lod, bhatt_lod, gsplat::{GsplatArray, GaussianKernel}, rad::RadEncoder,
    decoder::{ChunkReceiver, MultiDecoder}};
use std::{fs, path::Path, time::Instant};
use spark_lib::tsplat::TsplatArray;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut decoder = MultiDecoder::new(GsplatArray::new(), None, Some(&args[1]));
    decoder.push(&fs::read(&args[1])?)?;
    decoder.finish()?;
    let mut splats = decoder.into_splats();
    let started = Instant::now();
    let method = args[3].as_str();
    match method {
        "tiny" => tiny_lod::compute_lod_tree(&mut splats, 1.75, true, |message| println!("{message}")),
        "bhatt-k2" => {
            splats.kernel = GaussianKernel::K2;
            bhatt_lod::compute_lod_tree(&mut splats, 1.75, |message| println!("{message}"));
        },
        "bhatt" => bhatt_lod::compute_lod_tree(&mut splats, 1.75, |message| println!("{message}")),
        _ => anyhow::bail!("expected tiny, bhatt or bhatt-k2"),
    }
    println!("{method} merge: {:.3}s", started.elapsed().as_secs_f64());
    chunk_tree::chunk_tree(&mut splats, 0, |_| {});
    splats.encode_lod_opacity();
    let mut encoder = RadEncoder::new(splats);
    encoder.resolve_encoding();
    {
        let name = method;
        let directory = Path::new(&args[2]).join(&name);
        fs::create_dir_all(&directory)?;
        let mut header = Vec::new();
        let chunks = encoder.encode_with_chunks(&mut header, "bay-lod-")?;
        fs::write(directory.join("bay-lod.rad"), header)?;
        for (name, data) in chunks { fs::write(directory.join(name), data)?; }
    }
    println!("Build and export: {:.3}s", started.elapsed().as_secs_f64());
    Ok(())
}
