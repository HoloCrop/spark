mod trained;
use pyo3::prelude::*;


#[pymodule]
mod py_build_lod {
    #[pymodule_export]
    use crate::trained::{TrainedLevel, encode_trained_arrays, encode_trained_archive};
    use pyo3::prelude::*;
    use pyo3::exceptions::PyValueError;
    use spark_lib::decoder::SplatGetter;

    use std::fs;
    use std::path::PathBuf;
    use std::fs::File;
    use std::io::{BufReader, BufWriter, Read, Write};

    use spark_lib::{chunk_tree};
    use spark_lib::rad::RadEncoder;
    use spark_lib::{
        decoder::{ChunkReceiver, MultiDecoder},
        gsplat::GsplatArray,
        tsplat::{Tsplat, TsplatArray},
        bhatt_lod
    };

    fn read_file_chunks(filename: &str, decoder: &mut impl ChunkReceiver) -> anyhow::Result<()> {
        const CHUNK_SIZE: usize = 1024 * 1024; // 1 MiB
        let mut reader = BufReader::new(File::open(filename).unwrap());
        let mut buffer = vec![0u8; CHUNK_SIZE];
        loop {
            let bytes_read = reader.read(&mut buffer).unwrap();
            if bytes_read == 0 {
                break;
            }
            decoder.push(&buffer[..bytes_read])?;
        }
        decoder.finish()
    }

    fn encoder_snapshot<T: SplatGetter>(e: &RadEncoder<T>) -> serde_json::Value {
        serde_json::json!({
            "center": e.center_encoding, "alpha": e.alpha_encoding, "rgb": e.rgb_encoding,
            "scales": e.scales_encoding, "orientation": e.orientation_encoding,
            "label": e.label_encoding, "instance_label": e.instance_encoding,
            "sh": e.sh_encoding, "encoding": e.encoding, "sh_label": e.sh_label_encoding,
        })
    }


    #[pyfunction]
    fn encode_trained_rad(directory: &str, output_dir: &str, levels: usize,
                          moment_factor: f32, resolution_factor: f32) -> PyResult<()> {
        spark_lib::trained_lod::encode_levels(
            std::path::Path::new(directory), std::path::Path::new(output_dir),
            levels, moment_factor, resolution_factor,
        ).map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[pyfunction]
    fn encode_rad(input_file: &str, output_dir: &str) -> PyResult<()> {
        let splats: GsplatArray = GsplatArray::new();

        let mut decoder = MultiDecoder::new(splats, None, Some(&input_file));
        let mut splats = match read_file_chunks(&input_file, &mut decoder) {
            Ok(_) => {
                println!("Detected file type: {:?}", decoder.file_type.unwrap());
                decoder.into_splats()
            }
            Err(error) => {
                eprintln!("Decoding failed: {:?}", error);
                return Err(PyValueError::new_err(format!("Decoding failed: {:?}", error)));
            }
        };

        let input_splat_count = splats.len();
        println!("Read: num_splats: {}", input_splat_count);

        let mut zero_opacity = 0;
        let mut zero_scale = 0;
        let mut invalid_quat = 0;

        splats.retain(|splat| {
            zero_opacity += if splat.opacity() > 0.0 { 0 } else { 1 };
            zero_scale += if splat.max_scale() > 0.0 { 0 } else { 1 };
            invalid_quat += if splat.quaternion().is_finite() && splat.quaternion().length() > 0.0 { 0 } else { 1 };
            (splat.opacity() > 0.0) && (splat.max_scale() > 0.0) &&
            (splat.quaternion().is_finite() && splat.quaternion().length() > 0.0)
        });

        let mut description = serde_json::Map::new();
        if input_splat_count != splats.len() {
            println!("zero_opacity: {}, zero_scale: {}, invalid_quat: {}", zero_opacity, zero_scale, invalid_quat);
            println!("Removed {} empty splats, remaining splats.len={}", input_splat_count - splats.len(), splats.len());
            description.insert("empty_splat_count".to_string(), serde_json::Value::Number((input_splat_count - splats.len()).into()));
            description.insert("initial_splat_count".to_string(), serde_json::Value::Number(input_splat_count.into()));
        }

        let start_time = std::time::Instant::now();

        let lod_base = 1.75;
        description.insert("method".to_string(), serde_json::Value::String(format!("BhattLod: {:?}", lod_base)));
        bhatt_lod::compute_lod_tree(&mut splats, lod_base, |s| println!("{}", s));

        let lod_duration = start_time.elapsed();
        let lod_secs = lod_duration.as_secs_f64();
        description.insert(
            "lod_duration".to_string(),
            serde_json::Number::from_f64(lod_secs).map(serde_json::Value::Number).unwrap_or(serde_json::Value::Null),
        );

        let final_splat_count = splats.len();
        description.insert("final_splat_count".to_string(), serde_json::Value::Number(final_splat_count.into()));

        let start_time = std::time::Instant::now();

        chunk_tree::chunk_tree(&mut splats, 0, |s| println!("{}", s));

        let chunk_duration = start_time.elapsed();
        description.insert("chunk_duration".to_string(), serde_json::Number::from_f64(chunk_duration.as_secs_f64()).into());
        splats.encode_lod_opacity();
    
        let mut encoder = RadEncoder::new(splats);
        let input_encoding = encoder_snapshot(&encoder);
        
        description.insert("input_encoding".to_string(), input_encoding);

        encoder.resolve_encoding();
        let resolved_encoding = encoder_snapshot(&encoder);
        description.insert("resolved_encoding".to_string(), resolved_encoding);

        println!("Encoding RAD file with center={:?}, alpha={:?}, rgb={:?}, scales={:?}, orientation={:?}, sh={:?}", encoder.center_encoding, encoder.alpha_encoding, encoder.rgb_encoding, encoder.scales_encoding, encoder.orientation_encoding, encoder.sh_encoding);
        if let Some(encoding) = encoder.encoding.as_ref() {
            println!("Splat Encoding: {:?}", encoding);
        }

        let comment = serde_json::to_string_pretty(&description).unwrap();
        println!("Comment: {}", comment);
        let mut encoder = encoder.with_comment(comment);

        let stem = std::path::Path::new(input_file)
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy();
        let output_filename = format!("{}-lod", stem);
        let binding = std::path::PathBuf::from(&output_filename);
        let output_filename = binding.file_name().unwrap().to_str().unwrap();
        
        let output_dir_path = PathBuf::from(output_dir);
        fs::create_dir_all(&output_dir_path)?;
        let mut output_path = output_dir_path.join(output_filename);

        let filename_ext = format!("{}.rad", output_path.to_str().unwrap());
        let mut writer = BufWriter::new(File::create(&filename_ext).unwrap());

        let filename_only = output_path.file_name().unwrap().to_str().unwrap();
        let chunk_prefix = format!("{}-", filename_only);
        let chunks = encoder.encode_with_chunks(&mut writer, &chunk_prefix).unwrap();
        for (filename, chunk) in chunks {
            output_path.set_file_name(&filename);
            let mut chunk_writer = BufWriter::new(File::create(&output_path).unwrap());
            chunk_writer.write_all(&chunk).unwrap();
            chunk_writer.flush()?;
            println!("Wrote {} ({} bytes)", filename, chunk.len());
        }
        
        println!("Wrote {}", filename_ext);
        Ok(())
    }

}
