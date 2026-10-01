//! Write encoded RAD bytes directly to their final directory or upload archive.
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use zip::{write::SimpleFileOptions, CompressionMethod, ZipWriter};

pub enum RadOutput<'a> {
    Directory(&'a Path),
    Archive(&'a Path),
}

impl RadOutput<'_> {
    pub fn write(self, header: Vec<u8>, chunks: Vec<(String, Vec<u8>)>) -> anyhow::Result<()> {
        let files = std::iter::once(("bay-lod.rad".to_owned(), header)).chain(chunks);
        match self {
            Self::Directory(path) => {
                fs::create_dir_all(path)?;
                for (name, bytes) in files {
                    fs::write(path.join(name), bytes)?;
                }
            }
            Self::Archive(path) => {
                let mut archive = ZipWriter::new(BufWriter::new(File::create(path)?));
                let options = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
                for (name, bytes) in files {
                    archive.start_file(name, options)?;
                    archive.write_all(&bytes)?;
                }
                archive.finish()?.flush()?;
            }
        }
        Ok(())
    }
}
