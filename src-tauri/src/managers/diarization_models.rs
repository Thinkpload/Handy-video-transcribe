//! Download + path resolution for diarization ONNX models.
//!
//! Two files are needed:
//!   - segmentation.onnx  (pyannote-segmentation-3.0 ONNX export)
//!   - embedding.onnx     (wespeaker raw-waveform ONNX export)
//!
//! Default URLs point at community/sherpa-onnx mirrors of the pyannote models.
//! If they 404, the user can replace the files manually under
//!   <app_data>/models/diarization/{segmentation,embedding}.onnx

use anyhow::{anyhow, Context, Result};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use tauri::AppHandle;

const SEG_URL: &str =
    "https://huggingface.co/onnx-community/pyannote-segmentation-3.0/resolve/main/onnx/model.onnx";
const EMB_URL: &str =
    "https://huggingface.co/deepghs/pyannote-wespeaker-voxceleb-resnet34-LM/resolve/main/speaker-embedding.onnx";

pub fn diarization_dir(app: &AppHandle) -> Result<PathBuf> {
    let dir = crate::portable::app_data_dir(app)
        .map_err(|e| anyhow!("app data dir: {}", e))?
        .join("models")
        .join("diarization");
    if !dir.exists() {
        fs::create_dir_all(&dir)?;
    }
    Ok(dir)
}

pub fn segmentation_path(app: &AppHandle) -> Result<PathBuf> {
    Ok(diarization_dir(app)?.join("segmentation.onnx"))
}

pub fn embedding_path(app: &AppHandle) -> Result<PathBuf> {
    Ok(diarization_dir(app)?.join("embedding.onnx"))
}

pub fn models_present(app: &AppHandle) -> bool {
    matches!(
        (segmentation_path(app), embedding_path(app)),
        (Ok(s), Ok(e)) if s.exists() && e.exists()
    )
}

fn download_to(url: &str, dest: &Path) -> Result<()> {
    log::info!("downloading {} -> {}", url, dest.display());
    let resp = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()?
        .get(url)
        .send()
        .with_context(|| format!("GET {}", url))?;
    if !resp.status().is_success() {
        return Err(anyhow!("download failed {}: {}", url, resp.status()));
    }
    let bytes = resp.bytes()?;
    let tmp = dest.with_extension("part");
    let mut f = fs::File::create(&tmp)?;
    f.write_all(&bytes)?;
    f.sync_all()?;
    drop(f);
    fs::rename(&tmp, dest)?;
    Ok(())
}

pub fn download_models(app: &AppHandle) -> Result<()> {
    let seg = segmentation_path(app)?;
    let emb = embedding_path(app)?;
    if !seg.exists() {
        download_to(SEG_URL, &seg)?;
    }
    if !emb.exists() {
        download_to(EMB_URL, &emb)?;
    }
    Ok(())
}
