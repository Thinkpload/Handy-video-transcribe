//! Long-form video meeting transcription pipeline.
//!
//! Pipeline: video file -> ffmpeg (16kHz mono f32 PCM) -> chunking on silence ->
//! Whisper transcription per chunk -> heuristic speaker labeling -> segments.
//!
//! Diarization here is a placeholder: it assigns speaker IDs based on long
//! silence gaps between utterances. The real solution is a pyannote ONNX
//! pipeline (segmentation + speaker embeddings + clustering); this module
//! exposes `assign_speakers` as the integration point.

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::Arc;
use tauri::{AppHandle, Emitter};

use crate::managers::transcription::TranscriptionManager;

pub const SAMPLE_RATE: u32 = 16_000;
const CHUNK_TARGET_SECS: f32 = 28.0;
const CHUNK_MAX_SECS: f32 = 35.0;
const SILENCE_RMS: f32 = 0.005;
const SILENCE_MIN_SECS: f32 = 0.6;
const NEW_SPEAKER_GAP_SECS: f32 = 1.5;

#[derive(Debug, Clone, Serialize, Deserialize, specta::Type)]
pub struct MeetingSegment {
    pub start: f32,
    pub end: f32,
    pub speaker: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, specta::Type)]
pub struct MeetingProgress {
    pub job_id: String,
    pub stage: String,
    pub processed_secs: f32,
    pub total_secs: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, specta::Type)]
pub struct MeetingResult {
    pub job_id: String,
    pub duration_secs: f32,
    pub segments: Vec<MeetingSegment>,
}

pub fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Decode any media file to 16kHz mono f32 PCM via ffmpeg.
fn extract_pcm(input: &Path) -> Result<Vec<f32>> {
    let mut child = Command::new("ffmpeg")
        .args(["-nostdin", "-hide_banner", "-loglevel", "error", "-i"])
        .arg(input)
        .args([
            "-ac",
            "1",
            "-ar",
            &SAMPLE_RATE.to_string(),
            "-f",
            "f32le",
            "-",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("Failed to launch ffmpeg. Is it installed and on PATH?")?;

    let mut stdout = child.stdout.take().ok_or_else(|| anyhow!("no stdout"))?;
    let mut buf = Vec::with_capacity(SAMPLE_RATE as usize * 4 * 600);
    stdout.read_to_end(&mut buf)?;
    let status = child.wait()?;
    if !status.success() {
        let mut err = String::new();
        if let Some(mut e) = child.stderr.take() {
            let _ = e.read_to_string(&mut err);
        }
        return Err(anyhow!("ffmpeg failed: {}", err));
    }

    if buf.len() % 4 != 0 {
        return Err(anyhow!("ffmpeg produced unaligned PCM output"));
    }
    let samples: Vec<f32> = buf
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok(samples)
}

/// Compute RMS energy per fixed-size frame.
fn frame_rms(samples: &[f32], frame: usize) -> Vec<f32> {
    samples
        .chunks(frame)
        .map(|c| {
            let sum: f32 = c.iter().map(|x| x * x).sum();
            (sum / c.len().max(1) as f32).sqrt()
        })
        .collect()
}

/// Find chunk boundaries (sample indices) that try to cut on silence near
/// CHUNK_TARGET_SECS and never exceed CHUNK_MAX_SECS.
fn find_chunk_bounds(samples: &[f32]) -> Vec<(usize, usize)> {
    let frame = (SAMPLE_RATE as f32 * 0.02) as usize; // 20ms
    let rms = frame_rms(samples, frame);
    let silence_min_frames = (SILENCE_MIN_SECS / 0.02) as usize;
    let target = (CHUNK_TARGET_SECS * SAMPLE_RATE as f32) as usize;
    let max = (CHUNK_MAX_SECS * SAMPLE_RATE as f32) as usize;

    let mut bounds = Vec::new();
    let mut start = 0usize;
    while start < samples.len() {
        let hard_end = (start + max).min(samples.len());
        if hard_end == samples.len() {
            bounds.push((start, hard_end));
            break;
        }
        // Search for a silence region whose center is closest to start+target.
        let search_from = (start + target / 2) / frame;
        let search_to = hard_end / frame;
        let mut best_cut: Option<usize> = None;
        let mut run = 0usize;
        for i in search_from..search_to {
            if rms.get(i).copied().unwrap_or(1.0) < SILENCE_RMS {
                run += 1;
                if run >= silence_min_frames {
                    best_cut = Some(i * frame);
                }
            } else if best_cut.is_some() {
                break;
            } else {
                run = 0;
            }
        }
        let cut = best_cut.unwrap_or(hard_end);
        bounds.push((start, cut));
        start = cut;
    }
    bounds
}

/// Placeholder diarization: assign a new speaker whenever there is a long
/// inter-segment gap. Real implementation should run pyannote segmentation
/// + speaker embedding + clustering on `samples` and assign speakers per
/// time interval, then merge with the transcribed segments.
fn assign_speakers(segments: &mut [MeetingSegment]) {
    let mut speaker_idx = 0u32;
    let mut prev_end = 0.0f32;
    for (i, seg) in segments.iter_mut().enumerate() {
        if i > 0 && (seg.start - prev_end) >= NEW_SPEAKER_GAP_SECS {
            speaker_idx = (speaker_idx + 1) % 4;
        }
        seg.speaker = format!("Speaker {}", speaker_idx + 1);
        prev_end = seg.end;
    }
}

pub fn transcribe_video(
    app: &AppHandle,
    transcription: &Arc<TranscriptionManager>,
    job_id: String,
    video_path: &Path,
) -> Result<MeetingResult> {
    if !ffmpeg_available() {
        return Err(anyhow!(
            "ffmpeg not found on PATH. Install ffmpeg to transcribe video files."
        ));
    }

    let _ = app.emit(
        "meeting-progress",
        MeetingProgress {
            job_id: job_id.clone(),
            stage: "extract".into(),
            processed_secs: 0.0,
            total_secs: 0.0,
        },
    );

    let samples = extract_pcm(video_path)?;
    let total_secs = samples.len() as f32 / SAMPLE_RATE as f32;

    let bounds = find_chunk_bounds(&samples);
    let mut segments: Vec<MeetingSegment> = Vec::with_capacity(bounds.len());

    for (start, end) in bounds {
        let start_secs = start as f32 / SAMPLE_RATE as f32;
        let end_secs = end as f32 / SAMPLE_RATE as f32;

        let _ = app.emit(
            "meeting-progress",
            MeetingProgress {
                job_id: job_id.clone(),
                stage: "transcribe".into(),
                processed_secs: start_secs,
                total_secs,
            },
        );

        let chunk = samples[start..end].to_vec();
        let text = match transcription.transcribe(chunk) {
            Ok(t) => t,
            Err(e) => {
                log::error!("chunk transcription failed at {:.1}s: {}", start_secs, e);
                String::new()
            }
        };
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            segments.push(MeetingSegment {
                start: start_secs,
                end: end_secs,
                speaker: String::new(),
                text: trimmed.to_string(),
            });
        }
    }

    assign_speakers(&mut segments);

    let _ = app.emit(
        "meeting-progress",
        MeetingProgress {
            job_id: job_id.clone(),
            stage: "done".into(),
            processed_secs: total_secs,
            total_secs,
        },
    );

    Ok(MeetingResult {
        job_id,
        duration_secs: total_secs,
        segments,
    })
}
