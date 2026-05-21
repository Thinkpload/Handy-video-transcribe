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

use std::sync::atomic::{AtomicBool, Ordering};

use crate::audio_toolkit::diarization::{assign_speaker_to_asr, Diarizer};
use crate::managers::diarization_models;
use crate::managers::transcription::TranscriptionManager;

/// Returned by every cancellation check; mapped to a user-visible error.
fn cancelled() -> anyhow::Error {
    anyhow!("Cancelled")
}

pub fn is_cancelled_error(e: &anyhow::Error) -> bool {
    e.to_string() == "Cancelled"
}

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

/// Decode any media file to 16kHz mono f32 PCM via ffmpeg, streaming the
/// output as it arrives so we can report progress and abort early on cancel.
///
/// Diarization needs the full waveform so we still accumulate to a Vec, but
/// the read happens incrementally rather than blocking on `read_to_end`.
fn extract_pcm<F: FnMut(usize)>(
    input: &Path,
    cancel: &AtomicBool,
    mut on_progress: F,
) -> Result<Vec<f32>> {
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
    let mut samples: Vec<f32> = Vec::new();
    // 64 KiB == 16 384 f32 samples == ~1.024s at 16kHz: a good progress cadence.
    let mut buf = [0u8; 64 * 1024];
    let mut carry: [u8; 4] = [0; 4];
    let mut carry_len: usize = 0;
    let mut last_reported_s: usize = 0;

    loop {
        if cancel.load(Ordering::Relaxed) {
            let _ = child.kill();
            return Err(cancelled());
        }
        let n = stdout.read(&mut buf)?;
        if n == 0 {
            break;
        }
        let mut start = 0usize;
        if carry_len > 0 {
            let need = 4 - carry_len;
            let take = need.min(n);
            carry[carry_len..carry_len + take].copy_from_slice(&buf[..take]);
            carry_len += take;
            start = take;
            if carry_len == 4 {
                samples.push(f32::from_le_bytes(carry));
                carry_len = 0;
            }
        }
        let aligned_end = start + ((n - start) / 4) * 4;
        for c in buf[start..aligned_end].chunks_exact(4) {
            samples.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
        }
        let rem = n - aligned_end;
        if rem > 0 {
            carry[..rem].copy_from_slice(&buf[aligned_end..n]);
            carry_len = rem;
        }

        let now_s = samples.len() / SAMPLE_RATE as usize;
        if now_s > last_reported_s {
            on_progress(samples.len());
            last_reported_s = now_s;
        }
    }

    let status = child.wait()?;
    if !status.success() {
        let mut err = String::new();
        if let Some(mut e) = child.stderr.take() {
            let _ = e.read_to_string(&mut err);
        }
        return Err(anyhow!("ffmpeg failed: {}", err));
    }
    if carry_len != 0 {
        return Err(anyhow!("ffmpeg produced unaligned PCM output"));
    }
    on_progress(samples.len());
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

/// Fallback diarization when ONNX models are not present: rotate speaker
/// labels on long inter-segment gaps.
fn assign_speakers_heuristic(segments: &mut [MeetingSegment]) {
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

/// Run pyannote ONNX diarization on the full audio and assign speaker labels
/// to ASR segments by maximum time overlap.
fn assign_speakers_pyannote(
    app: &AppHandle,
    samples: &[f32],
    segments: &mut [MeetingSegment],
    num_speakers: Option<usize>,
    cancel: &AtomicBool,
) -> Result<()> {
    let seg_path = diarization_models::segmentation_path(app)?;
    let emb_path = diarization_models::embedding_path(app)?;
    let mut diar = Diarizer::new(&seg_path, &emb_path)?;
    let turns = diar.diarize(samples, num_speakers, cancel)?;
    log::info!("pyannote diarization produced {} turns", turns.len());
    for seg in segments.iter_mut() {
        let spk = assign_speaker_to_asr(&turns, seg.start, seg.end).unwrap_or(0);
        seg.speaker = format!("Speaker {}", spk + 1);
    }
    Ok(())
}

pub fn transcribe_video(
    app: &AppHandle,
    transcription: &Arc<TranscriptionManager>,
    job_id: String,
    video_path: &Path,
    num_speakers: Option<usize>,
    cancel: Arc<AtomicBool>,
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

    let samples = {
        let app_for_cb = app.clone();
        let job_id_for_cb = job_id.clone();
        extract_pcm(video_path, &cancel, |samples_so_far| {
            let secs = samples_so_far as f32 / SAMPLE_RATE as f32;
            let _ = app_for_cb.emit(
                "meeting-progress",
                MeetingProgress {
                    job_id: job_id_for_cb.clone(),
                    stage: "extract".into(),
                    processed_secs: secs,
                    total_secs: 0.0,
                },
            );
        })?
    };
    let total_secs = samples.len() as f32 / SAMPLE_RATE as f32;

    let bounds = find_chunk_bounds(&samples);
    let mut segments: Vec<MeetingSegment> = Vec::with_capacity(bounds.len());

    for (start, end) in bounds {
        if cancel.load(Ordering::Relaxed) {
            return Err(cancelled());
        }
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

    if diarization_models::models_present(app) {
        let _ = app.emit(
            "meeting-progress",
            MeetingProgress {
                job_id: job_id.clone(),
                stage: "diarize".into(),
                processed_secs: total_secs,
                total_secs,
            },
        );
        match assign_speakers_pyannote(app, &samples, &mut segments, num_speakers, &cancel) {
            Ok(()) => {}
            Err(e) if is_cancelled_error(&e) => return Err(e),
            Err(e) => {
                log::error!("pyannote diarization failed, falling back: {}", e);
                assign_speakers_heuristic(&mut segments);
            }
        }
    } else {
        assign_speakers_heuristic(&mut segments);
    }

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
