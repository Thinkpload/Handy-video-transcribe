//! Speaker diarization using pyannote ONNX models.
//!
//! Pipeline:
//!   1. Sliding-window inference of pyannote/segmentation-3.0 over the full audio.
//!      Each 10s window outputs a [num_frames, 7] tensor of powerset class logits.
//!   2. Powerset → multilabel decoding into per-frame activity for up to 3 local speakers.
//!   3. Stitch overlapping windows by greedy Hungarian-style local-speaker matching
//!      (overlap-add of binary activations).
//!   4. Extract contiguous turns per local speaker.
//!   5. Embed each turn with a wespeaker-style ONNX model (raw 16kHz waveform → 256-dim).
//!   6. Agglomerative clustering (cosine distance) → global speaker IDs.
//!
//! ⚠️  This module is written against the ONNX exports:
//!     - onnx-community/pyannote-segmentation-3.0  (input "input_values": [B,1,160000], output "logits": [B,589,7])
//!     - a wespeaker ONNX taking raw waveform [B, N] → [B, 256]
//!
//!     I/O tensor names and exact frame counts vary between exports — adjust
//!     `SEG_INPUT_NAME` / `EMB_INPUT_NAME` / `SEG_NUM_FRAMES` if your models differ.
//!     Run `Diarizer::new` once with a real model to check shapes via the logs.

use anyhow::{anyhow, Context, Result};
use ndarray::{Array1, Array2, Array3, ArrayView2, Axis};
use ort::session::{builder::GraphOptimizationLevel, Session};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

pub mod cluster;

pub const SAMPLE_RATE: usize = 16_000;
const SEG_WINDOW_SECS: f32 = 10.0;
const SEG_STEP_SECS: f32 = 1.0;
const SEG_WINDOW_SAMPLES: usize = (SEG_WINDOW_SECS as usize) * SAMPLE_RATE;
const SEG_STEP_SAMPLES: usize = (SEG_STEP_SECS as usize) * SAMPLE_RATE;
/// Empirical for pyannote/segmentation-3.0 ONNX export. Adjust if your model
/// outputs a different number of frames per 10s window.
const SEG_NUM_FRAMES: usize = 589;
const MAX_LOCAL_SPEAKERS: usize = 3;
const EMB_DIM_HINT: usize = 256;

/// Powerset classes for pyannote-3.0 (silence + 3 speakers, up to 2 simultaneous).
/// Index → set of active local speaker indices.
const POWERSET: [&[usize]; 7] = [&[], &[0], &[1], &[2], &[0, 1], &[0, 2], &[1, 2]];

#[derive(Debug, Clone)]
pub struct DiarSegment {
    pub start: f32,
    pub end: f32,
    /// Global cluster id, 0-indexed.
    pub speaker: usize,
}

pub struct Diarizer {
    segmentation: Session,
    embedding: Session,
    seg_input_name: String,
    seg_output_name: String,
    emb_input_name: String,
    emb_output_name: String,
}

impl Diarizer {
    pub fn new(seg_path: &Path, emb_path: &Path) -> Result<Self> {
        let segmentation = Session::builder()
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .commit_from_file(seg_path)
            .with_context(|| format!("loading segmentation model {}", seg_path.display()))?;
        let embedding = Session::builder()
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .commit_from_file(emb_path)
            .with_context(|| format!("loading embedding model {}", emb_path.display()))?;

        let seg_input_name = segmentation.inputs()[0].name().to_string();
        let seg_output_name = segmentation.outputs()[0].name().to_string();
        let emb_input_name = embedding.inputs()[0].name().to_string();
        let emb_output_name = embedding.outputs()[0].name().to_string();

        log::info!(
            "Diarizer loaded: seg in={} out={}, emb in={} out={}",
            seg_input_name,
            seg_output_name,
            emb_input_name,
            emb_output_name
        );

        Ok(Self {
            segmentation,
            embedding,
            seg_input_name,
            seg_output_name,
            emb_input_name,
            emb_output_name,
        })
    }

    pub fn diarize(
        &mut self,
        samples: &[f32],
        num_speakers: Option<usize>,
        cancel: &AtomicBool,
    ) -> Result<Vec<DiarSegment>> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }

        // 1+2+3: run sliding-window segmentation and stitch into a global activity
        //         matrix [total_frames, MAX_LOCAL_SPEAKERS].
        let (activity, frame_dur) = self.run_segmentation(samples, cancel)?;

        // 4: extract contiguous turns per local-speaker channel.
        let turns = extract_turns(&activity, frame_dur, 0.5, 0.2);
        log::info!("diarization: {} raw turns", turns.len());
        if turns.is_empty() {
            return Ok(Vec::new());
        }

        // 5: embeddings per turn.
        let mut embeddings: Vec<Array1<f32>> = Vec::with_capacity(turns.len());
        for t in &turns {
            if cancel.load(Ordering::Relaxed) {
                return Err(anyhow!("Cancelled"));
            }
            let s = (t.start * SAMPLE_RATE as f32) as usize;
            let e = ((t.end * SAMPLE_RATE as f32) as usize).min(samples.len());
            if e <= s + SAMPLE_RATE / 4 {
                // <0.25s — too short to embed reliably; reuse previous or zero.
                embeddings.push(Array1::zeros(EMB_DIM_HINT));
                continue;
            }
            let emb = self.embed(&samples[s..e])?;
            embeddings.push(emb);
        }

        // 6: cluster.
        let labels = cluster::agglomerative(&embeddings, num_speakers, 0.7155);

        let mut segs: Vec<DiarSegment> = turns
            .into_iter()
            .zip(labels.into_iter())
            .map(|(t, l)| DiarSegment {
                start: t.start,
                end: t.end,
                speaker: l,
            })
            .collect();

        merge_adjacent(&mut segs, 0.5);
        Ok(segs)
    }

    /// Run segmentation across the audio with overlap-add stitching.
    /// Returns (activity matrix [total_frames, MAX_LOCAL_SPEAKERS], frame_duration_secs).
    fn run_segmentation(
        &mut self,
        samples: &[f32],
        cancel: &AtomicBool,
    ) -> Result<(Array2<f32>, f32)> {
        let frame_dur = SEG_WINDOW_SECS / SEG_NUM_FRAMES as f32;
        let frames_per_step = (SEG_STEP_SAMPLES as f32 / SAMPLE_RATE as f32 / frame_dur) as usize;
        let total_secs = samples.len() as f32 / SAMPLE_RATE as f32;
        let total_frames = ((total_secs / frame_dur).ceil() as usize).max(1);

        let mut accum: Array2<f32> = Array2::zeros((total_frames, MAX_LOCAL_SPEAKERS));
        let mut counts: Array1<f32> = Array1::zeros(total_frames);

        let mut start = 0usize;
        while start < samples.len() {
            if cancel.load(Ordering::Relaxed) {
                return Err(anyhow!("Cancelled"));
            }
            let end = (start + SEG_WINDOW_SAMPLES).min(samples.len());
            // Pad the window with zeros if shorter than expected.
            let mut window = vec![0.0f32; SEG_WINDOW_SAMPLES];
            window[..end - start].copy_from_slice(&samples[start..end]);

            let input = Array3::from_shape_vec((1, 1, SEG_WINDOW_SAMPLES), window)?;
            let input_value = ort::value::Tensor::<f32>::from_array(input)?;
            let outputs = self.segmentation.run(ort::inputs![
                self.seg_input_name.as_str() => input_value
            ])?;
            let logits = outputs[self.seg_output_name.as_str()]
                .try_extract_array::<f32>()?
                .into_dimensionality::<ndarray::Ix3>()?
                .to_owned();
            // shape: [1, num_frames, 7]
            let frames = logits.shape()[1];
            let classes = logits.shape()[2];
            if classes != POWERSET.len() {
                return Err(anyhow!(
                    "unexpected segmentation output classes: {} (expected {})",
                    classes,
                    POWERSET.len()
                ));
            }

            // softmax + powerset → multilabel
            let multilabel = powerset_to_multilabel(logits.index_axis(Axis(0), 0));

            // Match local-speaker channels in this window to global channels in `accum`
            // by binary overlap with the already-accumulated activations in the
            // overlapping region.
            let frame_offset = start / (SAMPLE_RATE / (SEG_NUM_FRAMES / SEG_WINDOW_SECS as usize));
            // Simpler: derive from time.
            let win_start_secs = start as f32 / SAMPLE_RATE as f32;
            let frame_offset = (win_start_secs / frame_dur).round() as usize;

            let perm = best_permutation(&accum, &counts, &multilabel, frame_offset);

            for f in 0..frames {
                let global_f = frame_offset + f;
                if global_f >= total_frames {
                    break;
                }
                for local_spk in 0..MAX_LOCAL_SPEAKERS {
                    let global_spk = perm[local_spk];
                    accum[(global_f, global_spk)] += multilabel[(f, local_spk)];
                }
                counts[global_f] += 1.0;
            }

            if end == samples.len() {
                break;
            }
            start += SEG_STEP_SAMPLES;
            // touch unused to silence compiler when feature-gated logging is off
            let _ = frames_per_step;
        }

        // Normalize by counts to get average activity.
        for f in 0..total_frames {
            let c = counts[f].max(1.0);
            for s in 0..MAX_LOCAL_SPEAKERS {
                accum[(f, s)] /= c;
            }
        }
        Ok((accum, frame_dur))
    }

    fn embed(&mut self, samples: &[f32]) -> Result<Array1<f32>> {
        // Many wespeaker ONNX exports want [batch, num_samples] f32 mono 16kHz.
        let input = Array2::from_shape_vec((1, samples.len()), samples.to_vec())?;
        let input_value = ort::value::Tensor::<f32>::from_array(input)?;
        let outputs = self.embedding.run(ort::inputs![
            self.emb_input_name.as_str() => input_value
        ])?;
        let emb = outputs[self.emb_output_name.as_str()]
            .try_extract_array::<f32>()?
            .into_dimensionality::<ndarray::Ix2>()?
            .to_owned();
        let v = emb.index_axis(Axis(0), 0).to_owned();
        Ok(l2_normalize(v))
    }
}

fn l2_normalize(mut v: Array1<f32>) -> Array1<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    v.mapv_inplace(|x| x / norm);
    v
}

/// Convert powerset logits [num_frames, 7] to per-speaker activity
/// [num_frames, MAX_LOCAL_SPEAKERS] via softmax + sum of classes containing speaker k.
fn powerset_to_multilabel(logits: ArrayView2<f32>) -> Array2<f32> {
    let n = logits.shape()[0];
    let mut out = Array2::<f32>::zeros((n, MAX_LOCAL_SPEAKERS));
    for f in 0..n {
        // softmax
        let row = logits.index_axis(Axis(0), f);
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|x| x / sum).collect();
        for (class_idx, members) in POWERSET.iter().enumerate() {
            for &spk in *members {
                out[(f, spk)] += probs[class_idx];
            }
        }
    }
    out
}

/// Find the local→global speaker permutation that maximises overlap with the
/// already-accumulated activity in the overlapping region.
fn best_permutation(
    accum: &Array2<f32>,
    counts: &Array1<f32>,
    window: &Array2<f32>,
    frame_offset: usize,
) -> [usize; MAX_LOCAL_SPEAKERS] {
    let frames = window.shape()[0];
    // Score matrix: score[local][global] = sum over overlap frames of
    //   (avg activity in accum[global]) * window[local].
    let mut score = [[0.0f32; MAX_LOCAL_SPEAKERS]; MAX_LOCAL_SPEAKERS];
    for f in 0..frames {
        let gf = frame_offset + f;
        if gf >= accum.shape()[0] || counts[gf] < 1.0 {
            continue;
        }
        for l in 0..MAX_LOCAL_SPEAKERS {
            let w = window[(f, l)];
            if w < 0.01 {
                continue;
            }
            for g in 0..MAX_LOCAL_SPEAKERS {
                let a = accum[(gf, g)] / counts[gf].max(1.0);
                score[l][g] += a * w;
            }
        }
    }

    // Brute-force best permutation (6 perms for 3 speakers).
    let perms: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let mut best = [0, 1, 2];
    let mut best_score = f32::NEG_INFINITY;
    for p in perms.iter() {
        let s = score[0][p[0]] + score[1][p[1]] + score[2][p[2]];
        if s > best_score {
            best_score = s;
            best = *p;
        }
    }
    best
}

#[derive(Debug, Clone)]
struct Turn {
    start: f32,
    end: f32,
    local_speaker: usize,
}

/// Extract contiguous turns per local-speaker channel.
/// `onset` / `offset` are hysteresis thresholds on the per-frame activity.
fn extract_turns(activity: &Array2<f32>, frame_dur: f32, onset: f32, offset: f32) -> Vec<Turn> {
    let mut turns = Vec::new();
    for s in 0..activity.shape()[1] {
        let mut active = false;
        let mut t_start = 0.0f32;
        for f in 0..activity.shape()[0] {
            let v = activity[(f, s)];
            let t = f as f32 * frame_dur;
            if !active && v > onset {
                active = true;
                t_start = t;
            } else if active && v < offset {
                active = false;
                if t - t_start >= 0.25 {
                    turns.push(Turn {
                        start: t_start,
                        end: t,
                        local_speaker: s,
                    });
                }
            }
        }
        if active {
            let t_end = activity.shape()[0] as f32 * frame_dur;
            if t_end - t_start >= 0.25 {
                turns.push(Turn {
                    start: t_start,
                    end: t_end,
                    local_speaker: s,
                });
            }
        }
    }
    turns.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap());
    turns
}

fn merge_adjacent(segs: &mut Vec<DiarSegment>, max_gap: f32) {
    if segs.is_empty() {
        return;
    }
    segs.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap());
    let mut out: Vec<DiarSegment> = Vec::with_capacity(segs.len());
    for s in segs.drain(..) {
        if let Some(last) = out.last_mut() {
            if last.speaker == s.speaker && s.start - last.end <= max_gap {
                last.end = s.end.max(last.end);
                continue;
            }
        }
        out.push(s);
    }
    *segs = out;
}

/// Public helper: given diarization segments and a list of (start, end) ASR
/// segments, assign a speaker label to each ASR segment by maximum overlap.
pub fn assign_speaker_to_asr(diar: &[DiarSegment], asr_start: f32, asr_end: f32) -> Option<usize> {
    let mut best: Option<(f32, usize)> = None;
    for d in diar {
        let ov = (asr_end.min(d.end) - asr_start.max(d.start)).max(0.0);
        if ov > 0.0 {
            match best {
                Some((b, _)) if b >= ov => {}
                _ => best = Some((ov, d.speaker)),
            }
        }
    }
    best.map(|(_, s)| s)
}
