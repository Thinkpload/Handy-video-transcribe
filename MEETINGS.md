# Meetings — Long-form Video Transcription with Diarization

This document describes the **Meetings** feature added to the fork
`thinkpload/handy-video-transcribe`. It is not part of upstream Handy. Upstream
Handy is a real-time dictation tool; this fork adds a separate workflow for
transcribing long, pre-recorded meeting videos with speaker labels.

---

## What it does

Pick a meeting video (or audio) file → get a timed transcript split by
speaker, exportable in five formats and stored locally for later.

End-to-end pipeline:

```
video/audio file
   │
   ▼  ffmpeg sidecar (system binary, streaming stdout)
16 kHz mono f32 PCM
   │
   ▼  silence-aware chunking (~28 s windows, max 35 s)
audio chunks
   │
   ▼  TranscriptionManager (existing Whisper pipeline, language-aware)
ASR segments with timestamps
   │
   ▼  pyannote ONNX diarization on the full waveform
       (sliding-window segmentation → powerset decoding →
        wespeaker embeddings → agglomerative clustering)
speaker turns
   │
   ▼  assign Speaker N to each ASR segment by max time overlap
   │
   ▼  persist row in meetings.db
final meeting result (segments, speakers, duration)
```

Speaker labels gracefully degrade: when the diarization ONNX models are not
installed, a heuristic (new speaker on long silence gaps) is used. Everything
else still works.

---

## Requirements

| Component             | Where it comes from                                              |
| --------------------- | ---------------------------------------------------------------- |
| `ffmpeg`              | **System binary on PATH.** Install with `brew install ffmpeg`, `apt install ffmpeg`, or download from [ffmpeg.org](https://ffmpeg.org/download.html). Not bundled. |
| Whisper model         | Same flow as upstream Handy — pick a multilingual model in Models tab (Small / Medium / Turbo / Large). English-only models will not work on non-English meetings. |
| Diarization models    | Downloaded on demand from the Meetings tab (~30 MB total). See "Diarization models" below. |

The first run will prompt you to download diarization models. Skipping that is
fine — the heuristic labeler will be used.

---

## Using it

1. Open the **Meetings** tab in the sidebar.
2. (Optional) Click **Download (~30 MB)** to fetch diarization models.
3. Pick a **Language** — `Auto-detect` or one of the explicit options.
   Selecting an explicit language is recommended for long recordings because
   Whisper occasionally drifts on silent/musical sections under `auto`.
4. Optionally set **Speakers** (0 = auto-detect cluster count).
5. Click **Choose video or audio…** and pick a file. Supported by ffmpeg
   defaults: `mp4`, `mkv`, `mov`, `webm`, `avi`, `m4a`, `mp3`, `wav`, `flac`, `ogg`.
6. Watch the progress bar — stages are `extract → transcribe → diarize`.
7. When done, the transcript appears below, grouped per segment with speaker
   label and clickable timestamp. Use the format buttons (TXT / SRT / VTT /
   MD / JSON) to export.

The meeting is **auto-saved** to `meetings.db` when transcription succeeds.
Saved meetings appear in the list above the transcript view; click to open,
✕ to delete.

---

## Export formats

| Format | Use case                                          |
| ------ | ------------------------------------------------- |
| TXT    | Plain reading, paste into notes                   |
| SRT    | Subtitle file for video editors / VLC             |
| VTT    | Web-friendly subtitles (`<v Speaker>…` markup)    |
| MD     | Markdown, segments collapsed per consecutive speaker, `### Speaker [HH:MM]` headers |
| JSON   | Raw `MeetingSegment[]` for downstream tooling     |

Export is client-side — the buttons trigger browser downloads via blob URLs.

---

## Diarization models

The Meetings tab downloads two ONNX files into `<app_data>/models/diarization/`:

| File                   | Source (default URL)                                                                                                                | Notes |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ----- |
| `segmentation.onnx`    | `huggingface.co/onnx-community/pyannote-segmentation-3.0`                                                                            | 10 s windows, output `[B, 589, 7]` powerset logits (silence + 3 speakers, up to 2 simultaneous). |
| `embedding.onnx`       | `huggingface.co/deepghs/pyannote-wespeaker-voxceleb-resnet34-LM`                                                                     | Raw 16 kHz waveform → 256-dim embedding. Language-agnostic. |

If those URLs 404 or you want to swap models, drop replacement files at the
same paths and restart. Names of input/output tensors are read at session-load
time, so most ONNX variants of the same architecture will work without code
changes. If a model uses a different frame count per 10 s window, edit
`SEG_NUM_FRAMES` in `src-tauri/src/audio_toolkit/diarization/mod.rs`.

### How the diarization pipeline works

Implemented in `src-tauri/src/audio_toolkit/diarization/`:

1. **Sliding-window segmentation** — 10 s windows, 1 s step, applied across
   the whole audio. For each window the ONNX model returns 7-class powerset
   logits over ~589 frames.
2. **Powerset → multilabel** — softmax over classes, then summing the
   probability of each class that contains speaker *k* gives a per-frame
   activity for up to 3 local speakers.
3. **Overlap-add stitching** — windows overlap by 9 s. Local speaker
   channels in a new window are matched to global channels in the accumulator
   by brute-forcing all 6 permutations and picking the one with maximum
   activity overlap on the shared frames. Activations are then averaged.
4. **Turn extraction** — per global-speaker channel, hysteresis thresholds
   (onset 0.5, offset 0.2) produce contiguous speech turns ≥0.25 s.
5. **Embedding** — each turn's audio is fed to the wespeaker model, L2-
   normalised → 256-dim cosine-comparable vector.
6. **Agglomerative clustering** — single linkage on cosine distance, with
   either a fixed *k* (user-selected speaker count) or a default 0.7155
   threshold. Labels are densely remapped to `0..K`.
7. **Merging** — adjacent same-speaker segments within 0.5 s are merged.
8. **Assigning to ASR** — each Whisper segment gets the diarization speaker
   with maximum time overlap.

### Known quality issues

- **Single linkage** can over-merge when there are short cross-talk turns.
  If your meetings have heavy interruption patterns, switching to average or
  complete linkage may help (`cluster.rs`).
- **Powerset model** caps simultaneous speakers at 2 within a 10 s window;
  rare 3+ way overlap will be dropped to silence.
- **Auto cluster count** uses a fixed threshold derived from voxceleb-like
  data; if you know the exact number of speakers, set it explicitly.
- The pipeline has been written carefully but **not run end-to-end during
  development** (the build environment lacks the native deps and network
  access to fetch ONNX runtime). Expect to validate locally and tune.

---

## Persistence

| Path                                        | Contents                                                  |
| ------------------------------------------- | --------------------------------------------------------- |
| `<app_data>/meetings.db`                    | SQLite. One row per meeting; segments stored as JSON.     |
| `<app_data>/models/diarization/*.onnx`      | Downloaded diarization models.                            |

`<app_data>` resolves via `crate::portable::app_data_dir()` — same rules as
the rest of Handy (portable mode → `~/Handy-data`, otherwise platform default).

Schema (`meetings_store.rs`):

```sql
CREATE TABLE meetings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    INTEGER NOT NULL,    -- unix seconds
    source_path   TEXT    NOT NULL,    -- original picked file path
    file_name     TEXT    NOT NULL,    -- basename, used as display name
    language      TEXT    NOT NULL,    -- "ru", "en", "auto", …
    duration_secs REAL    NOT NULL,
    segments_json TEXT    NOT NULL     -- serde_json of Vec<MeetingSegment>
);
```

Migrations follow the same `rusqlite_migration` pattern as the existing
`history.db`.

---

## Architecture map

### Rust (`src-tauri/src/`)

| File                                              | Role                                                                 |
| ------------------------------------------------- | -------------------------------------------------------------------- |
| `managers/meetings.rs`                            | Top-level pipeline: ffmpeg extract, chunking, transcribe, diarize.   |
| `managers/meetings_store.rs`                      | SQLite persistence for completed meetings.                           |
| `managers/diarization_models.rs`                  | Resolves paths under `<app_data>/models/diarization/` and downloads. |
| `audio_toolkit/diarization/mod.rs`                | `Diarizer`: ONNX inference, stitching, embedding extraction.         |
| `audio_toolkit/diarization/cluster.rs`            | Single-linkage agglomerative clustering with cosine distance.        |
| `commands/meetings.rs`                            | Tauri commands exposed to the frontend.                              |

### Tauri commands

| Command                          | Notes                                                              |
| -------------------------------- | ------------------------------------------------------------------ |
| `check_ffmpeg_available`         | Returns `bool`. Used to render the install hint.                   |
| `diarization_models_present`     | Returns `bool`.                                                    |
| `download_diarization_models`    | Async. Downloads the two ONNX files if missing.                    |
| `transcribe_meeting_video`       | Async. Runs the full pipeline; auto-saves to `meetings.db`.        |
| `list_meetings`                  | `Vec<MeetingSummary>` ordered newest first.                        |
| `get_meeting`                    | `Option<StoredMeeting>` by id.                                     |
| `delete_meeting`                 | Removes the row.                                                   |

### Events

| Event              | Payload          | When                                                |
| ------------------ | ---------------- | --------------------------------------------------- |
| `meeting-progress` | `MeetingProgress` | Stage transitions and ~1s ticks during extract/transcribe/diarize. |

### Frontend (`src/`)

| File                                                   | Role                                                       |
| ------------------------------------------------------ | ---------------------------------------------------------- |
| `components/Sidebar.tsx`                               | Adds the `meetings` section to `SECTIONS_CONFIG`.          |
| `components/settings/meetings/MeetingsSettings.tsx`    | The Meetings page: picker, progress, saved list, exports.  |
| `components/settings/meetings/exporters.ts`            | Pure TS export functions for TXT/SRT/VTT/MD/JSON.          |
| `i18n/locales/en/translation.json`                     | Translation keys under `meetings.*` and `sidebar.meetings`. |
| `bindings.ts`                                          | Auto-generated by `tauri-specta` on `cargo build`. Includes the new commands and types. |

---

## Dependencies added to this fork

```toml
ort     = { version = "=2.0.0-rc.10", default-features = false, features = ["ndarray"] }
ndarray = "0.16"
reqwest = { version = "0.12", features = ["json", "stream", "blocking"] }   # blocking feature added
```

`ort` is the ONNX Runtime binding. It pulls `ort-sys`, which downloads
prebuilt ONNX Runtime binaries during build. If your network blocks
`cdn.pyke.io`, set `ORT_STRATEGY=system` and install ONNX Runtime as a
system package, or vendor the prebuilt archive locally and point `ort` at
it. The pinned version (`=2.0.0-rc.10`) must match whatever `transcribe-rs`
brings in — bump if Cargo complains about version unification.

---

## Limitations and known gaps

1. **Diarization quality is unvalidated.** The pipeline was written without
   end-to-end runs. Expect to tune thresholds and possibly switch to average
   linkage on real data.
2. **Memory.** The full waveform is held in RAM because diarization needs
   the whole audio. A 2-hour meeting is ~460 MB of f32. Multi-hour recordings
   may be a problem on low-RAM machines. Streaming diarization is a possible
   future improvement.
3. **No cancellation** of an in-flight transcription job. The only way to
   stop a 2-hour job in progress is to quit the app.
4. **ffmpeg is not bundled.** Cross-platform sidecar packaging is a separate
   piece of work (involves `tauri.conf.json` externalBin entries and CI
   plumbing).
5. **No edit / merge speakers UI.** If the cluster count is wrong you re-run
   the transcription with an explicit `Speakers` value rather than fixing
   labels by hand.
6. **i18n.** Only English strings were added for the new feature. Other
   locales fall back to English.
7. **Real-time dictation flow is untouched.** All upstream Handy features
   continue to work; Meetings lives in its own tab and uses its own SQLite
   database and storage location.

---

## Extending

Common extensions, in rough order of cost:

- **Tune diarization thresholds** — `extract_turns(onset, offset)` and the
  agglomerative `threshold` in `cluster.rs`.
- **Switch clustering** — replace `cluster::agglomerative` with average /
  complete linkage or spectral clustering.
- **Add a new export format** — implement `to<Format>(segments)` in
  `exporters.ts` and add a button in `MeetingsSettings.tsx`.
- **Bundle ffmpeg as a Tauri sidecar** — see Tauri docs on `externalBin`.
  Add per-platform binaries under `src-tauri/binaries/`, update
  `tauri.conf.json`, and replace `Command::new("ffmpeg")` with the
  sidecar-aware launcher.
- **Cancellation** — add an `AtomicBool` keyed by `job_id` into a shared
  registry, check it between chunks in `transcribe_video`, expose a
  `cancel_meeting` command.
- **Search saved meetings** — `segments_json` already contains the text;
  either filter client-side over `list_meetings` or add an FTS5 virtual
  table mirroring the segments.
