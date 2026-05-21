use std::path::PathBuf;
use std::sync::Arc;
use tauri::AppHandle;

use crate::managers::meetings::{self, MeetingResult};
use crate::managers::transcription::TranscriptionManager;

#[tauri::command]
#[specta::specta]
pub fn check_ffmpeg_available() -> bool {
    meetings::ffmpeg_available()
}

#[tauri::command]
#[specta::specta]
pub async fn transcribe_meeting_video(
    app: AppHandle,
    job_id: String,
    video_path: String,
) -> Result<MeetingResult, String> {
    let path = PathBuf::from(&video_path);
    if !path.exists() {
        return Err(format!("File not found: {}", video_path));
    }
    let transcription = {
        let state = tauri::Manager::state::<Arc<TranscriptionManager>>(&app);
        state.inner().clone()
    };

    tauri::async_runtime::spawn_blocking(move || {
        meetings::transcribe_video(&app, &transcription, job_id, &path)
    })
    .await
    .map_err(|e| format!("join error: {}", e))?
    .map_err(|e| e.to_string())
}
