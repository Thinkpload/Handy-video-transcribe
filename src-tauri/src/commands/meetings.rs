use std::path::PathBuf;
use std::sync::Arc;
use tauri::AppHandle;

use crate::managers::diarization_models;
use crate::managers::meeting_jobs::MeetingJobs;
use crate::managers::meetings::{self, MeetingResult};
use crate::managers::meetings_store::{MeetingSummary, MeetingsStore, StoredMeeting};
use crate::managers::transcription::TranscriptionManager;
use crate::settings::get_settings;

#[tauri::command]
#[specta::specta]
pub fn check_ffmpeg_available() -> bool {
    meetings::ffmpeg_available()
}

#[tauri::command]
#[specta::specta]
pub fn diarization_models_present(app: AppHandle) -> bool {
    diarization_models::models_present(&app)
}

#[tauri::command]
#[specta::specta]
pub async fn download_diarization_models(app: AppHandle) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || diarization_models::download_models(&app))
        .await
        .map_err(|e| format!("join error: {}", e))?
        .map_err(|e| e.to_string())
}

#[tauri::command]
#[specta::specta]
pub async fn transcribe_meeting_video(
    app: AppHandle,
    job_id: String,
    video_path: String,
    num_speakers: Option<u32>,
) -> Result<MeetingResult, String> {
    let path = PathBuf::from(&video_path);
    if !path.exists() {
        return Err(format!("File not found: {}", video_path));
    }
    let transcription = {
        let state = tauri::Manager::state::<Arc<TranscriptionManager>>(&app);
        state.inner().clone()
    };
    let store = tauri::Manager::try_state::<Arc<MeetingsStore>>(&app).map(|s| s.inner().clone());
    let num = num_speakers.map(|n| n as usize).filter(|n| *n > 0);
    let language = get_settings(&app).selected_language.clone();
    let file_name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| video_path.clone());
    let source_path = video_path.clone();

    let jobs = tauri::Manager::state::<Arc<MeetingJobs>>(&app)
        .inner()
        .clone();
    let cancel = jobs.register(&job_id);

    let app_for_job = app.clone();
    let path_for_job = path.clone();
    let job_id_for_job = job_id.clone();
    let result_res: Result<MeetingResult, String> =
        tauri::async_runtime::spawn_blocking(move || {
            meetings::transcribe_video(
                &app_for_job,
                &transcription,
                job_id_for_job,
                &path_for_job,
                num,
                cancel,
            )
        })
        .await
        .map_err(|e| format!("join error: {}", e))?
        .map_err(|e| e.to_string());

    jobs.done(&job_id);
    let result = result_res?;

    if let Some(store) = store {
        if let Err(e) = store.save(&source_path, &file_name, &language, &result) {
            log::error!("Failed to persist meeting: {}", e);
        }
    }
    Ok(result)
}

#[tauri::command]
#[specta::specta]
pub fn cancel_meeting_job(app: AppHandle, job_id: String) -> bool {
    tauri::Manager::state::<Arc<MeetingJobs>>(&app).cancel(&job_id)
}

#[tauri::command]
#[specta::specta]
pub fn list_meetings(app: AppHandle) -> Result<Vec<MeetingSummary>, String> {
    let store = tauri::Manager::state::<Arc<MeetingsStore>>(&app);
    store.list().map_err(|e| e.to_string())
}

#[tauri::command]
#[specta::specta]
pub fn get_meeting(app: AppHandle, id: i64) -> Result<Option<StoredMeeting>, String> {
    let store = tauri::Manager::state::<Arc<MeetingsStore>>(&app);
    store.get(id).map_err(|e| e.to_string())
}

#[tauri::command]
#[specta::specta]
pub fn delete_meeting(app: AppHandle, id: i64) -> Result<(), String> {
    let store = tauri::Manager::state::<Arc<MeetingsStore>>(&app);
    store.delete(id).map_err(|e| e.to_string())
}
