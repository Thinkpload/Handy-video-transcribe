//! Persistent storage for completed meeting transcriptions.
//!
//! Each meeting is stored as a row with its segments serialised as JSON.
//! Schema is kept intentionally simple — we don't anticipate complex queries.

use anyhow::Result;
use rusqlite::{params, Connection};
use rusqlite_migration::{Migrations, M};
use serde::{Deserialize, Serialize};
use specta::Type;
use std::path::PathBuf;
use std::sync::Mutex;
use tauri::AppHandle;

use crate::managers::meetings::{MeetingResult, MeetingSegment};

static MIGRATIONS: &[M] = &[M::up(
    "CREATE TABLE IF NOT EXISTS meetings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at INTEGER NOT NULL,
        source_path TEXT NOT NULL,
        file_name TEXT NOT NULL,
        language TEXT NOT NULL,
        duration_secs REAL NOT NULL,
        segments_json TEXT NOT NULL
    );",
)];

#[derive(Clone, Debug, Serialize, Deserialize, Type)]
pub struct MeetingSummary {
    pub id: i64,
    pub created_at: i64,
    pub file_name: String,
    pub source_path: String,
    pub language: String,
    pub duration_secs: f32,
    pub segment_count: i64,
    pub speaker_count: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize, Type)]
pub struct StoredMeeting {
    pub id: i64,
    pub created_at: i64,
    pub file_name: String,
    pub source_path: String,
    pub language: String,
    pub duration_secs: f32,
    pub segments: Vec<MeetingSegment>,
}

pub struct MeetingsStore {
    conn: Mutex<Connection>,
}

impl MeetingsStore {
    pub fn new(app: &AppHandle) -> Result<Self> {
        let db_path: PathBuf = crate::portable::app_data_dir(app)?.join("meetings.db");
        let mut conn = Connection::open(&db_path)?;
        Migrations::new(MIGRATIONS.to_vec()).to_latest(&mut conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    pub fn save(
        &self,
        source_path: &str,
        file_name: &str,
        language: &str,
        result: &MeetingResult,
    ) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        let segments_json = serde_json::to_string(&result.segments)?;
        let now = chrono::Utc::now().timestamp();
        conn.execute(
            "INSERT INTO meetings (created_at, source_path, file_name, language, duration_secs, segments_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![now, source_path, file_name, language, result.duration_secs, segments_json],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub fn list(&self) -> Result<Vec<MeetingSummary>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, created_at, file_name, source_path, language, duration_secs, segments_json
             FROM meetings ORDER BY created_at DESC",
        )?;
        let rows = stmt.query_map([], |row| {
            let segments_json: String = row.get(6)?;
            let segments: Vec<MeetingSegment> =
                serde_json::from_str(&segments_json).unwrap_or_default();
            let speaker_count = {
                let mut set = std::collections::HashSet::new();
                for s in &segments {
                    set.insert(s.speaker.clone());
                }
                set.len() as i64
            };
            Ok(MeetingSummary {
                id: row.get(0)?,
                created_at: row.get(1)?,
                file_name: row.get(2)?,
                source_path: row.get(3)?,
                language: row.get(4)?,
                duration_secs: row.get(5)?,
                segment_count: segments.len() as i64,
                speaker_count,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn get(&self, id: i64) -> Result<Option<StoredMeeting>> {
        let conn = self.conn.lock().unwrap();
        let row = conn
            .query_row(
                "SELECT id, created_at, file_name, source_path, language, duration_secs, segments_json
                 FROM meetings WHERE id = ?1",
                params![id],
                |row| {
                    let segments_json: String = row.get(6)?;
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, String>(4)?,
                        row.get::<_, f32>(5)?,
                        segments_json,
                    ))
                },
            )
            .ok();
        Ok(row.map(
            |(id, created_at, file_name, source_path, language, duration_secs, json)| {
                let segments: Vec<MeetingSegment> = serde_json::from_str(&json).unwrap_or_default();
                StoredMeeting {
                    id,
                    created_at,
                    file_name,
                    source_path,
                    language,
                    duration_secs,
                    segments,
                }
            },
        ))
    }

    pub fn delete(&self, id: i64) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute("DELETE FROM meetings WHERE id = ?1", params![id])?;
        Ok(())
    }
}
