//! Registry of in-flight meeting transcription jobs, used for cooperative
//! cancellation. Each running job registers an `AtomicBool` cancel flag keyed
//! by its `job_id`; the pipeline polls the flag between chunks / windows.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Default)]
pub struct MeetingJobs {
    flags: Mutex<HashMap<String, Arc<AtomicBool>>>,
}

impl MeetingJobs {
    pub fn register(&self, job_id: &str) -> Arc<AtomicBool> {
        let flag = Arc::new(AtomicBool::new(false));
        self.flags
            .lock()
            .unwrap()
            .insert(job_id.to_string(), flag.clone());
        flag
    }

    pub fn cancel(&self, job_id: &str) -> bool {
        if let Some(flag) = self.flags.lock().unwrap().get(job_id) {
            flag.store(true, Ordering::SeqCst);
            true
        } else {
            false
        }
    }

    pub fn done(&self, job_id: &str) {
        self.flags.lock().unwrap().remove(job_id);
    }
}
