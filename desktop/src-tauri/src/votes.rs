//! Opt-in sending of blind A/B votes to jective.nz (maintainer, 2026-10-26). The endpoint
//! contract (the maintainer's "Eluate vote endpoint spec", kept outside the repo): POST JSON
//! {schema: 1, app, votes[1..50]} over HTTPS, answers 200 {accepted, duplicates, rejected},
//! 400 / 415 / 413 / 429 (Retry-After) / 5xx as read below; votes are deduplicated on `id`.
//!
//! The page queues a vote only after the user said yes to sharing (`queue_vote`); the
//! vote is appended to OUTBOX_FILE in the app's data folder, so votes made offline or
//! before a crash are sent later. One sender thread at a time posts the outbox in batches
//! of up to MAX_BATCH as `{schema, app, votes}` and reads the answer as the spec has it:
//! 200 clears the batch (accepted, duplicates and rejected alike; a 200 whose body names an
//! `error` is that error), 400 and 415 keep the votes and stop for this run, 413 halves the
//! batch, 429 waits Retry-After, a 5xx or no connection retries after BACKOFF_S. A vote is
//! one JSON object with a string `id` (the page's UUID); nothing in it names a file.
//! `clear_outbox` (sharing switched off) deletes what is still waiting.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{json, Value};
use tauri::{AppHandle, Manager, State};

pub const VOTE_URL: &str = "https://jective.nz/api/eluate/votes";
const OUTBOX_FILE: &str = "vote-outbox.jsonl";
const MAX_BATCH: usize = 50;
const VOTE_MAX_BYTES: usize = 4096;
const TIMEOUT_S: u64 = 15;
const BACKOFF_S: [u64; 4] = [60, 600, 3600, 86_400];
const RETRY_AFTER_MAX_S: u64 = 3600;

/// The outbox lock (queue and send never interleave a file write) and the one sender.
#[derive(Default)]
pub struct Outbox {
    file: Arc<Mutex<()>>,
    sending: Arc<AtomicBool>,
}

fn outbox_path(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir.join(OUTBOX_FILE))
}

/// A vote the outbox accepts: a JSON object under VOTE_MAX_BYTES with a string `id`.
fn check_vote(vote: &str) -> Result<Value, String> {
    if vote.len() > VOTE_MAX_BYTES {
        return Err("Vote too large".into());
    }
    let v: Value = serde_json::from_str(vote).map_err(|_| "Not a vote".to_string())?;
    match v.get("id").and_then(Value::as_str) {
        Some(id) if !id.is_empty() && v.is_object() => Ok(v),
        _ => Err("Not a vote".into()),
    }
}

/// The outbox's votes, oldest first, one per id (a line that does not parse is dropped).
fn read_outbox(p: &Path) -> Vec<Value> {
    let text = std::fs::read_to_string(p).unwrap_or_default();
    let mut seen = std::collections::HashSet::new();
    text.lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .filter(|v| v.get("id").and_then(Value::as_str).is_some_and(|id| seen.insert(id.to_string())))
        .collect()
}

/// Rewrites the outbox without the votes whose ids are in `sent` (atomic: temp + rename).
fn drop_sent(p: &Path, sent: &[String]) -> Result<(), String> {
    let keep: Vec<String> = read_outbox(p)
        .into_iter()
        .filter(|v| !sent.iter().any(|s| v.get("id").and_then(Value::as_str) == Some(s.as_str())))
        .map(|v| v.to_string())
        .collect();
    if keep.is_empty() {
        let _ = std::fs::remove_file(p);
        return Ok(());
    }
    let tmp = p.with_extension("jsonl.tmp");
    std::fs::write(&tmp, keep.join("\n") + "\n").map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, p).map_err(|e| e.to_string())
}

/// The request body for one batch (spec: Request body).
pub fn envelope(votes: &[Value]) -> Value {
    json!({
        "schema": 1,
        "app": { "name": "eluate", "version": env!("CARGO_PKG_VERSION"), "os": std::env::consts::OS },
        "votes": votes,
    })
}

/// What one POST came back as.
#[derive(Debug, PartialEq)]
pub enum Outcome {
    Sent,              // 200: drop the batch from the outbox
    Stop(String),      // 400, 415, 403: keep the votes, send nothing more this run
    TooLarge,          // 413: halve the batch
    Wait(u64),         // 429: seconds to wait
    Retry,             // 5xx, no connection, a timeout: back off
}

/// The spec's status codes, and a 200 whose body names an error (the Apps Script variant).
pub fn outcome(status: u16, body: &str, retry_after: Option<&str>) -> Outcome {
    let err = serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| v.get("error").and_then(Value::as_str).map(str::to_string));
    match (status, err.as_deref()) {
        (200, None) => Outcome::Sent,
        (200, Some("rate_limited")) | (429, _) => Outcome::Wait(
            retry_after.and_then(|s| s.trim().parse().ok()).unwrap_or(60).min(RETRY_AFTER_MAX_S),
        ),
        (200, Some("too_large")) | (413, _) => Outcome::TooLarge,
        (s, _) if s >= 500 => Outcome::Retry,
        (_, e) => Outcome::Stop(format!("{status} {}", e.unwrap_or(""))),
    }
}

/// POSTs one batch to `url`.
pub fn post(url: &str, votes: &[Value]) -> Outcome {
    let agent = ureq::AgentBuilder::new().timeout(Duration::from_secs(TIMEOUT_S)).build();
    let ua = format!("Eluate/{} ({})", env!("CARGO_PKG_VERSION"), std::env::consts::OS);
    match agent
        .post(url)
        .set("Content-Type", "application/json")
        .set("User-Agent", &ua)
        .send_string(&envelope(votes).to_string())
    {
        Ok(r) => {
            let status = r.status();
            let ra = r.header("Retry-After").map(str::to_string);
            outcome(status, &r.into_string().unwrap_or_default(), ra.as_deref())
        }
        Err(ureq::Error::Status(code, r)) => {
            let ra = r.header("Retry-After").map(str::to_string);
            outcome(code, &r.into_string().unwrap_or_default(), ra.as_deref())
        }
        Err(_) => Outcome::Retry,
    }
}

/// Sends the outbox until it is empty or the endpoint says stop; backs off on failures.
/// Runs on its own thread; a second call while one runs does nothing.
fn send_all(path: PathBuf, lock: Arc<Mutex<()>>, sending: Arc<AtomicBool>) {
    if sending.swap(true, Ordering::SeqCst) {
        return;
    }
    std::thread::spawn(move || {
        let mut batch = MAX_BATCH;
        let mut failures = 0usize;
        loop {
            let votes: Vec<Value> = {
                let _g = lock.lock().unwrap();
                read_outbox(&path).into_iter().take(batch).collect()
            };
            if votes.is_empty() {
                break;
            }
            match post(VOTE_URL, &votes) {
                Outcome::Sent => {
                    let ids: Vec<String> =
                        votes.iter().filter_map(|v| v.get("id").and_then(Value::as_str).map(str::to_string)).collect();
                    let _g = lock.lock().unwrap();
                    if drop_sent(&path, &ids).is_err() {
                        break;
                    }
                    failures = 0;
                    batch = MAX_BATCH;
                }
                Outcome::TooLarge if batch > 1 => batch = (batch / 2).max(1),
                Outcome::TooLarge => {
                    // one vote is far below the 64 KB limit; drop it rather than loop
                    let ids: Vec<String> =
                        votes.iter().filter_map(|v| v.get("id").and_then(Value::as_str).map(str::to_string)).collect();
                    let _g = lock.lock().unwrap();
                    let _ = drop_sent(&path, &ids);
                }
                Outcome::Wait(s) => std::thread::sleep(Duration::from_secs(s)),
                Outcome::Retry => {
                    std::thread::sleep(Duration::from_secs(BACKOFF_S[failures.min(BACKOFF_S.len() - 1)]));
                    failures += 1;
                }
                Outcome::Stop(_) => break,
            }
        }
        sending.store(false, Ordering::SeqCst);
    });
}

fn start(app: &AppHandle, outbox: &Outbox) -> Result<(), String> {
    send_all(outbox_path(app)?, outbox.file.clone(), outbox.sending.clone());
    Ok(())
}

/// Appends a vote (sharing is on) and starts sending.
#[tauri::command]
pub fn queue_vote(app: AppHandle, outbox: State<'_, Outbox>, vote: String) -> Result<(), String> {
    let v = check_vote(&vote)?;
    {
        let _g = outbox.file.lock().unwrap();
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(outbox_path(&app)?)
            .map_err(|e| e.to_string())?;
        writeln!(f, "{v}").map_err(|e| e.to_string())?;
    }
    start(&app, &outbox)
}

/// Sends whatever is waiting (the page calls it at start when sharing is on).
#[tauri::command]
pub fn flush_votes(app: AppHandle, outbox: State<'_, Outbox>) -> Result<(), String> {
    start(&app, &outbox)
}

/// Sharing switched off: what is still waiting is deleted, not sent.
#[tauri::command]
pub fn clear_outbox(app: AppHandle, outbox: State<'_, Outbox>) -> Result<(), String> {
    let _g = outbox.file.lock().unwrap();
    match std::fs::remove_file(outbox_path(&app)?) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e.to_string()),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("eluate-votes-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d.join(OUTBOX_FILE)
    }

    #[test]
    fn outcomes_follow_the_spec() {
        assert_eq!(outcome(200, r#"{"accepted":1,"duplicates":0,"rejected":[]}"#, None), Outcome::Sent);
        assert_eq!(outcome(200, r#"{"error":"bad_envelope"}"#, None), Outcome::Stop("200 bad_envelope".into()));
        assert!(matches!(outcome(400, r#"{"error":"bad_envelope"}"#, None), Outcome::Stop(_)));
        assert!(matches!(outcome(415, r#"{"error":"unsupported_media_type"}"#, None), Outcome::Stop(_)));
        assert_eq!(outcome(413, "", None), Outcome::TooLarge);
        assert_eq!(outcome(429, "", Some("120")), Outcome::Wait(120));
        assert_eq!(outcome(429, "", Some("999999")), Outcome::Wait(RETRY_AFTER_MAX_S));
        assert_eq!(outcome(429, "", None), Outcome::Wait(60));
        assert_eq!(outcome(503, "oops", None), Outcome::Retry);
    }

    #[test]
    fn outbox_dedupes_and_drops_what_was_sent() {
        let p = tmp("outbox");
        std::fs::write(&p, "{\"id\":\"a\",\"choice\":\"A\"}\n{\"id\":\"b\"}\nnot json\n{\"id\":\"a\"}\n").unwrap();
        let ids: Vec<_> = read_outbox(&p).iter().map(|v| v["id"].as_str().unwrap().to_string()).collect();
        assert_eq!(ids, ["a", "b"]);
        drop_sent(&p, &["a".to_string()]).unwrap();
        assert_eq!(read_outbox(&p).len(), 1);
        drop_sent(&p, &["b".to_string()]).unwrap();
        assert!(!p.exists());
    }

    #[test]
    fn a_vote_needs_an_id_and_a_size_limit() {
        assert!(check_vote(r#"{"id":"x","choice":"A"}"#).is_ok());
        assert!(check_vote(r#"{"choice":"A"}"#).is_err());
        assert!(check_vote(r#"{"id":""}"#).is_err());
        assert!(check_vote("[1]").is_err());
        assert!(check_vote(&format!(r#"{{"id":"x","p":"{}"}}"#, "a".repeat(VOTE_MAX_BYTES))).is_err());
        let e = envelope(&[json!({"id": "x"})]);
        assert_eq!(e["schema"], 1);
        assert_eq!(e["app"]["name"], "eluate");
        assert_eq!(e["votes"].as_array().unwrap().len(), 1);
    }

    /// Acceptance test 1 against the live endpoint: `cargo test -- --ignored live_endpoint`.
    /// Sends one vote marked `"test": true` with a fixed id, so a rerun is a duplicate.
    #[test]
    #[ignore]
    fn live_endpoint_acceptance() {
        let vote = json!({
            "id": "00000000-0000-4000-8000-0000000000a1", "test": true,
            "time": "2026-10-27T00:00:00.000Z", "choice": "A", "master": "B", "preferred": "premaster",
            "switches": 7, "listened_s": 94,
            "settings": { "loud": "-10.0", "tilt": "0.0", "preserve": false, "delivery": "master",
                          "character": "auto", "profile": "live", "bpm": null, "xo": null, "sideHp": null },
            "measures": { "loudness": -10.0, "true_peak": -0.3, "crest": { "before": 14.9, "after": 13.0 },
                          "lra": { "before": 4.7, "after": 3.4 }, "tilt": { "before": -4.77, "after": -4.84 } }
        });
        assert_eq!(post(VOTE_URL, &[vote]), Outcome::Sent);
    }
}
