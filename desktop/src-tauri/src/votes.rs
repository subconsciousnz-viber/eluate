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
//! `error` is that error), 413 halves the batch, 429 waits Retry-After, a 5xx or no
//! connection retries after BACKOFF_S. 400 and 415 (the votes refused) halve the batch down
//! to the one vote at fault, which is set aside in REJECTED_FILE once it has been refused
//! REJECT_AFTER times (counted across runs in REJECTS_FILE), so one bad vote cannot wedge
//! the queue; any other 4xx stops sending for this run.
//!
//! What leaves the machine is only what `wire_vote` builds (audit desktop #10): the shell
//! does not trust the page's JSON, it copies the known fields, type-checked, into a new
//! object (enums as enums, numbers rounded, the time cut to the minute) and drops anything
//! else, so a page bug or injected script cannot send a path or a file name through it.
//! The envelope adds the app's name, version and OS (`envelope`), the User-Agent repeats
//! version and OS. Nothing in a vote names a file.
//!
//! `clear_outbox` (sharing switched off) deletes what is still waiting (and the set-aside
//! votes) and moves the generation on: a sender stops at its next step (before a post,
//! or within a second of a wait) instead of sending what was read before. A POST already
//! on the wire (at most TIMEOUT_S) cannot be recalled.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{json, Map, Value};
use tauri::{AppHandle, Manager, State};

pub const VOTE_URL: &str = "https://jective.nz/api/eluate/votes";
const OUTBOX_FILE: &str = "vote-outbox.jsonl";
/// Votes the endpoint refused REJECT_AFTER times, kept (not sent) for the maintainer.
const REJECTED_FILE: &str = "vote-outbox.rejected.jsonl";
/// "<id> <count>": how often the vote at the head of the outbox was refused.
const REJECTS_FILE: &str = "vote-outbox.rejects";
const REJECT_AFTER: u32 = 3;
const MAX_BATCH: usize = 50;
const VOTE_MAX_BYTES: usize = 4096;
const TIMEOUT_S: u64 = 15;
const BACKOFF_S: [u64; 4] = [60, 600, 3600, 86_400];
const RETRY_AFTER_MAX_S: u64 = 3600;

/// The page's choices (ui/app.js castVote, keyOf; planner.CHARACTERS, USER_PROFILES).
const CHOICES: [&str; 3] = ["A", "B", "none"];
const SIDES: [&str; 2] = ["A", "B"];
const PREFERRED: [&str; 3] = ["master", "premaster", "none"];
const DELIVERIES: [&str; 2] = ["master", "streaming"];
const CHARACTERS: [&str; 6] = ["auto", "transparent", "slow", "moderate", "fast", "aggressive"];
const PROFILES: [&str; 3] = ["electronic", "live", "stems"];
const MAX_CROSSOVERS: usize = 8;

/// The outbox lock (queue and send never interleave a file write), the one sender, and
/// the generation `clear_outbox` moves on.
#[derive(Default)]
pub struct Outbox {
    file: Arc<Mutex<()>>,
    sending: Arc<AtomicBool>,
    generation: Arc<AtomicU64>,
}

fn outbox_path(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir.join(OUTBOX_FILE))
}

/// A vote the outbox accepts: a JSON object under VOTE_MAX_BYTES that `wire_vote` takes;
/// returns the wire form.
fn check_vote(vote: &str) -> Result<Value, String> {
    if vote.len() > VOTE_MAX_BYTES {
        return Err("Vote too large".into());
    }
    let v: Value = serde_json::from_str(vote).map_err(|_| "Not a vote".to_string())?;
    wire_vote(&v)
}

fn bad(field: &str) -> String {
    format!("Not a vote: {field}")
}

/// One of `allowed`, exactly.
fn one_of(v: Option<&Value>, allowed: &[&str], field: &str) -> Result<Value, String> {
    match v.and_then(Value::as_str) {
        Some(s) if allowed.contains(&s) => Ok(json!(s)),
        _ => Err(bad(field)),
    }
}

/// A lower-case UUID (8-4-4-4-12 hex digits).
fn uuid(v: Option<&Value>) -> Result<Value, String> {
    let s = v.and_then(Value::as_str).ok_or_else(|| bad("id"))?;
    let ok = s.len() == 36
        && s.bytes().enumerate().all(|(i, b)| {
            if matches!(i, 8 | 13 | 18 | 23) { b == b'-' } else { b.is_ascii_hexdigit() }
        });
    if ok { Ok(json!(s.to_ascii_lowercase())) } else { Err(bad("id")) }
}

/// A UTC ISO time (Date.toISOString) cut to the minute: "2026-10-27T13:05:00.000Z".
fn minute(v: Option<&Value>) -> Result<Value, String> {
    let s = v.and_then(Value::as_str).ok_or_else(|| bad("time"))?;
    let b = s.as_bytes();
    let shape = b.len() >= 17
        && b[..16].iter().enumerate().all(|(i, c)| match i {
            4 | 7 => *c == b'-',
            10 => *c == b'T',
            13 => *c == b':',
            _ => c.is_ascii_digit(),
        })
        && s.ends_with('Z');
    if shape { Ok(json!(format!("{}:00.000Z", &s[..16]))) } else { Err(bad("time")) }
}

/// A finite number (a numeric string too, as keyOf writes loud and tilt), or None.
fn number(v: &Value) -> Option<f64> {
    let x = match v {
        Value::Number(n) => n.as_f64()?,
        Value::String(s) => s.trim().parse().ok()?,
        _ => return None,
    };
    x.is_finite().then_some(x)
}

/// `v` rounded to `decimals` within [lo, hi]; null stays null (`nullable`).
fn rounded(v: Option<&Value>, decimals: i32, lo: f64, hi: f64, nullable: bool, field: &str) -> Result<Value, String> {
    match v {
        None | Some(Value::Null) if nullable => Ok(Value::Null),
        Some(v) => {
            let x = number(v).filter(|x| (lo..=hi).contains(x)).ok_or_else(|| bad(field))?;
            let k = 10f64.powi(decimals);
            let r = (x * k).round() / k;
            Ok(if decimals == 0 { json!(r as i64) } else { json!(r) })
        }
        None => Err(bad(field)),
    }
}

/// keyOf's one-decimal string ("-10.0") from a number or a numeric string.
fn decimal_string(v: Option<&Value>, lo: f64, hi: f64, field: &str) -> Result<Value, String> {
    match v {
        None | Some(Value::Null) => Ok(Value::Null),
        Some(v) => match number(v).filter(|x| (lo..=hi).contains(x)) {
            Some(x) => Ok(json!(format!("{x:.1}"))),
            None => Err(bad(field)),
        },
    }
}

/// The render's settings (keyOf): every field optional (null), typed when present.
fn settings(v: Option<&Value>) -> Result<Value, String> {
    let s = match v {
        None | Some(Value::Null) => return Ok(Value::Null),
        Some(Value::Object(s)) => s,
        Some(_) => return Err(bad("settings")),
    };
    let opt = |k: &str, allowed: &[&str]| match s.get(k) {
        None | Some(Value::Null) => Ok(Value::Null),
        v => one_of(v, allowed, k),
    };
    let preserve = match s.get("preserve") {
        None | Some(Value::Null) => Value::Null,
        Some(Value::Bool(b)) => json!(b),
        Some(_) => return Err(bad("preserve")),
    };
    let xo = match s.get("xo") {
        None | Some(Value::Null) => Value::Null,
        Some(Value::Array(a)) if a.len() <= MAX_CROSSOVERS => Value::Array(
            a.iter().map(|f| rounded(Some(f), 0, 1.0, 30_000.0, false, "xo")).collect::<Result<_, _>>()?,
        ),
        Some(_) => return Err(bad("xo")),
    };
    Ok(json!({
        "loud": decimal_string(s.get("loud"), -100.0, 100.0, "loud")?,
        "tilt": decimal_string(s.get("tilt"), -100.0, 100.0, "tilt")?,
        "preserve": preserve,
        "delivery": opt("delivery", &DELIVERIES)?,
        "character": opt("character", &CHARACTERS)?,
        "profile": opt("profile", &PROFILES)?,
        "bpm": rounded(s.get("bpm"), 2, 1.0, 1000.0, true, "bpm")?,
        "xo": xo,
        "sideHp": rounded(s.get("sideHp"), 0, 0.0, 30_000.0, true, "sideHp")?,
    }))
}

/// Eluate's level measurements (castVote's `measures`), rounded to 0.01.
fn measures(v: Option<&Value>) -> Result<Value, String> {
    let m = match v {
        None | Some(Value::Null) => return Ok(Value::Null),
        Some(Value::Object(m)) => m,
        Some(_) => return Err(bad("measures")),
    };
    let level = |v: Option<&Value>, field: &str| rounded(v, 2, -1000.0, 1000.0, true, field);
    let mut out = Map::new();
    out.insert("loudness".into(), level(m.get("loudness"), "loudness")?);
    out.insert("true_peak".into(), level(m.get("true_peak"), "true_peak")?);
    for k in ["crest", "lra", "tilt"] {
        let pair = match m.get(k) {
            None | Some(Value::Null) => Value::Null,
            Some(Value::Object(p)) => json!({
                "before": level(p.get("before"), k)?,
                "after": level(p.get("after"), k)?,
            }),
            Some(_) => return Err(bad(k)),
        };
        out.insert(k.into(), pair);
    }
    Ok(Value::Object(out))
}

/// The wire form of a vote: the known fields of `v`, type-checked, in a new object; any
/// other field is dropped. id, time, choice, master and preferred are required.
fn wire_vote(v: &Value) -> Result<Value, String> {
    let v = v.as_object().ok_or_else(|| "Not a vote".to_string())?;
    Ok(json!({
        "id": uuid(v.get("id"))?,
        "time": minute(v.get("time"))?,
        "choice": one_of(v.get("choice"), &CHOICES, "choice")?,
        "master": one_of(v.get("master"), &SIDES, "master")?,
        "preferred": one_of(v.get("preferred"), &PREFERRED, "preferred")?,
        "switches": rounded(v.get("switches"), 0, 0.0, 100_000.0, true, "switches")?,
        "listened_s": rounded(v.get("listened_s"), 0, 0.0, 86_400.0, true, "listened_s")?,
        "settings": settings(v.get("settings"))?,
        "measures": measures(v.get("measures"))?,
    }))
}

/// The outbox's votes in their wire form, oldest first, one per id (a line that does not
/// parse or is not a vote, written by an older version perhaps, is dropped).
fn read_outbox(p: &Path) -> Vec<Value> {
    let text = std::fs::read_to_string(p).unwrap_or_default();
    let mut seen = std::collections::HashSet::new();
    text.lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .filter_map(|v| wire_vote(&v).ok())
        .filter(|v| v.get("id").and_then(Value::as_str).is_some_and(|id| seen.insert(id.to_string())))
        .collect()
}

fn ids(votes: &[Value]) -> Vec<String> {
    votes.iter().filter_map(|v| v.get("id").and_then(Value::as_str).map(str::to_string)).collect()
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

/// Counts one more refusal of the vote `id` (REJECTS_FILE beside the outbox `p`) and
/// returns the count so far.
fn count_reject(p: &Path, id: &str) -> u32 {
    let f = p.with_file_name(REJECTS_FILE);
    let text = std::fs::read_to_string(&f).unwrap_or_default();
    let before = match text.trim().split_once(' ') {
        Some((i, n)) if i == id => n.parse().unwrap_or(0),
        _ => 0,
    };
    let n = before + 1;
    let _ = std::fs::write(&f, format!("{id} {n}\n"));
    n
}

/// Moves `votes` from the outbox `p` to REJECTED_FILE beside it.
fn set_aside(p: &Path, votes: &[Value]) -> Result<(), String> {
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(p.with_file_name(REJECTED_FILE))
        .map_err(|e| e.to_string())?;
    for v in votes {
        writeln!(f, "{v}").map_err(|e| e.to_string())?;
    }
    let _ = std::fs::remove_file(p.with_file_name(REJECTS_FILE));
    drop_sent(p, &ids(votes))
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
    Reject(String),    // 400, 415, a 200 naming another error: the votes were refused
    Stop(String),      // 401, 403 and any other 4xx: keep the votes, send nothing more this run
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
        (200 | 400 | 415, e) => Outcome::Reject(format!("{status} {}", e.unwrap_or(""))),
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

/// Waits `s` seconds; false as soon as the generation is no longer `gen` (sharing off).
fn pause(s: u64, generation: &AtomicU64, gen: u64) -> bool {
    for _ in 0..s {
        if generation.load(Ordering::SeqCst) != gen {
            return false;
        }
        std::thread::sleep(Duration::from_secs(1));
    }
    generation.load(Ordering::SeqCst) == gen
}

/// Sends the outbox at `path` with `post` until it is empty, the endpoint says stop or
/// sharing is switched off; backs off on failures.
fn send_loop(path: &Path, lock: &Mutex<()>, generation: &AtomicU64, mut post: impl FnMut(&[Value]) -> Outcome) {
    let mut batch = MAX_BATCH;
    let mut failures = 0usize;
    loop {
        let (gen, votes): (u64, Vec<Value>) = {
            let _g = lock.lock().unwrap();
            (generation.load(Ordering::SeqCst), read_outbox(path).into_iter().take(batch).collect())
        };
        if votes.is_empty() || generation.load(Ordering::SeqCst) != gen {
            break;
        }
        match post(&votes) {
            Outcome::Sent => {
                let _g = lock.lock().unwrap();
                if drop_sent(path, &ids(&votes)).is_err() {
                    break;
                }
                failures = 0;
                batch = MAX_BATCH;
            }
            Outcome::TooLarge | Outcome::Reject(_) if votes.len() > 1 => batch = (votes.len() / 2).max(1),
            Outcome::TooLarge => {
                // one vote is far below the 64 KB limit; drop it rather than loop
                let _g = lock.lock().unwrap();
                let _ = drop_sent(path, &ids(&votes));
            }
            Outcome::Reject(_) => {
                // the one vote at fault: set aside after REJECT_AFTER refusals, else
                // try again on a later run
                let _g = lock.lock().unwrap();
                if generation.load(Ordering::SeqCst) != gen
                    || count_reject(path, &ids(&votes).concat()) < REJECT_AFTER
                    || set_aside(path, &votes).is_err()
                {
                    break;
                }
                batch = MAX_BATCH;
            }
            Outcome::Wait(s) => {
                if !pause(s, generation, gen) {
                    break;
                }
            }
            Outcome::Retry => {
                if !pause(BACKOFF_S[failures.min(BACKOFF_S.len() - 1)], generation, gen) {
                    break;
                }
                failures += 1;
            }
            Outcome::Stop(_) => break,
        }
    }
}

/// Runs `send_loop` against VOTE_URL on its own thread; a second call while one runs does
/// nothing.
fn send_all(path: PathBuf, lock: Arc<Mutex<()>>, sending: Arc<AtomicBool>, generation: Arc<AtomicU64>) {
    if sending.swap(true, Ordering::SeqCst) {
        return;
    }
    std::thread::spawn(move || {
        send_loop(&path, &lock, &generation, |votes| post(VOTE_URL, votes));
        sending.store(false, Ordering::SeqCst);
    });
}

fn start(app: &AppHandle, outbox: &Outbox) -> Result<(), String> {
    send_all(outbox_path(app)?, outbox.file.clone(), outbox.sending.clone(), outbox.generation.clone());
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

/// Sharing switched off: what is still waiting is deleted, not sent, and a running sender
/// stops (the page awaits this before it shows sharing as off).
#[tauri::command]
pub fn clear_outbox(app: AppHandle, outbox: State<'_, Outbox>) -> Result<(), String> {
    let _g = outbox.file.lock().unwrap();
    outbox.generation.fetch_add(1, Ordering::SeqCst);
    clear_files(&outbox_path(&app)?)
}

/// Deletes the outbox `p` and the files beside it that hold votes or their counts.
fn clear_files(p: &Path) -> Result<(), String> {
    for f in [p.to_path_buf(), p.with_file_name(REJECTED_FILE), p.with_file_name(REJECTS_FILE)] {
        match std::fs::remove_file(f) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(e.to_string()),
            _ => {}
        }
    }
    Ok(())
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

    /// A vote as castVote makes it (`v` included: the page strips it, the shell drops it).
    fn page_vote(id: &str) -> Value {
        json!({
            "v": 1, "id": id, "time": "2026-10-27T13:05:42.123Z", "choice": "A", "master": "B",
            "preferred": "premaster", "switches": 7, "listened_s": 94,
            "settings": { "loud": "-10.0", "tilt": "0.0", "preserve": false, "delivery": "master",
                          "character": "auto", "profile": "live", "bpm": 127.996, "xo": [120.4, 2000],
                          "sideHp": 100.2 },
            "measures": { "loudness": -10.004, "true_peak": -0.3, "crest": { "before": 14.9, "after": 13.0 },
                          "lra": { "before": 4.7, "after": 3.4 }, "tilt": { "before": -4.77, "after": -4.84 } }
        })
    }

    fn uid(n: u32) -> String {
        format!("00000000-0000-4000-8000-{n:012}")
    }

    fn write_outbox(p: &Path, n: u32) {
        let lines: Vec<String> = (1..=n).map(|i| page_vote(&uid(i)).to_string()).collect();
        std::fs::write(p, lines.join("\n") + "\n").unwrap();
    }

    #[test]
    fn outcomes_follow_the_spec() {
        assert_eq!(outcome(200, r#"{"accepted":1,"duplicates":0,"rejected":[]}"#, None), Outcome::Sent);
        assert_eq!(outcome(200, r#"{"error":"bad_envelope"}"#, None), Outcome::Reject("200 bad_envelope".into()));
        assert!(matches!(outcome(400, r#"{"error":"bad_envelope"}"#, None), Outcome::Reject(_)));
        assert!(matches!(outcome(415, r#"{"error":"unsupported_media_type"}"#, None), Outcome::Reject(_)));
        assert!(matches!(outcome(403, "", None), Outcome::Stop(_)));
        assert_eq!(outcome(413, "", None), Outcome::TooLarge);
        assert_eq!(outcome(429, "", Some("120")), Outcome::Wait(120));
        assert_eq!(outcome(429, "", Some("999999")), Outcome::Wait(RETRY_AFTER_MAX_S));
        assert_eq!(outcome(429, "", None), Outcome::Wait(60));
        assert_eq!(outcome(503, "oops", None), Outcome::Retry);
    }

    #[test]
    fn outbox_dedupes_and_drops_what_was_sent() {
        let p = tmp("outbox");
        let (a, b) = (page_vote(&uid(1)), page_vote(&uid(2)));
        std::fs::write(&p, format!("{a}\n{b}\nnot json\n{{\"id\":\"x\"}}\n{a}\n")).unwrap();
        let ids: Vec<_> = read_outbox(&p).iter().map(|v| v["id"].as_str().unwrap().to_string()).collect();
        assert_eq!(ids, [uid(1), uid(2)]);
        drop_sent(&p, &[uid(1)]).unwrap();
        assert_eq!(read_outbox(&p).len(), 1);
        drop_sent(&p, &[uid(2)]).unwrap();
        assert!(!p.exists());
    }

    #[test]
    fn only_the_known_fields_go_on_the_wire() {
        let mut v = page_vote("6F9619FF-8B86-4011-B42D-00C04FC964FF");
        v["path"] = json!("C:/Users/me/Music/secret.wav");
        v["settings"]["file"] = json!("secret.wav");
        v["measures"]["name"] = json!("secret");
        let w = check_vote(&v.to_string()).unwrap();
        let keys: Vec<&str> = w.as_object().unwrap().keys().map(String::as_str).collect();
        assert_eq!(keys.len(), 9, "{keys:?}");
        assert!(!w.to_string().contains("secret") && w.get("v").is_none());
        assert_eq!(w["id"], "6f9619ff-8b86-4011-b42d-00c04fc964ff");
        assert_eq!(w["time"], "2026-10-27T13:05:00.000Z");
        assert_eq!((w["switches"].as_i64(), w["listened_s"].as_i64()), (Some(7), Some(94)));
        let s = &w["settings"];
        assert_eq!((s["loud"].as_str(), s["tilt"].as_str(), s["profile"].as_str()), (Some("-10.0"), Some("0.0"), Some("live")));
        assert_eq!((s["bpm"].as_f64(), s["sideHp"].as_i64()), (Some(128.0), Some(100)));
        assert_eq!(s["xo"], json!([120, 2000]));
        assert_eq!(w["measures"]["loudness"].as_f64(), Some(-10.0));
        assert_eq!(w["measures"]["tilt"]["after"].as_f64(), Some(-4.84));
        // a vote without settings or measures (no render key yet) is still a vote
        let mut bare = page_vote(&uid(1));
        bare["settings"] = Value::Null;
        bare.as_object_mut().unwrap().remove("measures");
        let w = check_vote(&bare.to_string()).unwrap();
        assert!(w["settings"].is_null() && w["measures"].is_null());
    }

    #[test]
    fn a_vote_with_a_wrong_field_is_refused() {
        let base = page_vote(&uid(1));
        let cases: [(&str, Value); 12] = [
            ("id", json!("x")),
            ("id", json!(5)),
            ("time", json!("yesterday")),
            ("time", json!("2026-10-27T13:05:42.123+13:00")),
            ("choice", json!("C:/x.wav")),
            ("master", json!("none")),
            ("preferred", json!("A")),
            ("switches", json!(-1)),
            ("listened_s", json!("long")),
            ("settings", json!("C:/x.wav")),
            ("measures", json!([1])),
            ("id", Value::Null),
        ];
        for (k, bad) in cases {
            let mut v = base.clone();
            v[k] = bad.clone();
            assert!(check_vote(&v.to_string()).is_err(), "{k} = {bad}");
        }
        for (k, bad) in [("character", json!("C:/x")), ("xo", json!(["C:/x"])), ("bpm", json!("fast")),
                         ("preserve", json!("yes")), ("loud", json!("C:/x")), ("delivery", json!(1))] {
            let mut v = base.clone();
            v["settings"][k] = bad.clone();
            assert!(check_vote(&v.to_string()).is_err(), "settings.{k} = {bad}");
        }
        let mut v = base.clone();
        v["measures"]["crest"]["before"] = json!("C:/x");
        assert!(check_vote(&v.to_string()).is_err());
        assert!(check_vote("[1]").is_err());
        assert!(check_vote(&format!(r#"{{"id":"x","p":"{}"}}"#, "a".repeat(VOTE_MAX_BYTES))).is_err());
        let e = envelope(&[json!({"id": "x"})]);
        assert_eq!(e["schema"], 1);
        assert_eq!(e["app"]["name"], "eluate");
        assert_eq!(e["votes"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn a_refused_vote_is_set_aside_and_the_queue_moves_on() {
        let p = tmp("reject");
        write_outbox(&p, 5);
        let (lock, generation) = (Mutex::new(()), AtomicU64::new(0));
        // the endpoint refuses vote 3 only, the others are accepted
        let bad = uid(3);
        let mut posts = 0;
        let mut endpoint = |votes: &[Value]| {
            posts += 1;
            if ids(votes).contains(&bad) { Outcome::Reject("400 bad_vote".into()) } else { Outcome::Sent }
        };
        for run in 1..=REJECT_AFTER {
            send_loop(&p, &lock, &generation, &mut endpoint);
            let left = ids(&read_outbox(&p));
            if run < REJECT_AFTER {
                // the batch is halved down to the one vote at fault, which stays queued
                assert_eq!(left.first(), Some(&bad), "run {run}: {left:?}");
            }
        }
        // refused REJECT_AFTER times: moved aside, and everything else was sent
        assert!(read_outbox(&p).is_empty());
        let aside = std::fs::read_to_string(p.with_file_name(REJECTED_FILE)).unwrap();
        assert_eq!(aside.lines().count(), 1);
        assert!(aside.contains(&bad));
        assert!(!p.with_file_name(REJECTS_FILE).exists());
        assert!(posts < 30, "{posts} posts");
        clear_files(&p).unwrap();
        assert!(!p.with_file_name(REJECTED_FILE).exists());
    }

    #[test]
    fn switching_sharing_off_stops_the_sender() {
        let p = tmp("generation");
        write_outbox(&p, 3);
        let (lock, generation) = (Mutex::new(()), AtomicU64::new(0));
        let mut posts = 0;
        // sharing is switched off while the first post is out (clear_outbox); the endpoint
        // is down, so the sender would otherwise wait BACKOFF_S and post again
        send_loop(&p, &lock, &generation, |_: &[Value]| {
            posts += 1;
            generation.fetch_add(1, Ordering::SeqCst);
            clear_files(&p).unwrap();
            Outcome::Retry
        });
        assert_eq!(posts, 1);
        assert!(!pause(5, &generation, 0));
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
