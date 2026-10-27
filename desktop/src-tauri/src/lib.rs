//! eluate desktop shell: a WebView2 window over ui/, and the engine process.
//!
//! The engine (`eluate-engine`, a PyInstaller build of eluate.app_engine) runs in
//! slots, one child each: a render pool ("render-<job>" per `master`, at most MAX_RENDERS
//! at once; the page decides how many run in parallel), "view" (`preview`, `ui-data`),
//! "list" (`summary`), "analyse" (`analyse`: the analysis a master would make, cached in
//! the work dir before Render; maintainer feedback 2026-10-24) and "stems-<job>" (`stems-sum`:
//! a stem set's sum, Open stems). Starting a job stops only that
//! slot's previous job, so selecting a track never touches a running render or analysis;
//! `stop_job` with "render" stops every render (not the analysis: its own slot).
//! Engine threads (report 8): the page sends how many renders it runs at once
//! (`start_job`'s parallel) and the shell alone sets a render's (and an analysis')
//! `--threads`: all cores but two, split evenly across those renders (`engine_threads`).
//! "Limit CPU" (limit_cpu, sprint WP8): `--threads 2`, and on Windows every engine child
//! starts at below-normal priority. Every engine child gets the BLAS / OpenMP pools pinned
//! to one thread (BLAS_THREAD_VARS): the engine's own pool does the parallel work.
//! Its stdout lines prefixed with ENGINE_PREFIX are JSON messages, passed to the page as
//! `engine` events tagged with the page's job id; when the process ends an `exit` event
//! follows with its code, whether it was stopped, and the tail of stderr. Files the engine reports in a `data` message (the master and the A/B pair)
//! are added to the asset protocol scope so the page can play them; nothing else is readable.
//! Masters are written next to the premaster, in "<name> - eluate/".
//!
//! Granted inputs (docs/batch-mode.md 2): the engine only ever reads audio the user chose
//! here, through `pick_audio`, `pick_folders` or a drop on the window (handled in Rust, not
//! by the page). Folders are listed top level only (`list_inputs`). `start_job` refuses any
//! other input for every kind (`job_inputs`); `master` and `analyse` write only into the
//! input's work dir, the `-o` the shell appends last (`engine_args`), and `ui-data` gets
//! the granted input, never a folder: the engine derives the work dir (uidata.work_dir,
//! the same rule as work_dir here), reads that input and refuses a render whose
//! source.json names another file (audit 13).
//! `write_summary` writes eluate-batch.csv only into a folder the user chose or that
//! holds a chosen file. While any render runs, Windows is asked not to sleep, and closing
//! the window asks first. Blind A/B votes are saved beside their master (`write_vote`) and,
//! only when the user opted in, queued and sent to jective.nz (votes.rs).

mod votes;

use std::collections::{HashMap, HashSet, VecDeque};
use std::ffi::OsString;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};
use tauri::{AppHandle, DragDropEvent, Emitter, Manager, State, WindowEvent};
use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};
use tauri_plugin_opener::OpenerExt;

const ENGINE_PREFIX: &str = "@@eluate ";
const STDERR_TAIL: usize = 40;
const LOG_MAX_BYTES: u64 = 1 << 20; // engine.log rolls over to engine.log.1 past this
const AUDIO_EXT: [&str; 5] = ["wav", "flac", "aif", "aiff", "w64"];
const OUT_SUFFIX: &str = " - eluate";
/// Output folders written before the rename (2026-10-26): never read as inputs either.
const LEGACY_OUT_SUFFIX: &str = " - distiller";
const MAX_INPUTS: usize = 200; // a guard against a wrong folder (a sample library)
const SUMMARY_FILE: &str = "eluate-batch.csv";
const SUMMARY_MAX_BYTES: usize = 1 << 20;
/// Blind A/B votes, one JSON object per line, in the render's output folder.
const VOTES_FILE: &str = "ab-votes.jsonl";
const VOTE_MAX_BYTES: usize = 4096;
const MAX_RENDERS: usize = 8; // a guard; the page runs at most 3 at once
/// Thread-count variables of the BLAS / OpenMP builds numpy and scipy may load.
const BLAS_THREAD_VARS: [&str; 4] =
    ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"];
const RENDER: &str = "render";

struct Job {
    id: u64,
    child: Arc<Mutex<Option<Child>>>,
    killed: Arc<AtomicBool>,
}

#[derive(Default)]
struct Jobs(Mutex<HashMap<String, Job>>);

/// Paths the user chose: audio files, and the folders a CSV may be written into; and the
/// stem sets chosen with Open stems, by their work dir.
#[derive(Default)]
struct Granted {
    files: Mutex<HashSet<PathBuf>>,
    dirs: Mutex<HashSet<PathBuf>>,
    sets: Mutex<HashMap<PathBuf, StemSet>>,
}

/// A stem set (Open stems): its stems and the optional mix for the null test, all chosen
/// by the user. The engine reads them only through arguments the shell adds itself.
#[derive(Clone, Debug, PartialEq)]
struct StemSet {
    stems: Vec<PathBuf>,
    mix: Option<PathBuf>,
}

/// The file a stem set's sum is written to, in the set's work dir (cli.STEMS_SUM).
const STEMS_SUM: &str = "stems-sum.wav";
/// Engine flags whose values are paths the shell supplies (audit 13 #4): never the page's.
const SHELL_FLAGS: [&str; 2] = ["--stems", "--mix"];

/// Sends "a render is running" to the thread that holds the Windows sleep request
/// (SetThreadExecutionState is per thread, so one long-lived thread owns it).
struct KeepAwake(Mutex<Sender<bool>>);

fn slot_of(kind: &str, job: u64) -> Option<String> {
    match kind {
        "master" => Some(format!("{RENDER}-{job}")),
        "preview" | "ui-data" => Some("view".into()),
        "summary" => Some("list".into()),
        "analyse" => Some("analyse".into()),
        "stems-sum" => Some(format!("stems-{job}")),
        _ => None,
    }
}

fn is_render(slot: &str) -> bool {
    slot.starts_with(RENDER)
}

/// Engine threads for a render: all cores but two shared by the `parallel` renders the
/// page runs at once (at least one each), or two under "Limit CPU".
fn engine_threads(limit_cpu: bool, cores: usize, parallel: usize) -> usize {
    if limit_cpu {
        cores.clamp(1, 2)
    } else {
        (cores.saturating_sub(2) / parallel.clamp(1, MAX_RENDERS)).max(1)
    }
}

/// Windows creation flags for an engine child: no console window, and below-normal
/// priority under "Limit CPU". Only called from the `#[cfg(windows)]` spawn path, but
/// deliberately not gated: the unit test below covers the flag arithmetic on every
/// platform. The allow keeps that coverage without a dead-code warning off Windows.
#[cfg_attr(not(windows), allow(dead_code))]
fn creation_flags(limit_cpu: bool) -> u32 {
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    const BELOW_NORMAL_PRIORITY_CLASS: u32 = 0x0000_4000;
    CREATE_NO_WINDOW | if limit_cpu { BELOW_NORMAL_PRIORITY_CLASS } else { 0 }
}

/// A canonical path without Windows' verbatim prefix, so the engine, the opener and the
/// page see an ordinary path (C:\..., \\server\share\...).
fn canon(p: &Path) -> Option<PathBuf> {
    let c = std::fs::canonicalize(p).ok()?;
    let s = c.to_string_lossy();
    if let Some(rest) = s.strip_prefix(r"\\?\UNC\") {
        return Some(PathBuf::from(format!(r"\\{rest}")));
    }
    if let Some(rest) = s.strip_prefix(r"\\?\") {
        if rest.as_bytes().get(1) == Some(&b':') {
            return Some(PathBuf::from(rest));
        }
    }
    Some(c)
}

fn is_out_dir(p: &Path) -> bool {
    p.file_name()
        .map(|n| {
            let n = n.to_string_lossy();
            n.ends_with(OUT_SUFFIX) || n.ends_with(LEGACY_OUT_SUFFIX)
        })
        .unwrap_or(false)
}

/// An audio file the engine did not write (not master.wav, not an A/B file, not inside
/// a "* - eluate" folder).
fn is_input_audio(p: &Path) -> bool {
    let ext_ok = p
        .extension()
        .map(|e| AUDIO_EXT.contains(&e.to_string_lossy().to_ascii_lowercase().as_str()))
        .unwrap_or(false);
    let name = p.file_name().map(|n| n.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
    ext_ok
        && name != "master.wav"
        && !name.ends_with("_matched.wav")
        && !p.parent().map(is_out_dir).unwrap_or(false)
}

/// Files and folders as chosen (picked or dropped) to the inputs they stand for: a file
/// as itself, a folder as its audio files, top level only, sorted by name; an engine
/// output folder contributes nothing. Canonical, first occurrence kept, at most
/// MAX_INPUTS (`capped` says the rest were left out).
fn list_inputs(chosen: &[PathBuf]) -> (Vec<PathBuf>, bool) {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    let mut capped = false;
    let mut push = |p: PathBuf, out: &mut Vec<PathBuf>| {
        if out.len() >= MAX_INPUTS {
            capped = true;
        } else if seen.insert(p.clone()) {
            out.push(p);
        }
    };
    for c in chosen {
        let Some(c) = canon(c) else { continue };
        if c.is_dir() {
            if is_out_dir(&c) {
                continue;
            }
            let mut files: Vec<PathBuf> = std::fs::read_dir(&c)
                .map(|rd| rd.filter_map(Result::ok).map(|e| e.path()).collect())
                .unwrap_or_default();
            files.retain(|f| f.is_file() && is_input_audio(f));
            files.sort_by_key(|f| f.file_name().map(|n| n.to_string_lossy().to_lowercase()));
            for f in files {
                push(f, &mut out);
            }
        } else if c.is_file() && is_input_audio(&c) {
            push(c, &mut out);
        }
    }
    (out, capped)
}

/// Grants the inputs `chosen` stands for (and their folders, for the CSV) and returns
/// what the page gets: {"inputs": [...], "capped": bool}.
fn grant(granted: &Granted, chosen: &[PathBuf]) -> Value {
    let (inputs, capped) = list_inputs(chosen);
    {
        let mut dirs = granted.dirs.lock().unwrap();
        dirs.extend(chosen.iter().filter_map(|c| canon(c)).filter(|c| c.is_dir()));
        dirs.extend(inputs.iter().filter_map(|f| f.parent().map(Path::to_path_buf)));
    }
    granted.files.lock().unwrap().extend(inputs.iter().cloned());
    json!({"inputs": inputs.iter().map(|p| p.to_string_lossy()).collect::<Vec<_>>(),
           "capped": capped})
}

/// The inputs of a job: one granted file (`summary`: one or more), else an error.
fn job_inputs(granted: &Granted, kind: &str, inputs: &[String]) -> Result<Vec<PathBuf>, String> {
    if inputs.is_empty() || (kind != "summary" && inputs.len() != 1) {
        return Err("Wrong number of inputs".into());
    }
    if kind == "stems-sum" {
        // the one input is the set's work dir: the shell knows its stems
        let w = canon(Path::new(&inputs[0])).ok_or("Folder not found")?;
        let known = granted.sets.lock().unwrap().contains_key(&w);
        return if known { Ok(vec![w]) } else { Err("Not a chosen stem set".into()) };
    }
    inputs.iter().map(|i| check_granted(granted, i)).collect()
}

/// The engine's arguments for a job. The page's arguments go first for `master` and
/// `analyse`: argparse keeps the last -o and --threads, so the output stays in work_dir and
/// the threads follow limit_cpu and parallel whatever the page sends.
fn engine_args(kind: &str, paths: &[PathBuf], args: &[String], threads: usize,
               set: Option<&StemSet>) -> Result<Vec<OsString>, String> {
    let mut v: Vec<OsString> = vec![kind.into()];
    let mix = |v: &mut Vec<OsString>, s: &StemSet| {
        if let Some(m) = &s.mix {
            v.extend(["--mix".into(), m.clone().into()]);
        }
    };
    match kind {
        "stems-sum" => {
            let s = set.ok_or("Not a chosen stem set")?;
            v.extend(s.stems.iter().map(|p| p.clone().into()));
            mix(&mut v, s);
            v.extend(["-o".into(), paths[0].clone().into()]);
        }
        "master" | "analyse" => {
            v.push(paths[0].clone().into());
            v.extend(args.iter().map(OsString::from));
            // a stem set's master runs the stems flow on the files the user chose
            if let (Some(s), "master") = (set, kind) {
                v.push("--stems".into());
                v.extend(s.stems.iter().map(|p| p.clone().into()));
                mix(&mut v, s);
            }
            v.extend(["--threads".into(), threads.to_string().into(), "-o".into()]);
            v.push(work_dir(&paths[0])?.into());
        }
        "preview" => {
            v.push(paths[0].clone().into());
            v.extend(args.iter().map(OsString::from));
        }
        "ui-data" => v.push(paths[0].clone().into()),
        _ => v.extend(paths.iter().map(|p| p.clone().into())),
    }
    Ok(v)
}

fn check_granted(granted: &Granted, input: &str) -> Result<PathBuf, String> {
    let p = canon(Path::new(input)).ok_or("File not found")?;
    if granted.files.lock().unwrap().contains(&p) || (is_stems_sum(&p) && set_of(granted, &p).is_some()) {
        return Ok(p);
    }
    Err("Not a chosen file".into())
}

/// The stem set whose sum `p` is, or whose work dir `p` is (stems-sum's input).
fn set_of(granted: &Granted, p: &Path) -> Option<StemSet> {
    let sets = granted.sets.lock().unwrap();
    if is_stems_sum(p) {
        return sets.get(p.parent()?).cloned();
    }
    sets.get(p).cloned()
}

/// A chosen file, or a path inside the work dir of one (audit 13 #5: reveal and open_log
/// act only on what the user chose or the engine wrote for it).
fn in_granted_scope(granted: &Granted, path: &str) -> Result<PathBuf, String> {
    let p = canon(Path::new(path)).ok_or("File not found")?;
    let files = granted.files.lock().unwrap();
    let ok = files.contains(&p)
        || files.iter().any(|f| work_dir(f).ok().and_then(|w| canon(&w)).is_some_and(|w| p.starts_with(&w)))
        || granted.sets.lock().unwrap().keys().any(|w| p.starts_with(w));
    if ok { Ok(p) } else { Err("Not a chosen file".into()) }
}

fn engine_command(app: &AppHandle) -> Result<Command, String> {
    let exe = if cfg!(windows) { "eluate-engine.exe" } else { "eluate-engine" };
    if let Ok(res) = app.path().resource_dir() {
        let p = res.join("engine").join(exe);
        if p.is_file() {
            return Ok(Command::new(p));
        }
    }
    let python = std::env::var_os("ELUATE_PYTHON").map(PathBuf::from).or_else(|| {
        // a development run from the repository uses its virtual environment
        let venv = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..").join(".venv");
        let p = if cfg!(windows) { venv.join("Scripts").join("python.exe") } else { venv.join("bin").join("python") };
        (cfg!(debug_assertions) && p.is_file()).then_some(p)
    });
    match python {
        Some(py) => {
            let mut c = Command::new(py);
            c.args(["-m", "eluate.app_engine"]);
            Ok(c)
        }
        None => Err("Engine not found".into()),
    }
}

/// The rule uidata.work_dir follows too: `<name> - eluate/` beside the input, and a stem
/// set's sum is already in its set's work dir.
fn work_dir(input: &Path) -> Result<PathBuf, String> {
    let parent = input.parent().ok_or("No folder")?;
    if is_stems_sum(input) {
        return Ok(parent.to_path_buf());
    }
    let stem = input.file_stem().ok_or("No file name")?.to_string_lossy().into_owned();
    Ok(parent.join(format!("{stem}{OUT_SUFFIX}")))
}

fn is_stems_sum(p: &Path) -> bool {
    p.file_name().is_some_and(|n| n == STEMS_SUM) && p.parent().is_some_and(is_out_dir)
}

/// A stem set's name: the stems' common file-name prefix without trailing separators
/// ("128 drift Am bass", "128 drift Am kik" -> "128 drift Am"), else their folder's name.
fn set_name(stems: &[PathBuf]) -> String {
    let names: Vec<String> =
        stems.iter().filter_map(|p| p.file_stem().map(|n| n.to_string_lossy().into_owned())).collect();
    let mut prefix: Vec<char> = names.first().map(|n| n.chars().collect()).unwrap_or_default();
    for n in names.iter().skip(1) {
        let k = prefix.iter().zip(n.chars()).take_while(|(a, b)| **a == *b).count();
        prefix.truncate(k);
    }
    let seps: &[char] = &[' ', '-', '_', '.'];
    let name = prefix.into_iter().collect::<String>().trim_end_matches(seps).to_string();
    if name.chars().count() >= 2 {
        return name;
    }
    stems
        .first()
        .and_then(|p| p.parent())
        .and_then(|d| d.file_name())
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "Stems".into())
}

/// The page's flags pass through to the engine except the shell's own (SHELL_FLAGS).
fn page_args_ok(args: &[String]) -> Result<(), String> {
    let bad = args.iter().any(|a| SHELL_FLAGS.iter().any(|f| a == f || a.starts_with(&format!("{f}="))));
    if bad { Err("Not allowed".into()) } else { Ok(()) }
}

fn allow_media(app: &AppHandle, ui: &Value) {
    let scope = app.asset_protocol_scope();
    let mut paths: Vec<&str> = Vec::new();
    if let Some(m) = ui.get("master").and_then(Value::as_str) {
        paths.push(m);
    }
    if let Some(ab) = ui.get("ab").and_then(Value::as_object) {
        paths.extend(ab.values().filter_map(Value::as_str));
    }
    for p in paths {
        let _ = scope.allow_file(p);
    }
}

/// Stops the job in `slot` only ("render": every render); the other slots' children are
/// not touched.
fn stop_slot(jobs: &Jobs, slot: &str) -> bool {
    let mut map = jobs.0.lock().unwrap();
    let keys: Vec<String> = if slot == RENDER {
        map.keys().filter(|k| is_render(k)).cloned().collect()
    } else {
        vec![slot.to_string()]
    };
    let mut any = false;
    for k in keys {
        if let Some(job) = map.remove(&k) {
            job.killed.store(true, Ordering::SeqCst);
            if let Some(c) = job.child.lock().unwrap().as_mut() {
                let _ = c.kill();
            }
            any = true;
        }
    }
    any
}

fn stop_all(jobs: &Jobs) {
    let slots: Vec<String> = jobs.0.lock().unwrap().keys().cloned().collect();
    for s in slots {
        stop_slot(jobs, &s);
    }
}

fn renders(jobs: &Jobs) -> usize {
    jobs.0.lock().unwrap().keys().filter(|k| is_render(k)).count()
}

fn render_busy(app: &AppHandle) -> bool {
    renders(&app.state::<Jobs>()) > 0
}

fn set_awake(app: &AppHandle) {
    let _ = app.state::<KeepAwake>().0.lock().unwrap().send(render_busy(app));
}

#[tauri::command]
#[allow(clippy::too_many_arguments)]
fn start_job(
    app: AppHandle,
    jobs: State<'_, Jobs>,
    granted: State<'_, Granted>,
    job: u64,
    kind: String,
    inputs: Vec<String>,
    args: Vec<String>,
    limit_cpu: Option<bool>,
    parallel: Option<usize>,
) -> Result<(), String> {
    let slot = slot_of(&kind, job).ok_or("Unknown job")?;
    let limit_cpu = limit_cpu.unwrap_or(false);
    let parallel = parallel.unwrap_or(1);
    page_args_ok(&args)?;
    let paths = job_inputs(&granted, &kind, &inputs)?;
    let set = set_of(&granted, &paths[0]);
    stop_slot(&jobs, &slot);
    if is_render(&slot) && renders(&jobs) >= MAX_RENDERS {
        return Err("Too many renders".into());
    }
    let mut cmd = engine_command(&app)?;
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    cmd.args(engine_args(&kind, &paths, &args, engine_threads(limit_cpu, cores, parallel), set.as_ref())?);
    if let Ok(cache) = app.path().app_cache_dir() {
        cmd.env("NUMBA_CACHE_DIR", cache.join("numba"));
    }
    for v in BLAS_THREAD_VARS {
        cmd.env(v, "1");
    }
    cmd.env("PYTHONUNBUFFERED", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(creation_flags(limit_cpu));
    }
    let mut child = cmd.spawn().map_err(|e| format!("Engine did not start: {e}"))?;
    let stdout = child.stdout.take().ok_or("No engine output")?;
    let stderr = child.stderr.take().ok_or("No engine output")?;
    let child = Arc::new(Mutex::new(Some(child)));
    let killed = Arc::new(AtomicBool::new(false));
    jobs.0.lock().unwrap().insert(slot.clone(), Job { id: job, child: child.clone(), killed: killed.clone() });
    if is_render(&slot) {
        set_awake(&app);
    }

    let tail = Arc::new(Mutex::new(VecDeque::<String>::new()));
    let log_path = app.path().app_log_dir().ok().map(|d| {
        let _ = std::fs::create_dir_all(&d);
        d.join("engine.log")
    });
    let tail_w = tail.clone();
    let err_thread = std::thread::spawn(move || {
        let mut log = log_path.and_then(|p| {
            if std::fs::metadata(&p).map(|m| m.len() > LOG_MAX_BYTES).unwrap_or(false) {
                let _ = std::fs::rename(&p, p.with_extension("log.1"));
            }
            std::fs::OpenOptions::new().create(true).append(true).open(p).ok()
        });
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            if let Some(f) = log.as_mut() {
                let _ = writeln!(f, "{line}");
            }
            let mut t = tail_w.lock().unwrap();
            t.push_back(line);
            if t.len() > STDERR_TAIL {
                t.pop_front();
            }
        }
    });

    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            let Some(body) = line.strip_prefix(ENGINE_PREFIX) else { continue };
            let Ok(mut msg) = serde_json::from_str::<Value>(body) else { continue };
            if msg.get("event").and_then(Value::as_str) == Some("data") {
                if let Some(ui) = msg.get("ui") {
                    allow_media(&app, ui);
                }
            }
            msg["job"] = json!(job);
            let _ = app.emit("engine", msg);
        }
        let _ = err_thread.join();
        let code = child
            .lock()
            .unwrap()
            .take()
            .and_then(|mut c| c.wait().ok())
            .and_then(|s| s.code())
            .unwrap_or(-1);
        {
            // the slot is free unless a newer job already took it
            let jobs = app.state::<Jobs>();
            let mut map = jobs.0.lock().unwrap();
            if map.get(&slot).map(|j| j.id) == Some(job) {
                map.remove(&slot);
            }
        }
        if is_render(&slot) {
            set_awake(&app);
        }
        let stderr_tail: Vec<String> = tail.lock().unwrap().iter().cloned().collect();
        let _ = app.emit(
            "engine",
            json!({"event": "exit", "job": job, "slot": slot, "code": code,
                   "killed": killed.load(Ordering::SeqCst), "stderr": stderr_tail}),
        );
    });
    Ok(())
}

#[tauri::command]
fn stop_job(app: AppHandle, jobs: State<'_, Jobs>, slot: String) {
    if stop_slot(&jobs, &slot) && is_render(&slot) {
        set_awake(&app);
    }
}

#[tauri::command]
async fn pick_audio(app: AppHandle) -> Value {
    let chosen: Vec<PathBuf> = app
        .dialog()
        .file()
        .add_filter("Audio", &AUDIO_EXT)
        .blocking_pick_files()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|p| p.into_path().ok())
        .collect();
    grant(&app.state::<Granted>(), &chosen)
}

/// Open stems: the stems (two or more), then optionally the stereo mix for the null test
/// (Cancel skips it). Grants them, makes the set's work dir beside the first stem, and
/// returns {name, stems, mix, work, sum}, or null when fewer than two stems were chosen.
#[tauri::command]
async fn pick_stems(app: AppHandle) -> Value {
    let stems: Vec<PathBuf> = app
        .dialog()
        .file()
        .set_title("Choose the stems")
        .add_filter("Audio", &AUDIO_EXT)
        .blocking_pick_files()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|p| p.into_path().ok())
        .filter_map(|p| canon(&p))
        .collect();
    if stems.len() < 2 {
        return Value::Null;
    }
    let mix = app
        .dialog()
        .file()
        .set_title("Optional: the stereo mix, to check the stems sum to it (Cancel to skip)")
        .add_filter("Audio", &AUDIO_EXT)
        .blocking_pick_file()
        .and_then(|p| p.into_path().ok())
        .and_then(|p| canon(&p))
        .filter(|m| !stems.contains(m));
    register_set(&app.state::<Granted>(), stems, mix)
}

/// Grants a chosen stem set, makes its work dir beside the first stem and records it
/// (pick_stems after its dialogs; smoke_stems).
fn register_set(granted: &Granted, stems: Vec<PathBuf>, mix: Option<PathBuf>) -> Value {
    let name = set_name(&stems);
    let Some(parent) = stems.first().and_then(|p| p.parent()).map(Path::to_path_buf) else { return Value::Null };
    let work = parent.join(format!("{name}{OUT_SUFFIX}"));
    if std::fs::create_dir_all(&work).is_err() {
        return Value::Null;
    }
    let Some(work) = canon(&work) else { return Value::Null };
    granted.files.lock().unwrap().extend(stems.iter().cloned().chain(mix.clone()));
    granted.dirs.lock().unwrap().insert(parent);
    granted.sets.lock().unwrap().insert(work.clone(), StemSet { stems: stems.clone(), mix: mix.clone() });
    let s = |p: &Path| p.to_string_lossy().into_owned();
    json!({"name": name, "stems": stems.iter().map(|p| s(p)).collect::<Vec<_>>(),
           "mix": mix.as_deref().map(s), "work": s(&work), "sum": s(&work.join(STEMS_SUM))})
}

#[tauri::command]
async fn pick_folders(app: AppHandle) -> Value {
    let chosen: Vec<PathBuf> = app
        .dialog()
        .file()
        .blocking_pick_folders()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|p| p.into_path().ok())
        .collect();
    grant(&app.state::<Granted>(), &chosen)
}

/// Writes the batch summary (R4) as SUMMARY_FILE into `folder`, which must be a chosen
/// folder or hold a chosen file. Returns the file's path.
#[tauri::command]
fn write_summary(granted: State<'_, Granted>, folder: String, csv: String) -> Result<String, String> {
    let dir = canon(Path::new(&folder)).ok_or("Folder not found")?;
    if !granted.dirs.lock().unwrap().contains(&dir) {
        return Err("Not a chosen folder".into());
    }
    if csv.len() > SUMMARY_MAX_BYTES {
        return Err("Summary too large".into());
    }
    let p = dir.join(SUMMARY_FILE);
    std::fs::write(&p, csv).map_err(|e| e.to_string())?;
    Ok(p.to_string_lossy().into_owned())
}

/// Appends one blind A/B vote (a JSON object, one line) to VOTES_FILE in `dir`, which must
/// be an output folder.
fn append_vote(dir: &Path, vote: &str) -> Result<PathBuf, String> {
    if !is_out_dir(dir) {
        return Err("Not an output folder".into());
    }
    if vote.len() > VOTE_MAX_BYTES {
        return Err("Vote too large".into());
    }
    let v: Value = serde_json::from_str(vote).map_err(|_| "Not a vote".to_string())?;
    if !v.is_object() {
        return Err("Not a vote".into());
    }
    let p = dir.join(VOTES_FILE);
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&p).map_err(|e| e.to_string())?;
    use std::io::Write;
    writeln!(f, "{v}").map_err(|e| e.to_string())?;
    Ok(p)
}

/// Saves a blind A/B vote beside the master it was cast on (`master`: the render's
/// master.wav, in a chosen file's work dir).
#[tauri::command]
fn write_vote(granted: State<'_, Granted>, master: String, vote: String) -> Result<String, String> {
    let p = in_granted_scope(&granted, &master)?;
    let dir = p.parent().ok_or("No folder")?;
    append_vote(dir, &vote).map(|p| p.to_string_lossy().into_owned())
}

#[tauri::command]
fn reveal(app: AppHandle, granted: State<'_, Granted>, path: String) -> Result<(), String> {
    let p = in_granted_scope(&granted, &path)?;
    app.opener().reveal_item_in_dir(p).map_err(|e| e.to_string())
}

/// Opens a render's process-log.html (in a chosen file's work dir) in the default browser;
/// any other file is refused.
#[tauri::command]
fn open_log(app: AppHandle, granted: State<'_, Granted>, path: String) -> Result<(), String> {
    let p = in_granted_scope(&granted, &path).map_err(|_| "No process log".to_string())?;
    if p.file_name().and_then(|n| n.to_str()) != Some("process-log.html") || !p.is_file() {
        return Err("No process log".into());
    }
    app.opener().open_path(p.to_string_lossy(), None::<&str>).map_err(|e| e.to_string())
}

/// Debug builds only (scripts/smoke.mjs): grants `paths` as a drop would and sends the
/// page the same `inputs` event, since a native dialog or drop cannot be scripted.
/// Debug builds only (scripts/smoke.mjs): pick_stems without its dialogs.
#[cfg(debug_assertions)]
#[tauri::command]
fn smoke_stems(app: AppHandle, stems: Vec<String>, mix: Option<String>) -> Value {
    let stems: Vec<PathBuf> = stems.iter().filter_map(|p| canon(Path::new(p))).collect();
    let mix = mix.and_then(|m| canon(Path::new(&m)));
    register_set(&app.state::<Granted>(), stems, mix)
}

#[cfg(debug_assertions)]
#[tauri::command]
fn smoke_grant(app: AppHandle, paths: Vec<String>) -> Value {
    let chosen: Vec<PathBuf> = paths.iter().map(PathBuf::from).collect();
    let got = grant(&app.state::<Granted>(), &chosen);
    let _ = app.emit("inputs", got.clone());
    got
}

#[cfg(windows)]
fn keep_awake_thread() -> Sender<bool> {
    #[link(name = "kernel32")]
    extern "system" {
        fn SetThreadExecutionState(flags: u32) -> u32;
    }
    const ES_CONTINUOUS: u32 = 0x8000_0000;
    const ES_SYSTEM_REQUIRED: u32 = 0x0000_0001;
    let (tx, rx) = channel::<bool>();
    std::thread::spawn(move || {
        for busy in rx {
            let flags = if busy { ES_CONTINUOUS | ES_SYSTEM_REQUIRED } else { ES_CONTINUOUS };
            unsafe { SetThreadExecutionState(flags) };
        }
    });
    tx
}

#[cfg(not(windows))]
fn keep_awake_thread() -> Sender<bool> {
    let (tx, rx) = channel::<bool>();
    std::thread::spawn(move || for _ in rx {});
    tx
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let builder = tauri::Builder::default();
    #[cfg(debug_assertions)]
    let builder = builder.invoke_handler(tauri::generate_handler![
        start_job, stop_job, pick_audio, pick_folders, pick_stems, write_summary, write_vote, reveal,
        open_log, votes::queue_vote, votes::flush_votes, votes::clear_outbox, smoke_grant, smoke_stems
    ]);
    #[cfg(not(debug_assertions))]
    let builder = builder.invoke_handler(tauri::generate_handler![
        start_job, stop_job, pick_audio, pick_folders, pick_stems, write_summary, write_vote, reveal,
        open_log, votes::queue_vote, votes::flush_votes, votes::clear_outbox
    ]);
    builder
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .manage(Jobs::default())
        .manage(Granted::default())
        .manage(votes::Outbox::default())
        .manage(KeepAwake(Mutex::new(keep_awake_thread())))
        .on_window_event(|window, event| match event {
            WindowEvent::DragDrop(DragDropEvent::Drop { paths, .. }) => {
                let app = window.app_handle();
                let got = grant(&app.state::<Granted>(), paths);
                let _ = app.emit("inputs", got);
            }
            WindowEvent::CloseRequested { api, .. } if render_busy(window.app_handle()) => {
                api.prevent_close();
                let w = window.clone();
                window
                    .dialog()
                    .message(if renders(&window.state::<Jobs>()) > 1 {
                        "Renders are running. Stop them and close?"
                    } else {
                        "A render is running. Stop it and close?"
                    })
                    .kind(MessageDialogKind::Warning)
                    .buttons(MessageDialogButtons::OkCancelCustom("Stop and close".into(), "Cancel".into()))
                    .show(move |ok| {
                        if ok {
                            stop_all(&w.state::<Jobs>());
                            set_awake(w.app_handle());
                            let _ = w.destroy();
                        }
                    });
            }
            WindowEvent::Destroyed => stop_all(&window.state::<Jobs>()),
            _ => {}
        })
        .run(tauri::generate_context!())
        .expect("error while running eluate");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("eluate-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn touch(p: &Path) {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, b"x").unwrap();
    }

    fn names(v: &[PathBuf]) -> Vec<String> {
        v.iter().map(|p| p.file_name().unwrap().to_string_lossy().into_owned()).collect()
    }

    #[test]
    fn folder_listing_rules() {
        let d = tmp("list");
        for f in ["b.wav", "A.flac", "c.AIFF", "notes.txt", "master.wav", "x_matched.wav",
                  "a - eluate/master.wav", "a - eluate/ab/original_matched.wav",
                  "sub/deep.wav"] {
            touch(&d.join(f));
        }
        let (got, capped) = list_inputs(&[d.clone(), d.join("b.wav"), d.join("a - eluate")]);
        // top level, sorted by name, audio only, engine outputs and duplicates skipped
        assert_eq!(names(&got), ["A.flac", "b.wav", "c.AIFF"]);
        assert!(!capped);
        // a file chosen inside an output folder is not an input either
        let (got, _) = list_inputs(&[d.join("a - eluate").join("master.wav")]);
        assert!(got.is_empty());
        assert!(got.iter().all(|p| !p.to_string_lossy().starts_with(r"\\?\")));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn votes_append_one_line_each_in_output_folders_only() {
        let d = tmp("votes");
        let out = d.join("song - eluate");
        std::fs::create_dir_all(&out).unwrap();
        let p = append_vote(&out, r#"{"choice":"A","master":"B"}"#).unwrap();
        append_vote(&out, r#"{"choice":"none","master":"A"}"#).unwrap();
        let text = std::fs::read_to_string(&p).unwrap();
        assert_eq!(text.lines().count(), 2);
        assert!(text.lines().all(|l| serde_json::from_str::<Value>(l).unwrap().is_object()));
        assert!(append_vote(&d, r#"{"choice":"A"}"#).is_err());          // not an output folder
        assert!(append_vote(&out, "[1, 2]").is_err());                     // not an object
        assert!(append_vote(&out, "not json").is_err());
        assert!(append_vote(&out, &format!(r#"{{"x":"{}"}}"#, "a".repeat(VOTE_MAX_BYTES))).is_err());
    }

    #[test]
    fn legacy_output_folders_are_not_inputs() {
        assert!(is_out_dir(Path::new("C:/m/Track 1 - distiller")));
        assert!(is_out_dir(Path::new("C:/m/Track 1 - eluate")));
        assert!(!is_out_dir(Path::new("C:/m/Track 1")));
    }

    #[test]
    fn listing_is_capped() {
        let d = tmp("cap");
        for i in 0..MAX_INPUTS + 5 {
            touch(&d.join(format!("t{i:03}.wav")));
        }
        let (got, capped) = list_inputs(&[d.clone()]);
        assert_eq!(got.len(), MAX_INPUTS);
        assert!(capped);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn only_granted_inputs_and_folders() {
        let d = tmp("grant");
        touch(&d.join("song.wav"));
        touch(&d.join("other.wav"));
        let g = Granted::default();
        let v = grant(&g, &[d.join("song.wav")]);
        assert_eq!(v["inputs"].as_array().unwrap().len(), 1);
        assert!(check_granted(&g, &d.join("song.wav").to_string_lossy()).is_ok());
        assert_eq!(check_granted(&g, &d.join("other.wav").to_string_lossy()).unwrap_err(),
                   "Not a chosen file");
        assert!(check_granted(&g, &d.join("none.wav").to_string_lossy()).is_err());
        // the chosen file's folder may take the CSV; its parent may not
        assert!(g.dirs.lock().unwrap().contains(&canon(&d).unwrap()));
        assert!(!g.dirs.lock().unwrap().contains(&canon(d.parent().unwrap()).unwrap()));
        let _ = std::fs::remove_dir_all(&d);
    }

    fn mk(id: u64) -> Job {
        Job { id, child: Arc::new(Mutex::new(None)), killed: Arc::new(AtomicBool::new(false)) }
    }

    #[test]
    fn reveal_and_log_scope() {
        let d = tmp("scope");
        touch(&d.join("song.wav"));
        touch(&d.join("song - eluate").join("process-log.html"));
        touch(&d.join("other - eluate").join("process-log.html"));
        let g = Granted::default();
        grant(&g, &[d.join("song.wav")]);
        assert!(in_granted_scope(&g, &d.join("song.wav").to_string_lossy()).is_ok());
        assert!(in_granted_scope(&g, &d.join("song - eluate").join("process-log.html").to_string_lossy()).is_ok());
        assert!(in_granted_scope(&g, &d.join("other - eluate").join("process-log.html").to_string_lossy()).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn slots_are_isolated() {
        let jobs = Jobs::default();
        let (render, view) = (mk(1), mk(2));
        let (rk, vk) = (render.killed.clone(), view.killed.clone());
        jobs.0.lock().unwrap().insert(slot_of("master", 1).unwrap(), render);
        jobs.0.lock().unwrap().insert("view".into(), view);
        assert!(stop_slot(&jobs, "view"));
        assert!(vk.load(Ordering::SeqCst) && !rk.load(Ordering::SeqCst));
        assert!(jobs.0.lock().unwrap().contains_key("render-1"));
        assert!(!stop_slot(&jobs, "list"));
        assert_eq!(slot_of("master", 7).as_deref(), Some("render-7"));
        assert_eq!(slot_of("ui-data", 7).as_deref(), Some("view"));
        assert_eq!(slot_of("summary", 7).as_deref(), Some("list"));
        assert_eq!(slot_of("analyse", 7).as_deref(), Some("analyse"));
        assert_eq!(slot_of("rm", 7), None);
    }

    #[test]
    fn analysis_has_its_own_slot() {
        // not a render (no sleep request, no close prompt) and not stopped with them
        let jobs = Jobs::default();
        let (r, a) = (mk(1), mk(2));
        let (rk, ak) = (r.killed.clone(), a.killed.clone());
        jobs.0.lock().unwrap().insert(slot_of("master", 1).unwrap(), r);
        jobs.0.lock().unwrap().insert(slot_of("analyse", 2).unwrap(), a);
        assert!(!is_render("analyse"));
        assert_eq!(renders(&jobs), 1);
        assert!(stop_slot(&jobs, "render"));
        assert!(rk.load(Ordering::SeqCst) && !ak.load(Ordering::SeqCst));
        assert!(stop_slot(&jobs, "analyse") && ak.load(Ordering::SeqCst));
    }

    #[test]
    fn analyse_is_grant_checked_and_writes_the_work_dir() {
        let d = tmp("analyse");
        touch(&d.join("song.wav"));
        touch(&d.join("other.wav"));
        let g = Granted::default();
        grant(&g, &[d.join("song.wav")]);
        let song = d.join("song.wav").to_string_lossy().into_owned();
        let other = d.join("other.wav").to_string_lossy().into_owned();
        for kind in ["analyse", "master", "preview", "ui-data"] {
            assert_eq!(job_inputs(&g, kind, &[other.clone()]).unwrap_err(), "Not a chosen file");
            assert_eq!(job_inputs(&g, kind, &[song.clone(), song.clone()]).unwrap_err(),
                       "Wrong number of inputs");
            assert_eq!(job_inputs(&g, kind, &[]).unwrap_err(), "Wrong number of inputs");
        }
        let paths = job_inputs(&g, "analyse", &[song.clone()]).unwrap();
        // the page's -o and --threads come first, so the shell's own win (argparse keeps
        // the last)
        let page: Vec<String> = ["--bpm", "128", "-o", "C:/elsewhere", "--threads", "64"]
            .iter().map(|s| s.to_string()).collect();
        for kind in ["analyse", "master"] {
            let v = engine_args(kind, &paths, &page, 6, None).unwrap();
            let s: Vec<String> = v.iter().map(|a| a.to_string_lossy().into_owned()).collect();
            assert_eq!(s[0], kind);
            assert_eq!(s[1..3], [paths[0].to_string_lossy().into_owned(), "--bpm".into()]);
            let n = s.len();
            assert_eq!(s[n - 4..n - 1], ["--threads".to_string(), "6".into(), "-o".into()]);
            assert_eq!(PathBuf::from(&s[n - 1]), work_dir(&paths[0]).unwrap());
        }
        assert_eq!(engine_args("ui-data", &paths, &page, 6, None).unwrap().len(), 2);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn stem_sets_name_work_dir_and_args() {
        let set = StemSet {
            stems: vec![PathBuf::from("C:/s/128 drift Am bass.wav"), PathBuf::from("C:/s/128 drift Am kik.wav")],
            mix: Some(PathBuf::from("C:/s/mix.wav")),
        };
        assert_eq!(set_name(&set.stems), "128 drift Am");
        assert_eq!(set_name(&[PathBuf::from("C:/proj/Song/a.wav"), PathBuf::from("C:/proj/Song/b.wav")]), "Song");
        let work = PathBuf::from("C:/s/128 drift Am - eluate");
        let sum = work.join(STEMS_SUM);
        assert!(is_stems_sum(&sum) && !is_stems_sum(&PathBuf::from("C:/s").join(STEMS_SUM)));
        assert_eq!(work_dir(&sum).unwrap(), work);
        // the shell's own flags never come from the page, in either spelling
        let ok: Vec<String> = vec!["--bpm".into(), "172".into()];
        assert!(page_args_ok(&ok).is_ok());
        for bad in [["--stems", "C:/x.wav"], ["--mix=C:/x.wav", "1"], ["--mix", "C:/x.wav"]] {
            assert!(page_args_ok(&bad.map(String::from)).is_err());
        }
        let s = |v: Vec<OsString>| v.iter().map(|a| a.to_string_lossy().into_owned()).collect::<Vec<_>>();
        // stems-sum: the set's files and work dir, nothing from the page
        let a = s(engine_args("stems-sum", &[work.clone()], &[], 6, Some(&set)).unwrap());
        assert_eq!(a, ["stems-sum", "C:/s/128 drift Am bass.wav", "C:/s/128 drift Am kik.wav", "--mix",
                       "C:/s/mix.wav", "-o", &work.to_string_lossy()]);
        // master of the sum: the page's flags, then the set's --stems / --mix, then -o work
        let m = s(engine_args("master", &[sum.clone()], &ok, 6, Some(&set)).unwrap());
        let i = m.iter().position(|x| x == "--stems").unwrap();
        assert!(i > m.iter().position(|x| x == "--bpm").unwrap());
        assert_eq!(m[i + 1..i + 5], ["C:/s/128 drift Am bass.wav", "C:/s/128 drift Am kik.wav", "--mix", "C:/s/mix.wav"]);
        assert_eq!(PathBuf::from(m.last().unwrap()), work);
        // analyse of the sum gets no stem flags (the analysis is the sum's)
        assert!(!s(engine_args("analyse", &[sum], &ok, 6, Some(&set)).unwrap()).contains(&"--stems".to_string()));
        assert!(engine_args("stems-sum", &[work], &[], 6, None).is_err());
        assert_eq!(slot_of("stems-sum", 3).as_deref(), Some("stems-3"));
        assert!(!is_render("stems-3"));
    }

    #[test]
    fn render_pool() {
        let jobs = Jobs::default();
        let (a, b, v) = (mk(1), mk(2), mk(3));
        let (ak, bk, vk) = (a.killed.clone(), b.killed.clone(), v.killed.clone());
        for (slot, j) in [(slot_of("master", 1), a), (slot_of("master", 2), b), (slot_of("preview", 3), v)] {
            jobs.0.lock().unwrap().insert(slot.unwrap(), j);
        }
        assert_eq!(renders(&jobs), 2);
        // one render stops alone, by its own slot
        assert!(stop_slot(&jobs, "render-1"));
        assert!(ak.load(Ordering::SeqCst) && !bk.load(Ordering::SeqCst));
        assert_eq!(renders(&jobs), 1);
        // "render" stops every render and nothing else
        jobs.0.lock().unwrap().insert("render-4".into(), mk(4));
        assert!(stop_slot(&jobs, "render"));
        assert!(bk.load(Ordering::SeqCst) && !vk.load(Ordering::SeqCst));
        assert_eq!(renders(&jobs), 0);
        assert!(!stop_slot(&jobs, "render"));
        stop_all(&jobs);
        assert!(vk.load(Ordering::SeqCst) && jobs.0.lock().unwrap().is_empty());
    }

    #[test]
    fn limit_cpu_settings() {
        assert_eq!(engine_threads(false, 32, 1), 30);
        assert_eq!(engine_threads(true, 32, 1), 2);
        assert_eq!(engine_threads(false, 2, 1), 1);
        assert_eq!(engine_threads(true, 1, 1), 1);
        // parallel renders share the cores: (cores - 2) / parallel, at least one each
        assert_eq!(engine_threads(false, 32, 3), 10);
        assert_eq!(engine_threads(false, 24, 3), 7);
        assert_eq!(engine_threads(false, 16, 2), 7);
        assert_eq!(engine_threads(false, 4, 3), 1);
        assert_eq!(engine_threads(true, 32, 3), 2);
        // a page value of 0 or past the render guard is clamped
        assert_eq!(engine_threads(false, 32, 0), 30);
        assert_eq!(engine_threads(false, 66, 100), 8);
        assert_eq!(creation_flags(false), 0x0800_0000);
        assert_eq!(creation_flags(true), 0x0800_4000);
    }

    #[test]
    fn work_dir_matches_the_engine() {
        assert_eq!(work_dir(Path::new("C:/m/Track 1.wav")).unwrap(),
                   Path::new("C:/m").join("Track 1 - eluate"));
    }
}
