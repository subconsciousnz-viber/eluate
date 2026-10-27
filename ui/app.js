"use strict";

/* Runs in the desktop shell (window.__TAURI__, engine via IPC) or in a browser, where it
   draws ui/data.js if present (python -m eluate.uidata <work> ui/data.js). */

const TAURI = window.__TAURI__;
const IS_APP = !!(TAURI && TAURI.core);
const NS = "http://www.w3.org/2000/svg";
const $ = (id) => document.getElementById(id);
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

// Profiles (planner.PROFILES): the loudness default and the slider range differ
const PROFILES = {
  electronic: { loud: -4.1, range: [-9, -2] },
  live: { loud: -10, range: [-14, -6] },         // phase 1, uncalibrated (R9)
};
const DEFAULT_LOUD = PROFILES.electronic.loud;
const TILT_REF = -4.5;
// the slope Tilt moves from: under Live the mix's own (planner.target_slope), else TILT_REF
const tiltRef = () => (state.profile === "live" && isRender() && D.tilt && D.tilt.before
  ? D.tilt.before.stereo : TILT_REF);
// Character = the engine's timing presets (docs/gain-staging.md 7); auto picks from tempo
const CHARACTERS = ["auto", "transparent", "slow", "moderate", "fast", "aggressive"];
const REF_CREST = 7.1;                 // fallback only: evaluate.FLAG_CREST_DB
const AUDIO_EXT = /\.(wav|flac|aiff?|w64)$/i;
// the source offer (AGPL-3.0-or-later): this repository at the release tag v<app version>
const SOURCE_REPO = "https://github.com/subconsciousnz-viber/eluate";

const minus = (s) => String(s).replace("-", "\u2212");
const fmt = (v, d = 1, sign = false) =>
  v == null || !Number.isFinite(+v) ? "\u2013" : minus((sign && v > 0 ? "+" : "") + Number(v).toFixed(d));
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
const clock = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
const el = (tag, attrs = {}, text) => {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (text != null) n.textContent = text;
  return n;
};

/* ------------------------------------------------------------------- state */

let D = null;                            // the selected track's uidata: null | preview | render
const state = {
  loud: DEFAULT_LOUD, tilt: 0, preserve: false, delivery: "master", character: "auto",
  profile: "electronic", limitCpu: false,
  bpm: null, xo: null, sideHp: null,     // the selected track's own; null: the engine's
  view: "stereo", listen: "after",       //   (detected tempo, key-locked crossovers, 100 Hz)
  blind: true, abSel: "y",               // blind A/B on by default; abSel: the A/B button ("x" | "y")
};
// The track list (docs/batch-mode.md). status: queued | running | done | error.
// ui / preview: its uidata; row: the list's numbers (from summary or ui); rendered: the
// settings key of the master on disk; bpm, xo, sideHp: its own controls.
const tracks = [];
let selected = -1;
const sel = () => tracks[selected] || null;
// Engine slots in the shell: "render-<id>" (the queue's renders, up to parallelRenders()
// at once), view (the selected track), list (rows of added tracks), analyse (the analysis
// a track's master would make, run after its preview so Render reuses it). Each holds
// {id, kind, track, stage, step, steps, render, rendersMax, pass, passesMax, eta, etaAt,
// then, t0, error, key, akey, logs} while its job runs (pass: a stem set's correction pass).
const slots = { view: null, list: null, analyse: null };
const isRenderSlot = (k) => k.startsWith("render");
const renderJobs = () => Object.keys(slots).filter(isRenderSlot).map((k) => slots[k]);
const renderOf = (t) => (t && renderJobs().find((j) => j.track === t)) || null;
// a stem set's summing job (Open stems): slot "stems-<job>", never stopped by another track
const stemsOf = (t) => (t && Object.keys(slots).filter((k) => k.startsWith("stems-"))
  .map((k) => slots[k]).find((j) => j && j.track === t)) || null;
// One render at a time (maintainer, 2026-10-22: serial by default, a render must respect the
// machine's memory), so it gets all cores but two; Limit CPU also lowers the engine's
// threads and priority in the shell. PARALLEL_RENDERS true restores the pool: up to 3 at
// once, one per 8 cores. Sent with each job: the shell splits the engine threads across
// that many renders (lib.rs engine_threads)
const PARALLEL_RENDERS = false;
const CORES = navigator.hardwareConcurrency || 1;
const parallelRenders = () => (!PARALLEL_RENDERS || state.limitCpu
  ? 1 : Math.max(1, Math.min(3, Math.floor(CORES / 8))));
let jobSeq = 0;
let queueOn = false, queuePass = 0;
let queueFirst = null;                   // the track whose analysis the queue waited for
const analysePending = [];               // previewed tracks waiting for the analyse slot
const summaryPending = new Set();        // inputs whose row the list slot has not read yet
let progress = 1;                        // fraction of the after layers revealed

const stem = (p) => p.split(/[\\/]/).pop().replace(/\.[^.]+$/, "");
const folderOf = (p) => p.replace(/[\\/][^\\/]*$/, "");
const samePath = (a, b) => a.toLowerCase() === b.toLowerCase();

// One comparable key per settings set: what the page sends equals what run.json records.
function keyOf(s) {
  return JSON.stringify({
    loud: (+s.loud).toFixed(1), tilt: (+s.tilt || 0).toFixed(1), preserve: !!s.preserve,
    delivery: s.delivery, character: s.character, profile: s.profile || "electronic", bpm: s.bpm ?? null,
    xo: s.xo ? s.xo.map((f) => Math.round(f)) : null, sideHp: s.sideHp == null ? null : Math.round(s.sideHp),
  });
}
function keyOfControls(c) {
  return keyOf({
    loud: c.rms_target ?? c.target_rms ?? DEFAULT_LOUD, tilt: c.tilt_db || 0, preserve: c.preserve_mix,
    delivery: c.streaming ? "streaming" : "master", character: c.character || "auto",
    profile: c.profile || "electronic",
    bpm: c.bpm, xo: c.crossovers_hz, sideHp: c.subside_hz,
  });
}
function saveTrackParams() {
  const t = sel();
  if (t) { t.bpm = state.bpm; t.xo = state.xo; t.sideHp = state.sideHp; }
}
function settingsFor(t) {
  saveTrackParams();
  return keyOf({ ...state, bpm: t.bpm, xo: t.xo, sideHp: t.sideHp });
}
// a stem set renders, analyses and previews only once its sum exists (t.unsummed)
const needsRender = (t) => !t.unsummed && (t.status !== "done" || t.rendered !== settingsFor(t));

const isRender = () => D && D.kind === "render";
const crossovers = () => state.xo || (isRender() ? D.crossovers : null);
const sideHp = () => state.sideHp ?? (isRender() ? D.sideHp : null) ?? 100;
const bpmNow = () => state.bpm ?? (isRender() ? D.bpm : null);

/* ------------------------------------------------------------------- engine */

// the flags the analysis depends on, the same for `analyse` and `master` (cli._matches)
function analyseArgs(t) {
  const a = [];
  if (t.bpm) a.push("--bpm", String(t.bpm));
  if (t.xo) a.push("--xover", t.xo.map((f) => Math.round(f)).join(","));
  return a;
}
const analysisKey = (t) => JSON.stringify(analyseArgs(t));

function masterArgs(t) {
  const a = ["--target-rms", state.loud.toFixed(1), ...analyseArgs(t)];
  if (state.tilt) a.push("--tilt", state.tilt.toFixed(1));
  if (state.preserve) a.push("--preserve");
  if (state.character !== "auto") a.push("--character", state.character);
  if (t.sideHp) a.push("--subside", String(Math.round(t.sideHp)));
  if (state.delivery === "streaming") a.push("--streaming");
  if (state.profile !== "electronic") a.push("--profile", state.profile);
  return a;
}

const FIRST_STAGE = { master: "Analysing", analyse: "Analysing", preview: "Reading track",
                      "ui-data": "Loading", summary: "Reading", "stems-sum": "Summing stems" };

async function startJob(slot, kind, t, inputs) {
  if (!IS_APP) return;
  const id = ++jobSeq;
  const j = { id, kind, track: t, inputs, stage: FIRST_STAGE[kind], t0: performance.now(), error: null,
              logs: [] };
  if (kind === "master") { j.key = settingsFor(t); j.step = 1; j.steps = 4; }
  if (kind === "analyse") { saveTrackParams(); j.akey = analysisKey(t); }
  slots[slot === "render" ? `render-${id}` : slot === "stems" ? `stems-${id}` : slot] = j;
  const args = kind === "master" ? masterArgs(t) : kind === "analyse" ? analyseArgs(t)
    : kind === "preview" ? ["--target-rms", state.loud.toFixed(1)] : [];
  drawStage(); drawButtons(); drawList();
  try {
    await TAURI.core.invoke("start_job", { job: id, kind, inputs: inputs || [t.input], args,
                                           limitCpu: state.limitCpu, parallel: parallelRenders() });
  } catch (e) {
    onEngine({ job: id, event: "error", msg: String(e) });
    onEngine({ job: id, event: "exit", code: 2, killed: false });
  }
}

function onEngine(p) {
  const slot = Object.keys(slots).find((k) => slots[k] && slots[k].id === p.job);
  if (!slot) return;                     // a stopped job's last messages
  const j = slots[slot];
  if (p.event === "stage") {
    Object.assign(j, { stage: p.stage, step: p.step || null, steps: p.steps || null,
                       render: p.render || null, rendersMax: p.renders_max || null,
                       pass: p.pass || null, passesMax: p.passes_max || null, then: p.then_s ?? null });
    if (p.eta_s != null) { j.eta = p.eta_s; j.etaAt = performance.now(); }
    if (isRenderSlot(slot)) j.track.stage = p.stage;
  } else if (p.event === "log") { j.logs.push(p.msg); if (j.logs.length > 200) j.logs.shift(); }
  else if (p.event === "data") onData(p.ui, j);
  else if (p.event === "error") j.error = p.msg;
  else if (p.event === "exit") {
    if (isRenderSlot(slot) || slot.startsWith("stems-")) delete slots[slot]; else slots[slot] = null;
    onExit(slot, j, p);
    return;
  }
  drawStage(); drawList();
}

function rowOf(ui) {
  if (ui.kind === "summary") {
    return { loudness: ui.loudness, tilt: ui.tilt, crest: ui.crest, crestBefore: ui.crestBefore,
             truePeak: ui.truePeak, flags: ui.flags || [] };
  }
  return { loudness: ui.loudness.reached, tilt: ui.tilt.after.stereo, crest: ui.crest.after,
           crestBefore: ui.crest.before, truePeak: ui.truePeak, flags: ui.flags || [] };
}

function onData(ui, j) {
  if (ui.kind === "stems") { j.track.nullTest = ui.null; return; }
  if (ui.kind === "summary") {
    summaryPending.delete(ui.input.toLowerCase());
    const t = tracks.find((x) => samePath(x.input, ui.input));
    if (!t) return;
    if (ui.state === "done" && t.status === "queued") {
      t.status = "done"; t.row = rowOf(ui);
      const c = ui.controls || {};
      // the master on disk was made with these per-track controls: adopt them, so its
      // settings compare equal and Render skips it (R3)
      if (t.bpm == null && t.xo == null && t.sideHp == null) {
        t.bpm = c.bpm ?? null; t.xo = c.crossovers_hz || null; t.sideHp = c.subside_hz ?? null;
        if (t === sel()) { state.bpm = t.bpm; state.xo = t.xo; state.sideHp = t.sideHp; }
      }
      t.rendered = ui.controls ? keyOfControls(c) : null;
      if (t === sel() && !t.ui) startJob("view", "ui-data", t);
    }
    drawAll();
    return;
  }
  const t = j.track;
  if (ui.kind === "render") { t.ui = ui; t.row = rowOf(ui); }
  else if (ui.kind === "preview") t.preview = ui;
  if (t === sel() && (ui.kind === "render" || !t.ui)) setData(ui, j.kind === "master");
}

function onExit(slot, j, p) {
  const t = j.track;
  const failed = j.error || (p.code !== 0 && !p.killed);
  if (isRenderSlot(slot)) {
    t.stage = "";
    if (p.killed) { if (!t.removed) restorePrior(t); }
    else if (failed) { t.status = "error"; t.error = j.error || `Engine stopped (${p.code})`; }
    else { t.status = "done"; t.error = null; t.rendered = j.key; }
    // a Render pressed after Cancel, before the stopped render ended, starts here
    if (queueOn) runNext();
  } else if (slot.startsWith("stems-")) {
    // the sum is the track from here: its summary (an earlier master of the set), preview
    // and background analysis run as for any track
    if (failed || p.killed) {
      t.status = "error";
      t.error = p.killed ? "Stopped" : j.error || `Engine stopped (${p.code})`;
    } else if (!t.removed) {
      t.unsummed = false; t.status = "queued";
      readRows([t]);
      if (t === sel()) startJob("view", "preview", t);
    }
  } else if (slot === "analyse") {
    // a failed analysis is not shown: the track's master meets the same error and shows it
    if (!failed && !p.killed) t.analysedKey = j.akey;
    if (queueOn) runNext(); else pumpAnalyse();
  } else if (slot === "view") {
    if (failed) t.error = j.error || `Engine stopped (${p.code})`;
    t.previewed = true;
    if (j.kind === "preview" && !failed && !p.killed) wantAnalyse(t);
  }
  drawAll();
}

// Render starts the queue. Nothing renders before it is pressed, one track or many (maintainer,
// 2026-10-24: "single track should wait"); the analysis made after the preview is reused.
function render() {
  queueOn = true; queuePass += 1;
  analysePending.length = 0;             // each master analyses (or reuses) its own
  if (slots.analyse) queueFirst = slots.analyse.track;
  runNext();
  drawAll();
}

// Hands queued tracks to free render slots; the queue ends when none is left or running.
// An analysis still running is waited for (its track renders first and reuses it), so
// nothing is analysed twice and no render shares the machine with it.
function runNext() {
  if (queueOn && slots.analyse) { drawAll(); return; }
  while (queueOn && renderJobs().length < parallelRenders()) {
    const due = (x) => x.pass !== queuePass && needsRender(x);
    const t = (queueFirst && tracks.includes(queueFirst) && due(queueFirst) ? queueFirst : null)
      || tracks.find(due);
    queueFirst = null;
    if (!t) break;
    t.prior = { status: t.status };
    t.pass = queuePass; t.status = "running"; t.stage = "Analysing"; t.error = null;
    startJob("render", "master", t);
  }
  if (queueOn && !renderJobs().length) { queueOn = false; writeSummaries(); drawAll(); }
}

// Cancel: the queue stops, a running render or analysis is stopped. The stopped track goes
// back to what it was before Render: Queued, or, when it was Done, whatever its work dir
// now says (the engine clears a render's completion marker as it starts, cli.do_master,
// so a Done track is Queued once its re-render got going).
async function cancelJobs() {
  queueOn = false; queueFirst = null; analysePending.length = 0;
  const waits = [];
  if (IS_APP && slots.analyse) {
    slots.analyse.track.noAuto = true;   // not analysed again by itself until its Tempo changes
    waits.push(TAURI.core.invoke("stop_job", { slot: "analyse" }).catch(() => {}));
  }
  if (IS_APP && renderJobs().length) waits.push(TAURI.core.invoke("stop_job", { slot: "render" }).catch(() => {}));
  drawAll();
  await Promise.all(waits);
}

function restorePrior(t) {
  const was = (t.prior && t.prior.status) || "queued";
  t.status = "queued"; t.error = null;
  if (was === "done") {
    t.ui = null; t.row = null; t.rendered = null;
    if (t === sel()) {
      D = t.preview || null; pauseAll(); $("play").disabled = true;
      if (!D) startJob("view", "preview", t);
    }
    readRows([t]);
  }
}

// After its preview, a track that will need rendering is analysed in the analyse slot,
// one at a time and never while the queue runs; a Tempo change runs it again.
function wantAnalyse(t) {
  if (!IS_APP || queueOn || t.noAuto || !tracks.includes(t) || !needsRender(t)) return;
  const a = slots.analyse;
  if (t.analysedKey === analysisKey(t) || (a && a.track === t && a.akey === analysisKey(t))) return;
  if (a && a.track === t) { startJob("analyse", "analyse", t); return; }   // replaces the stale one
  if (!analysePending.includes(t)) analysePending.push(t);
  pumpAnalyse();
}

function pumpAnalyse() {
  if (queueOn || slots.analyse) return;
  while (analysePending.length) {
    const t = analysePending.shift();
    if (tracks.includes(t) && needsRender(t) && t.analysedKey !== analysisKey(t)) {
      startJob("analyse", "analyse", t);
      return;
    }
  }
}

function queuePosition() {
  const doneThisPass = tracks.filter((x) => x.pass === queuePass).length;
  const left = tracks.filter((x) => x.pass !== queuePass && needsRender(x)).length;
  return `track ${Math.max(1, doneThisPass)} of ${Math.max(1, doneThisPass + left)}`;
}

function readRows(list) {
  for (const t of list) if (!t.unsummed) summaryPending.add(t.input.toLowerCase());
  const inputs = tracks.filter((t) => summaryPending.has(t.input.toLowerCase())).map((t) => t.input);
  if (inputs.length) startJob("list", "summary", null, inputs);
}

/* -------------------------------------------------------------------- list */

function addInputs(paths) {
  const fresh = [];
  for (const p of paths || []) {
    if (!AUDIO_EXT.test(p) || tracks.some((t) => samePath(t.input, p))) continue;
    const t = { input: p, name: stem(p), status: "queued", stage: "", bpm: null, xo: null, sideHp: null,
                ui: null, preview: null, row: null, rendered: null, error: null };
    tracks.push(t); fresh.push(t);
  }
  if (!fresh.length) return;
  readRows(fresh);
  if (selected < 0) select(tracks.indexOf(fresh[0]));
  if (queueOn) runNext();
  drawAll();
}

// The hero card's track row names the selected track in full (the title holds it too, for
// a window too narrow to show it)
function showName(name) {
  const n = $("track");
  n.textContent = name || "No track";
  n.classList.toggle("none", !name);
  if (name) n.title = name; else n.removeAttribute("title");
  document.title = name || "Eluate";
}

function select(i) {
  saveTrackParams();
  pauseAll();
  selected = i;
  const t = sel();
  showName(t ? t.name : null);
  if (!t) { D = null; drawAll(); return; }
  state.bpm = t.bpm; state.xo = t.xo; state.sideHp = t.sideHp;
  D = t.ui || t.preview || null;
  if (t.unsummed) { drawAll(); return; }   // a stem set still summing: nothing to read yet
  if (isRender()) setAudio();
  if (!t.ui && t.status === "done") startJob("view", "ui-data", t);
  else if (!t.ui && !t.preview && !(slots.view && slots.view.track === t)) startJob("view", "preview", t);
  drawAll();
}

async function removeTrack(i) {
  const t = tracks[i];
  if (!t) return;
  const j = renderOf(t);
  if (j) {
    t.removed = true;
    if (IS_APP) await TAURI.core.invoke("stop_job", { slot: `render-${j.id}` }).catch(() => {});
  }
  if (slots.analyse && slots.analyse.track === t) {
    t.removed = true;
    if (IS_APP) await TAURI.core.invoke("stop_job", { slot: "analyse" }).catch(() => {});
  }
  const sj = stemsOf(t);
  if (sj) {
    t.removed = true;
    if (IS_APP) await TAURI.core.invoke("stop_job", { slot: `stems-${sj.id}` }).catch(() => {});
  }
  summaryPending.delete(t.input.toLowerCase());
  tracks.splice(i, 1);
  const next = selected === i ? Math.min(i, tracks.length - 1) : selected > i ? selected - 1 : selected;
  selected = -1;
  select(next);
}

function setData(ui, animate) {
  D = ui;
  if (isRender()) {
    setAudio();
    if (animate) reveal();
  }
  drawAll();
}

const STATUS = { queued: "Queued", done: "Done", error: "Error", summing: "Summing stems" };
const shortFlags = (r) => (r && r.flags ? r.flags.filter((f) => f.short) : []);

// The engine's stage names (app_engine) in the page's words.
const WORDS = { Report: "Writing report", Loading: "Loading graphs" };
const word = (j) => WORDS[j.stage] || j.stage;
const analysisOf = (t) => (t && slots.analyse && slots.analyse.track === t ? slots.analyse : null);

// Seconds the job has left (the engine's estimate counted down), or null: no estimate,
// or past it. An analysis the queue waits for adds the master after it (then_s).
function secondsLeft(j) {
  if (j.eta == null) return null;
  let s = j.eta - (performance.now() - j.etaAt) / 1000;
  if (j.kind === "analyse" && queueOn && j.then != null) s = Math.max(0, s) + j.then;
  return s > 0 ? s : null;
}
const aboutLeft = (s) => (s == null ? "" : s < 45 ? "under a minute left"
  : `about ${Math.max(1, Math.round(s / 60))} min left`);
const shortLeft = (s) => (s == null ? "" : s < 45 ? "<1 min" : `~${Math.max(1, Math.round(s / 60))} min`);

function rowState(t) {
  const j = renderOf(t) || analysisOf(t) || stemsOf(t);
  if (j) {
    const left = shortLeft(secondsLeft(j));
    return left ? `${j.stage} · ${left}` : `${j.stage} ${clock((performance.now() - j.t0) / 1000)}`;
  }
  return t.status === "running" ? "Analysing" : STATUS[t.status];
}

function drawList() {
  const list = $("list"), rows = $("rows");
  list.hidden = tracks.length < 2;
  if (list.hidden) { rows.innerHTML = ""; return; }
  rows.innerHTML = tracks.map((t, i) => {
    const r = t.row, fl = shortFlags(r), has = (k) => fl.some((f) => f.kind === k);
    const cell = (v, short, extra = "") => `<span class="num${short ? " short" : ""}">${v}${extra}</span>`;
    return `<div role="option" id="row-${i}" data-i="${i}" aria-selected="${i === selected}"` +
      ` class="trk st-${analysisOf(t) ? "running" : t.status}"${t.error ? ` title="${esc(t.error)}"` : ""}>` +
      `<span class="nm">${esc(t.name)}</span><span class="st">${esc(rowState(t))}</span>` +
      (r ? cell(fmt(r.loudness), has("loudness")) + cell(fmt(r.tilt), has("tilt")) +
           cell(fmt(r.crest), has("crest"), r.crestBefore != null ? `<small>${fmt(r.crestBefore)}</small>` : "") +
           cell(fmt(r.truePeak, 2), has("tp")) +
           `<span class="num${fl.length ? " short" : ""}"${fl.length ? ` title="${esc(fl.map((f) => f.text || f.kind).join("\n"))}"` : ""}>${fl.length || ""}</span>`
         : "<span></span>".repeat(5)) +
      `<button type="button" class="rm" data-rm="${i}" tabindex="-1" aria-hidden="true">×</button></div>`;
  }).join("");
  rows.setAttribute("aria-activedescendant", selected >= 0 ? `row-${selected}` : "");
}

function bindList() {
  const rows = $("rows");
  rows.addEventListener("click", (e) => {
    const rm = e.target.closest("[data-rm]");
    if (rm) { removeTrack(+rm.dataset.rm); return; }
    const row = e.target.closest("[data-i]");
    if (row && +row.dataset.i !== selected) select(+row.dataset.i);
  });
  rows.addEventListener("keydown", (e) => {
    const n = tracks.length;
    let i = selected;
    if (e.key === "ArrowDown") i = Math.min(n - 1, i + 1);
    else if (e.key === "ArrowUp") i = Math.max(0, i - 1);
    else if (e.key === "Home") i = 0;
    else if (e.key === "End") i = n - 1;
    else if (e.key === "Delete") { e.preventDefault(); removeTrack(selected); return; }
    else return;
    e.preventDefault();
    if (i !== selected) select(i);
    const row = $(`row-${selected}`);
    if (row) row.scrollIntoView({ block: "nearest" });
  });
}

// The batch summary (R4): one eluate-batch.csv per folder of the list's tracks, written
// by the shell when a queue of two or more tracks ends.
function writeSummaries() {
  if (!IS_APP || tracks.length < 2) return;
  const q = (s) => `"${String(s ?? "").replace(/"/g, '""')}"`;
  const n = (v, d = 2) => (v == null ? "" : Number(v).toFixed(d));
  const head = "Name,State,Loudness (dB RMS),Tilt (dB/oct),Crest (dB),Crest before (dB),True peak (dBTP),Flags";
  const byFolder = new Map();
  for (const t of tracks) {
    const f = folderOf(t.input);
    if (!byFolder.has(f)) byFolder.set(f, []);
    const r = t.row || {};
    byFolder.get(f).push([q(t.name), q(STATUS[t.status] || t.status), n(r.loudness), n(r.tilt), n(r.crest),
      n(r.crestBefore), n(r.truePeak), q(shortFlags(r).map((x) => x.kind).join("; "))].join(","));
  }
  for (const [folder, lines] of byFolder) {
    TAURI.core.invoke("write_summary", { folder, csv: [head, ...lines].join("\r\n") + "\r\n" }).catch(() => {});
  }
}

/* ------------------------------------------------------------------- header */

// What the engine is doing, in plain words (maintainer feedback 2026-10-24): the stage, its
// step, the solver's render and the time left. Only the stage line is a live region; the
// time beside it ticks without being announced.
function stageLine(j) {
  let step = j.step, steps = j.steps;
  if (j.kind === "analyse" && queueOn) { step = 1; steps = 4; }   // Render waits for it
  let text = step ? `Step ${step} of ${steps}: ${word(j)}` : word(j);
  if (j.render) text += ` · render ${j.render} of up to ${j.rendersMax}`;
  if (j.pass) text += ` · stems pass ${j.pass} of up to ${j.passesMax}`;
  const long = j.kind === "master" || j.kind === "analyse";
  const time = aboutLeft(secondsLeft(j)) || (long ? clock((performance.now() - j.t0) / 1000) : "");
  return { text, time };
}

const setText = (n, v) => { if (n.textContent !== v) n.textContent = v; };

// The status sits under the selected track's name (the hero card's track row), so it is
// that track's: its render or analysis, else its view job, else its state. While the
// queue renders another track, it adds which of the queue's tracks is rendering.
function drawStage() {
  const s = $("stage"), live = $("stage-text"), t = sel();
  const r = renderOf(t) || analysisOf(t) || stemsOf(t);
  const v = slots.view && slots.view.track === t ? slots.view : null;
  const j = r || v;
  const err = !j && t && t.error;
  const busy = queueOn && tracks.length > 1;
  let text = "", time = "";
  if (j) {
    ({ text, time } = stageLine(j));
    if (r && busy) text += ` · ${queuePosition()}`;
  } else if (err) text = t.error;
  else if (t && t.status === "done" && !needsRender(t)) text = "Done";
  else if (t && busy) text = STATUS[t.status] || "Queued";
  else if (t && (t.previewed || t.ui)) text = "Ready";
  if (!r && !err && t && busy) text += ` · rendering ${queuePosition()}`;
  s.classList.toggle("error", !!err);
  s.classList.toggle("running", !!j);
  if (live.getAttribute("role") !== (err ? "alert" : "status")) {
    live.setAttribute("role", err ? "alert" : "status");
    live.setAttribute("aria-live", err ? "assertive" : "polite");
  }
  setText(live, text);
  setText($("stage-time"), time);
  s.title = [text, time].filter(Boolean).join(" · ");
}

// While the queue runs (a render, or the analysis it waits for) the settings that would
// change it are locked; playback, the graphs and the list stay free. Cancel shows while
// the engine works on a render or an analysis.
const locked = () => queueOn;

function drawLock() {
  const lk = locked();
  for (const id of ["tilt", "loud", "bpm", "preserve"]) $(id).disabled = lk;
  for (const id of ["character", "delivery", "profile"]) {
    const g = $(id);
    g.setAttribute("aria-disabled", String(lk));
    for (const b of g.children) b.disabled = lk;
  }
  const c = $("cancel"), focused = document.activeElement === c;
  c.hidden = !(IS_APP && (queueOn || renderJobs().length || analysisOf(sel())));
  if (focused && c.hidden && !$("render").disabled) $("render").focus();
}

function drawButtons() {
  const r = $("render"), t = sel();
  const mine = !!renderOf(t);
  r.disabled = !IS_APP || !tracks.length || locked();
  drawLock();
  $("reveal").hidden = !(IS_APP && isRender() && !mine);
  $("log").hidden = !(isRender() && D.log && !mine);
  $("open").disabled = !IS_APP && !D;
  $("folder").hidden = !IS_APP;
  $("stems").hidden = !IS_APP;
  $("cpu-wrap").hidden = !IS_APP;
  // a stem set masters under the stems profile (cli.do_master_stems), whatever the toggle says
  const stemSet = !!(t && t.stems);
  $("profile").hidden = stemSet;
  $("profile-stems").hidden = !stemSet;
  markChanged();
}

function markChanged() {
  $("render").classList.toggle("changed", !locked() && tracks.some(needsRender));
}

/* ------------------------------------------------------------------ targets */

function timingFor(bpm) {
  return bpm < 70 ? "Slow" : bpm <= 140 ? "Moderate" : "Fast";
}

// Amber comes from the engine's checks (report.json flags_structured), the same list
// report.md reads; the page's own thresholds are only for data made before them.
function isShort(kind, fallback) {
  if (!D.flags) return fallback;
  return D.flags.some((f) => f.kind === kind && f.short);
}

function reached() {
  if (!isRender()) return null;
  const tg = tiltRef() + state.tilt;
  return {
    loud: D.loudness.reached, loudTarget: D.loudness.target,
    loudShort: isShort("loudness", D.loudness.target - D.loudness.reached > D.loudness.tol),
    tilt: D.tilt.after.stereo, tiltTarget: tg,
    tiltShort: isShort("tilt", Math.abs(D.tilt.after.stereo - tg) > D.tilt.tol),
  };
}

function drawTargets() {
  const r = reached();
  const [lo, hi] = PROFILES[state.profile].range;
  $("loud").min = lo; $("loud").max = hi;
  $("loud").value = state.loud;
  $("loud-v").textContent = fmt(state.loud);
  const lg = $("loud-got");
  lg.className = "got" + (r && r.loudShort ? " short" : "");
  lg.innerHTML = r ? `Reached ${fmt(r.loud)} dB RMS` + (r.loudShort && D.loudness.heldBy ? `<small>held by limiter</small>` : "") : "";

  $("tilt").value = state.tilt;
  $("tilt-v").textContent = fmt(state.tilt, 1, true);
  const t = $("tilt-got");
  t.className = "got" + (r && r.tiltShort ? " short" : "");
  t.innerHTML = r ? `Reached ${fmt(r.tilt)} <span class="of">/ Target ${fmt(r.tiltTarget)} dB/oct</span>` : "";

  $("preserve").setAttribute("aria-checked", String(state.preserve));
  $("preserve-v").textContent = state.preserve ? "On" : "Off";

  const b = bpmNow();
  $("bpm").value = state.bpm ?? "";
  $("bpm").placeholder = b ? fmt(b, b % 1 ? 1 : 0) : "Auto";
  $("bpm-v").textContent = state.bpm ? fmt(state.bpm, state.bpm % 1 ? 1 : 0) : "Auto";
  $("bpm-got").innerHTML = b && state.character === "auto" ? `<span class="of">${timingFor(b)}</span>` : "";

  const sm = $("summary");
  sm.innerHTML = r
    ? `<span class="item"><span class="k">Loudness</span><span class="v${r.loudShort ? " short" : ""}">${fmt(r.loud)}</span><span class="of">target ${fmt(r.loudTarget)} dB RMS</span></span>` +
      `<span class="item"><span class="k">Tilt</span><span class="v${r.tiltShort ? " short" : ""}">${fmt(r.tilt)}</span><span class="of">target ${fmt(r.tiltTarget)} dB/oct</span></span>`
    : "";
}

function bindTargets() {
  const changed = () => { drawTargets(); markChanged(); };
  $("loud").addEventListener("input", (e) => {
    state.loud = +e.target.value; changed(); drawTimeline();
  });
  $("tilt").addEventListener("input", (e) => { state.tilt = +e.target.value; changed(); drawSpectrum(); });
  $("preserve").addEventListener("click", () => { state.preserve = !state.preserve; changed(); });
  const setBpm = (v) => {
    state.bpm = Number.isFinite(v) && v > 0 ? Math.min(220, Math.max(40, Math.round(v * 2) / 2)) : null;
    changed();
    // the analysis depends on the tempo: the selected track's runs again with it
    const t = sel();
    if (t) { saveTrackParams(); t.noAuto = false; if (t.previewed) wantAnalyse(t); }
  };
  $("bpm").addEventListener("change", (e) => setBpm(e.target.value === "" ? NaN : +e.target.value));
  $("disclosure").addEventListener("click", () => {
    const btn = $("disclosure"), open = btn.getAttribute("aria-expanded") !== "true";
    btn.setAttribute("aria-expanded", String(open));
    $("controls-body").hidden = !open;
  });
}

/* ---------------------------------------------------------------- character */

function drawCharacter() {
  const seg = $("character");
  if (!seg.children.length) {
    for (const name of CHARACTERS) {
      const b = document.createElement("button");
      b.type = "button"; b.setAttribute("role", "radio"); b.dataset.v = name;
      b.textContent = name[0].toUpperCase() + name.slice(1);
      seg.append(b);
    }
  }
  for (const b of seg.children) {
    const checked = b.dataset.v === state.character;
    b.setAttribute("aria-checked", String(checked));
    b.tabIndex = checked ? 0 : -1;
  }
  bindRadioGroup(seg, (value) => {
    state.character = value; drawCharacter(); drawTargets(); markChanged();
  });
}

/* ----------------------------------------------------------------- timeline */

// the legend is HTML above the chart (.view-head), so the level strip starts near the top
const TL = { L: 52, R: 10, land: [10, 180], sec: [198, 36], gr: [242, 50], axis: 312 };
const LAND_DB = [-36, 7];
const GR_DB = 6;
let tlGeom = null;

function drawTimeline() {
  const host = $("tl");
  const w = host.clientWidth, h = host.clientHeight;
  if (!w) return;
  const pw = w - TL.L - TL.R;
  const seekable = isRender();
  host.setAttribute("aria-valuemin", "0");
  host.setAttribute("aria-valuemax", seekable ? String(D.duration) : "0");
  host.setAttribute("aria-disabled", String(!seekable));
  host.tabIndex = seekable ? 0 : -1;
  const svg = el("svg", { viewBox: `0 0 ${w} ${h}`, role: "img",
    "aria-label": "Track level chart" });
  const g = el("g");
  svg.append(g);

  const [ly, lh] = TL.land;
  const LY = (d) => ly + (1 - Math.max(0, Math.min(1, (d - LAND_DB[0]) / (LAND_DB[1] - LAND_DB[0])))) * lh;
  for (const d of [0, -12, -24]) {
    g.append(el("line", { x1: TL.L, x2: w - TL.R, y1: LY(d), y2: LY(d), stroke: "var(--rule)",
      "stroke-dasharray": d ? "1 4" : "none" }));
    g.append(el("text", { x: TL.L - 8, y: LY(d) + 4, "text-anchor": "end", class: "tl-axis" }, fmt(d, 0)));
  }
  g.append(el("line", { x1: TL.L, x2: w - TL.R, y1: LY(state.loud), y2: LY(state.loud),
    stroke: "var(--reach)", "stroke-width": 1.25 }));
  g.append(el("text", { x: TL.L - 8, y: LY(state.loud) + 4, "text-anchor": "end", class: "tl-axis",
    style: "fill: var(--reach)" }, fmt(state.loud)));
  const [gy, gh] = TL.gr;
  const base = gy + gh;
  g.append(el("line", { x1: TL.L, x2: w - TL.R, y1: base, y2: base, stroke: "var(--rule)" }));
  g.append(el("text", { x: TL.L - 8, y: base - 2, "text-anchor": "end", class: "tl-label" }, "GR"));

  if (!D) { host.replaceChildren(svg); tlGeom = null; return; }

  const n = D.env.before.length;
  const X = (i) => TL.L + (i + 0.5) * pw / n;
  const T = (t) => TL.L + t / D.duration * pw;
  const defs = el("defs");
  const clip = el("clipPath", { id: "done" });
  clip.append(el("rect", { id: "done-r", x: 0, y: -20, width: TL.L + progress * pw, height: 400 }));
  defs.append(clip); svg.prepend(defs);

  const sections = D.sections || [];
  for (const sec of sections) {
    if (!sec.flag) continue;
    g.prepend(el("rect", { x: T(sec.start), y: TL.land[0], width: T(sec.end) - T(sec.start),
      height: base - TL.land[0], fill: "var(--amber)", opacity: 0.045 }));
  }

  const lbase = ly + lh;
  const area = (arr) => `M${X(0)},${lbase}` +
    arr.map((v, i) => `L${X(i).toFixed(1)},${LY(v).toFixed(1)}`).join("") + `L${X(n - 1)},${lbase}Z`;
  const line = (arr) => arr.map((v, i) => `${i ? "L" : "M"}${X(i).toFixed(1)},${LY(v).toFixed(1)}`).join("");
  const after = el("g", { "clip-path": "url(#done)" });
  if (D.env.after) {
    after.append(el("path", { d: area(D.env.after), fill: "var(--ice)", opacity: 0.22 }));
    after.append(el("path", { d: area(D.meter.after), fill: "var(--ice)", opacity: 0.85 }));
  }
  g.append(after);
  g.append(el("path", { d: line(D.env.before), fill: "none", stroke: "var(--graphite)",
    "stroke-width": 0.75, opacity: 0.8 }));
  g.append(el("path", { d: line(D.meter.before), fill: "none", stroke: "var(--graphite)", "stroke-width": 1.5 }));
  // the target line on top of the curves
  g.append(el("line", { x1: TL.L, x2: w - TL.R, y1: LY(state.loud), y2: LY(state.loud),
    stroke: "var(--reach)", "stroke-width": 1.25 }));

  // sections
  const [sy, sh] = TL.sec;
  for (const sec of sections) {
    const x0 = T(sec.start), x1 = T(sec.end);
    g.append(el("line", { x1: x0, x2: x0, y1: sy, y2: sy + sh, stroke: "var(--rule)" }));
    if (sec.flag) g.append(el("rect", { x: x0, y: sy, width: x1 - x0, height: 2, fill: "var(--amber)" }));
    if (x1 - x0 < 30) continue;
    const label = x1 - x0 < 62 ? sec.label.slice(0, 5) : sec.label;
    g.append(el("text", { x: x0 + 6, y: sy + 15, class: sec.ref ? "sec-ref" : "sec-name" }, label));
    if (sec.flag) g.append(el("text", { x: x0 + 6, y: sy + 31, class: "sec-val" }, fmt(sec.change, 1, true)));
  }

  // limiter gain reduction, rising from its baseline
  if (D.limiterGr) {
    const GY = (d) => base - Math.min(1, Math.max(0, d) / GR_DB) * gh;
    const gp = `M${X(0)},${base}` + D.limiterGr.map((v, i) => `L${X(i).toFixed(1)},${GY(v).toFixed(1)}`).join("")
      + `L${X(n - 1)},${base}Z`;
    const gg = el("g", { "clip-path": "url(#done)" });
    gg.append(el("path", { d: gp, fill: "var(--ice)", opacity: 0.55 }));
    g.append(gg);
  }

  const step = D.duration > 600 ? 60 : 30;
  for (let t = 0; t <= D.duration; t += step) {
    g.append(el("text", { x: T(t), y: TL.axis, "text-anchor": t ? "middle" : "start", class: "tl-axis" }, clock(t)));
  }

  const head = el("line", { id: "head", x1: TL.L, x2: TL.L, y1: TL.land[0], y2: base, stroke: "var(--reach)",
    "stroke-width": 1, visibility: isRender() ? "visible" : "hidden" });
  const hover = el("line", { id: "hover", x1: 0, x2: 0, y1: TL.land[0], y2: base, stroke: "var(--dim)",
    "stroke-width": 1, "stroke-dasharray": "2 2", visibility: "hidden" });
  g.append(hover, head);

  host.replaceChildren(svg);
  tlGeom = { pw, T, n };
  placeHead();
}

function reveal() {
  if (reduceMotion) { progress = 1; return; }
  progress = 0;
  const t0 = performance.now(), total = 700;
  const step = (now) => {
    progress = Math.min(1, (now - t0) / total);
    const r = document.querySelector("#done-r");
    if (r && tlGeom) r.setAttribute("width", TL.L + (1 - (1 - progress) ** 3) * tlGeom.pw);
    if (progress < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function bindTimeline() {
  const host = $("tl"), ro = $("readout");
  host.addEventListener("pointermove", (e) => {
    if (!tlGeom) return;
    const r = host.getBoundingClientRect();
    const x = e.clientX - r.left;
    const f = (x - TL.L) / tlGeom.pw;
    const hl = host.querySelector("#hover");
    if (f < 0 || f > 1) { hl.setAttribute("visibility", "hidden"); return; }
    const i = Math.min(tlGeom.n - 1, Math.floor(f * tlGeom.n));
    hl.setAttribute("x1", x); hl.setAttribute("x2", x); hl.setAttribute("visibility", "visible");
    const t = f * D.duration;
    const sec = (D.sections || []).find((s) => t >= s.start && t < s.end);
    ro.innerHTML = `<b>${clock(t)}</b>` +
      (D.meter.after ? `<span>After RMS</span><b>${fmt(D.meter.after[i])} dBFS</b>` : "") +
      `<span>Before RMS</span><b>${fmt(D.meter.before[i])} dBFS</b>` +
      (D.limiterGr ? `<span>GR</span><b>${fmt(D.limiterGr[i])} dB</b>` : "") +
      (sec && sec.change != null ? `<span>${esc(sec.label)}</span><b>${fmt(sec.change, 1, true)} LU</b>` : "");
  });
  // the readout keeps the last position (CSS dims it once the pointer leaves)
  host.addEventListener("pointerleave", () => {
    const hl = host.querySelector("#hover");
    if (hl) hl.setAttribute("visibility", "hidden");
  });
  host.addEventListener("click", (e) => {
    if (!tlGeom || !isRender()) return;
    const r = host.getBoundingClientRect();
    const f = (e.clientX - r.left - TL.L) / tlGeom.pw;
    if (f >= 0 && f <= 1) { seek(f * D.duration); host.focus({ preventScroll: true }); }
  });
  host.addEventListener("keydown", (e) => {
    if (!tlGeom || !isRender()) return;
    const t = current().currentTime || 0, step = e.shiftKey ? 30 : 5;
    let next;
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") next = t - step;
    else if (e.key === "ArrowRight" || e.key === "ArrowUp") next = t + step;
    else if (e.key === "PageDown") next = t - 30;
    else if (e.key === "PageUp") next = t + 30;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = D.duration;
    else return;
    e.preventDefault();
    seek(next);
  });
}

/* ----------------------------------------------------------------- spectrum */

const SP = { L: 52, R: 10, T: 26, B: 22, f0: 20, f1: 20000 };

function fitLine(hz, db, lo = 100, hi = 10000) {
  const xs = [], ys = [];
  hz.forEach((f, i) => {
    if (f >= lo && f <= hi && Number.isFinite(db[i])) { xs.push(Math.log2(f / 1000)); ys.push(db[i]); }
  });
  const mx = xs.reduce((a, b) => a + b, 0) / xs.length, my = ys.reduce((a, b) => a + b, 0) / ys.length;
  let num = 0, den = 0;
  xs.forEach((x, i) => { num += (x - mx) * (ys[i] - my); den += (x - mx) ** 2; });
  const slope = num / den;
  return { slope, mx, my, at: (f) => my + slope * (Math.log2(f / 1000) - mx) };
}

function drawSpectrum() {
  $("spectrum").hidden = !isRender();
  $("graph").hidden = !isRender();
  if (!isRender()) { showGraph("level"); return; }
  const host = $("sp");
  const w = host.clientWidth, h = host.clientHeight;
  if (!w) return;
  const v = state.view;
  // the tilt fit reads the third-octave LTAS; the curves draw the 1/24-octave one when the data has it
  const hz = D.ltas.hz, after = D.ltas.after[v];
  const fine = D.ltasFine || D.ltas;
  const dhz = fine.hz, dAfter = fine.after[v];
  const dBefore = fine.before[v].map((d) => d + D.matchDb);
  const pw = w - SP.L - SP.R, ph = h - SP.T - SP.B;
  const lf0 = Math.log10(SP.f0), lf1 = Math.log10(SP.f1);
  const X = (f) => SP.L + (Math.log10(f) - lf0) / (lf1 - lf0) * pw;
  const F = (x) => 10 ** (lf0 + (x - SP.L) / pw * (lf1 - lf0));
  const vals = [];
  dhz.forEach((f, i) => { if (f >= 30) for (const d of [dBefore[i], dAfter[i]]) if (Number.isFinite(d)) vals.push(d); });
  const top = Math.ceil(Math.max(...vals) / 6) * 6 + 3, bot = Math.floor(Math.min(...vals) / 6) * 6 - 3;
  const Y = (d) => SP.T + (top - d) / (top - bot) * ph;

  const svg = el("svg", { viewBox: `0 0 ${w} ${h}`, role: "img",
    "aria-label": `Spectrum chart, ${v}` });
  for (const f of [50, 100, 200, 500, 1000, 2000, 5000, 10000]) {
    svg.append(el("line", { x1: X(f), x2: X(f), y1: SP.T, y2: SP.T + ph, stroke: "var(--rule)", opacity: 0.5 }));
    svg.append(el("text", { x: X(f), y: h - 4, "text-anchor": "middle", class: "tl-axis" },
      f >= 1000 ? `${f / 1000}k` : String(f)));
  }
  for (let d = Math.ceil((bot + 3) / 12) * 12; d <= top - 3; d += 24) {
    svg.append(el("line", { x1: SP.L, x2: w - SP.R, y1: Y(d), y2: Y(d), stroke: "var(--rule)", opacity: 0.5,
      "stroke-dasharray": "1 4" }));
    svg.append(el("text", { x: SP.L - 8, y: Y(d) + 4, "text-anchor": "end", class: "tl-axis" }, fmt(d, 0)));
  }
  svg.append(el("text", { x: SP.L - 8, y: h - 4, "text-anchor": "end", class: "tl-axis" }, "Hz"));

  const fit = fitLine(hz, after);
  const tgSlope = tiltRef() + state.tilt;
  const tgAt = (f) => fit.my + tgSlope * (Math.log2(f / 1000) - fit.mx);
  const got = D.tilt.after[v];
  const short = Math.abs(got - tgSlope) > D.tilt.tol;
  const fr = [100, 10000];
  if (short) {
    svg.append(el("polygon", { points: [
      [X(fr[0]), Y(fit.at(fr[0]))], [X(fr[1]), Y(fit.at(fr[1]))],
      [X(fr[1]), Y(tgAt(fr[1]))], [X(fr[0]), Y(tgAt(fr[0]))]].map((p) => p.join(",")).join(" "),
      fill: "var(--amber)", opacity: 0.16 }));
  }
  svg.append(el("line", { x1: X(fr[0]), x2: X(fr[1]), y1: Y(tgAt(fr[0])), y2: Y(tgAt(fr[1])),
    stroke: "var(--reach)", "stroke-width": 1.5 }));
  svg.append(el("line", { x1: X(fr[0]), x2: X(fr[1]), y1: Y(fit.at(fr[0])), y2: Y(fit.at(fr[1])),
    stroke: short ? "var(--amber)" : "var(--reach)", "stroke-width": 1, "stroke-dasharray": "4 3" }));

  const path = (arr) => arr.map((d, i) => [dhz[i], d]).filter(([f, d]) => f >= 30 && Number.isFinite(d))
    .map(([f, d], i) => `${i ? "L" : "M"}${X(f).toFixed(1)},${Y(d).toFixed(1)}`).join("");
  svg.append(el("path", { d: path(dBefore), fill: "none", stroke: "var(--graphite)", "stroke-width": 1.25,
    "stroke-linejoin": "round" }));
  svg.append(el("path", { d: path(dAfter), fill: "none", stroke: "var(--ice)", "stroke-width": 1.75,
    "stroke-linejoin": "round" }));

  // the presence and air shelves of the channels this view shows (air: "airM" / "airS")
  const chans = v === "stereo" ? ["M", "S"] : [v];
  chans.flatMap((c) => [c, "air" + c]).forEach((ch, k) => {
    const s = D.shelves && D.shelves[ch];
    if (!s || !s.gain_db) return;
    const atMax = s.gain_db >= D.shelves.max[ch] - 1e-6;
    const j = Math.max(0, dhz.findIndex((f) => f >= s.fc_hz));
    const x = X(s.fc_hz), y = Y(dAfter[j]) - 18 - k * 16;
    const col = atMax ? "var(--amber)" : "var(--ice)";
    svg.append(el("circle", { cx: x, cy: y - 4, r: 3.5, fill: "none", stroke: col, "stroke-width": 1.5 }));
    svg.append(el("text", { x: x + 9, y, class: "tl-axis", style: `fill:${col}` },
      `${v === "stereo" ? ch.replace("air", "") + " " : ""}${fmt(s.gain_db, 1, true)}${atMax ? " max" : ""}`));
  });

  const xo = crossovers() || [];
  const handles = xo.map((f, i) => ({ f, i, kind: "xo" }));
  if (v !== "M") handles.push({ f: sideHp(), i: 0, kind: "side" });
  for (const hd of handles) {
    const side = hd.kind === "side";
    const lo = side ? 20 : (hd.i ? xo[hd.i - 1] * 1.5 : 40);
    const hi = side ? 300 : (hd.i < xo.length - 1 ? xo[hd.i + 1] / 1.5 : 16000);
    const lk = locked();
    const gx = el("g", { class: (side ? "xo side" : "xo") + (lk ? " locked" : ""), tabindex: lk ? -1 : 0,
      role: "slider", "aria-disabled": String(lk),
      "aria-label": side ? "Side high-pass" : `Crossover ${hd.i + 1}`,
      "aria-orientation": "horizontal", "aria-valuemin": Math.round(lo), "aria-valuemax": Math.round(hi),
      "aria-valuenow": Math.round(hd.f), "aria-valuetext": `${Math.round(hd.f)} Hz` });
    const x = X(hd.f);
    gx.append(el("rect", { x: x - 6, y: SP.T - 18, width: 12, height: ph + 18, fill: "transparent" }));
    gx.append(el("line", { x1: x, x2: x, y1: side ? SP.T + ph * 0.55 : SP.T - 4, y2: SP.T + ph,
      stroke: side ? "var(--graphite)" : "var(--dim)", "stroke-width": 1, "stroke-dasharray": side ? "1 3" : "none" }));
    gx.append(el("text", { x, y: side ? SP.T + ph * 0.55 - 6 : SP.T - 9, "text-anchor": "middle",
      class: "tl-axis" }, side ? `Side ${Math.round(hd.f)}` : `${Math.round(hd.f)}`));
    const set = (f) => {
      if (locked()) return;
      if (side) state.sideHp = Math.min(300, Math.max(20, f));
      else {
        const cur = (state.xo = (state.xo || xo).slice());
        cur[hd.i] = Math.min(hi, Math.max(lo, f));
      }
      markChanged(); drawSpectrum();
      const again = [...$("sp").querySelectorAll(".xo")][side ? xo.length : hd.i];
      if (again) again.focus({ preventScroll: true });
    };
    gx.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      if (locked()) return;
      const r = host.getBoundingClientRect();
      const move = (ev) => set(F(ev.clientX - r.left));
      const up = () => { removeEventListener("pointermove", move); removeEventListener("pointerup", up); };
      addEventListener("pointermove", move); addEventListener("pointerup", up);
    });
    gx.addEventListener("keydown", (e) => {
      const stp = e.shiftKey ? 2 ** (1 / 3) : 2 ** (1 / 12);
      if (e.key === "ArrowRight" || e.key === "ArrowUp") { e.preventDefault(); set(hd.f * stp); }
      if (e.key === "ArrowLeft" || e.key === "ArrowDown") { e.preventDefault(); set(hd.f / stp); }
      if (e.key === "Home") { e.preventDefault(); set(lo); }
      if (e.key === "End") { e.preventDefault(); set(hi); }
    });
    svg.append(gx);
  }

  host.replaceChildren(svg);
  const st = $("spec-tilt");
  st.className = "spec-tilt" + (short ? " short" : "");
  st.innerHTML = `<span class="of">Measured</span> ${fmt(got, 2)} <span class="of">/ Target</span> ${fmt(tgSlope, 2)} dB/oct`;
  st.setAttribute("aria-label", `Tilt ${fmt(got, 2)}, target ${fmt(tgSlope, 2)} dB/oct`);
  for (const b of $("view").children) b.setAttribute("aria-checked", String(b.dataset.v === v));
}

// Track level and Spectrum share one card; both stay laid out, so each draws at full width
function showGraph(v) {
  document.querySelector(".stage-card").dataset.view = v;
  for (const b of $("graph").children) {
    b.setAttribute("aria-checked", String(b.dataset.v === v));
    b.tabIndex = b.dataset.v === v ? 0 : -1;
  }
}

// hovering or focusing a crossover / side high-pass handle names it in the spectrum's
// readout slot; the handles are redrawn on every drag, so this delegates from #sp
function bindSpectrum() {
  const sp = $("sp"), out = $("sp-readout");
  const show = (g) => {
    if (!g) { out.replaceChildren(); return; }
    const side = g.classList.contains("side");
    out.innerHTML = `<span>${side ? "Side high-pass" : "Crossover frequency"}</span><b>${esc(g.getAttribute("aria-valuetext"))}</b>`;
  };
  sp.addEventListener("pointerover", (e) => show(e.target.closest(".xo")));
  sp.addEventListener("pointermove", (e) => { const g = e.target.closest(".xo"); if (g) show(g); });
  sp.addEventListener("pointerleave", () => show(null));
  sp.addEventListener("focusin", (e) => show(e.target.closest(".xo")));
  sp.addEventListener("focusout", () => show(null));
}

/* ---------------------------------------------------------------- transport */

const audio = { before: $("a-before"), after: $("a-after") };
const current = () => audio[state.listen];

// Blind A/B (maintainer, 2026-10-26): the two level-matched files play as A and B in a random
// order per master, the vote waits until both were heard, and it is saved beside the
// master (shell write_vote, ab-votes.jsonl) before the answer shows. A vote holds no file
// name or path. Blind off before a vote shows the answer and casts none.
let blind = null;                 // { master, map: {x, y}, heard, switches, t0, done, vote }
const blindOn = () => state.blind && isRender() && blind && blind.master === D.master && !blind.done;
const sideOf = (v) => (blindOn() ? blind.map[v] : v === "x" ? "before" : "after");
const buttonOf = (side) => (blindOn() ? (blind.map.x === side ? "x" : "y") : side === "before" ? "x" : "y");

function newBlind() {
  const flip = Math.random() < 0.5;
  blind = { master: D.master, map: flip ? { x: "after", y: "before" } : { x: "before", y: "after" },
            heard: new Set(), switches: 0, t0: Date.now(), done: false, vote: null };
}

// plays button v's file from the same position, carrying on if playing
function listenTo(v) {
  const from = current(), playing = !from.paused, t = from.currentTime;
  state.abSel = v;
  state.listen = sideOf(v);
  const to = current();
  if (to !== from) {
    if (playing) from.pause();
    try { to.currentTime = t; } catch (_) { /* not loaded */ }
    if (playing) to.play().catch(() => {});
  }
  if (playing && blindOn() && !blind.heard.has(v)) { blind.heard.add(v); drawAB(); }
}

function pickAB(v) {
  for (const b of $("ab").children) {
    b.setAttribute("aria-checked", String(b.dataset.v === v));
    b.tabIndex = b.dataset.v === v ? 0 : -1;
  }
  state.abSel = v;
}

function drawAB() {
  const on = blindOn();
  const [x, y] = $("ab").children;
  x.textContent = on ? "A" : "Before";
  y.textContent = on ? "B" : "After";
  $("blind-wrap").hidden = !isRender();
  $("blind").setAttribute("aria-checked", String(state.blind));
  $("vote").hidden = !on;
  const both = on && blind.heard.size === 2;
  for (const b of $("vote").querySelectorAll("button")) b.disabled = !both;
  $("vote").title = on && !both ? "Play both A and B first" : "";
  const r = blind && blind.master === (D && D.master) && blind.done ? blind.reveal : "";
  $("ab-reveal").textContent = isRender() ? r || "" : "";
  $("share-wrap").hidden = !isRender();
  $("share").setAttribute("aria-checked", String(consent() === "yes"));
}

async function castVote(choice) {
  if (!blindOn() || blind.heard.size < 2) return;
  const masterSide = blind.map.x === "after" ? "A" : "B";
  const t = sel();
  const vote = {
    v: 1, id: crypto.randomUUID(), time: new Date().toISOString(), choice, master: masterSide,
    preferred: choice === "none" ? "none" : choice === masterSide ? "master" : "premaster",
    switches: blind.switches, listened_s: Math.round((Date.now() - blind.t0) / 1000),
    settings: t && t.rendered ? JSON.parse(t.rendered) : null,
    measures: {
      loudness: D.loudness.reached, true_peak: D.truePeak,
      crest: { before: D.crest.before, after: D.crest.after }, lra: { before: D.lra.before, after: D.lra.after },
      tilt: { before: +D.tilt.before.stereo.toFixed(2), after: +D.tilt.after.stereo.toFixed(2) },
    },
  };
  blind.vote = vote;
  if (IS_APP) await TAURI.core.invoke("write_vote", { master: D.master, vote: JSON.stringify(vote) }).catch(() => {});
  const said = choice === "none" ? "You couldn't tell them apart."
    : vote.preferred === "master" ? "You preferred the master." : "You preferred the original.";
  const c = consent();
  if (c === "yes") await queueVote(vote);
  endBlind(`${masterSide} was the master. ${said}${c === "yes" ? " Vote shared." : ""}`);
  if (c === null) showConsent(vote);       // asked once, after the first vote
}

// Sharing votes with jective.nz (maintainer, 2026-10-26): off unless the user says yes in the
// consent panel; the answer is kept in the app's storage. A shared vote is the local one
// minus its `v` (the endpoint's envelope carries the schema); the shell queues and sends it
// (votes.rs). Switching sharing off deletes what is still waiting to be sent.
const CONSENT_KEY = "eluate.shareVotes";
let pendingShare = null;                  // the vote that prompted the question

function consent() {
  try { return localStorage.getItem(CONSENT_KEY); } catch (_) { return null; }
}

function setConsent(v) {
  try { localStorage.setItem(CONSENT_KEY, v); } catch (_) { /* storage blocked: asked again next run */ }
  drawAB();
}

async function queueVote(vote) {
  const { v: _schema, ...wire } = vote;
  if (IS_APP) await TAURI.core.invoke("queue_vote", { vote: JSON.stringify(wire) }).catch(() => {});
}

function showConsent(vote) {
  pendingShare = vote;
  $("consent").hidden = false;
  $("consent-yes").focus();
}

function bindConsent() {
  $("consent-yes").addEventListener("click", async () => {
    $("consent").hidden = true;
    setConsent("yes");
    if (pendingShare) {
      await queueVote(pendingShare);
      if (blind && blind.vote === pendingShare) { blind.reveal += " Vote shared."; drawAB(); }
    }
    pendingShare = null;
  });
  $("consent-no").addEventListener("click", () => {
    $("consent").hidden = true;
    pendingShare = null;
    setConsent("no");
  });
  $("share").addEventListener("click", () => {
    if (consent() === "yes") {
      setConsent("no");
      if (IS_APP) TAURI.core.invoke("clear_outbox").catch(() => {});
    } else {
      showConsent(null);                  // switching on is the same yes, with the same words
    }
  });
  if (IS_APP && consent() === "yes") TAURI.core.invoke("flush_votes").catch(() => {});
}

// ends the blind session for this master: the buttons go back to Before / After with the
// same file still selected
function endBlind(reveal) {
  const side = state.listen;
  blind.done = true;
  blind.reveal = reveal;
  pickAB(buttonOf(side));
  drawAB();
}

function mediaUrl(path) {
  if (IS_APP) return TAURI.core.convertFileSrc(path);
  return "file:///" + path.replace(/\\/g, "/").split("/").map(encodeURIComponent).join("/").replace(/^([A-Za-z])%3A/, "$1:");
}

function setAudio() {
  pauseAll();
  for (const k of ["before", "after"]) {
    const p = D.ab && D.ab[k];
    audio[k].src = p ? mediaUrl(p) : "";
  }
  $("play").disabled = !(D.ab && D.ab.before && D.ab.after);
  // a new master starts a new blind session, on A
  if (isRender() && (!blind || blind.master !== D.master)) {
    newBlind();
    if (state.blind) { pickAB("x"); state.listen = sideOf("x"); }
    else state.listen = sideOf(state.abSel);
  }
  drawAB();
}

function pauseAll() {
  for (const a of Object.values(audio)) { a.pause(); }
  setPlayIcon(false);
}

function placeHead() {
  const hd = document.querySelector("#head");
  if (!hd || !tlGeom) return;
  const duration = D.duration || 0;
  const t = Math.min(duration, current().currentTime || 0);
  const x = tlGeom.T(t);
  hd.setAttribute("x1", x); hd.setAttribute("x2", x);
  const slider = $("tl");
  slider.setAttribute("aria-valuemax", String(isRender() ? duration : 0));
  slider.setAttribute("aria-valuenow", String(isRender() ? t : 0));
  slider.setAttribute("aria-valuetext", isRender() ? `${clock(t)} of ${clock(duration)}` : "Not available");
  $("pos").textContent = `${clock(t)} / ${clock(duration)}`;
}

function seek(t) {
  const at = Math.max(0, Math.min(D.duration, t));
  for (const a of Object.values(audio)) { try { a.currentTime = at; } catch (_) { /* not loaded */ } }
  placeHead();
}

function setPlayIcon(playing) {
  $("play").setAttribute("aria-label", playing ? "Pause" : "Play");
  $("play").innerHTML = playing
    ? '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2.5h3v11H4zM9 2.5h3v11H9z"/></svg>'
    : '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2.5v11l9-5.5z"/></svg>';
}

function bindTransport() {
  $("play").addEventListener("click", async () => {
    const a = current();
    if (a.paused) {
      try { await a.play(); setPlayIcon(true); } catch (_) { $("play").disabled = true; }
    } else { a.pause(); setPlayIcon(false); }
  });
  for (const a of Object.values(audio)) {
    a.addEventListener("timeupdate", () => { if (a === current()) placeHead(); });
    a.addEventListener("ended", () => setPlayIcon(false));
    // a side counts as heard once it has played
    a.addEventListener("play", () => {
      if (blindOn() && a === current() && !blind.heard.has(state.abSel)) { blind.heard.add(state.abSel); drawAB(); }
    });
  }
  segment("ab", (v) => {
    if (blindOn() && v !== state.abSel) blind.switches++;
    listenTo(v);
  });
  $("blind").addEventListener("click", () => {
    state.blind = !state.blind;
    if (isRender() && blind && blind.master === D.master && !blind.done) {
      if (!state.blind) {
        // switching off before a vote gives the answer away: this master's session ends
        endBlind(`${blind.map.x === "after" ? "A" : "B"} was the master. No vote.`);
      } else {
        // switching on mid-listen: a fresh random order, from A
        newBlind();
        pickAB("x");
        listenTo("x");
      }
    }
    drawAB();
  });
  for (const b of $("vote").querySelectorAll("button")) b.addEventListener("click", () => castVote(b.dataset.v));
}

function drawStats() {
  if (!isRender()) { $("stats").innerHTML = ""; return; }
  const crestLow = isShort("crest", D.crest.after != null && D.crest.after < REF_CREST);
  const noiseShort = isShort("noise", !!(D.noise && D.noise.status));
  $("stats").innerHTML =
    `<div><dt>Crest</dt><dd${crestLow ? ' class="short"' : ""}>${fmt(D.crest.after)} dB<small>Before ${fmt(D.crest.before)} dB</small></dd></div>` +
    `<div><dt>Range</dt><dd${isShort("lra", false) ? ' class="short"' : ""}>${fmt(D.lra.after)} LU<small>Before ${fmt(D.lra.before)} LU</small></dd></div>` +
    `<div><dt>True peak</dt><dd>${fmt(D.truePeak, 2)} dBTP</dd></div>` +
    (D.noise && D.noise.after_db != null ? `<div><dt>Noise</dt><dd${noiseShort ? ' class="short"' : ""}>${fmt(D.noise.after_db, 0)} dBFS` +
      `<small>Before ${fmt(D.noise.before_db, 0)} dBFS</small></dd></div>` : "");
}

/* ------------------------------------------------------------------- shared */

function bindRadioGroup(seg, onPick) {
  if (seg.dataset.radioBound) return;
  const buttons = () => [...seg.querySelectorAll('[role="radio"]')];
  const select = (button, focus = false) => {
    for (const b of buttons()) {
      const checked = b === button;
      b.setAttribute("aria-checked", String(checked));
      b.tabIndex = checked ? 0 : -1;
    }
    if (focus) button.focus();
    onPick(button.dataset.v);
  };
  for (const button of buttons()) {
    button.addEventListener("click", () => select(button));
    button.addEventListener("keydown", (e) => {
      const list = buttons(), i = list.indexOf(button);
      let next = i;
      if (e.key === "ArrowRight" || e.key === "ArrowDown") next = (i + 1) % list.length;
      else if (e.key === "ArrowLeft" || e.key === "ArrowUp") next = (i - 1 + list.length) % list.length;
      else if (e.key === "Home") next = 0;
      else if (e.key === "End") next = list.length - 1;
      else return;
      e.preventDefault();
      select(list[next], true);
    });
  }
  seg.dataset.radioBound = "true";
  const selected = buttons().find((b) => b.getAttribute("aria-checked") === "true") || buttons()[0];
  if (selected) {
    for (const b of buttons()) b.tabIndex = b === selected ? 0 : -1;
  }
}

function segment(id, onPick) {
  bindRadioGroup($(id), onPick);
}

function drawAll() {
  drawList(); drawTargets(); drawCharacter(); drawTimeline(); drawSpectrum(); drawStats(); drawButtons(); drawStage();
  drawAB();
}

async function openTrack() {
  if (!IS_APP) return;
  const got = await TAURI.core.invoke("pick_audio").catch(() => null);
  if (got) addInputs(got.inputs);
}

// Open stems: the stems (and optionally the mix, for the null test) become one track named
// after their common prefix; the shell owns the files (pick_stems) and the engine sums them
// into the set's work dir (stems-sum), then the sum is mastered under the stems profile
// with the stems report (master --stems, the shell adds the files).
async function openStems(picked) {
  if (!IS_APP) return;
  const got = picked || await TAURI.core.invoke("pick_stems").catch(() => null);
  if (!got) return;
  let t = tracks.find((x) => samePath(x.input, got.sum));
  const fresh = { stems: got.stems, mix: got.mix, unsummed: true, status: "summing", ui: null, preview: null,
                  row: null, error: null, nullTest: null };
  if (t) Object.assign(t, fresh);
  else {
    t = { input: got.sum, name: `${got.name} (${got.stems.length} stems)`, stage: "", bpm: null, xo: null,
          sideHp: null, rendered: null, ...fresh };
    tracks.push(t);
  }
  select(tracks.indexOf(t));
  startJob("stems", "stems-sum", t, [got.work]);
  drawAll();
}

async function openFolder() {
  if (!IS_APP) return;
  const got = await TAURI.core.invoke("pick_folders").catch(() => null);
  if (got) addInputs(got.inputs);
}

async function init() {
  if (!IS_APP) {
    await new Promise((res) => {
      const s = Object.assign(document.createElement("script"), { src: "data.js" });
      s.onload = s.onerror = res;
      document.head.append(s);
    });
    if (window.DRAFT) { D = window.DRAFT; showName(D.track); setAudio(); }
  }
  $("open").addEventListener("click", openTrack);
  $("folder").addEventListener("click", openFolder);
  $("stems").addEventListener("click", () => openStems());
  $("reveal").addEventListener("click", () => {
    if (isRender()) TAURI.core.invoke("reveal", { path: D.master }).catch(() => {});
  });
  $("log").addEventListener("click", () => {
    if (!isRender() || !D.log) return;
    if (IS_APP) TAURI.core.invoke("open_log", { path: D.log }).catch(() => {});
    else window.open(mediaUrl(D.log), "_blank");
  });
  $("render").addEventListener("click", () => {
    if (locked()) return;
    render();
    if (!$("cancel").hidden) $("cancel").focus();     // Render is locked now; Cancel is next
  });
  $("cancel").addEventListener("click", async () => {
    await cancelJobs();
    if (!$("render").disabled) $("render").focus();
  });
  $("cpu").addEventListener("click", () => {
    state.limitCpu = !state.limitCpu;
    $("cpu").setAttribute("aria-checked", String(state.limitCpu));
    if (queueOn) runNext();
  });
  segment("delivery", (v) => { state.delivery = v; markChanged(); });
  segment("profile", (v) => {
    // the loudness follows the profile's default unless it was moved off the old one
    if (state.loud === PROFILES[state.profile].loud) state.loud = PROFILES[v].loud;
    state.profile = v;
    const [lo, hi] = PROFILES[v].range;
    state.loud = Math.min(hi, Math.max(lo, state.loud));
    drawTargets(); drawTimeline(); markChanged();
  });
  segment("view", (v) => { state.view = v; drawSpectrum(); });
  segment("graph", showGraph);
  bindTargets(); bindTimeline(); bindSpectrum(); bindTransport(); bindConsent(); bindList();
  drawAbout();
  if (/controls/.test(location.hash)) $("disclosure").click();

  if (IS_APP) {
    const ev = TAURI.event;
    await ev.listen("engine", (e) => onEngine(e.payload));
    const tl = document.querySelector(".timeline");
    await ev.listen("tauri://drag-enter", () => tl.classList.add("drop"));
    await ev.listen("tauri://drag-leave", () => tl.classList.remove("drop"));
    // drops are expanded and granted by the shell (Rust), which sends the inputs
    await ev.listen("inputs", (e) => { tl.classList.remove("drop"); addInputs(e.payload && e.payload.inputs); });
    setInterval(() => { if (renderJobs().length || slots.view || slots.analyse) { drawStage(); drawList(); } }, 1000);
  }
  new ResizeObserver(() => { drawTimeline(); drawSpectrum(); }).observe(document.querySelector(".app"));
  drawAll();
}

// About line: the version is the shell's own (tauri.conf.json), so the tag is never written twice
async function drawAbout() {
  let v = null;
  try { if (IS_APP && TAURI.app) v = await TAURI.app.getVersion(); } catch { /* browser preview */ }
  const src = v ? `${SOURCE_REPO}/tree/v${v}` : SOURCE_REPO;
  $("about").innerHTML = `Eluate${v ? " " + esc(v) : ""} &middot; AGPL-3.0-or-later &middot; `
    + `Source: <span class="url">${esc(src)}</span>`;
}

document.fonts.ready.then(init);
