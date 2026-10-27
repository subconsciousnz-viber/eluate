// Batch-mode smoke test in the real app (sprint WP3): a debug build of the shell, driven
// through WebView2's DevTools port. Three short tracks, the middle one corrupt; the first
// has a long name (A). A opened alone (maintainer feedback 2026-10-24): the status says
// "Reading track", then its analysis runs in the background ("Analysing"), then "Ready",
// and it waits there for Render ("single track should wait"). b and c join the list.
// Render: A's master reuses that analysis, the settings are locked while it runs
// (playback, graph tabs and the list are not), and Cancel stops it with A back to Queued
// (no run.json, no error). Render again must finish A and c, mark b Error and move on,
// write eluate-batch.csv, and show the selected track's graphs; one render at a time
// (with PARALLEL_RENDERS in ui/app.js set and 16 or more cores, A and c at once, WP8).
// Then a second Render, under Limit CPU, must skip the Done tracks (R3). Along the way it
// checks that the header and the hero card's track row do not cut the long name or a long
// status, at the default window size and at 900 px, and saves screenshots of the page
// (loaded, rendering, done) as PNGs.
// Usage, from desktop/: node scripts/smoke.mjs [--shots <dir>]   (rebuild the engine
// first: npm run engine). Screenshots go to <dir>, else to shots/ in the smoke's temp dir.
import { execFileSync, spawn } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const desktop = join(here, "..");
const root = join(desktop, "..");
const PORT = 9333;
const TIMEOUT_MS = 30 * 60 * 1000;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const dir = mkdtempSync(join(tmpdir(), "eluate-smoke-"));
const at = process.argv.indexOf("--shots");
const shots = at > 0 ? process.argv[at + 1] : join(dir, "shots");
mkdirSync(shots, { recursive: true });
// a long file name of the usual kind (artist - title (remix) premaster rate depth)
const A = "Halvard Moss - Tidal Ledger (Quiet Hours remix) premaster 44.1k 24bit";
const py = join(root, ".venv", "Scripts", "python.exe");
execFileSync(py, ["-c", `
import sys, soundfile as sf
from fixtures.e_signals import short_track
x, _ = short_track(44100)
sf.write(sys.argv[1], x, 44100, subtype="PCM_24")
sf.write(sys.argv[2], 0.7 * x, 44100, subtype="PCM_24")
# a stem set that sums to x (Open stems), one stem with a silent render tail
import numpy as np
d = sys.argv[3]
sf.write(d + "/song stem drums.wav", 0.6 * x, 44100, subtype="FLOAT")
sf.write(d + "/song stem keys.wav", np.vstack([0.4 * x, np.full((481, 2), 1e-5)]), 44100, subtype="FLOAT")
sf.write(d + "/song mix.wav", x, 44100, subtype="FLOAT")
`, join(dir, `${A}.wav`), join(dir, "c.wav"), dir], { cwd: root, stdio: "inherit" });
writeFileSync(join(dir, "b.wav"), "RIFF not audio");
const paths = [`${A}.wav`, "b.wav", "c.wav"].map((f) => join(dir, f));

execFileSync("node", [join(here, "stage-ui.mjs")], { stdio: "inherit" });
execFileSync("cargo", ["build", "--manifest-path", join(desktop, "src-tauri", "Cargo.toml")], { stdio: "inherit" });
// its own WebView2 profile: with the installed app open, a shared profile would join that
// app's browser process, which has no DevTools port
const app = spawn(join(desktop, "src-tauri", "target", "debug", "eluate.exe"), [], {
  env: { ...process.env, WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS: `--remote-debugging-port=${PORT}`,
         WEBVIEW2_USER_DATA_FOLDER: join(dir, "webview2") },
  stdio: "ignore",
});

let ws, seq = 0;
const pending = new Map();
async function connect() {
  for (let i = 0; i < 60; i++) {
    // WebView2 may listen on IPv6 only (::1), so both loopbacks are tried
    for (const host of ["127.0.0.1", "[::1]"]) try {
      const pages = await (await fetch(`http://${host}:${PORT}/json`, { signal: AbortSignal.timeout(2000) })).json();
      const page = pages.find((p) => p.type === "page");
      if (page) {
        ws = new WebSocket(page.webSocketDebuggerUrl);
        await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; setTimeout(rej, 5000); });
        ws.onmessage = (m) => {
          const d = JSON.parse(m.data);
          if (pending.has(d.id)) { pending.get(d.id)(d); pending.delete(d.id); }
        };
        return;
      }
    } catch { /* not up yet */ }
    await sleep(1000);
  }
  throw new Error("no DevTools page");
}
async function cdp(method, params = {}) {
  const id = ++seq;
  ws.send(JSON.stringify({ id, method, params }));
  const d = await new Promise((r) => pending.set(id, r));
  if (d.error) throw new Error(`${method}: ${JSON.stringify(d.error)}`);
  return d.result;
}
async function js(expr) {
  const r = await cdp("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true });
  if (r.exceptionDetails) throw new Error(JSON.stringify(r.exceptionDetails));
  return r.result.value;
}
const saved = [];
async function shot(name) {
  const { data } = await cdp("Page.captureScreenshot", { format: "png" });
  const f = join(shots, `${name}.png`);
  writeFileSync(f, Buffer.from(data, "base64"));
  saved.push(f);
}
// Nothing in the header or the track row is cut or spills: no element narrower than its
// text, the header's controls in one row, the status on one line, no sideways scroll.
// At 900 px (below the window's minimum) rows may wrap but nothing may be cut.
const FITS = `(() => {
  const cut = [...document.querySelectorAll(".bar > :not([hidden]), .bar button, #track, #stage, .track-actions > :not([hidden])")]
    .filter((n) => n.scrollWidth > n.clientWidth + 1).map((n) => n.id || n.className);
  const top = (id) => document.getElementById(id).getBoundingClientRect().top;
  const st = document.getElementById("stage"), lh = parseFloat(getComputedStyle(st).lineHeight);
  return { cut, sideways: document.documentElement.scrollWidth > innerWidth,
           barRows: Math.abs(top("render") - top("open")) > 8 ? 2 : 1,
           stageLines: Math.round(st.getBoundingClientRect().height / lh),
           name: document.getElementById("track").textContent };
})()`;
// a worst-case status in place of the live one while the layout is measured and shot
const LONG_STATUS = ["Step 3 of 4: Mastering \u00b7 render 2 of up to 7 \u00b7 track 1 of 3", "about 12 min left"];
async function fitsAt(width, name) {
  await js(`window.__draw = drawStage; drawStage = () => {};
    document.getElementById("stage-text").textContent = "${LONG_STATUS[0]}";
    document.getElementById("stage-time").textContent = "${LONG_STATUS[1]}"`);
  if (width) await cdp("Emulation.setDeviceMetricsOverride", { width, height: 980, deviceScaleFactor: 0, mobile: false });
  await sleep(300);
  const f = await js(FITS);
  await shot(name);
  if (width) await cdp("Emulation.clearDeviceMetricsOverride");
  await js("drawStage = __draw; drawStage()");
  return f;
}
async function until(what, expr, ms = TIMEOUT_MS, every = 2000) {
  const t0 = Date.now();
  for (;;) {
    const v = await js(expr);
    if (v) return v;
    if (Date.now() - t0 > ms) throw new Error(`timed out: ${what}`);
    await sleep(every);
  }
}
const rows = "tracks.map((t) => t.name + ':' + t.status).join(' ')";

const fails = [];
const check = (ok, what) => { console.log(`${ok ? "ok  " : "FAIL"} ${what}`); if (!ok) fails.push(what); };
try {
  await connect();
  await until("page ready", "typeof tracks !== 'undefined' && document.readyState === 'complete'", 60000);
  // every status line the page shows, in order
  await js(`window.__stages = []; new MutationObserver(() => {
    const s = document.getElementById("stage-text").textContent;
    if (__stages[__stages.length - 1] !== s) __stages.push(s);
  }).observe(document.getElementById("stage-text"), { childList: true, characterData: true, subtree: true })`);
  const size = await js("[innerWidth, innerHeight, devicePixelRatio]");
  console.log("     window:", size.join(" x "));
  check(size[0] === 1440, `the window opens at its default width (${size[0]} css px)`);
  check(await js(`document.getElementById("track").textContent === "No track"
    && document.getElementById("open").textContent.trim() === "Open" && !document.getElementById("empty")
    && !document.querySelector(".timeline button")`), "empty: track row says No track, Open is plain, no button in the graph");
  const empty = await js(FITS);
  await shot("0-empty");
  check(!empty.cut.length && !empty.sideways && empty.barRows === 1, `empty: nothing cut (${JSON.stringify(empty)})`);

  // A alone: loads, analyses, then waits for Render
  const got = await js(`__TAURI__.core.invoke("smoke_grant", { paths: ${JSON.stringify(paths.slice(0, 1))} })`);
  check(got.inputs.length === 1, "one input granted");
  const refused = await js(`__TAURI__.core.invoke("start_job", { job: 999, kind: "preview",
    inputs: [${JSON.stringify(join(root, "README.md"))}], args: [] }).then(() => "ran", (e) => String(e))`);
  check(refused === "Not a chosen file", `an input never chosen is refused (${refused})`);
  const refusedA = await js(`__TAURI__.core.invoke("start_job", { job: 998, kind: "analyse",
    inputs: [${JSON.stringify(join(root, "README.md"))}], args: [] }).then(() => "ran", (e) => String(e))`);
  check(refusedA === "Not a chosen file", `analyse is grant-checked too (${refusedA})`);

  // load: preview ("Reading track"), the background analysis, then Ready
  await until("A previewed and analysed", "tracks[0].previewed && tracks[0].analysedKey != null && !slots.analyse",
              300000, 250);
  const seen = await js("__stages");
  console.log("     status:", seen.join(" | "));
  check(seen.includes("Reading track"), "status says Reading track while the track loads");
  check(seen.some((x) => x.startsWith("Analysing")), "status says Analysing while the analysis runs");
  await sleep(3000);
  check(await js(`!queueOn && !renderJobs().length && tracks[0].status === "queued"
    && document.getElementById("list").hidden`), "one track waits for Render (no render started)");
  check(await js(`document.getElementById("stage-text").textContent`) === "Ready", "status Ready after load");
  const loaded = await js(FITS);
  await shot("1-loaded");
  check(loaded.name === A && !loaded.cut.length && !loaded.sideways && loaded.barRows === 1,
        `loaded: full name shown, nothing cut (${JSON.stringify(loaded)})`);

  await js(`__TAURI__.core.invoke("smoke_grant", { paths: ${JSON.stringify(paths.slice(1))} })`);
  await until("rows", "tracks.length === 3 && !slots.list", 60000);
  check(!(await js(`document.getElementById("list").hidden`)), "list shown for three tracks");
  check(await js(`!queueOn && !renderJobs().length && !slots.analyse`), "adding b and c starts no render");

  // Render: the master reuses the analysis; the settings are locked; Cancel stops it
  await js(`document.getElementById("render").click()`);
  check(await js(`document.activeElement.id`) === "cancel", "Render moves focus to Cancel");
  await until("A mastering", `(renderOf(tracks[0]) || {}).stage === "Mastering"`, 300000, 250);
  const lock = await js(`(() => {
    const d = (id) => document.getElementById(id).disabled;
    const seg = [...document.querySelectorAll("#character button, #delivery button, #profile button")];
    return { settings: ["tilt", "loud", "bpm", "preserve"].every(d) && seg.length > 8 && seg.every((b) => b.disabled),
             render: d("render"), cancel: !document.getElementById("cancel").hidden,
             free: [...document.querySelectorAll("#graph button, #ab button")].every((b) => !b.disabled),
             reused: renderOf(tracks[0]).logs.includes("analysis reused"),
             line: document.getElementById("stage-text").textContent,
             time: document.getElementById("stage-time").textContent };
  })()`);
  console.log("     status:", lock.line, "|", lock.time);
  check(lock.reused, "Render reused the analysis made after load (master log: analysis reused)");
  check(lock.settings && lock.render, "settings and Render disabled while rendering");
  check(lock.cancel && lock.free, "Cancel shown; graph tabs and A/B still usable");
  check(/^Step 3 of 4: Mastering · render \d of up to 7/.test(lock.line), `stage and step shown (${lock.line})`);
  check(/left$|^\d+:\d\d$/.test(lock.time), `time left shown (${lock.time})`);
  await shot("2-rendering");
  const wide = await fitsAt(0, "2-rendering-long-status");
  check(wide.name === A && !wide.cut.length && !wide.sideways && wide.barRows === 1 && wide.stageLines === 1,
        `rendering, long status: nothing cut, status on one line (${JSON.stringify(wide)})`);
  const narrow = await fitsAt(900, "2-rendering-900px");
  check(!narrow.cut.length && !narrow.sideways, `at 900 px: wraps, nothing cut (${JSON.stringify(narrow)})`);
  await js(`document.getElementById("row-2").click()`);
  check(await js(`selected === 2 && !!renderOf(tracks[0])`), "selecting another row leaves the render running");
  const other = await js(`[document.getElementById("track").textContent, document.getElementById("stage-text").textContent,
    !document.getElementById("cancel").hidden]`);
  check(other[0] === "c" && /^(Queued|Reading track) · rendering track 1 of 3$/.test(other[1]) && other[2],
        `the track row shows the selected track, the queue's progress and Cancel (${other.join(" | ")})`);
  await js(`document.getElementById("row-0").click()`);
  await js(`document.getElementById("cancel").click()`);
  await until("render stopped", "!queueOn && !renderJobs().length", 60000, 250);
  check(await js(`tracks[0].status === "queued" && !tracks[0].error`), "Cancel: A back to Queued, not Error");
  check(!existsSync(join(dir, `${A} - eluate`, "run.json")), "a stopped master leaves no run.json");
  check(await js(`!document.getElementById("tilt").disabled && !document.getElementById("render").disabled
    && document.getElementById("cancel").hidden`), "settings and Render free again after Cancel");

  await js(`window.__most = 0; setInterval(() => { __most = Math.max(__most, renderJobs().length); }, 100)`);
  await js(`document.getElementById("render").click()`);
  await until("queue end", "!queueOn && !renderJobs().length");
  console.log("     rows:", await js(rows));
  const [most, want] = await js(`[__most, parallelRenders()]`);
  check(most === Math.min(want, 3), `renders at once: ${most} of ${want}`);
  check(await js(`tracks[0].status === "done" && tracks[2].status === "done"`), "A and c done");
  check(await js(`tracks[1].status === "error"`), "corrupt b marked Error, queue moved on");
  const runA = JSON.parse(readFileSync(join(dir, `${A} - eluate`, "run.json"), "utf8"));
  check(runA.phases.analyse_s === 0, `A's master reused its analysis (analyse_s ${runA.phases.analyse_s})`);
  const csv = join(dir, "eluate-batch.csv");
  check(existsSync(csv) && readFileSync(csv, "utf8").trim().split(/\r?\n/).length === 4, "batch CSV written");
  await js(`document.getElementById("row-2").click()`);
  await until("view of c", `!slots.view && D && D.kind === "render" && D.track === "c"`, 120000);
  check(await js(`document.querySelectorAll("#tl path").length > 3`), "timeline drawn for the selected track");
  await js(`document.getElementById("row-0").click()`);
  await until("view of A", `!slots.view && D && D.kind === "render" && D.track === ${JSON.stringify(A)}`, 120000);
  check(true, "switching rows switches the graphs");
  check(await js(`document.getElementById("stage-text").textContent`) === "Done", "status Done for the selected track");
  const done = await js(FITS);
  await shot("3-done");
  check(done.name === A && !done.cut.length && !done.sideways && done.barRows === 1,
        `done: nothing cut (${JSON.stringify(done)})`);
  await js(`document.getElementById("cpu").click()`);
  check(await js(`state.limitCpu && parallelRenders() === 1`), "Limit CPU: one render at a time");
  await js(`document.getElementById("render").click()`);
  await until("second queue end", "!queueOn && !renderJobs().length");
  const again = await js(`tracks.filter((t) => t.pass === queuePass).map((t) => t.name).join(",")`);
  check(again === "b" && (await js(`tracks[1].status === "error"`)), `second Render retries only the Error track (${again})`);

  // Open stems: two stems (one with a silent render tail) and their mix become one track,
  // summed into the set's folder, null-tested, then mastered as a set with its stems report
  const S = ["song stem drums.wav", "song stem keys.wav"].map((f) => join(dir, f));
  const set = await js(`__TAURI__.core.invoke("smoke_stems", { stems: ${JSON.stringify(S)},
    mix: ${JSON.stringify(join(dir, "song mix.wav"))} })`);
  check(set && set.name === "song stem", `stem set named from the common prefix (${set && set.name})`);
  const sneaky = await js(`__TAURI__.core.invoke("start_job", { job: 998, kind: "master",
    inputs: [${JSON.stringify(set.sum)}], args: ["--stems", ${JSON.stringify(join(root, "README.md"))}] })
    .then(() => "ran", (e) => String(e))`);
  check(sneaky.includes("Not allowed"), `the page cannot pass --stems itself (${sneaky})`);
  await js(`openStems(${JSON.stringify(set)})`);
  await until("stems summed or failed",
              "tracks.some((t) => t.stems && (t.status === 'error' || (!t.unsummed && t.previewed)))", 300000);
  const st = await js(`(() => { const t = tracks.find((x) => x.stems);
    return { name: t.name, error: t.error, verdict: t.nullTest && t.nullTest.verdict }; })()`);
  check(st.name === "song stem (2 stems)" && !st.error && st.verdict === "pass",
        `stem set summed, null test passed against the mix (${JSON.stringify(st)})`);
  check(await js(`document.getElementById("profile").hidden && !document.getElementById("profile-stems").hidden`),
        "a selected stem set shows the stems profile, not the Electronic / Live toggle");
  await shot("4-stems-loaded");
  await js(`document.getElementById("render").click()`);
  await until("stems master", "!queueOn && !renderJobs().length");
  check(await js(`tracks.find((x) => x.stems).status === "done"`), "stem set mastered");
  check(existsSync(join(set.work, "stems-report.md")) && existsSync(join(set.work, "master.wav")),
        "master and stems report in the set's folder");
  check(readFileSync(join(set.work, "report.md"), "utf8").includes("Profile: stems"), "mastered under the stems profile");
  await shot("5-stems-done");
} catch (e) {
  fails.push(String(e));
  console.log("FAIL", e);
} finally {
  app.kill();
}
for (const f of saved) console.log("     shot:", f);
console.log(fails.length ? `\n${fails.length} failed` : "\nsmoke passed");
process.exit(fails.length ? 1 : 0);
