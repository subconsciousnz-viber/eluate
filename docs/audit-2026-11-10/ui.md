# Eluate UI audit (ui/index.html, ui/style.css, ui/app.js), 2026-11-10

Scope: ui/ at branch audit/2026-11-10 (fb082d1), the numbers from src/eluate/uidata.py, the protocol in
src/eluate/app_engine.py, and the shell commands in desktop/src-tauri/src/lib.rs and votes.rs. Read-only.

Method: code tracing; `node --check ui/app.js` (OK); headless Edge (puppeteer-core) on a copy of ui/
in the scratch dir with a synthetic `data.js` that follows `uidata.from_render`'s schema, plus a
harness page (`page/app.html` + `page/mock.js`) that fakes `window.__TAURI__` so the app-only state
machine (list, queue, Delete, blind votes, consent) can be driven by emitting engine events.
Scripts (the audit's session scratch, not in the repository): `audit/ui/gen.js`, `shots.js`, `test.js`, `test2.js`, `contrast.js`; screenshots `audit/ui/shot-*.png`.

Severity: S1 wrong numbers / wrong state / data loss / privacy promise broken; S2 broken interaction or
a11y blocker; S3 misleading or missing must-know state; S4 code quality; S5 nit.
Counts: **S1 4 · S2 6 · S3 13 · S4 7 · S5 8** (38 findings).

---

## S1

### 1. The tilt target shown is the current controls', not the displayed master's (and changes with the Profile toggle)
- **Where:** app.js:20-21 (`tiltRef` reads `state.profile`), 619 (`tg = tiltRef() + state.tilt`), 623-624, 642, 656, 927-930, 1015; profile handler 1407-1414 (doesn't redraw the spectrum).
- **What:** uidata sends the render's target, `tilt.target` (uidata.py:240), but the page never reads it. It rebuilds the target from the *current* Tilt slider and the *current* Profile. If you move Tilt or switch Profile after a render, the "Target" next to the reached value changes, but the master on screen hasn't changed. The amber state then contradicts the number: `tiltShort` comes from the engine flag (still not short), while the spectrum's polygon uses the page threshold against the new target.
- **Evidence (harness, synthetic render with target −4.5):** set Tilt to +1 and the summary reads `Tilt −4.4 target −3.5 dB/oct` in reach green, the spectrum reads `Target −3.50`, and the amber polygon appears. Then click Live: the summary reads `target −3.9` while the spectrum still reads `−3.50`, because it wasn't redrawn. Loudness is handled the other way (the summary uses `D.loudness.target`, the render's), so the two rows of the same summary follow different rules.
- **Fix:** show the displayed master's target. Use the controls only for the pending render (the Render `changed` state already covers that).
  ```js
  const shownTiltTarget = () => (isRender() && D.tilt.target != null ? D.tilt.target : tiltRef() + state.tilt);
  // reached(): const tg = shownTiltTarget();   drawSpectrum(): const tgSlope = shownTiltTarget();
  // keep tiltRef() only for a preview (no render yet)
  ```
  Also call `drawSpectrum()` in the Profile handler (1413).
- **Confidence:** high (reproduced).

### 2. "Slow" is shown for a detected tempo under 70 BPM, but the engine renders Moderate
- **Where:** app.js:606-608 (`timingFor`), 647-651.
- **What:** `planner.character_choice` (planner.py:1734-1739) only resolves Auto to slow for a tempo the user *gave*. A detected tempo under 70 reads moderate ("slow needs a given tempo"). The page's hint uses `bpmNow()`, which falls back to the detected `D.bpm`, and prints "Slow". The value next to Tempo is wrong whenever detection lands under 70, which is common because of octave errors on half-time material.
- **Evidence:** a synthetic render with `bpm: 62` and Tempo field empty shows `bpm-got = "Slow"` (test.js / test2.js; visible in shot-960-controls.png: "62 Auto Slow").
- **Fix (minimal):**
  ```js
  function timingFor(bpm, given) { return bpm < 70 ? (given ? "Slow" : "Moderate") : bpm <= 140 ? "Moderate" : "Fast"; }
  // drawTargets: timingFor(b, state.bpm != null)
  ```
  Better: have uidata send the resolved preset from the render (solve/run record) and display that, so the page doesn't duplicate planner thresholds.
- **Confidence:** high.

### 3. Reopening a mastered folder: Done rows aren't treated as done, and Render silently re-masters (overwrites) them with defaults
- **Where:** app.js:202-212 (adopts only bpm/xo/sideHp from `controls`), 114 (`needsRender` compares to the shared `state`), 549, 600-601.
- **What:** `summary` reads the master's `controls` (target_rms, tilt_db, preserve, streaming, character, profile). The page adopts only the per-track ones. The shared ones stay at the app defaults, so any folder mastered at non-default settings comes back with list rows saying **Done**, the status line saying **Ready**, and Render lit as `changed`. One Render press re-masters every track at defaults into the same `<name> - eluate/` folder, and the earlier master.wav, report and votes context are overwritten. batch-mode.md 3 promises that reopening "shows its results without rendering".
- **Evidence (harness):** two summaries `state:"done"` with `controls.target_rms:-6, tilt_db:1` give rows `st-done … Done` and `render.changed = true` (test.js S2). After setting `state.loud=-6, state.tilt=1` by hand, `changed = false` (S3).
- **Fix:** when no render has run in this session and no control has been touched, adopt the first Done track's shared controls. If reopened tracks disagree, keep the controls but mark those rows "Done · other settings" and never count them as needing a render until the user edits a control:
  ```js
  if (!touched && c) { state.loud = c.target_rms ?? state.loud; state.tilt = c.tilt_db || 0; state.preserve = !!c.preserve_mix;
    state.delivery = c.streaming ? "streaming" : "master"; state.character = c.character || "auto";
    if (c.profile && c.profile in PROFILES) state.profile = c.profile; syncSegs(); }
  ```
  (`touched` is set by any control handler. `syncSegs` sets aria-checked/tabIndex on #delivery/#profile.)
- **Confidence:** high (behaviour); the severity depends on whether you see the overwrite as data loss (I do: the old master is gone).

### 4. Pressing Delete twice on a rendering row removes two tracks
- **Where:** app.js:390-412 (`removeTrack` awaits `stop_job`, then `tracks.splice(i, 1)` by the stale index), 487.
- **What:** while the first call awaits the shell, a second Delete calls `removeTrack(selected)` with the same index. Both finish with `splice(i,1)`, which removes the next track too, which may be a Done track. Files aren't touched, but the list loses a track and its state, and `stop_job` is sent twice.
- **Evidence (harness, stop_job resolving after 200 ms):** a 2-track list rendering track a; two Delete keydowns leave `tracks.length = 0` and the calls show `stop_job:render-4` twice (test.js S6).
- **Fix:**
  ```js
  async function removeTrack(i) {
    const t = tracks[i]; if (!t || t.removing) return; t.removing = true;
    … await stops …
    const k = tracks.indexOf(t); if (k < 0) return;
    tracks.splice(k, 1); const cur = sel(); … select(cur && cur !== t ? tracks.indexOf(cur) : Math.min(k, tracks.length - 1));
  }
  ```
- **Confidence:** high (reproduced). (It's S1 only for the list state; files are never deleted.)

---

## S2

### 5. A master can be voted on again after its answer was revealed
- **Where:** app.js:1054-1063 (`blind` is one global object), 1205-1209 (`setAudio` starts a new session whenever `blind.master !== D.master`).
- **What:** select track A, vote, select B, then back to A, and a fresh blind session starts with a new random order, `done=false` and the vote buttons back. The user already knows which file is the master, so a second vote is contaminated, and with sharing on it goes to jective.nz. Restarting the app has the same effect.
- **Evidence:** test2.js prints "back on A: blindOn=true done=false votebuttons hidden=false".
- **Fix:** keep sessions per master (`const blinds = new Map()` keyed by `D.master`), and read `ab-votes.jsonl` presence (the shell can return whether the file exists from `ui-data`, e.g. `voted: true` in uidata) so a voted master opens unblinded.
- **Confidence:** high.

### 6. The crossover / side high-pass sliders sit inside `<svg role="img">`, so assistive tech can't reach them
- **Where:** app.js:912-913 (svg `role: "img"`), 972-976 (focusable `g role="slider"` children).
- **What:** children of `role="img"` are presentational, so screen readers don't expose the sliders or their aria-valuetext, although they're in the tab order.
- **Fix:** `el("svg", { viewBox, role: "group", "aria-label": "Spectrum" })` and put the curves in a `<g aria-hidden="true">`.
- **Confidence:** high.

### 7. Consent panel: focus isn't returned, there's no Escape, and focus is lost
- **Where:** app.js:1150-1178; index.html:171.
- **What:** `showConsent` focuses "Send my votes". "No thanks" or "Send" hides the panel, and focus stays on a hidden element (test2.js: `activeElement` still `consent-yes` after hiding), which drops it to `<body>`. Escape does nothing. When opened from the Share switch, focus should go back to the switch.
- **Fix:**
  ```js
  let consentReturn = null;
  function showConsent(vote) { pendingShare = vote; consentReturn = document.activeElement; $("consent").hidden = false; $("consent-yes").focus(); }
  function closeConsent() { $("consent").hidden = true; (consentReturn && consentReturn.isConnected ? consentReturn : $("share")).focus(); }
  $("consent").addEventListener("keydown", (e) => { if (e.key === "Escape") $("consent-no").click(); });
  ```
- **Confidence:** high.

### 8. The minimum window (960×640) isn't really usable, and the play button is squashed at every size
- **Where:** style.css:306 (`.row` columns 7.5rem / 1fr / 4.5rem / 9.5rem), 365-366 (`.seg.wide` six buttons, no wrap), 375-387 (transport doesn't wrap above 900 px), 380 (`.play` lacks `flex: none`), 408-413 / 422-424 (stats wrap into a column).
- **Evidence:** shot-960-level.png: the sticky transport is about 200 px of the 640 px window (stats stacked four high) and covers most of the level chart. shot-960-controls.png: Tilt and Loudness sliders are about 50 px wide, "Aggressive" overflows the card, and `scrollWidth` is 965 > 960. shot-1440-controls.png: the play button is an oval and "0:00 / 4:05" wraps onto two lines.
- **Fix:**
  ```css
  .play { flex: none; }  .pos { white-space: nowrap; }
  @media (max-width: 1240px) { .transport { flex-wrap: wrap; row-gap: .5rem; } .stats { flex-wrap: nowrap; } .stats dd small { display: none; } }
  @media (max-width: 1100px) { .controls-body { grid-template-columns: 1fr; } }   /* sliders get the full width */
  .seg.wide { flex-wrap: wrap; }
  ```
  Remove the 1100 px `.stats { flex-wrap: wrap }` that stacks the stats.
- **Confidence:** high (screenshots).

### 9. Amber ("short") is the only cue in most places, and screen readers get nothing (WCAG 1.4.1)
- **Where:** style.css:159 (`.trk .num.short`), 262 (`.spec-tilt.short`), 296 (`.summary .v.short`), 412 (`.stats dd.short`), 254 (`.sec-val`). Only `.got.short::before` (style.css:321) adds a dot.
- **What:** Loudness, Tilt, Crest, Range and Noise turn amber with no shape or text. Nothing in the accessible text says "short".
- **Fix:** reuse the dot everywhere, and add hidden text:
  ```css
  .short::before { content: ""; display: inline-block; width: .4rem; height: .4rem; border-radius: 50%; background: var(--amber); margin-right: .4rem; vertical-align: .12em; }
  ```
  ```js
  const SR_SHORT = '<span class="sr-only"> (flagged)</span>';   // append inside every .short value
  ```
- **Confidence:** high.

### 10. Rows in the track list have no column meaning for screen readers, and the list is rebuilt every second
- **Where:** app.js:452-470 (`role="option"` content is bare numbers; the header is `aria-hidden`), 1429 (`drawList()` every second while any job runs, which rewrites `innerHTML`).
- **What:** a screen reader reads "a Done −6.1 −3.4 8.0 10.0 −0.40 1". The flag reasons exist only in a `title`. The second-by-second rebuild also resets hover (`.rm` flickers) and can make the active option re-announce.
- **Fix:** give each option an `aria-label`, e.g. `` `${t.name}, ${rowState(t)}, loudness ${fmt(r.loudness)} dB RMS, tilt …, ${fl.length} flags` ``. Only update the `.st` cell text in the 1 s tick (`rows.querySelector(`#row-${i} .st`).textContent = …`) instead of re-rendering the list.
- **Confidence:** high (code); medium on the SR churn.

---

## S3

### 11. Engine flags that never show for a single track: true peak, stereo/mono, sub, crest loss, and every flag's text
- **Where:** app.js:1282-1292 (`drawStats` only checks crest, lra, noise; True peak has no `short` class); the flag text appears only in the list's Flags-count `title` (465), and the list exists only with 2+ tracks.
- **What:** `flags_structured` kinds are loudness | tilt | tp | section | lra | sub | stereo | crest | crest_loss | noise (evaluate.py:33-35). With one track open, a master that went out of phase in a band ("now negative: check mono") or missed the true-peak ceiling shows nothing. The engine already writes plain-language `text` for each.
- **Fix:** add `isShort("tp")` on the True peak `dd`. Add a "Flags n" stat (amber when n > 0) that opens a small list of the short flags' `text`, as must-know state (see Q2).
- **Confidence:** high.

### 12. Spectrum amber uses the page's own threshold, and compares the Mid or Side slope to the stereo target
- **Where:** app.js:929-930, 941, 1014.
- **What:** `short = |D.tilt.after[v] − tgSlope| > tol`. It doesn't use `isShort("tilt")`, so the spectrum can be amber while the summary is reach (finding 1 shows this). In the Mid or Side view it compares that channel's slope against the *stereo* target. Side slopes are normally flatter, so Side shows amber when nothing was missed.
- **Fix:** `const short = v === "stereo" ? isShort("tilt", fallback) : false;`. In the M/S views draw the target line without the polygon.
- **Confidence:** high.

### 13. Level timeline: the target line follows the slider, not the master; the preview curve doesn't follow the slider
- **Where:** app.js:734-737, 774-775 (`LY(state.loud)`); preview is shifted to the target at preview time (uidata.py:154; app.js:155).
- **What:** after a render, moving Loudness moves the "Target RMS" line and its label, but the summary's target and the curves stay the render's. Before a render, the "Before" curve is drawn shifted to the target at the moment of preview. Switching Electronic to Live (−4.1 to −10) moves the target line 6 dB but leaves the curve where it was. That suggests the track is 6 dB above target, which isn't true.
- **Fix:** render view: draw `D.loudness.target`, plus a faint dashed line at `state.loud` only when it differs. Preview: draw `meter.before + (state.loud − previewTarget)`, storing `previewTarget = D.matchDb` reference, or re-run the preview, which is cheap.
- **Confidence:** high.

### 14. Status line goes stale after a control edit, and elapsed time looks like time left
- **Where:** app.js:661 (`changed()` = drawTargets + markChanged; no drawStage/drawList), 1406 (Delivery), 549; 528 (`time = aboutLeft(...) || clock(elapsed)`).
- **What:** (a) on a Done track, editing Loudness, Tilt, Delivery etc. leaves "Done" in the status while Render lights up `changed`. `drawStage` would say "Ready", but it isn't called. (b) When the estimate runs out, the time slot switches from "about 1 min left" to an unlabeled "7:42", which reads as minutes left when it's elapsed time.
- **Fix:** call `drawStage(); drawList();` in `changed()` and in the Delivery and Profile handlers. Use `` `${clock(el)} elapsed` `` for the fallback, or "finishing" once past the estimate.
- **Confidence:** high.

### 15. Raw engine and exception text in the error line
- **Where:** app.js:229, 238, 249 (`Engine stopped (${p.code})`, including `(-1)` when the shell couldn't read a code); app_engine.py:213 (`KeyError: 'crest_db'`, `[Errno 13] Permission denied: 'C:\\…'`).
- **What:** the error is the only must-know state on a failed track, and for a non-engineer it reads as gibberish. A path in an OSError also lands in the row `title`. That's local only, never sent, so it's fine for privacy.
- **Fix:** map the common cases in the page: OSError errno 2 → "File not found", 13 → "Can't write next to the file", MemoryError/memory guard → "Not enough memory", code −1/killed by the OS → "Engine stopped unexpectedly". Keep the raw text in the Log.
- **Confidence:** medium (the wording is your call).

### 16. Live region and labels that assistive tech ignores
- **Where:** app.js:555-558 (changes the `role` of `#stage-text` between status and alert); index.html:94 (`aria-label` on `.spec-tilt`, a plain div), 111, 136 (`aria-label` on div `.targets` / `.character`), 76 and 93 (readouts `aria-hidden`).
- **What:** switching `role` on an existing live region isn't reliably announced (NVDA/Chrome announce the new text as a polite status or not at all). `aria-label` on a role-less div isn't exposed (ARIA 1.2 prohibits names on generic). The charts have no data alternative: the timeline slider's valuetext is only the time.
- **Fix:** add a separate `<p id="stage-alert" role="alert" class="sr-only"></p>` and write errors there. Give `.targets` / `.character` `role="group"`. Make the `.tl` valuetext `` `${clock(t)} of ${clock(dur)}, after ${fmt(after)} dBFS, before ${fmt(before)} dBFS${sec ? ", " + sec.label : ""}` `` on keyboard seek.
- **Confidence:** high for the generic-name issue; medium on role switching.

### 17. Non-text contrast fails on the controls and the key chart line (WCAG 1.4.11)
- **Where:** style.css:329/334 (range track `--rule-hi` 1.52:1 on panel), 337-339 (number field border `--rule` 1.21:1, field bg `--well` on `--panel` 1.07:1), 266 (crossover handle line `--rule-hi` 1.52:1), 343 (toggle off border 1.21:1, though the graphite knob at 5.39:1 carries it); app.js:774 (target line reach over the After meter fill ice@0.85: **1.08:1**, it disappears wherever the After curve is above target), 770-772 (Before line graphite over ice@0.85: 2.22:1).
- **Evidence:** contrast table below; at 200 % text the slider tracks are invisible (shot-1440-text200.png).
- **Fix:** range track `background: var(--graphite)` (5.07:1); number input `border-color: var(--graphite)`; `.xo line { stroke: var(--graphite) }`. For the target line add a dark halo: draw it twice, first `stroke: var(--steel); stroke-width: 3.25`, then reach 1.25.
- **Confidence:** high.

### 18. The blind switch can read On while nothing is blind
- **Where:** app.js:1093, 1264-1277.
- **What:** after a vote (`blind.done`), switching Blind off and on again leaves `aria-checked="true"` with buttons labelled Before/After (test2.js).
- **Fix:** when `blind.done` for this master, disable the switch, or show it off, until the next master.
- **Confidence:** high.

### 19. The vote-sharing description doesn't name everything that is sent
- **Where:** index.html:173-175; app.js:1108-1118; votes.rs:85-90 (envelope adds `app.version` and `app.os`).
- **Verified:** no audio, file name or path is in the vote (`settings` is `keyOf`: loud, tilt, preserve, delivery, character, profile, bpm, xo, sideHp; `measures` are numbers; `write_vote` gets `D.master` only as a shell argument to locate the folder, not in the payload). The POST is made by the shell (votes.rs), not the page; the page's CSP `connect-src` is IPC only.
- **What:** the payload also carries a UUID, an ISO timestamp, the listening time, the switch count, and (from the shell) app version and OS. The consent text says "Only the vote is sent: your choice, the settings and Eluate's level measurements". That's not a breach of the release-note promise, but it isn't a complete list either.
- **Fix:** add "the time, how long you listened, and the app version and OS" to the consent paragraph and the privacy page, or drop `time` (the server can stamp receipt) and round `listened_s`.
- **Confidence:** high (facts); the severity is your judgement.

### 20. Loudness only in "dB RMS"; no integrated LUFS or streaming turn-down anywhere
- **Where:** uidata.py:241-246 (the report has `integrated_lufs` but uidata doesn't send it); app.js:1286-1291.
- **What:** the audience reads LUFS. Without integrated loudness, the user can't tell that a −4 dB RMS master (around −5 LUFS) will be turned down about 9 dB on a −14 LUFS service. This is must-know state for the "Streaming" delivery choice. See Q1.
- **Confidence:** high (that it's missing).

### 21. A failed vote save is hidden
- **Where:** app.js:1120 (`write_vote … .catch(() => {})`), then the answer is revealed.
- **What:** the README says the vote is saved "before the answer shows". On a failure (folder read-only, moved) the answer still shows and the vote is lost silently.
- **Fix:** `try { await invoke(...) } catch { endBlind(`${masterSide} was the master. Vote not saved.`); return; }`.
- **Confidence:** high.

### 22. The stems null-test result is read and dropped
- **Where:** app.js:197 (`j.track.nullTest = ui.null`), never displayed.
- **What:** a passing null test with a small residual, or a set summed without a mix, looks the same as one that matched. A failed test does surface, as an error.
- **Fix:** show it as state in the status when the sum completes, e.g. "Stems summed · null −62 dB" or "· no mix", or drop the field.
- **Confidence:** medium (it depends what `null` holds; stems.null_test).

### 23. 200 % text: the layout overflows sideways
- **Where:** style.css:375-379 (transport never wraps above 900 px), 365-366, 252-255 (chart text fixed at 11 px).
- **Evidence:** shot-1440-text200.png: the transport runs off past 2000 px, Character overflows its column, "Target" is clipped under it, and the chart labels stay 11 px.
- **Fix:** as in finding 8 (wrap the transport, wrap `.seg.wide`); set chart text with `font-size: var(--s-1)` in CSS (it scales with rem).
- **Confidence:** medium. WebView2's response to Windows text size isn't verified; this is the browser's root-font 200 %.

---

## S4

### 24. Crossover drag: window listeners without pointer capture, and no re-analysis
- **Where:** app.js:994-1001.
- **What:** `addEventListener("pointermove"/"pointerup")` on window, no `pointercancel`, no `setPointerCapture`. A release outside the WebView, or a cancel, leaves the listener live, and the handle follows the mouse. Crossovers feed the analysis (`analyseArgs`), but unlike Tempo (672) a drag doesn't call `wantAnalyse`, so the next Render analyses again.
- **Fix:** `gx.setPointerCapture(e.pointerId)` and listen on `gx` for move/up/cancel; on up, `saveTrackParams(); wantAnalyse(sel())`.

### 25. Arrowing through the list starts an engine process per keypress
- **Where:** app.js:386 (select starts a `preview` job), 480-493.
- **What:** holding ArrowDown over 20 queued tracks spawns 20 `preview` engines, each killed by the next (the shell stops the slot's previous job). Each spawn loads Python and numpy.
- **Fix:** debounce the view job in `select` (`clearTimeout(viewT); viewT = setTimeout(() => startJob(...), 250)`).

### 26. Spectrum and fit with empty or partial data
- **Where:** app.js:883-887 (`fitLine` divides by 0 when no bins in 100 Hz–10 kHz), 909 (`Math.max(...[])` = −Infinity gives NaN geometry), 955 (`D.shelves.max[ch]` throws if `max` is missing), 1116 (`D.tilt.before.stereo.toFixed` throws on null and blocks the vote).
- **Fix:** guard with `if (!vals.length) return;`, `den || 1`, `(D.shelves.max || {})[ch]`, and `fmt`-style null checks in `castVote`.

### 27. CSV formula injection in eluate-batch.csv
- **Where:** app.js:500-512.
- **What:** a file named `=HYPERLINK(...)`, `+…`, `-…` or `@…` becomes a formula in Excel. Low risk, since these are the user's own names.
- **Fix:** `q = (s) => { s = String(s ?? ""); if (/^[=+\-@\t\r]/.test(s)) s = "'" + s; return `"${s.replace(/"/g,'""')}"`; }`.

### 28. Re-picking a stem set that is rendering resets it under the running job
- **Where:** app.js:1352-1355 (`Object.assign(t, fresh)` sets `status:"summing"`, `ui:null` even while `renderOf(t)` runs; then a `stems-sum` writes into the same work dir).
- **Fix:** `if (t && (renderOf(t) || stemsOf(t))) { select(tracks.indexOf(t)); return; }`.

### 29. Duplicated and magic geometry
- **Where:** app.js:730-737 and 774-775 (the target line drawn twice, once under and once over the curves; the first is dead); 783-784 (30 / 62 px label cut-offs), 957 (18 / 16 px shelf offsets), 969-970 (1.5× crossover spacing, 40 / 16000 / 20 / 300 Hz bounds, which duplicate engine limits); 791 (`GR_DB` with no axis label); two separate "short tilt" computations (624 vs 930); `PROFILES` duplicates planner.PROFILES (12-16).
- **Fix:** move the chart numbers into `TL`/`SP` (`SP.xoGap`, `SP.xoLo`, `SP.xoHi`, `SP.sideRange`); remove the first target line; have uidata send the profile ranges and crossover bounds.

### 30. Dead or unused
- **What:** `t.nullTest` (finding 22); `.spec-label` in style.css:66 and 252 (never emitted); uidata fields the page ignores: `fs`, `input`, `work`, `controls` (render), `noise.threshold_db`, `tilt.target` (finding 1), summary `bpm`/`track`/`work`; `progress` global only used by the reveal clip.
- **Module split:** not recommended. ES modules don't load from `file://` in the browser preview that ui/README relies on, and the file is cohesive by section. If anything, move the pure helpers (`fmt`, `clock`, `keyOf`, `esc`) into a classic `util.js` loaded before app.js (CSP `script-src 'self'` allows it), so they can be unit-tested with node.

---

## S5

### 31. Negative zero and "+0.0"
- **Where:** app.js:30-31.
- **Evidence:** `fmt(-0.04)` → "−0.0" and `fmt(0.04,1,true)` → "+0.0" (test.js). This hits the section change (outro −0.04 LU), tilt, true peak and shelves.
- **Fix:**
  ```js
  const fmt = (v, d = 1, sign = false) => { if (v == null || !Number.isFinite(+v)) return "\u2013";
    let s = Number(v).toFixed(d); if (+s === 0) s = s.replace("-", ""); return minus((sign && +s > 0 ? "+" : "") + s); };
  ```

### 32. Decimals and case are inconsistent
- **What:** tilt has 2 decimals in the spectrum head (1015) and 1 in the summary, Controls and list. "target" (summary, 655-656) vs "Target" (Controls, 642). The BPM placeholder prints "128.0" for a detected 127.98 (649).
- **Fix:** 1 decimal everywhere; one casing; `Math.abs(b - Math.round(b)) < 0.05 ? 0 : 1` decimals.

### 33. Legend keys don't match the marks
- **Where:** index.html:71-74; app.js:764-767, 791-795.
- **What:** "After" uses a line key but is drawn as two filled areas (peak 0.22, RMS 0.85). "Limiter GR" uses a dashed-line key but is a filled area. GR has no scale (6 dB full scale is implicit).
- **Fix:** give `.key.after` and `.key.gr` filled swatches; label the GR strip "GR 6 dB" at its top.

### 34. Missing units and titles
- **What:** crossover and side labels ("120", "Side 100") lack "Hz" (982). List numbers lack units (the header only says "Loudness"). Long names in the list are cut with no `title` (461; only the hero name has one).
- **Fix:** add `title="${esc(t.name)}"` on `.nm`; write units in the list header ("Loudness dB RMS", "TP dBTP"); add Hz to the handle labels.

### 35. The graph tabs are a radio group
- **Where:** index.html:54.
- **What:** Track level / Spectrum switch panels, so `tablist` / `tab` / `tabpanel` with `aria-controls` is the right pattern.

### 36. Render is enabled with nothing to render, and rewrites the CSV
- **Where:** app.js:585, 281.
- **Fix:** `r.disabled = … || !tracks.some(needsRender)`.

### 37. Stem sets are marked "changed" by the Profile toggle, which they ignore
- **Where:** app.js:93, 114.
- **Fix:** `profile: t.stems ? "stems" : state.profile` in `settingsFor`.

### 38. Paths compared case-insensitively on every OS
- **Where:** app.js:87.
- **Fix:** case-insensitive on Windows only (`navigator.userAgent` or a flag the shell sends).

---

## Contrast (WCAG 2.2, computed in contrast.js)

Text needs 4.5:1 (3:1 at 18.66 px bold or 24 px); UI components and meaningful graphics need 3:1.

| fg \ bg | page #0D1011 | panel #111517 | well #0A0D0E | rule #1F2729 (checked seg) |
|---|---|---|---|---|
| ice #E3ECEF | 15.93 | 15.31 | 16.26 | 12.68 |
| dim #8A908F | 5.88 | 5.66 | 6.01 | 4.68 |
| graphite #75899A | 5.28 | 5.07 | 5.39 | 4.20 (text n/a) |
| reach #8FD0BF | 10.86 | 10.44 | 11.09 | 8.65 |
| amber #F4B84A | 10.74 | 10.32 | 10.96 | 8.55 |
| red #F0565B | 5.63 | 5.41 | 5.75 | 4.48 (text n/a) |
| mark #4ECFA8 | 9.84 | 9.46 | 10.04 | 7.83 |
| rule-hi #2D383B | **1.58** | **1.52** | **1.62** | 1.26 |
| rule #1F2729 | **1.26** | **1.21** | **1.28** | 1.00 |

- All text tokens pass on all grounds. Render text (steel on ice 15.93, on reach 10.86) passes.
- Fail as components: slider track (rule-hi 1.52), number-field border (rule 1.21), crossover handle line (rule-hi 1.52), toggle off border (rule 1.21; the knob, graphite on well at 5.39, carries the state, so it passes overall).
- Fail as graphics: Target RMS line over the After RMS fill (reach on ice@0.85 ≈ #C3CBCE: **1.08**); Before RMS line over the After fill (2.22); Before over the peak fill (2.72).
- Disabled (opacity 0.45) controls are exempt (vote buttons 2.10).
- Amber's only non-colour cue: the dot on `.got.short`, and the section's amber bar plus its number. Everywhere else it's colour only (finding 9).

---

## Questions for the maintainer (UX judgements under the GUI text rule)

Each answer below would be a control name, a live value or must-know state, not explanatory copy, unless it says otherwise.

1. **Streaming loudness.** Add "Integrated −7.9 LUFS" to the footer stats, and under Streaming a live value "Turn-down −6.1 dB" (to −14 LUFS)? *Recommend yes.* It's the number this audience checks first, and the report already has `integrated_lufs`.
2. **What a flag means.** The engine already writes one plain sentence per short flag. Show them as must-know state: a "Flags 3" stat that expands to the sentences, and the same text as each amber value's accessible description? *Recommend yes.* Without them, amber is a warning with no content.
3. **Delivery.** Show the ceiling and format as values in the control ("Master −0.3 dBTP · 24-bit" / "Streaming −1 dBTP · 16-bit")? *Recommend yes* (values, not hints).
4. **Where the output went.** On Done, should the status carry the path as state ("Done · master.wav in “<name> - eluate”")? *Recommend yes.* Open folder exists, but nothing says files were written next to the source.
5. **Gain applied.** The Before curve is level-matched, so the timeline never shows how much louder the master is. Add a "Gain +5.2 dB" stat (`matchDb`)? *Recommend yes.*
6. **Character / Preserve mix / Tilt.** Tilt already shows its target as a value; under Live that's the mix's own slope (finding 1). Character shows the resolved timing only under Auto. *Recommend:* always show the resolved preset as a value, and add no copy for Preserve mix (its effect is visible in the reached values).
7. **Empty state.** "No track" over an empty chart, with legend items that have no data. Allow one navigation line in the empty chart, "Open or drop audio files"? *Recommend yes* (navigation). Hide the legend until there is data.
8. **Controls collapsed by default.** First-time users render without ever seeing Loudness or Character. *Recommend:* open until the first render, then remember the user's choice.
9. **Tooltips that are hints:** `#stems` title ("One track from several stems …"), `#profile-stems` title, the vote group's "Play both A and B first". *Recommend:* drop the first two (or make Stems' name self-describing, e.g. "Open stems"); keep the vote one, because it's the reason a control is disabled (state). Your call.
10. **Blind default.** Say why it's blind? *Recommend no copy;* the revealed answer is enough.
11. **Done with other settings.** Should the status read "Done" or "Ready" when the controls no longer match the master (finding 14)? *Recommend "Done · settings changed",* which is state and keeps the master's existence visible.
12. **Consent text** is explanatory by necessity. *Recommend keeping it* (informed consent beats the rule), completed per finding 19.
13. **Delivery/Profile in the header.** They apply to the whole list, so the header placement is right. *Recommend* keeping them, and dimming them during renders as now.

## Verified correct

- `node --check ui/app.js` passes. No inline scripts or handlers in index.html. The fonts are local (`@font-face` fonts/Archivo.ttf). CSP (tauri.conf.json:29) `script-src 'self'`, `style-src 'self' 'unsafe-inline'` (covers the SVG `style=` attributes), `media-src asset:`. data.js loads only outside the app (app.js:1373-1380), and stage-ui.mjs strips it.
- Untrusted strings in the DOM: track names, stage text, errors, section labels, flag texts and the version go through `textContent` or `esc()`. The harness injected `<img onerror>` as an error and got 0 `img` elements. `title` attributes are escaped.
- The A/B files use `convertFileSrc` on paths the shell added to the asset scope from the engine's own data message (lib.rs:518-522). `reveal`, `open_log`, `write_vote` and `write_summary` all check the granted scope in Rust.
- The vote POST is done by the shell (votes.rs), and the payload holds no audio, file name or path.
- Stale-job guards: `onEngine` matches by job id, so replaced view, list and analyse jobs are ignored. The shell stops a slot's previous job (lib.rs:451). A killed render restores the prior state (`restorePrior`). Removed tracks are skipped on data and exit.
- Lock-out: Tilt, Loudness, Tempo, Preserve, Character, Delivery, Profile and the crossover/side handles are disabled while `queueOn`, which covers the analysis Render waits for. Playback, tabs, list, Open and Limit CPU stay free.
- Focus: Render moves focus to Cancel (harness: `activeElement = cancel`). Cancel returns it to Render, and so does the queue ending while Cancel has focus (app.js:577-579).
- Radio groups: roving tabindex; Arrow, Home and End move and select. Switches are native buttons with `role="switch"` and `aria-checked`. The listbox uses `aria-activedescendant`, with Arrow, Home, End and Delete. Tab order (harness) runs header → list → track actions → graph tabs → timeline slider → Controls → play → A/B → Blind → Share, which is logical. Hidden views use `visibility`, so their controls leave the tab order.
- A/B: both files seek together (`seek`), the switch keeps the position (`listenTo`), the order is random per master (`newBlind`), and votes stay disabled until both sides have played. Blind off before a vote reveals the answer and records no vote.
- GR sign: uidata sends positive dB; the page draws upward from the baseline and prints "GR x dB". Crest before/after and LRA before/after are labelled correctly in the stats.
- `prefers-reduced-motion`: CSS removes transitions and animations, and the JS reveal skips its animation. Focus styles are visible everywhere (`:focus-visible` reach 2 px, which passes at 10.4:1).
- Section flags on the timeline come from uidata's `flag`, which uses the same threshold and gate as evaluate's section check.

## Open questions

- Does WebView2 follow Windows "Text size"? Tauri disables zoom hotkeys by default, so finding 23 may only bite if it does.
- Is the engine's reported tilt the same unweighted least-squares fit over 100 Hz–10 kHz of the third-octave LTAS that `fitLine` draws? If not, the dashed "Measured tilt" line's slope can differ from the printed number. For Live, does `planner.target_slope` (from `plan_inputs`) equal `report.tilt.before.stereo`? Finding 1's fix (use `D.tilt.target`) avoids both questions.
- Does `grant()` add the parent folder of dropped or picked *files* to `granted.dirs`? `write_summary` needs it, or the CSV silently isn't written for lists built from Open rather than Folder (the error is swallowed, app.js:512).
- `stems.null_test` result shape: what should the status show for it (finding 22)?
- Not exercised against a real engine: the stems flow end-to-end, ETA behaviour, and real WebView2 screen-reader output.
