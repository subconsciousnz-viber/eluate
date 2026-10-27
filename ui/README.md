# UI

The desktop app's page (`desktop/` wraps it). In a browser it draws `data.js` if present:
`.venv/Scripts/python -m eluate.uidata work/renders/<dir> ui/data.js`, then open
`index.html` (A/B playback reads the render's `ab/` files).

- Controls are collapsed on open; the heading carries the reached loudness and tilt.
  Inside: Tilt, Preserve mix, Tempo (a BPM field; empty = detected) on the left;
  Character (auto or one of the engine's five timing presets, `--character`) and Loudness
  on the right. Punch, Hardness and Control are kept out of the UI (maintainer, 2026-10-20: no
  Ozone-style complexity; a fine-tune panel may come later). No genre presets.
- Amber comes from the engine's checks (`flags`, report.json `flags_structured`):
  loudness short (and the limiter holding it back), tilt short, sections that rose more
  than 3 LU against the loudest section, crest below the maintainer's masters, loudness range,
  and the noise floor in the footer. The page's own thresholds are only a fallback for
  data made before the engine sent flags.
- Batch (docs/batch-mode.md): Open (several files), Folder, or a drop adds tracks to a
  list; two or more show the track list (keyboard listbox; Delete removes a row). Render
  masters every track not Done with the current settings, one at a time (maintainer,
  2026-10-22; `PARALLEL_RENDERS` in app.js restores min(3, cores / 8) at once); Limit CPU
  beside Render gives a render two engine threads at below-normal priority (sprint WP8).
  Selecting a row shows that track. Tempo, crossovers and the side high-pass belong to the selected track,
  the other controls to the list. A finished list of two or more writes
  `eluate-batch.csv` into each tracks' folder.
- Log opens the render's process-log.html (eluate.processlog) in the browser.
- Stems (maintainer, 2026-10-27; docs/noise-and-stems.md, phase C): one track from several
  stems and an optional mix, named "<common prefix> (n stems)". Its sum is written to the
  set's work dir first ("Summing stems", null test included); the sum is then the track
  for preview, analysis, graphs and A/B, and Render masters the set under the stems profile
  with its stems report.
- Listening (maintainer, 2026-10-26): the transport plays the render's level-matched A/B files
  (ab/original_matched.wav, ab/master_matched.wav) from the same position. Blind is on by
  default: the two play as A and B in a random order per master, Prefer A / B / Can't
  tell is enabled once both have played, and the vote is appended to the master's folder
  (`ab-votes.jsonl`, shell write_vote) before the answer shows. A vote carries the choice,
  which side was the master, switches, listening time, the render's settings and the
  engine's before/after measures, and no file name or path. Blind off before a vote shows
  the answer and casts none.
- Sharing votes (maintainer, 2026-10-26): off unless the user says yes in the panel shown after
  their first vote (or on switching Share votes on); the answer is kept in the app's
  storage. A shared vote (the local record with its UUID, minus `v`) is queued by the shell
  and sent to https://jective.nz/api/eluate/votes in batches, retried when offline
  (desktop/src-tauri/src/votes.rs); switching sharing off deletes what is still waiting.
- Tilt under Live moves from the mix's own slope, not from -4.5 dB/oct
  (planner.target_slope); the spectrum's target line and the readout follow it.
- Layout (maintainer feedback 2026-10-24, docs/UI-brief.md): the header holds the app-level
  controls (Open, Folder, Stems, Delivery, Profile, Limit CPU, Render); the graph card starts with
  the track row: the selected track's full name, its status under it, and Cancel, Log and
  Open folder at the right. Nothing renders until Render, one track or many.
- Progress (maintainer feedback 2026-10-24, docs/UI-brief.md): the track row's status names the
  stage in plain words ("Reading track", "Analysing", "Step 3 of 4: Mastering · render 2
  of up to 7") with the engine's estimate of the time left (eluate.estimate). After a
  track's preview its analysis runs in the background (`app_engine analyse`, the shell's
  analyse slot), so Render reuses it; Render waits for one still running. While a render
  runs the settings are disabled, and Cancel stops it (the track goes back to Queued).

Design tokens (top of `style.css`): page `#0D1011`, panel `#111517`, well `#0A0D0E` (the
Jective Marks ground), graphite `#75899A` (before), ice `#E3ECEF` (after), reach `#8FD0BF`
(targets), amber `#F4B84A` (short but accepted), mark teal `#4ECFA8` / `#8EECC9` (the mark
only). Archivo (bundled, OFL, `fonts/`), semi-condensed for names, normal width with tabular
figures for values. The mark is the Eluate chromatogram: five elution peaks (one per band) in
layered traces over a mirror-symmetric lens of strands (the low end is mono and centred),
inline in `index.html`.
