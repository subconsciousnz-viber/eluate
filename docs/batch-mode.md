# Batch mode: spec (draft, 2026-10-20)

Master several tracks in one go with the same controls, then pick any of them to see the
same before / after view the app shows for a single track today.

## 1. Assessment

What exists and carries over unchanged:

- Every master is self-contained in `<input dir>/<stem> - eluate/` (master.wav, report,
  process log, A/B files, analysis cache). A batch is a list of single masters; no new
  output layout is needed.
- `app_engine master` and `app_engine ui-data <input>` already produce everything the view
  draws (`uidata.from_render`). The per-track view needs no engine change.
- The analysis cache (`features.npz`, keyed on the audio's SHA-256) makes a re-render of a
  batch track with new controls skip the analysis.

What has to change:

| Piece | Today | Batch needs |
|---|---|---|
| Shell file picking | `pick_audio`: one file | several files, or one or more folders scanned for audio |
| Shell jobs | one slot; a new job kills the running one | a render slot (the queue) and a view slot (preview / ui-data for the selected track) that do not kill each other |
| Page state | globals for one track (`input`, `D`, `rendered`, `job`) | a track list; the existing view draws the selected track |
| Queue | none | page-driven: on each `exit`, start the next queued track |

Cost: renders take 7 to 25 min each on one core (handoff, Still owed 3), so 10 tracks are
1 to 4 hours. Batch makes the performance work more valuable but does not depend on it.

## 2. Adding tracks

- **Open** (the header file button) opens a multi-select file dialog (`blocking_pick_files`,
  same filter as today: wav, flac, aif, aiff, w64). One file chosen = a list of one (no list
  panel), which, like a list, waits for Render (maintainer, 2026-10-24: "single track should
  wait"; it used to master straight after its preview).
- **Folder**: a second header button, folder dialog (`blocking_pick_folders`, several
  folders allowed). The shell lists the audio files in each folder, **top level only**,
  sorted by name. It skips:
  - anything inside a `* - eluate` folder (the engine's own outputs: master.wav, ab/*),
    so a folder is never re-mastered from its masters;
  - names the engine writes (`master.wav`, `*_matched.wav`);
  - duplicates (the same resolved path twice).
- **Drop**: several files and/or folders dropped on the window go through the same rules.
- Adding to a list that is already rendering appends to the queue.
- Limit: 200 tracks per list (a guard against a wrong folder, e.g. a sample library).

### Granted inputs (security, closes audit 08 #2)

The shell keeps the set of paths that came from its own dialogs or from a window drop
(Rust-side `WindowEvent::DragDrop`, not the page's event). `start_job` refuses any input
not in that set. The page never names a file the user did not choose, and folder listing
stays in Rust (no `list_dir` command for the page).

## 3. The track list

A list panel between the header and the timeline, shown only when the list has two or
more tracks (one track = today's layout exactly). One row per track:

| Column | Content |
|---|---|
| Name | file stem |
| State | Queued / the engine stage and the time left (or elapsed) / Done / Error |
| Loudness | reached, against the target (amber when short, as the Controls summary) |
| Tilt | reached |
| Crest | after, with before small (as the footer) |
| True peak | dBTP |
| Flags | count of amber items (loudness short, tilt off, sections moved, noise) |

Text follows the brief's GUI text rule: names, values, state only.

- Click (or arrow keys) selects a row; the timeline, spectrum, footer stats, A/B, Log and
  Open folder show that track, exactly as today. A queued track shows its preview.
- The row's numbers come from the `data` message of its master (kept in page memory, about
  100 kB per track). A track that already has a finished render on disk (same work dir,
  `run.json`, the completion marker, and `report.json` present) is shown as Done and its
  view is loaded with `ui-data` when selected, so reopening a mastered folder shows its
  results without rendering.
- Remove: Delete key or a row button removes a queued or finished track from the list
  (never deletes files). The running track is stopped first.

## 4. Rendering

- **Render** with a list renders every track that is not Done with the current settings
  (queued, failed, stopped, or done with other settings); with one track it is today's
  Render.
- One track at a time (maintainer, 2026-10-22): a render takes about 180 MiB per million stereo
  frames plus about 0.75 GB with threads (6 min at 48 kHz: ~3.7 to 4.1 GB), and gets all
  cores but two. The parallel pool stays in the shell and the page (`PARALLEL_RENDERS` in
  ui/app.js re-enables up to three at once, one per 8 cores). Each render checks the
  machine's free memory first (`eluate.memory`): one thread if the estimate is over
  85% of it, a plain error if it is over all of it.
- **Cancel** (maintainer feedback 2026-10-24; was Stop on the Render button) stops the running
  track and the queue; the stopped track goes back to Queued (a Done track too once its
  re-render had started, as the engine clears the completion marker first; the list
  re-reads its work dir) and the rest stay Queued; Render resumes. While the queue runs,
  the settings are disabled (Tempo included).
- After a track's preview its analysis runs in the shell's `analyse` slot (one at a time,
  never while the queue runs; a Tempo change runs it again), so its master reuses it;
  Render waits for an analysis still running and starts with that track.
- An error in one track marks it Error (the one-line message as the row's title) and the
  queue moves on.
- Settings are shared by the whole list. **BPM is per track**: the Tempo field edits the
  selected track's BPM (Auto by default), since tempo is a property of the track. Changing
  a control marks the Done tracks as changed (the Render button's existing `changed`
  state) without re-rendering anything until Render.
- Selecting a track while the queue runs loads its view in the view slot; it never touches
  the render.
- Progress in the status (the hero card's track row): "track 3 of 10" after the stage text.
- While a queue runs, the shell asks Windows not to sleep (`SetThreadExecutionState`
  with `ES_SYSTEM_REQUIRED`, cleared when the queue ends).
- Closing the window with a queue running asks first (native dialog: must-know state).

## 5. Shell changes (`desktop/src-tauri/src/lib.rs`)

- `pick_audio` returns a list (`blocking_pick_files`); new `pick_folders` returns the files
  found (rules in 2). Both add to the granted set.
- Rust-side drop handling: files and folders dropped on the window are expanded and
  granted; an `inputs` event gives the page the list.
- `start_job(job, kind, inputs, args, limit_cpu, parallel)`: the shell derives the slot
  from the kind; each slot keeps its own child and stops only its own previous job.
  Kinds: `master` (a render slot per job; the shell caps them at MAX_RENDERS, the page
  runs one), `preview` and `ui-data` (view slot), `summary` (list slot), `analyse`
  (analyse slot, 2026-10-24). For `ui-data` the page sends the **input**, the shell passes
  it on and the engine derives the work dir (`uidata.work_dir`, the same rule as the
  shell's) and reads only that input, refusing a source.json that names another file
  (audit 13), so the page cannot point the engine at another folder or file.
- Window close with a running render: confirm (dialog plugin `blocking_ask`).

## 6. Engine changes

None needed for the first version. Built since: `app_engine summary <input>` (sprint WP1:
what a row needs from report.json / run.json / source.json, so a reopened folder fills its
rows without one `ui-data` job per track) and `app_engine analyse` (2026-10-24: the
background analysis after a preview, which the master reuses; stage events carry the
time-left estimate, `eluate.estimate`).

## 7. Page changes (`ui/app.js`, `index.html`, `style.css`)

- State: `tracks = [{input, name, bpm, status, stage, t0, ui, preview, settings, error}]`,
  `selected`. `D` becomes `tracks[selected].ui || .preview`; the draw functions keep
  reading `D`.
- Job ids map to a track and slot; the stale-job guard checks the slot's current id.
- Queue runner: on a render-slot `exit`, store the result on its track and start the next
  track to render.
- New element ids: `folder` (header button), `list` (the track list), and per row the data
  attributes the CSS needs. Update `docs/UI-brief.md` and `ui/README.md`.

## 8. Tests

- Shell: folder scan rules (skips `* - eluate`, `master.wav`, duplicates; top level;
  sorted; the 200 limit) as Rust unit tests of the listing function; a page-chosen input
  that was never granted is refused.
- Page: by hand through WebView2 debugging, as for the current app: a three-track folder,
  cancel and resume, an error in the middle (a corrupt file), select while rendering,
  reopen a mastered folder.

## 9. Phases

1. Shell: multi-file and folder pick, Rust-side drop, granted set, two slots, ui-data by
   input. Page: track list, queue, selection, per-track BPM. (Most of the work.)
2. Reopen a mastered folder showing Done tracks; sleep prevention; close confirmation.
3. Later: `summary` engine command (built, WP1); parallel renders after profiling (the
   pool is built but off since 2026-10-22, `PARALLEL_RENDERS`); a batch summary file (CSV
   of the row numbers) in the chosen folder (built, WP2: `eluate-batch.csv`).

## 10. Open questions for the maintainer

1. Folder scan: top level only (proposed) or subfolders too?
2. BPM per track (proposed) or shared? The other controls are shared.
3. Should Render skip tracks already Done with the same settings (proposed), or always
   re-render the whole list?
4. A batch summary file (CSV / HTML of all rows) next to the tracks: wanted in phase 1?
