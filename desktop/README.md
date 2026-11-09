# Desktop app (Tauri, Windows)

A WebView2 window over `ui/`, running the engine (`eluate.app_engine`, frozen with
PyInstaller) as child processes in slots: a render pool (the queue; one render at a time
unless `PARALLEL_RENDERS` in ui/app.js is set, then up to three), view (the selected
track's preview or results), list (the rows of added tracks) and analyse (the background
analysis). Opening a track draws a preview and analyses it in the background; Render
masters it with the current controls. Several tracks (Open, Folder or a drop) make a list
that Render masters, one per free render slot. Stems makes one track of a stem set (its sum,
mastered under the stems profile with a stems report). The master and its files go next to the
premaster in `<name> - eluate/` (the same layout as `eluate master -o`).

## Build

Needs Rust, Node and the repo's Python 3.12 `.venv` with the package installed.

```
cd desktop
npm install
npm run engine      # dist-engine/eluate-engine/ (PyInstaller onedir; about 370 MB before
                    # tests and scikit-learn were dropped from the spec: re-measure)
npm run build       # src-tauri/target/release/bundle/nsis/Eluate_<v>_x64-setup.exe
```

Rebuild the engine after any change under `src/eluate/`; `npm run build` only bundles
what `dist-engine/` holds. `npm run dev` runs the shell against `.venv` directly
(`python -m eluate.app_engine`, or `ELUATE_PYTHON`), so engine changes need no
rebuild there.

### Linux

The same three commands, after the system packages. This is the set the beta was built
with on Ubuntu 24.04, not a minimised one:

```
sudo apt install build-essential libwebkit2gtk-4.1-dev librsvg2-dev patchelf libtbb12
```

`libtbb12` is not a Tauri dependency and is only needed for the AppImage. numba ships
`tbbpool.so`, which links `libtbb.so.12`; numba falls back to another threading layer
when it is absent and PyInstaller only warns, but linuxdeploy walks every ELF in the
frozen engine and fails on the first dependency it cannot resolve:

```
ERROR: Could not find dependency: libtbb.so.12
ERROR: Failed to deploy dependencies for existing files
```

`npm run build` writes both of

```
src-tauri/target/release/bundle/deb/Eluate_<v>_amd64.deb
src-tauri/target/release/bundle/appimage/Eluate_<v>_amd64.AppImage
```

The targets are in `src-tauri/tauri.linux.conf.json`, which tauri merges on Linux only,
so the base config keeps `nsis` and the Windows build is untouched. rpm is configured
there but kept out of `targets`: tauri puts the version straight into the RPM Version
field, which does not allow the `-` that every semver prerelease carries. At 0.1.0,
adding `"rpm"` back to `targets` is the whole change.

## Pieces

- `src-tauri/src/lib.rs`: commands `start_job` (master, analyse, preview, ui-data, summary; each
  slot stops only its own previous job; a master gets its own slot `render-<job>`, at most
  8 at once, and the page decides how many it runs), `stop_job(slot)` ("render" stops every
  render), `pick_audio` (several files),
  `pick_folders` (top level, audio only, engine outputs skipped, at most 200),
  `write_summary` (`eluate-batch.csv`, only into a chosen folder or one holding a chosen
  file), `reveal`, `open_log` (a render's process-log.html, nothing else). Drops are
  handled in Rust and sent as an `inputs` event. The engine only reads inputs the user
  chose (the granted set); `ui-data` gets the input; the engine derives the work dir and
  reads only that input (a source.json naming another file is refused, audit 13).
  Engine messages become `engine` events; the master and A/B files of a finished job are
  added to the asset scope so the page can play them. Windows is kept awake while any
  render runs; closing the window during one asks first and stops them all. Limit CPU
  (`start_job`'s `limitCpu`, sprint WP8): off, a master runs with `--threads` all cores
  but two split across the renders the page runs at once (`start_job`'s `parallel`; the page
  renders one at a time unless `PARALLEL_RENDERS` in ui/app.js is set); on,
  with `--threads 2`, and every engine child starts at below-normal priority
  (`BELOW_NORMAL_PRIORITY_CLASS`). Every engine child gets `OMP_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and `VECLIB_MAXIMUM_THREADS` set to 1 (the
  engine's own thread pool does the parallel work). `cargo test` (run by `npm run build`)
  covers the listing rules, the granted set (folder symlinks too), slot isolation (the
  analyse slot too), the render pool, the Limit CPU settings, the engine pipe reader
  (invalid UTF-8, non-object messages), the path normaliser and, on Windows, the Job Object.
  Engine stdout and stderr are read to their end whatever bytes they carry; a line that is
  not UTF-8 is read with replacement characters, and an `@@eluate` line that is not a JSON
  object is logged and skipped. Paths are canonical (symlinks resolved, so a symlinked
  file in a chosen folder is granted as its target); where `canonicalize` fails (some
  RAM-disk, virtual-drive, cloud-sync and SMB volumes) an existing path is made absolute
  instead. Paths past 259 characters keep the `\\?\` prefix.
  Engine children die with the shell: on Windows every child joins one Job Object with
  `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, so a crash or End task of `eluate.exe` kills them
  too; on Linux they get `PR_SET_PDEATHSIG` (SIGKILL). **macOS has no equivalent**: there an
  engine outlives a crashed shell until its next write to the closed pipe, which during a
  long analysis or solver pass can be minutes. Engine stderr is appended, each line tagged
  `[slot job]`, to
  `%LOCALAPPDATA%\io.github.subconsciousnz-viber.eluate\logs\engine.log` (the app
  identifier in `tauri.conf.json`; moved to `engine.log.1` once it passes 1 MB when a job
  starts). In development builds only, `ELUATE_PYTHON` or the repo's `.venv` stand in for a
  missing bundled engine; a release build never runs another interpreter.
- `src-tauri/src/votes.rs`: opt-in sending of blind A/B votes to jective.nz, the only
  network call. The shell rebuilds each vote from known fields only (id a UUID, time cut to
  the minute, choice / master / preferred, switches, listened_s, the settings and the
  level measurements, type-checked and rounded) and drops anything else; the envelope adds
  the app's name, version and OS. Votes wait in
  `%APPDATA%\io.github.subconsciousnz-viber.eluate\vote-outbox.jsonl`; a vote the endpoint
  refuses (400/415) three times moves to `vote-outbox.rejected.jsonl` beside it. Switching
  sharing off (`clear_outbox`) deletes both and stops a running sender before its next
  post; a POST already on the wire (15 s at most) cannot be recalled. The consent text in
  ui/index.html should say a shared vote also carries its time (to the minute), the
  listening time and number of A/B switches, and Eluate's version and OS.
- `engine/eluate-engine.spec`, `engine/entry.py`: the frozen engine. The shell
  sets `NUMBA_CACHE_DIR` to the app cache (librosa's numba cache needs a writable dir).
- `scripts/smoke.mjs`: batch smoke test in a debug build through WebView2's DevTools port
  (three tracks, one corrupt: Render the queue, Error handling, the CSV,
  row switching, the R3 skip under Limit CPU). `node scripts/smoke.mjs` after
  `npm run engine`; a few minutes. Debug builds add a `smoke_grant` command for it; release
  builds do not have it.
- `scripts/stage-ui.mjs`: copies `ui/` into `dist/` without `data.js` (private render
  numbers for browser previews) or the README.
- Installer: NSIS, per-user (no admin), WebView2 bootstrapper if missing. Unsigned, so
  SmartScreen warns on first run. Where WebView2 is missing (older Windows 10, LTSC) the
  bootstrapper needs the internet at install time; only `offlineInstaller` (about 130 MB
  more) avoids that.

## Release checklist

`THIRD_PARTY_NOTICES.txt` describes the Windows installer; CI checks only that its Rust
part matches `Cargo.lock`. Before a release (audit 2026-11-10, desktop S2 #3/#4; the same
list is in `tools/third_party_notices.py`):

1. Regenerate the notices from the venv the engine was built from
   (`python tools/third_party_notices.py`, after `npm run engine` so bundle extras are
   seen) and commit them with the release.
2. Per platform: a Python lock per platform (`requirements/lock-<os>-<arch>-py312.txt`)
   and the engine built from it, and a notices file generated for that platform's lock and
   Rust target, shipped in the macOS and Linux packages instead of the Windows one.
3. AppImage: credit the system libraries linuxdeploy bundles (GStreamer, WebKitGTK and
   their dependencies, LGPL-2.1+ among others), e.g. from the AppDir's `usr/lib` and the
   packages' copyright files.
4. Tag the exact commit each platform's packages were built from, so the AGPL
   Corresponding Source (build scripts and configs included) is at the tag the release
   notes name.
5. Attach the libsndfile 1.2.2, mpg123 1.32.9 and LAME 3.100 source tarballs (and the
   vcpkg port versions the DLL was built with) to the GitHub release, and point Part 2 of
   the notices at them (LGPL source offer).
6. Rebuild the engine (`npm run engine`), run `node scripts/smoke.mjs`, and re-measure the
   engine size above.
