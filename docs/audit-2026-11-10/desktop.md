# Eluate desktop shell, packaging, security, CI: audit (branch audit/2026-11-10, 2026-11-10)

Scope: desktop/src-tauri (lib.rs, votes.rs, configs, capabilities, Cargo), desktop/scripts, desktop/engine,
src/eluate/app_engine.py (protocol side), ui/app.js (vote payload, IPC calls), tools/third_party_notices.py,
THIRD_PARTY_NOTICES.txt, .github/workflows/tests.yml, origin/macos-linux-installers.

Commands run (logs in the audit's session scratch, not in the repository): `node desktop/scripts/stage-ui.mjs` + the CI engine placeholder,
`cargo test --locked` (cargo-test.log), `cargo clippy --all-targets` (clippy.log), a scratch crate probing
Windows `canonicalize` (canon/), a scratch crate checking serde_json / `lines()` behaviour (jsonpanic/),
crates.py (Cargo graph vs THIRD_PARTY_NOTICES), engine import timings from .venv.

## Results at a glance

- Findings: S1 0, S2 4, S3 5, S4 9, S5 13.
- **cargo test --locked** (cold, no cache, CI recipe): build 3 min 40 s (224 s wall). 16 passed, 0 failed, 1 ignored
  (`live_endpoint_acceptance`, which needs the network). 0 compiler warnings. The placeholder trick works.
- **cargo clippy --all-targets** (clippy 0.1.96): 165 s, exit 0. 5 warnings, all `clippy::cloned_ref_to_slice_refs` in
  test code (lib.rs:865, 950, 955, 993, 997). 0 in shipped code.
- **Privacy verdict:** the shell and page contact exactly one network endpoint, an opt-in, consent-gated POST of
  blind-A/B votes to https://jective.nz/api/eluate/votes. There is no telemetry, update check or install ID; votes
  carry a per-vote random UUID and no file names or paths. The promise holds today, but the shell enforces none of
  it: it forwards whatever JSON object the page hands it (finding 10).

---

## Findings (sorted by severity)

### S2: broken build/packaging/CI or licence non-compliance

**1. THIRD_PARTY_NOTICES.txt is stale: 12 shipped Rust crates are missing, including the whole TLS stack.**
- Where: THIRD_PARTY_NOTICES.txt:8724-8960 (Part 4), tools/third_party_notices.py:258-300, desktop/src-tauri/Cargo.toml:21 (`ureq = "2"`).
- Problem: the notices list 237 crates. The current release graph for x86_64-pc-windows-msvc has 249. Running
  the generator's own algorithm (crates.py) gives the missing set: `ureq 2.12.1`, `rustls 0.23.46`,
  `rustls-webpki 0.103.16 [ISC]`, `ring 0.17.14 [Apache-2.0 AND ISC]`, `untrusted 0.9.0 [ISC]`,
  `webpki-roots 0.26.11` and `1.0.11 [CDLA-Permissive-2.0]`, `rustls-pki-types`, `subtle [BSD-3-Clause]`,
  `zeroize`, `getrandom 0.2.17`, `base64 0.22.1`. None are listed as stale. The vote sender (votes.rs) and the
  notices were committed together in bf1abae, so the notices were generated before ureq was added. ring carries
  ISC and OpenSSL-derived notices. CDLA-Permissive-2.0 (webpki-roots' CA data) requires the agreement text to go
  with shared data.
- Evidence: `crates.py` output: "current 249 notices 237 / MISSING …" (above). `grep '^  ureq ' THIRD_PARTY_NOTICES.txt` finds nothing.
- Fix: rerun `tools/third_party_notices.py` and commit the result. Add a CI step that regenerates the file and fails on
  `git diff --exit-code THIRD_PARTY_NOTICES.txt` (at least Part 4, which needs only cargo).
- Confidence: high on staleness. Medium-high on the CDLA text obligation.

**2. Rust crate licence texts are not shipped (MIT, Apache-2.0, BSD-3, ISC notice conditions).**
- Where: tools/third_party_notices.py:283-289 ("Licence texts are not reproduced here … see crates.io").
- Problem: MIT requires "the above copyright notice and this permission notice shall be included in all copies or
  substantial portions". BSD-3 clause 2 requires reproduction "in the documentation and/or other materials
  provided with the distribution". ISC is the same. Apache-2.0 §4 requires a copy of the licence plus any NOTICE
  files. Linking to crates.io does not meet these for a binary distribution. The Python side does this correctly:
  it reproduces every LICENSE file.
- Fix: generate Part 4 with `cargo about generate` (or `cargo-bundle-licenses`) into the notices, or into a second
  shipped file such as `licenses/rust-crates.html` added to `bundle.resources`.
- Confidence: high.

**3. macOS/Linux packages ship Windows-only notices, unpinned dependencies, and source that is not at the release tag.**
- Where: tauri.linux.conf.json:22-24 (`bundleMediaFramework: true`), desktop/scripts/build-engine.mjs:14-17,
  tools/third_party_notices.py:39-43 (Windows lock, Windows DLL hash, `RUST_TARGET = x86_64-pc-windows-msvc`),
  docs/release-notes-0.1.0-beta.md "Source" section.
- Problems:
  - (a) The AppImage bundles GStreamer and, through linuxdeploy, WebKitGTK and other system `.so` files (LGPL-2.1+
    among others). None of them are credited. THIRD_PARTY_NOTICES.txt itself says "The Windows installer also ships…".
  - (b) There is only `requirements/lock-win-x64-py312.txt`, so the macOS/Linux engine is built from whatever
    pyproject's ranges resolve to that day. Its Python list and libsndfile build (different codecs and versions)
    are not the ones in the notices.
  - (c) The release notes name tag `v0.1.0-beta` (= bf1abae) as "the source of this release". The macOS/Linux build
    configuration (tauri.linux/macos.conf.json, build-engine.mjs) only exists from fb082d1 on main. AGPL §1
    "Corresponding Source" includes the scripts that control the build.
- Fix:
  - Generate per-platform notices (lock per platform, `--filter-platform` per target).
  - Add AppImage library credits, for example by listing `usr/lib` of the AppDir with package licences from dpkg.
  - Tag the commit the macOS/Linux packages were built from (for example `v0.1.0-beta.1`), or attach them only to a
    release whose tag contains their build files.
- Confidence: high for (b) and (c). Medium for (a): I could not build an AppImage here.

**4. LGPL source obligations for libsndfile, mpg123 and LAME rest on third-party URLs only.**
- Where: tools/third_party_notices.py:183-189, 207-214.
- Problem: the exact `libsndfile_x64.dll` (vcpkg build, codecs static) is shipped as a separate, replaceable DLL.
  That satisfies the relinking part of LGPL-2.1 §6. But §6(c)/(d) and LGPL-2.0 for LAME also require offering the
  library's source. The notices point at upstream repositories and at bastibe/libsndfile-binaries commit 1ac5f94,
  not at a source archive the distributor itself guarantees for 3 years. This is common practice, but strictly it
  is weaker than §6.
- Fix: put the libsndfile, mpg123 and LAME source tarballs, plus the vcpkg port versions, as assets on the GitHub
  release (or in a `third_party_src/` release asset). Add a sentence in the notices: "source available at <release
  URL> for 3 years".
- Confidence: medium (this is an interpretation question, not a mechanical one).

### S3: robustness (orphans, crashes, bad paths)

**5. One non-UTF-8 byte on engine stdout or stderr hangs the job forever. The engine blocks on a full pipe and keep-awake stays on.**
- Where: lib.rs:501 and lib.rs:514 (`BufReader::new(..).lines().map_while(Result::ok)`), lib.rs:525-532.
- Problem: `lines()` yields `Err(InvalidData)` on invalid UTF-8, and `map_while` then stops reading that pipe for good.
  - If it happens on stderr, nobody drains the pipe. Once the (≈4 KB) pipe buffer fills, the engine blocks in write.
    The stdout thread then blocks forever in `err_thread.join()`.
  - If it happens on stdout, the stdout thread goes straight to `join` and the engine blocks writing stdout.
  - Either way no `exit` event is ever sent, the render slot stays occupied, SetThreadExecutionState stays on, and
    the close dialog insists a render is running.
- Triggers: `PYTHONIOENCODING=utf-8` covers Python's own writes. It does not cover native libraries (OpenBLAS,
  LLVM/numba, libsndfile/mpg123 diagnostics) writing to fd 1/2 in the ANSI code page, for example a message
  quoting a path with "é".
- Evidence: scratch crate jsonpanic: "lines read before stop: 1 of 3" for input with one `\xff\xfe` line.
- Fix: read with `read_until(b'\n')` and `String::from_utf8_lossy`. Keep draining to EOF.
- Confidence: high on mechanism, low-medium on likelihood.

**6. An engine message that is valid JSON but not an object aborts the whole app (panic = "abort").**
- Where: lib.rs:516-522 (`msg["job"] = json!(job)`), Cargo.toml:27 (`panic = "abort"`).
- Problem: `serde_json::Value`'s `IndexMut<&str>` panics on numbers, strings, bools and arrays. A stdout line such as
  `@@eluate 5` or `@@eluate "x"` therefore panics the reader thread. The release profile aborts the process, every
  engine child is orphaned (finding 7), and the user loses the session.
- Evidence: jsonpanic: `5: PANIC — cannot access key "job" in JSON number`.
- Fix: `let Some(obj) = msg.as_object_mut() else { continue };` then `obj.insert("job".into(), json!(job))`.
- Confidence: high on mechanism. Low likelihood, since app_engine.emit always sends a dict.

**7. Engine children are not tied to the shell's lifetime (no Job Object). They are orphaned if the shell crashes or is killed.**
- Where: lib.rs:473-478 (spawn), lib.rs:790 (`Destroyed => stop_all`).
- Problem: clean shutdown is handled (Destroyed, and the close dialog's `stop_all`). A crash (finding 6), Task Manager
  "End task" on eluate.exe, or logoff kills only the shell. The engine (a single process: PyInstaller onedir, and
  the engine uses a ThreadPoolExecutor, not multiprocessing, per parallel.py:29) keeps running at full threads.
  It holds up to ~10 GB (release notes, Known limits) until its next `emit` hits the broken pipe. A long analysis
  or solver render can run minutes between emits. Solver spill directories are then left in %TEMP% until a later
  `master` sweeps them (6 h rule, solver.py:878).
- Fix: on Windows, create one Job Object at startup with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` and
  `AssignProcessToJobObject` each child after spawn (windows-sys, about 20 lines). On Unix, use `prctl(PR_SET_PDEATHSIG)`
  (Linux), or have the engine poll its stdin, which is already `Stdio::null()`. A pipe instead would show EOF when
  the parent dies.
- Confidence: high.

**8. Symlinked audio inside a chosen folder is listed but then refused at Render ("Not a chosen file").**
- Where: lib.rs:202-208 (`read_dir` entries are pushed as `c.join(name)`, not canonicalised), lib.rs:285-287
  (`check_granted` canonicalises before the lookup).
- Problem: for a file symlink in a dropped or picked folder, `granted.files` holds the link path. `start_job` then
  canonicalises the page's path to the link target, which is not in the set. The track shows in the list and fails
  on every job kind. Dedupe also misses a link and its target chosen together.
- Evidence: canon probe: `t\in\link.wav -> \\?\C:\…\t\outside\x.wav`.
- Fix: in `list_inputs`, `filter_map(|f| canon(&f))` before `is_input_audio` and push.
- Confidence: high.

**9. `std::fs::canonicalize` is the only path normaliser, and it fails on some volumes.**
- Where: lib.rs:144-156.
- Problem: GetFinalPathNameByHandle-based canonicalisation fails, or returns `\\?\Volume{GUID}\…`, on some
  RAM-disk, virtual-drive and cloud-sync filesystems and on some SMB servers. Every command then reports "File not
  found" for those inputs. Mapped drives come back as `\\server\share\…`, so the page shows UNC paths instead of Z:\.
  Stripping `\\?\` from a path longer than 260 characters hands the engine a non-verbatim long path; Python opens it
  only when LongPathsEnabled is set.
- Fix: keep canonicalize for symlink and `..` resolution, but fall back to `std::path::absolute` plus a
  case-normalised comparison when it fails (the `dunce` crate encodes the "strip only when safe" rule). Keep `\\?\`
  for paths longer than 259 characters.
- Confidence: medium (known Rust/Windows behaviour; not reproduced here on such a volume).

### S4: code quality, performance, bloat, hardening

**10. The vote privacy promise is enforced by page code only. The shell forwards any JSON object up to 4 KB.**
- Where: votes.rs:46-55 (`check_vote`: an object with a string `id`), votes.rs:84-90, ui/app.js:1108-1118, 1145-1148.
- Today's wire payload: `id` (crypto.randomUUID per vote), `time` (ISO, ms precision), `choice`, `master` (A/B),
  `preferred`, `switches`, `listened_s`, `settings` {loud, tilt, preserve, delivery, character, profile, bpm, xo,
  sideHp}, and `measures` {loudness, true_peak, crest/lra/tilt before and after}. The envelope adds
  `app.name/version/os`, and the User-Agent repeats `Eluate/<ver> (<os>)`.
  - No file names, paths, audio, machine IDs or install IDs. A per-vote UUID is not an install identifier.
  - The `time` + IP pair, and the seven measures plus bpm (which together nearly fingerprint a specific track), are
    inherent to the feature. The consent text discloses them as "settings and level measurements".
- Problem: any page bug or XSS could put arbitrary data (for example `D.master`, a full path) into the queue, and
  the shell would send it.
- Fix: whitelist the wire keys and types in Rust (`check_vote` building a new object from known fields, numbers
  rounded, strings limited to enums). Optionally truncate `time` to the minute.
- Confidence: high.

**11. Opting out can still send one in-flight batch, and a 400/415 "Stop" wedges the queue forever.**
- Where: votes.rs:151-158 (batch read into memory, then `post`), votes.rs:218-225 (`clear_outbox`), votes.rs:182.
- Problems:
  - If sharing is switched off while a POST is in flight (≤15 s, TIMEOUT_S), that batch of up to 50 votes still
    arrives. They were consented to when queued, so this is minor.
  - `Stop` keeps the batch. Every later flush re-sends the same first 50, gets the same 400, and stops, so every
    later vote stays stuck in `%APPDATA%\<identifier>\vote-outbox.jsonl` indefinitely.
- Fix:
  - Add a `generation: AtomicU64` bumped by `clear_outbox`. The sender checks it before `post`, and the page can
    `await clear_outbox` before reporting "off".
  - On 400/415, move the batch to `vote-outbox.rejected.jsonl` (or drop it) after N attempts.
- Confidence: high.

**12. Engine bundle bloat: about 60 MB of tests and unused scikit-learn.**
- Where: eluate-engine.spec:10-19 (`collect_all` over numpy, scipy, librosa, numba, llvmlite, sklearn, …).
- Measured in .venv: package tests directories are numpy 6.7 MB, scipy 16.6 MB, sklearn 5.7 MB, numba 4.7 MB
  (≈34 MB). scikit-learn is 26.9 MB in total, and librosa imports it only lazily, inside `librosa.decompose` and
  `librosa.segment` functions, which the engine never calls. The engine uses beat_track, onset_strength, pyin,
  mfcc, chroma_stft and chroma_cqt. llvmlite.dll is 115 MB and unavoidable while numba is used.
- Fix:
  - Replace `collect_all` with `collect_submodules(pkg, filter=lambda m: ".tests" not in m and ".testing" not in m)`
    plus `collect_data_files(pkg, excludes=["**/tests/**"])`.
  - Add `"sklearn"` to excludes, and verify with a frozen-engine smoke (`node scripts/smoke.mjs`).
  - Expected saving: roughly 50-60 MB of ~370 MB.
- Confidence: medium (sizes measured on site-packages, not on a built bundle; no PyInstaller in the venv).

**13. Every engine job, even `summary`, `ui-data` and `preview`, pays about 3-5 s of scientific-stack import.**
- Where: app_engine.py:60-61 (`_parser()` imports `eluate.cli` before parsing).
- Measured from source (warm, noisy, with Defender active):
  - `python -m eluate.app_engine --help`: 4.4-7.9 s (5 runs).
  - Bare interpreter: ~1 s. `import scipy.signal`: ~5 s, of which 1.4 s is `numpy.testing`, pulled in by
    `scipy._lib._array_api`. `import eluate.cli`: 2.6 s. `import eluate.uidata`: 3.1 s.
  - The frozen engine was not built here, so its time is an open question.
- Problem: selecting a track starts `ui-data` or `preview`. The import cost dominates their latency.
- Fix:
  - Build the parser without importing `cli`: move `_add_controls`, `_xover` and the choice lists into a light
    `eluate.cliargs` module.
  - Longer term, keep one warm engine process for the view/list slots (a JSON-lines request loop).
- Confidence: medium.

**14. Synchronous commands and the drop handler do filesystem work on the UI thread.**
- Where: lib.rs:304-311 (`in_granted_scope`, which canonicalises the work dir of every granted file on each call),
  lib.rs:640-651 (`write_summary`), lib.rs:676-697 (write_vote, reveal, open_log), lib.rs:765-768 (drop:
  `read_dir` + `is_file` stat of every entry before the 200 cap applies, lib.rs:202-205).
- Problem: in Tauri, a `#[tauri::command]` without `async` runs on the main thread (Tauri changelog / IPC guide,
  see Sources). With 200 tracks on an SMB share, one Reveal or vote does up to 400 network round trips while the
  window is frozen. A drop of a 10,000-file sample folder stats every file on the UI thread.
- Fix:
  - Make these commands `async` (or `#[tauri::command(async)]`).
  - Precompute the canonical work dir per granted file once, at grant time.
  - In the drop handler, spawn the listing on a thread and emit `inputs` when it is done.
  - Stop listing after MAX_INPUTS audio hits.
- Confidence: high on placement, medium on impact.

**15. The engine build is not reproducible, and it fails on uv-created venvs.**
- Where: build-engine.mjs:39 (`pip install -q "pyinstaller>=6.10"`), build-engine.ps1:7.
- Problems:
  - PyInstaller (and its hooks, which decide what gets bundled) is unpinned and not in any lock.
  - The repo .venv has no pip (`No module named pip`), so `npm run engine` fails at its first step.
  - The venv also contains ruff 0.17.4, vulture 2.16 and psutil 7.2.2, which appear in no lock or pyproject.
    Nothing checks that the build venv matches lock-win-x64-py312.txt. If psutil is importable at build time,
    a hook may bundle it, and it would not be in the notices.
- Fix:
  - Pin `pyinstaller==X` (with hashes) in `requirements/lock-build-win-x64-py312.txt`.
  - Before building, run `python -m pip install --require-hashes -r lock…` (or `uv pip sync`) into a clean venv,
    or at least `pip check` and compare `pip freeze` against the lock.
  - Use `python -m ensurepip` if pip is missing.
- Confidence: high.

**16. CI gaps.**
- Where: .github/workflows/tests.yml:3-11, 60-83.
- Gaps:
  - No `push:` trigger (deliberate while private, per lines 4-7), so main is never tested after merge.
  - No clippy, rustfmt, ruff, or `node --check ui/app.js` (CONTRIBUTING.md lists the last).
  - No notices freshness check (finding 1).
  - No macOS/Linux job, so the `#[cfg(not(windows))]` code (lib.rs:737-742) and tauri.linux/macos configs are never compiled in CI.
  - No release workflow: installers are built on a maintainer machine with no provenance or checksums.
  - Actions are pinned by tag, not SHA.
  - The Rust job does not run on the weekly schedule, so it never sees dependency drift.
- What is right: `--locked`; rust-cache keyed on the lockfile with workspace `desktop/src-tauri`; the placeholder
  trick (verified locally); `permissions: contents: read`; pip cache keyed on the locks; hash-checked lock install.
- Fix: the matrix proposed below.
- Confidence: high.

**17. Unit tests do not cover the riskiest code.**
- Where: lib.rs:797-1056, votes.rs:227-292.
- Covered: listing rules, grants, slot bookkeeping, argument building, thread arithmetic, vote file and outbox logic.
- Not covered:
  - the `start_job` spawn path, the reader threads, the `exit` event and slot release;
  - `allow_media`;
  - the `open_log` file-name check (ADS, case, `..` were verified manually below);
  - `write_summary` and its dirs rule;
  - `canon` prefix stripping for UNC;
  - symlinks (finding 8), non-object or non-UTF-8 engine lines (findings 5-6);
  - `send_all` (backoff, TooLarge, Stop, the clear race); `page_args_ok` with `--work`/`-o` overrides for `analyse`
    is covered, `preview` is not.
- Fix: factor the reader loop into `fn pump(reader, on_msg)` and test it on byte slices. Add a test with a fake
  engine (a tiny Rust test binary, or `cmd /c echo`) for spawn, kill and exit.
- Confidence: high.

**18. Mutex unwraps everywhere, and panic=abort turns any poisoned lock into a crash.**
- Where: lib.rs:222, 226, 239, 287, 295, 306, 309, 398, 408, 425, 433, 483, 505, 527-536; votes.rs:152, 162, 174, 199, 220.
- Problem: no lock is held across user-controlled fallible code, so poisoning needs a prior panic. Combined with
  finding 6 and `panic = "abort"`, though, every panic is fatal and kills the session with orphans (finding 7).
  No unwrap on user-controlled paths was found (path parsing goes through `ok_or`).
- Fix: consider `panic = "unwind"` in the release profile (cost: binary size) together with
  `lock().unwrap_or_else(PoisonError::into_inner)`. Or keep abort and fix finding 6.
- Confidence: high.

### S5: nits

19. **Wrong log path in the docs.** desktop/README.md:85 says `%LOCALAPPDATA%\org.eluate.desktop\logs\engine.log`. The
    identifier is `io.github.subconsciousnz-viber.eluate` (tauri.conf.json:6), so the real path is
    `%LOCALAPPDATA%\io.github.subconsciousnz-viber.eluate\logs\engine.log`. The vote outbox is under
    `%APPDATA%\<identifier>\`. High confidence.

20. **engine.log is hard to read and can grow without bound within a session.**
    - Where: lib.rs:489-510.
    - Every concurrent engine (up to 3 renders, view, analyse, stems) appends to the same file with no timestamp,
      job id or slot. Rotation is checked only when a job starts, so a long noisy job can grow it without bound, and
      only one generation is kept.
    - Fix: prefix each line with `[time slot job]` and check the size every N lines. High confidence.

21. **The asset-protocol scope only grows.**
    - Where: lib.rs:381-393.
    - Every master, A/B pair and stems sum the engine reports is added for the life of the process and never
      removed (Tauri has no remove, only `forbid_file`). The `allow_file` error is discarded.
    - Fine for 200 tracks. Optionally `forbid_file` when a track is removed. Low impact.

22. **`ELUATE_PYTHON` is honoured in release builds** when the bundled engine is missing (lib.rs:321-326). Only the venv
    fallback is gated by `debug_assertions`. Gate both. Low risk: setting env vars already implies local control.

23. **No working directory is set for the engine child** (lib.rs:459-472). It inherits the shell's CWD. In dev runs
    (`python -m eluate.app_engine`), `-m` puts CWD on `sys.path`. Set `current_dir(app_cache_dir)` for determinism.

24. **Neither the engine nor the CLI has `--version`.** `eluate --version` gives "the following arguments are
    required: cmd" (run here). The shell cannot detect a stale `dist-engine` next to a newer shell; README notes
    that `npm run build` bundles whatever `dist-engine` holds. Add `--version` to both, and have the shell log the
    engine's version.

25. **CSV formula injection.** In ui/app.js:499-509 a track named `=HYPERLINK(...)` becomes a formula in
    eluate-batch.csv when opened in Excel. `write_summary` (lib.rs:648-649) also overwrites an existing
    eluate-batch.csv silently. Prefix `'` to cells starting with `= + - @`.

26. **The installer needs the internet for WebView2.** `webviewInstallMode: downloadBootstrapper` (tauri.conf.json:54)
    needs internet at install time on machines without WebView2 (older Windows 10, LTSC, air-gapped studios), which
    sits awkwardly with an "offline" app. `embedBootstrapper` (+~2 MB) or `offlineInstaller` (+~130 MB) avoids it.
    Microsoft documents packaging the bootstrapper with the app (Sources).

27. **Unverified claims in the release notes.** "The AppImage runs on Fedora / openSUSE" (release notes, Platforms) sits
    under an "Untested" row. Reword it as "should run". The macOS "Builds" claim cannot be verified from the repo:
    there are no macOS build notes or CI.

28. **macOS/Linux behaviour differences are not stated in the release notes.** There is no keep-awake (lib.rs:737-742
    is a no-op off Windows), and Limit CPU there only reduces threads (no below-normal priority). macOS App Nap or
    idle sleep can stall a long render. Note it, or use `caffeinate`/IOPMAssertion and `nice`.

29. **Consent copy.** "we don't keep your IP address" (ui/index.html:175) is a server-side promise the client cannot
    enforce. The privacy URL is a non-clickable span. Make it an opener link, which is allowed through a shell
    command, and make sure the jective.nz policy states retention.

30. **deb version ordering.** The deb version `0.1.0-beta` sorts after `0.1.0` (documented in the release notes).
    For the next prerelease, use a version that maps to `0.1.0~beta` in the deb, or bump to 0.1.1 for the final.

31. **Clippy.** 5 `cloned_ref_to_slice_refs` warnings in tests (lib.rs:865, 950, 955, 993, 997). Use
    `std::slice::from_ref`.

32. **Stale branch.** `origin/macos-linux-installers` is already squash-merged into main as fb082d1 (see §8 below).
    Delete the branch.

33. **README nit.** `desktop/README.md:18-21` says "about 370 MB". Re-measure after finding 12.

---

## Section notes

### Command surface (task 1): traced and verified

- **Commands.** Release builds expose `start_job`, `stop_job`, `pick_audio`, `pick_folders`, `pick_stems`,
  `write_summary`, `write_vote`, `reveal`, `open_log`, `votes::queue_vote`, `votes::flush_votes` and
  `votes::clear_outbox`. `smoke_grant` and `smoke_stems` exist only under `cfg(debug_assertions)` (lib.rs:747-756).
- **Engine flag injection.** Every engine subparser sets `allow_abbrev=False` (app_engine.py:62-86), so `--stem` or
  `--mi` prefixes cannot reach `--stems` or `--mix`. `page_args_ok` blocks `--stems`, `--mix` and their `=` forms.
  The shell appends `--stems`/`--mix` itself, only for a registered stem set (lib.rs:266-271). The page's `-o`,
  `--work`, `-oX`, `--work=` and `--threads` all lose to the shell's trailing `--threads N -o <work_dir>`, because
  argparse keeps the last value of a dest.
  - Injecting `--` or a dangling `-o` only makes argparse fail; it is not a bypass.
  - `preview` accepts only `--target-rms` and writes nothing. `ui-data` and `stems-sum` take no page args.
  - `summary` passes only granted files.
- **Granted set.** It is keyed by canonical path. Canonicalisation gives the on-disk case and long names (8.3 →
  long, verified), resolves `..`, `::$DATA`, trailing dots and spaces, and resolves junctions and symlinks to
  their targets (verified). UNC paths keep the `\\server\share` form.
  - Gap: symlink entries in folder listings (finding 8).
  - Folders are listed top level only. The 200 cap applies per pick or drop; the granted set is cumulative.
- **open_log.**
  - Verified robust: `..`, upper case (canonical name comes back lower case), `::$DATA` (resolves to the file
    itself, so the name check passes on the real file), and 8.3 names.
  - Refused: a named ADS (`process-log.html:evil` canonicalises with its stream, so `file_name` ≠
    "process-log.html"); symlinks pointing outside a work dir (the target is outside scope).
- **reveal.** Accepts a chosen file, or anything under a chosen file's (or stem set's) work dir.
- **write_summary.** Writes only `<granted dir>/eluate-batch.csv`. A granted dir is a chosen folder or the parent of a
  chosen file (a chosen `* - eluate` folder is also accepted). Size ≤1 MB.
- **write_vote.** Appends one JSON object (≤4 KB) to `ab-votes.jsonl`, only in a `* - eluate` directory that is the
  parent of an in-scope path.
- **Environment.** BLAS/OpenMP thread vars (OMP, OPENBLAS, MKL, VECLIB) are set to 1. `NUMBA_CACHE_DIR` is
  `<app_cache_dir>/numba`, and entry.py falls back to `%TEMP%\eluate-numba`. `PYTHONUNBUFFERED`/`PYTHONIOENCODING` are
  set. stdin is null. `CREATE_NO_WINDOW`, plus `BELOW_NORMAL_PRIORITY_CLASS` under Limit CPU.
  - Rust std makes handles non-inheritable except the child's stdio. All spawns happen on the main thread, so no
    concurrent-spawn handle leak.
- **Termination.** Cancel calls `TerminateProcess` on the onedir engine, which is a single process, so there is no
  grandchild. Window close stops all slots (Destroyed). Gaps: crash or kill of the shell (finding 7).
- **Keep-awake.** A dedicated thread owns `SetThreadExecutionState`. It is cleared when the last render's reader
  thread ends, on every exit path, including spawn errors (spawn errors never set it). Windows also clears it at
  process exit.
- **Slot bookkeeping.** The `exit` path only frees the slot if `id` still matches, so a newer job keeps its slot. The
  child mutex is held across `wait()` only after both pipes have reached EOF, so there is no deadlock with
  `stop_slot` (which takes the jobs lock, then the child lock).

### Tauri configuration (task 3)

- **CSP** (tauri.conf.json:29): `default-src 'self'; script-src 'self'` (no inline or eval). Only
  `style-src 'unsafe-inline'`. `media-src` is limited to self and asset. `connect-src` is IPC only, so the page
  itself cannot fetch the network. Also `object-src 'none'`, `base-uri 'none'`, `frame-ancestors 'none'`. Good.
  - Page innerHTML: file names go through `esc()` (app.js:460-461), and the About version is escaped too.
- **Capabilities** (capabilities/main.json:6): `core:default` only, window `main`, no wildcards. No fs, shell, http,
  dialog or opener permissions for the page; Rust uses the dialog and opener plugins internally.
  - Note: app commands are allowed by default for all windows because build.rs uses no AppManifest. Acceptable with
    one local window.
  - tauri-plugin-fs is compiled in as a dependency of tauri-plugin-dialog but not registered or permitted.
- **withGlobalTauri: true.** Acceptable given the strict `script-src` and no remote content.
- **assetProtocol.** `enable: true`, `scope: []`, grown at runtime (finding 21).
- **Window and identity.** One window, `dragDropEnabled` (drops handled in Rust). Identifier
  `io.github.subconsciousnz-viber.eluate`.
- **Updater.** Absent (fine for a beta; users must recheck releases manually).
- **Windows bundle.** NSIS `currentUser` (no admin). WebView2 `downloadBootstrapper` silent (finding 26). Unsigned:
  SmartScreen warns, as documented. wry passes `--disable-features=…msSmartScreenProtection` by default, so page
  navigations are not sent to SmartScreen. This comes from memory of the wry defaults and was not re-verified here.
- **macOS.** `signingIdentity "-"` (ad-hoc) matches the notes' "ad-hoc signed, not notarised"; min 11.0;
  app + dmg targets.
- **Linux.** deb + AppImage; rpm configured but excluded (version-hyphen reason, documented).

### Privacy (task 2): details

- **Sending.** Batches of up to 50 votes. 200 clears the batch. 400/415/other-4xx stop the sender for this run.
  413 halves the batch, and a single 413 vote is dropped. 429 waits Retry-After (max 1 h, default 60 s). 5xx or no
  connection backs off 60 s, 10 min, 1 h, then 24 h repeating. One sender thread at a time. The outbox is
  deduplicated by id.
- **Offline.** Votes persist in `%APPDATA%\<identifier>\vote-outbox.jsonl` and are sent on a later launch (the page
  calls `flush_votes` at start only if consent is "yes").
- **Opt-out.** `clear_outbox` deletes the file (finding 11 covers the race).
- **TLS.** ureq 2.12.1 with rustls 0.23 and bundled webpki-roots (Mozilla CA set compiled in, OS store not used, so
  corporate TLS interception fails closed). No pinning. Overall timeout 15 s. No proxy support (ureq's
  proxy-from-env feature is off), so behind a mandatory proxy votes simply stay queued.
- **Other endpoints.** grep for `http`, `fetch`, `XMLHttpRequest`, `WebSocket`, `sendBeacon`, `reqwest`, `ureq`,
  `TcpStream` across ui/, desktop/src-tauri/src and src/eluate:
  - The only network call is votes.rs:24.
  - The page's GitHub URL is display text only.
  - process-log.html loads nothing external.
  - The engine uses no requests, urllib or pooch at runtime. requests, urllib3, certifi and pooch are bundled
    because librosa depends on pooch.
  - Outside Eluate's control: the WebView2 Evergreen runtime updates itself through Microsoft, and the NSIS
    bootstrapper downloads WebView2 at install time if it is missing.

### Engine packaging (task 4)

- **Spec.** Onedir, `console=True` started with `CREATE_NO_WINDOW`, UPX off. `collect_all` over the scientific stack
  (bloat: finding 12). Excludes are sensible and documented (matplotlib, pyloudnorm, pytest, PIL, setuptools,
  PyInstaller, …). pyloudnorm and matplotlib are dev-only and correctly excluded.
- **Hidden imports.** `collect_submodules("eluate")` covers the lazily imported modules (estimate, parallel, uidata, …).
- **entry.py.** Keeps a `NUMBA_CACHE_DIR` fallback. Whether numba's cache actually hits in a frozen app (source
  stamps of frozen modules) is unverified: see Open questions.
- **Startup.** See finding 13. A frozen onedir build usually starts somewhat faster than a venv (no .pth processing),
  but this was not measured.

### CI (task 6): proposed minimal matrix

```yaml
on: { pull_request: {}, push: { branches: [main] }, workflow_dispatch: {}, schedule: [{cron: "0 15 * * 0"}] }
jobs:
  lint:            # ubuntu-latest, ~1 min, free minutes
    - ruff check src tests tools (pin ruff in lock-dev)
    - node --check ui/app.js
    - cargo fmt --check; cargo clippy --locked --all-targets -- -D warnings  (desktop/src-tauri, needs
      libwebkit2gtk-4.1-dev; or run clippy in the windows rust job instead)
    - python tools/third_party_notices.py (Part 4 only, or full on windows) && git diff --exit-code THIRD_PARTY_NOTICES.txt
  python-fast:     # windows-latest (as now)
  python-full:     # schedule/dispatch (as now)
  rust:            # matrix: [windows-latest, ubuntu-24.04, macos-14]; stage UI + placeholder; cargo test --locked
                   # (also on schedule, to catch dependency drift)
  release:         # on: push tags v*: build engine from hashed locks + pinned PyInstaller, tauri build,
                   # sha256sums, upload to a draft release; per-platform notices; later: signing
```

### macos-linux-installers branch (task 8)

- **History.** `git log main..origin/macos-linux-installers` shows 5 commits: 2dfa24f (cross-platform
  build-engine.mjs, macOS targets), 3f30003 (stage UI before cargo test), 7affef7 (Linux deb/rpm/AppImage),
  53e0364 (drop rpm, stop repeating depends), 6beca6b (Linux build docs).
- **Already in main.** main's fb082d1 "macOS and Linux installers (untested), fresh-clone build fix" contains all of
  it: `git diff main origin/macos-linux-installers --stat` touches only README.md and the release notes, where main
  is newer.
  - Code risk is low: lib.rs only gains a doc comment and `#[cfg_attr(not(windows), allow(dead_code))]`.
  - package.json now stages the UI before `cargo test` (a fresh-clone fix, correct).
  - New platform JSON files are merged only on their own OS, so the Windows build is untouched.
- **Do not merge.** Delete the branch.
- **Accuracy of the untested-platform claims.**
  - "Untested" and "no render has been run" are honest.
  - "Builds" is believable for Linux (the libtbb12/linuxdeploy notes are evidence of real attempts) and unverifiable
    for macOS.
  - The surrounding problems are the notices and source-tag issues (finding 3), the unpinned non-Windows deps
    (finding 3b), and no keep-awake or priority off Windows (finding 28).

---

## Licence table

| Component | Licence | Bundled how | Credited? | Obligation met? |
|---|---|---|---|---|
| Eluate (app, engine, UI) | AGPL-3.0-or-later | — | LICENSE.txt in install; About shows source URL /tree/v<ver> | Yes for Windows; macOS/Linux source tag mismatch (finding 3) |
| numpy, scipy | BSD-3 (+0BSD/MIT/Zlib/CC0 parts) | PYZ + .pyd in _internal | Yes, full texts (Part 1) | Yes |
| OpenBLAS + gfortran runtime | BSD-3; GPL-3 w/ GCC RLE | DLLs in numpy.libs / scipy.libs | Yes (Part 2) | Yes |
| librosa | ISC | PYZ | Yes | Yes |
| numba / llvmlite (+LLVM) | BSD-2; Apache-2.0 WITH LLVM-exception | PYZ + llvmlite.dll | Yes | Yes |
| scikit-learn, joblib, soxr (libsoxr LGPL-2.1+, static in .pyd), pooch, requests, urllib3, certifi (MPL-2.0), … | various permissive; soxr LGPL | PYZ/.pyd | Yes (lock-driven) | Yes (soxr: source public, relinkable via rebuild) |
| soundfile | BSD-3 | PYZ | Yes | Yes |
| libsndfile 1.2.2 | LGPL-2.1-or-later | separate replaceable DLL (_soundfile_data) | Yes + DLL hash pinned | Relinking yes; source offer weak (finding 4) |
| FLAC / ogg / vorbis / opus | BSD-3 | static in libsndfile DLL | Yes (tools/licenses) | Yes |
| mpg123 1.32.9 / LAME 3.100 | LGPL-2.1 / LGPL-2.0+ | static in libsndfile DLL (replaceable as a whole) | Yes | Same as libsndfile (finding 4) |
| CPython 3.12 (+OpenSSL, libffi, zlib, …) | PSF-2.0 etc. | python312.dll + stdlib | Yes (Part 3) | Yes |
| MSVC runtime DLLs | MS redistributable terms | DLLs | Mentioned | Yes (redistributable) |
| PyInstaller bootloader | GPL-2.0+ with bootloader exception | eluate-engine.exe | No | Yes (the exception lifts conditions). Credit recommended. PyInstaller runtime hooks may be Apache-2.0 (unverified; see Open questions) |
| Tauri, wry, tao, plugins (237 crates listed) | MIT/Apache-2.0 mostly; 5 MPL-2.0 | static in eluate.exe | Names + SPDX only | No: texts missing (finding 2) |
| ureq, rustls, ring, webpki-roots, … (12 crates) | MIT/Apache/ISC; ring Apache-2.0 AND ISC; webpki-roots CDLA-Permissive-2.0 | static in eluate.exe | **No** | **No** (finding 1) |
| WebView2 Evergreen runtime | Microsoft software licence | Not bundled; bootstrapper downloaded by NSIS at install | Mentioned (Part 4 note) | Yes (no redistribution happens) |
| NSIS installer stub | zlib/libpng-style | installer | No | Yes (no notice requirement for binaries) |
| Archivo font | SIL OFL-1.1 | embedded in eluate.exe (dist/fonts) + licenses/Archivo-OFL.txt | Yes (separate file) | Yes (unmodified, RFN not used) |
| GStreamer / WebKitGTK etc. (AppImage only) | LGPL-2.1+ and others | bundled in AppImage | **No** | **No** (finding 3a) |
| "Jective" marks | trademark (no grant) | colour names in ui/README; jective.nz endpoint | No trademark statement | AGPL §7(e) allows declining trademark rights; consider a one-line TRADEMARKS note |
| Contributions | DCO 1.1, `git commit -s`, inbound = AGPL-3.0-or-later | — | CONTRIBUTING.md:17-21 | Fine. No CI DCO check (add the DCO app or action if the repo goes public) |

## Version consistency

| Place | Value | OK? |
|---|---|---|
| pyproject.toml:7 | 0.1.0b0 (PEP 440 spelling, commented) | Yes |
| tauri.conf.json:5 | 0.1.0-beta | Yes |
| Cargo.toml:3 (CARGO_PKG_VERSION, vote envelope, User-Agent) | 0.1.0-beta | Yes |
| package.json:3 | 0.1.0-beta | Yes |
| Release notes title / tag | 0.1.0-beta / v0.1.0-beta → bf1abae | Yes for Windows; macOS/Linux from fb082d1 (finding 3c) |
| About line (getVersion → /tree/v0.1.0-beta) | 0.1.0-beta | Yes |
| `eluate --version` | not supported (argparse error) | Missing (finding 24) |
| engine `--version` | not supported | Missing |
| deb package version | 0.1.0-beta (sorts above 0.1.0) | Documented quirk (finding 30) |

## Verified correct

- The CI recipe (stage-ui + placeholder engine) builds and tests the shell from a clean target dir with `--locked`.
  0 compiler warnings; clippy is clean outside tests.
- No engine flag abbreviation or injection route: `allow_abbrev=False` everywhere. Shell flags are appended last or
  shell-only, so `-o`, `--work` and `--threads` overrides from the page lose.
- The open_log name check resists case, `..`, `::$DATA`, named ADS, trailing dot/space and 8.3 names
  (probe in canon/).
- Junctions and symlinks are resolved before grant comparisons (except finding 8). Canonical case means the
  HashSet comparison is effectively case-insensitive on NTFS.
- Release builds do not contain `smoke_grant` or `smoke_stems`.
- No telemetry, update check, install ID or crash reporter. The vote UUID is per vote. Consent defaults to off. The
  consent text matches the payload.
- CSP and capabilities are minimal. The page gets no fs, shell, http, dialog or opener access.
- Engine parallelism is threads only, so Cancel cannot orphan worker processes. BLAS pools are pinned to 1.
- Keep-awake is cleared on every job end, including errors. A slot is released only by the job that owns it.
- The stem-set `--stems`/`--mix` come only from `register_set` (user dialogs), never from the page (audit 13 #4 holds).
- Python and native licence coverage is careful: full texts, the DLL hash pinned so codec drift is detected, codec
  texts kept in tools/licenses.

## Open questions

1. Frozen engine cold start and numba cache effectiveness in the PyInstaller build. Does `NUMBA_CACHE_DIR` actually
   get index hits for librosa's `@jit(cache=True)` functions when module sources are inside the PYZ? Needs
   `npm run engine` with PyInstaller pinned. Time `eluate-engine.exe summary x.wav` twice.
2. Actual built bundle contents and size (is psutil, cloudpickle or narwhals swept in? `bundle_extras` only reports
   if `dist-engine` existed when the notices were generated; the current notices say lock-only).
3. Does Tauri 2's `Scope::allow_file` escape glob metacharacters, so a work dir like `Track [Remix] - eluate` plays?
   If not, A/B playback silently fails for such names (the error is ignored at lib.rs:391).
4. macOS: does Gatekeeper's "Open Anyway" on the .app cover the nested ad-hoc-signed `engine/eluate-engine` and its
   dylibs? Does App Nap throttle a background render?
5. Linux WebKitGTK: does `<audio>` seeking work over the `asset://` custom protocol (range requests) for A/B?
6. PyInstaller runtime hooks' licence (Apache-2.0 per PyInstaller's COPYING?): if so, add a PyInstaller entry to the notices.
7. Server side: does jective.nz actually drop IPs, and what is the retention? The client promise depends on it.

## Sources

- Tauri: non-async commands run on the main thread: [tauri-macros v1.0.0-beta.0 notes](https://v2.tauri.app/release/tauri-macros/v1.0.0-beta.0/), [Tauri IPC guide (async commands on thread pool)](https://jonaskruckenberg.github.io/tauri-docs-wip/development/inter-process-communication.html), [freeze fix example](https://github.com/diffplug/dormouse/pull/321)
- WebView2 distribution (bootstrapper may be linked or packaged): [Microsoft Learn: Distribute your app and the WebView2 Runtime](https://learn.microsoft.com/microsoft-edge/webview2/concepts/distribution), [Evergreen vs fixed version](https://learn.microsoft.com/microsoft-edge/webview2/concepts/evergreen-vs-fixed-version)
- webpki-roots licence CDLA-Permissive-2.0: [rustls/webpki-roots](https://github.com/rustls/webpki-roots), [docs.rs README 1.0.7](https://docs.rs/crate/webpki-roots/1.0.7/source/README.md)
