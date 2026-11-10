# Audio-rate gain harness

The harness behind [docs/audit-2026-11-10/audio-rate.md](../../docs/audit-2026-11-10/audio-rate.md)
and the before / after table in [docs/gain-staging.md](../../docs/gain-staging.md) §21. It drives
every dynamic stage the way the planner drives it (multiband compressor and ceilings, speclim, glue,
final limiter, clipper, M/S link, notch envelopes) at all five characters, on synthetic signals
only, and measures the applied gain signal and the output: ripple, gain velocity, the share of the
gain's energy above 100 / 500 Hz, sideband lines, THD / H2 / H3 and non-harmonic products.

Run it after any change to the dynamics stages or the envelope smoother:

    .venv/Scripts/python.exe tools/audio_rate/run.py all      # or h1 .. h7, fixes, mbkick, shape
    .venv/Scripts/python.exe tools/audio_rate/run.py h1 --chars fast,aggressive --freqs 30,40

Each experiment writes `tools/audio_rate/out/<exp>.json` and `<exp>.md` (git-ignored) and prints
its tables. `ELUATE_REPO=<path>` measures another checkout (for instance a worktree at fb082d1, to
reproduce the "before" column); by default it measures this one. `drive.py` uses the stages' newer
parameters (limiter aim / hold / re-limit, band detector windows, ceiling release, crossover phase)
when the code under test has them and the original calls otherwise, so one harness reads both.

| exp | what |
|---|---|
| h1 | steady 30 / 40 / 60 Hz sines in the sub band, per stage x character (ripple, THD) |
| h2 | PCHIP across the 1 ms knots: knot lines, the gain's energy above 500 Hz |
| h3 | ms_link (live profile): the M / S minimum at audio rate vs at the control rate |
| h4 | glue on a 40 Hz sine plus a 128 BPM kick, and on the kick alone: onset splatter, largest 1 ms fall |
| h5 | notch depth envelopes and the composite clamp s(t) (quantised vs not) |
| h6 | clipper and limiter: span clipping vs the whole-signal curve, clipper THD, re-limit |
| h7 | the 44.1 kHz control grid against 48 kHz |
| fixes | the prototypes in `harness/fixes.py` (monkey-patched, never touching the repo) |
| mbkick, shape | sub / low band gains around kicks; the sub-band gain over one 40 Hz period |

`harness/fixes.py` still holds two glue prototypes that were measured but not adopted: a trailing
hold on the glue's required gain (F4 b) and gain applied above a split only (F4 c). The glue's
steady-sub distortion that they address is the one audio-rate item left (gain-staging §21 table).
