"""Synthetic test signals for eluate tests and later workstreams.

Contract: every generator is deterministic for a given `seed`, returns float64
audio shaped (n, 2) (unless stated) plus, where useful, a metadata dict holding
the ground truth a test needs (onset times, roots, ring extent). No file I/O.

- `noise_psd_slope`: Gaussian noise whose PSD falls at `slope_db_oct` dB/octave
  (-3.0103 = pink) between `f_lo` and fs/2, zero below `f_lo`; RMS set exactly.
- `impulse_train`: unit impulses (times `amplitude`) every `period_s`.
- `tone_sequence`: stereo sine segments [(dBFS peak, seconds), ...], the form of
  the EBU Tech 3341/3342 1 kHz test signals.
- `kick_sub`: four-on-the-floor kick with a 250 ms pitched sub tail plus a
  sidechain-ducked sub bassline following given roots per beat.
- `add_ring`: inject a narrowband resonance (+gain_db peaking boost, given Q)
  between two times, with 10 ms raised-cosine fades inside the span.
- `synth_note`: sustained harmonic note (partials k f0 at 1/k amplitude up to
  `n_harm` or 0.95 fs/2) over a quiet pink-noise bed.
- `sectioned_track`: kick, off-beat hats, sub bass and a pad, identical
  throughout, with a level step at every section start (bars_per_section
  bars each); meta holds the true boundaries, beats and bars.
- `chord_loop`: sustained triads (stacked harmonic notes) cycling a chord list
  per bar over a bass root, for key and chord tests.
- `bassline`: short-attack bass notes (off-beat, eighths or on the beat) with
  or without a kick on every beat: bass attacks that must not read as kicks.
- `drum_loop`: four-on-the-floor, DnB / hip-hop (kicks on a quarter of the
  beats) or kickless (snare + hats) drum patterns over a steady sub.
- `groove_track`: kick, off-beat hats, an unducked bass root and pad triads
  per bar (the kick + bass + pad material that hid kicks, audit features F5).
- `ambient_pad`, `sustained_chord`: beatless material (no metrical evidence).
"""

from __future__ import annotations

import numpy as np
from scipy import signal


def _rng(seed):
    return np.random.default_rng(seed)


def noise_psd_slope(
    n: int,
    fs: float,
    slope_db_oct: float = -3.0103,
    rms_dbfs: float = -20.0,
    f_lo: float = 10.0,
    seed: int | None = 0,
    channels: int = 2,
    correlation: float = 0.0,
) -> np.ndarray:
    """Noise with PSD proportional to f ** (slope_db_oct / (10 log10 2))."""
    rng = _rng(seed)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    expo = slope_db_oct / (20.0 * np.log10(2.0))  # amplitude exponent
    shape = np.zeros_like(f)
    band = f >= f_lo
    shape[band] = (f[band] / f_lo) ** expo
    out = np.empty((n, channels))
    white = []
    for _ in range(channels):
        spec = rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size)
        white.append(np.fft.irfft(spec * shape, n))
    if channels == 2 and correlation:
        a, b = white
        white = [a, correlation * a + np.sqrt(1.0 - correlation**2) * b]
    for c, w in enumerate(white):
        out[:, c] = w
    target = 10.0 ** (rms_dbfs / 20.0)
    out *= target / np.sqrt(np.mean(out**2))
    return out


def pink_noise(n, fs, rms_dbfs=-20.0, seed=0, channels=2):
    return noise_psd_slope(n, fs, -3.0103, rms_dbfs, seed=seed, channels=channels)


def impulse_train(n: int, fs: float, period_s: float, amplitude: float = 1.0,
                  offset_s: float = 0.0, channels: int = 2) -> np.ndarray:
    x = np.zeros((n, channels))
    idx = np.round(np.arange(offset_s, n / fs, period_s) * fs).astype(int)
    idx = idx[idx < n]
    x[idx, :] = amplitude
    return x


def tone_sequence(segments, fs: float, freq: float = 1000.0) -> np.ndarray:
    """Stereo sine, phase-continuous, level stepping per (dBFS_peak, seconds)."""
    parts = []
    for level_db, dur in segments:
        parts.append(np.full(int(round(dur * fs)), 10.0 ** (level_db / 20.0)))
    amp = np.concatenate(parts)
    t = np.arange(amp.size) / fs
    x = amp * np.sin(2.0 * np.pi * freq * t)
    return np.stack([x, x], axis=1)


def kick(fs: float, f_start: float = 150.0, f_end: float = 50.0,
         sweep_s: float = 0.03, tail_s: float = 0.25, level: float = 0.9) -> np.ndarray:
    """Mono kick: exponential pitch sweep f_start -> f_end over sweep_s, then a
    pitched sub tail at f_end decaying to -60 dB at tail_s."""
    n = int(round(tail_s * fs))
    t = np.arange(n) / fs
    k = np.log(f_end / f_start) / sweep_s
    f = np.where(t < sweep_s, f_start * np.exp(k * np.minimum(t, sweep_s)), f_end)
    phase = 2.0 * np.pi * np.cumsum(f) / fs
    env = np.exp(-t * np.log(1000.0) / tail_s)
    click = np.exp(-t / 0.002) * 0.3
    y = level * env * np.sin(phase) + click * np.sin(2 * np.pi * 1500 * t)
    fade = min(n, int(0.005 * fs))
    y[-fade:] *= np.cos(np.linspace(0, np.pi / 2, fade)) ** 2
    return y


def kick_sub(
    fs: float,
    bpm: float = 126.0,
    roots_hz=(55.0, 55.0, 61.74, 49.0),
    bars: int = 4,
    duck_db: float = -18.0,
    duck_release_s: float = 0.2,
    sub_level: float = 0.5,
    seed: int | None = 0,
):
    """Kick on every beat plus a sub bassline, sidechain-ducked by the kick.

    `roots_hz` lists the bass root per beat and cycles. Returns (x, meta) with
    meta = {"beat_times", "kick_onsets", "roots_per_beat", "bpm"}.
    """
    beat = 60.0 / bpm
    n_beats = bars * 4
    n = int(round(n_beats * beat * fs))
    onsets = np.arange(n_beats) * beat
    roots = np.array([roots_hz[i % len(roots_hz)] for i in range(n_beats)])
    # sub bassline: phase-continuous sine + 2nd harmonic, root per beat
    t = np.arange(n) / fs
    beat_idx = np.minimum((t / beat).astype(int), n_beats - 1)
    f_inst = roots[beat_idx]
    ph = 2.0 * np.pi * np.cumsum(f_inst) / fs
    sub = sub_level * (np.sin(ph) + 0.15 * np.sin(2 * ph))
    # sidechain gain: drop to duck_db at each onset, raised-cosine recovery
    g = np.ones(n)
    rel = int(round(duck_release_s * fs))
    floor = 10.0 ** (duck_db / 20.0)
    curve = floor + (1 - floor) * 0.5 * (1 - np.cos(np.pi * np.arange(rel) / rel))
    for o in onsets:
        i0 = int(round(o * fs))
        seg = curve[: max(0, min(rel, n - i0))]
        g[i0 : i0 + seg.size] = np.minimum(g[i0 : i0 + seg.size], seg)
    x = sub * g
    kk = kick(fs)
    for o in onsets:
        i0 = int(round(o * fs))
        m = min(kk.size, n - i0)
        x[i0 : i0 + m] += kk[:m]
    meta = {"beat_times": onsets, "kick_onsets": onsets, "roots_per_beat": roots,
            "bpm": bpm, "sidechain_gain": g}
    return np.stack([x, x], axis=1), meta


def add_ring(x: np.ndarray, fs: float, f0: float = 7000.0, q: float = 12.0,
             gain_db: float = 9.0, start_s: float = 0.0, end_s: float | None = None,
             fade_s: float = 0.01):
    """Add a peaking resonance y = x + (A - 1) * w(t) * BP(x), A = 10^(gain/20).

    BP is the RBJ constant-0-dB-peak bandpass at f0, Q (causal); with w = 1 the
    result is a peaking boost of exactly gain_db at f0. w(t) is 1 between
    start_s and end_s (clamped to the signal) with raised-cosine fades of
    fade_s inside that span; fades multiply, so spans shorter than two fades
    stay continuous. Returns (y, meta).
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    end_s = n / fs if end_s is None else end_s
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * q)
    b = np.array([alpha, 0.0, -alpha])
    a = np.array([1 + alpha, -2 * np.cos(w0), 1 - alpha])
    bp = signal.lfilter(b / a[0], a / a[0], x, axis=0)
    i0 = int(np.clip(round(start_s * fs), 0, n))
    i1 = int(np.clip(round(end_s * fs), i0, n))
    w = np.zeros(n)
    w[i0:i1] = 1.0
    nf = int(round(fade_s * fs))
    if nf > 0 and i1 > i0:
        ramp = 0.5 * (1 - np.cos(np.pi * np.arange(1, nf + 1) / (nf + 1)))
        up = np.ones(i1 - i0)
        k = min(nf, i1 - i0)
        up[:k] = ramp[:k]
        down = np.ones(i1 - i0)
        down[-k:] = ramp[:k][::-1]
        w[i0:i1] = up * down  # multiplied, so short spans keep both fades
    amp = 10.0 ** (gain_db / 20.0) - 1.0
    wt = w[:, None] if x.ndim == 2 else w
    y = x + amp * wt * bp
    return y, {"f0": f0, "q": q, "gain_db": gain_db, "start_s": start_s, "end_s": end_s}


def synth_note(n: int, fs: float, f0: float = 220.0, n_harm: int = 12, peak: float = 0.2,
               bed_dbfs: float = -40.0, seed: int | None = 0) -> np.ndarray:
    """Sustained note, partial k at amplitude 1/k and phase 0.3 k, plus pink bed."""
    t = np.arange(n) / fs
    y = np.zeros(n)
    for k in range(1, n_harm + 1):
        if k * f0 < 0.95 * fs / 2:
            y += np.sin(2 * np.pi * k * f0 * t + 0.3 * k) / k
    y *= peak / np.max(np.abs(y))
    return np.stack([y, y], axis=1) + pink_noise(n, fs, rms_dbfs=bed_dbfs, seed=seed)


def _hat(fs: float, rng, dur_s: float = 0.03, level: float = 0.15) -> np.ndarray:
    m = int(round(dur_s * fs))
    sos = signal.butter(4, 7000.0, btype="highpass", fs=fs, output="sos")
    return level * signal.sosfilt(sos, rng.standard_normal(m)) * np.exp(-np.arange(m) / (0.008 * fs))


_NOTE = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9,
         "A#": 10, "B": 11}


def sectioned_track(fs: float, bpm: float = 128.0, levels_db=(-12.0, 0.0, -6.0, 0.0, -9.0),
                    bars_per_section: int = 8, root_hz: float = 55.0, seed: int | None = 0,
                    chords=None):
    """Identical material per section with a level step at each section start.
    With `chords` (one chord name per section, e.g. ("Am", "F", ...)), a
    sustained pad triad changes at every section start as well (audit 04 F13:
    a fixture whose boundaries are not level-only).

    Returns (x, meta), meta = {"boundaries", "section_starts", "levels_db",
    "beat_times", "bar_times", "bpm", "bars_per_section"}.
    """
    rng = _rng(seed)
    beat = 60.0 / bpm
    n_bars = bars_per_section * len(levels_db)
    n_beats = 4 * n_bars
    n = int(round(n_beats * beat * fs))
    t = np.arange(n) / fs
    beats = np.arange(n_beats) * beat
    x = 0.25 * np.sin(2 * np.pi * root_hz * t)
    x += pink_noise(n, fs, rms_dbfs=-30.0, seed=None if seed is None else seed + 1, channels=1)[:, 0]
    kk = kick(fs)
    for b in beats:
        i0 = int(round(b * fs))
        m = min(kk.size, n - i0)
        x[i0:i0 + m] += kk[:m]
        h = _hat(fs, rng)
        j0 = int(round((b + beat / 2) * fs))
        m = min(h.size, n - j0)
        if m > 0:
            x[j0:j0 + m] += h[:m]
    sec_len = bars_per_section * 4 * beat
    starts = np.arange(len(levels_db)) * sec_len
    if chords is not None:
        for s0, c in zip(starts, chords):
            minor = c.endswith("m")
            root = _NOTE[c[:-1] if minor else c]
            i0, i1 = int(round(s0 * fs)), min(n, int(round((s0 + sec_len) * fs)))
            tt = t[i0:i1] - s0
            env = np.minimum(1.0, np.minimum(tt, tt[-1] - tt) / 0.02)
            for iv in ((0, 3, 7) if minor else (0, 4, 7)):
                f = 440.0 * 2 ** ((57 + root + iv - 69) / 12.0)
                for k in (1, 2, 3):
                    x[i0:i1] += 0.04 / k * env * np.sin(2 * np.pi * k * f * tt)
    g = np.ones(n)
    for s0, lv in zip(starts, levels_db):
        g[int(round(s0 * fs)):] = 10.0 ** (lv / 20.0)
    x *= g * 0.5
    meta = {"boundaries": starts[1:], "section_starts": starts, "levels_db": tuple(levels_db),
            "beat_times": beats, "bar_times": beats[::4], "bpm": bpm,
            "bars_per_section": bars_per_section}
    return np.stack([x, x], axis=1), meta




def chord_loop(fs: float, chords=("Am", "F", "C", "G"), bpm: float = 120.0, loops: int = 4,
               level: float = 0.1, seed: int | None = 0):
    """Triads (root position, octave 4, 6 partials each) one per bar, plus the
    chord root two octaves down. Returns (x, meta) with meta["chords"] per bar."""
    bar = 4 * 60.0 / bpm
    seq = list(chords) * loops
    n = int(round(len(seq) * bar * fs))
    t = np.arange(n) / fs
    x = np.zeros(n)
    for i, c in enumerate(seq):
        minor = c.endswith("m")
        root = _NOTE[c[:-1] if minor else c]
        ivs = (0, 3, 7) if minor else (0, 4, 7)
        i0, i1 = int(round(i * bar * fs)), int(round((i + 1) * bar * fs))
        tt = t[i0:i1]
        env = np.minimum(1.0, np.minimum(tt - tt[0], tt[-1] - tt + 1 / fs) / 0.01)
        for iv in ivs:
            f = 440.0 * 2 ** ((60 + root + iv - 69) / 12.0)
            for k in range(1, 7):
                x[i0:i1] += env * np.sin(2 * np.pi * k * f * tt) / k
        fb = 440.0 * 2 ** ((36 + root - 69) / 12.0)
        x[i0:i1] += 1.5 * env * np.sin(2 * np.pi * fb * tt)
    x *= level / np.max(np.abs(x))
    x = np.stack([x, x], axis=1) + pink_noise(n, fs, rms_dbfs=-60.0, seed=seed)
    return x, {"chords": seq, "bar_times": np.arange(len(seq)) * bar, "bpm": bpm}


def bassline(fs: float, bpm: float = 128.0, pattern: str = "offbeat", bars: int = 8,
             roots_hz=(55.0, 55.0, 61.74, 49.0), kick_on_beats: bool = True,
             attack_s: float = 0.005, level: float = 0.4, seed: int | None = 0,
             partials=(1.0,)):
    """Bass notes with short attacks (low-passed, stationary: no sweep, no click),
    optionally with a kick on every beat. `pattern`: "offbeat" (one note on each
    off-beat 8th, 128 BPM house), "eighths" (a note on every 8th, 140 BPM), or
    "beats" (one note per beat, no kick needed). Root per beat cycles through
    `roots_hz`; `partials` are the amplitudes of harmonics 1, 2, ... (0 first
    for a missing fundamental). Returns (x, meta) with meta = {"beat_times", "kick_onsets",
    "note_onsets", "roots_per_beat", "bpm"}."""
    beat = 60.0 / bpm
    n_beats = 4 * bars
    n = int(round(n_beats * beat * fs))
    t = np.arange(n) / fs
    roots = np.array([roots_hz[i % len(roots_hz)] for i in range(n_beats)])
    if pattern == "offbeat":
        notes = np.arange(n_beats) * beat + beat / 2
        dur = beat / 2
    elif pattern == "eighths":
        notes = np.arange(2 * n_beats) * beat / 2
        dur = beat / 2
    elif pattern == "beats":
        notes = np.arange(n_beats) * beat
        dur = beat
    else:
        raise ValueError(pattern)
    y = np.zeros(n)
    for s0 in notes:
        i0 = int(round(s0 * fs))
        m = min(int(round(dur * fs)), n - i0)
        tt = np.arange(m) / fs
        f = roots[min(int(np.floor(s0 / beat + 1e-9)), n_beats - 1)]  # floor of 2.9999 must be 3
        env = np.minimum(1.0, tt / attack_s) * np.minimum(1.0, (dur - tt) / 0.005)
        tone = sum(a * np.sin(2 * np.pi * (k + 1) * f * tt) for k, a in enumerate(partials))
        y[i0:i0 + m] += level * env * tone / max(1e-12, float(np.sum(np.abs(partials))))
    kicks = np.arange(n_beats) * beat if kick_on_beats else np.zeros(0)
    kk = kick(fs)
    for o in kicks:
        i0 = int(round(o * fs))
        m = min(kk.size, n - i0)
        y[i0:i0 + m] += kk[:m]
    return np.stack([y, y], axis=1), {"beat_times": np.arange(n_beats) * beat, "kick_onsets": kicks,
                                      "note_onsets": notes, "roots_per_beat": roots, "bpm": bpm}


def _snare(fs: float, rng, level: float = 0.4, lowpass_hz: float | None = None) -> np.ndarray:
    """Noise burst (40 ms decay) plus a 200 Hz body; `lowpass_hz` band-limits the
    noise (and holds its PSD per Hz constant across rates, for rate checks)."""
    m = int(round(0.15 * fs))
    t = np.arange(m) / fs
    w = rng.standard_normal(m)
    if lowpass_hz is not None:
        w = signal.sosfilt(signal.butter(4, lowpass_hz, "lowpass", fs=fs, output="sos"), w)
        w *= np.sqrt(fs / 48000.0)
    return level * (w * np.exp(-t / 0.04) + np.sin(2 * np.pi * 200 * t) * np.exp(-t / 0.03))


def drum_loop(fs: float, bpm: float = 128.0, pattern: str = "four", seconds: float = 32.0,
              sub_hz: float | None = 55.0, seed: int | None = 0):
    """Drum patterns for beat-grid tests (audit features F1 / F6). `pattern`:
    "four" (kick on every beat, off-beat hat), "dnb" / "hiphop" (kick on 1 and
    the "and" of 3, snare on 2 and 4, 8th hats: kicks on a quarter of the beats),
    "nokick" (snare on 2 and 4, 8th hats, no kick). A steady sine at `sub_hz`
    (None: none) sits under it. Peak 0.5. Returns (x, meta) with meta =
    {"beat_times", "kick_onsets", "bpm"}."""
    rng = _rng(seed)
    n = int(round(seconds * fs))
    beat = 60.0 / bpm
    x = np.zeros(n)
    kk = kick(fs)

    def put(sig, t0):
        i0 = int(round(t0 * fs))
        m = min(sig.size, n - i0)
        if m > 0:
            x[i0:i0 + m] += sig[:m]
    beats = np.arange(int(seconds / beat)) * beat
    kicks = []
    for b, t0 in enumerate(beats):
        if pattern == "four":
            put(kk, t0)
            kicks.append(t0)
            put(_hat(fs, rng), t0 + beat / 2)
            continue
        if pattern in ("dnb", "hiphop"):
            if b % 4 == 0:
                put(kk, t0)
                kicks.append(t0)
            if b % 4 == 2:
                put(kk, t0 + beat / 2)
                kicks.append(t0 + beat / 2)
        elif pattern != "nokick":
            raise ValueError(pattern)
        if b % 2 == 1:
            put(_snare(fs, rng), t0)
        put(_hat(fs, rng), t0)
        put(_hat(fs, rng, level=0.08 if pattern == "dnb" else 0.1), t0 + beat / 2)
    if sub_hz is not None:
        x += 0.15 * np.sin(2 * np.pi * sub_hz * np.arange(n) / fs)
    x *= 0.5 / np.max(np.abs(x))
    return np.stack([x, x], axis=1), {"beat_times": beats, "kick_onsets": np.array(kicks), "bpm": bpm}


def groove_track(fs: float, seconds: float = 20.0, bpm: float = 128.0, chords=("Am", "F", "C", "G"),
                 kick_on: bool = True, hats: bool = True, bass: bool = True, pad: bool = True,
                 level: float = 0.5, seed: int | None = 0):
    """Kick on every beat, off-beat hats, a sustained (not sidechained) bass root
    per bar (sine plus 2nd harmonic, 65 to 123 Hz) and pad triads per bar, over
    a -55 dBFS stereo pink bed: the kick + bass + pad material of audit features
    F5, where the unducked bass and pad hide the kicks' flux. Returns (x, meta)
    with meta = {"beat_times", "bar_times", "chords" (per bar), "bpm"}."""
    rng = _rng(seed)
    n = int(round(seconds * fs))
    t = np.arange(n) / fs
    beat = 60.0 / bpm
    beats = np.arange(int(np.ceil(seconds / beat))) * beat
    beats = beats[beats < seconds]
    bars = beats[::4]
    x = np.zeros(n)
    kk = kick(fs)
    for b in beats:
        i0 = int(round(b * fs))
        if kick_on:
            m = min(kk.size, n - i0)
            x[i0:i0 + m] += kk[:m]
        if hats:
            h = _hat(fs, rng)
            j0 = int(round((b + beat / 2) * fs))
            m = min(h.size, n - j0)
            if m > 0:
                x[j0:j0 + m] += h[:m]
    per_bar = []
    for i, b0 in enumerate(bars):
        c = chords[i % len(chords)]
        per_bar.append(c)
        minor = c.endswith("m")
        root = _NOTE[c[:-1] if minor else c]
        i0, i1 = int(round(b0 * fs)), min(n, int(round((b0 + 4 * beat) * fs)))
        tt = t[i0:i1] - b0
        env = np.minimum(1.0, np.minimum(tt, tt[-1] - tt) / 0.02)
        if pad:
            for iv in ((0, 3, 7) if minor else (0, 4, 7)):
                f = 440.0 * 2 ** ((60 + (root + iv) % 12 - 69) / 12.0)
                for k in (1, 2, 3, 4):
                    x[i0:i1] += 0.03 / k * env * np.sin(2 * np.pi * k * f * tt + k)
        if bass:
            fb = 440.0 * 2 ** ((36 + root - 69) / 12.0)
            x[i0:i1] += 0.25 * env * (np.sin(2 * np.pi * fb * tt) + 0.2 * np.sin(4 * np.pi * fb * tt))
    x *= level / max(1e-12, np.max(np.abs(x)))
    x = np.stack([x, x], axis=1) + pink_noise(n, fs, rms_dbfs=-55.0, seed=None if seed is None else seed + 7)
    return x, {"beat_times": beats, "bar_times": bars, "chords": per_bar, "bpm": bpm}


def ambient_pad(fs: float, seconds: float = 30.0, seed: int | None = 0) -> np.ndarray:
    """Beatless pad: four slowly swelling (0.05 Hz) notes with a slight vibrato over
    -40 dBFS pink air (audit features F1: no metrical evidence)."""
    n = int(round(seconds * fs))
    t = np.arange(n) / fs
    x = np.zeros(n)
    for midi, ph in ((57, 0.0), (60, 1.0), (64, 2.0), (69, 0.5)):
        f = 440.0 * 2 ** ((midi - 69) / 12.0)
        lfo = 0.5 + 0.5 * np.sin(2 * np.pi * 0.05 * t + ph)
        for k in (1, 2, 3):
            x += 0.05 / k * lfo * np.sin(2 * np.pi * k * f * (1 + 0.001 * np.sin(2 * np.pi * 0.2 * t)) * t)
    return np.stack([x, x * 0.95], axis=1) + pink_noise(n, fs, rms_dbfs=-40.0, seed=seed)


def sustained_chord(fs: float, seconds: float = 30.0, seed: int | None = 3) -> np.ndarray:
    """One A minor chord (A2 A3 C4 E4, 4 partials each) held throughout over a
    -45 dBFS pink bed: no onsets after the first."""
    n = int(round(seconds * fs))
    t = np.arange(n) / fs
    x = np.zeros(n)
    for midi in (45, 57, 60, 64):
        f = 440.0 * 2 ** ((midi - 69) / 12.0)
        for k in (1, 2, 3, 4):
            x += 0.08 / k * np.sin(2 * np.pi * k * f * t + k)
    return np.stack([x, x * 0.97], axis=1) + pink_noise(n, fs, rms_dbfs=-45.0, seed=seed)
