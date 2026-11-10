"""Audio-rate gain audit harness: CLI.

    python run.py h1 [--chars a,b] [--freqs 30,40,60] [--fs 48000]
    python run.py h2 | h3 | h4 | h5 | h6 | h7 | fixes | all

Writes out/<exp>.json and out/<exp>.md (tables) and prints the tables."""
from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from harness import common, drive, fixes, measure, signals  # noqa: E402
from harness.common import save_json, table, write_md  # noqa: E402
from eluate import envelope, planner  # noqa: E402
from eluate.plan import CONTROL_RATE_HZ, control_coord, control_length  # noqa: E402
from eluate.stages import glue as gl, limiter as lim, multiband as mb  # noqa: E402

CHARS = drive.CHARS
GDB = lambda g: 20.0 * np.log10(g)  # noqa: E731


def _r(d, nd=2):
    return {k: (round(v, nd) if isinstance(v, float) and math.isfinite(v) else v) for k, v in d.items()}


def pred_ripple(f0, att_ms, rel_ms, depth_db):
    """The architect's H1 formula: (1 - exp(-(T/2 - attack/2) / release)) depth."""
    half = 500.0 / f0
    span = max(0.0, half - att_ms / 2.0)
    return (1.0 - math.exp(-span / rel_ms)) * depth_db


def gain_row(stage, ch, f0, G_db, fs_g, y, fs, extra=None):
    g_lin = 10.0 ** (G_db / 20.0)
    r = {"stage": stage, "char": ch, "f0": f0, "gr_max_db": float(-G_db.min()),
         "ripple_db": measure.ripple_pp_db(G_db, fs_g)}
    r.update(measure.velocity(G_db, fs_g))
    r.update(measure.gain_spectrum(g_lin, fs_g))
    h = measure.harmonics(y, fs, f0, nh=min(200, int(20000 / f0)))
    r.update({k: v for k, v in h.items() if k != "nonharm_lines"})
    r["nonharm_lines"] = h["nonharm_lines"]
    r.update(measure.env_modulation(y, fs, f0))
    if extra:
        r.update(extra)
    return r


H1_COLS = ["stage", "char", "f0", "gr_max_db", "ripple_db", "pred_db", "vel_max", "vel_p999",
           "frac_gt_100", "frac_gt_500", "line_1000_db", "thd_db", "h2_db", "h3_db", "nonharm_db",
           "env_pp_db"]


# ---------------------------------------------------------------- H1

def exp_h1(fs, chars, freqs, seconds=12.0, lim_seconds=6.0):
    rows = []
    for f0 in freqs:
        x = signals.sine(f0, fs, seconds, -6.0)
        n = x.shape[0]
        hits = signals.hits_for(x, fs, 120.0)
        xl = signals.sine(f0, fs, lim_seconds, -6.0)
        for ch in chars:
            t = drive.timing(ch)
            t0 = time.time()
            # (a) compressor, budget-solved as the planner does
            p = drive.mb_realise(x, fs, drive.mb_plan(ch), hits)
            G = drive.mb_comp_gain_db(p, 0, 0, n, fs)
            y = mb.apply(x, fs, p)
            res = p.budget["result"][0][0]
            rows.append(gain_row("mb_comp", ch, f0, G, fs, y, fs, {
                "pred_db": pred_ripple(f0, t["mb_attack_ms"][0], t["mb_release_ms"][0], float(-G.min())),
                "att_ms": t["mb_attack_ms"][0], "rel_ms": t["mb_release_ms"][0],
                "threshold_db": res["threshold_db"], "comp_db": res["comp_db"]}))
            # (b) band ceiling as planned (lookahead per band, release 50 = MultibandPlan default)
            for label, look, rel in (("mb_ceiling(rel50)", t["band_lookahead_ms"], 50.0),
                                     ("mb_ceiling(rel100)", t["band_lookahead_ms"], 100.0),
                                     ("speclim(rel100)", t["speclim_lookahead_ms"], planner.SPECLIM_RELEASE_MS[0])):
                pc = drive.ceiling_only_plan(x, fs, 6.0, 0, look, rel)
                Gc = drive.mb_ceil_gain_db(pc, 0, 0, n, fs)
                yc = mb.apply(x, fs, pc)
                rows.append(gain_row(label, ch, f0, Gc, fs, yc, fs, {
                    "pred_db": pred_ripple(f0, look[0], rel, float(-Gc.min())),
                    "att_ms": look[0], "rel_ms": rel}))
            # (d) glue, budget-solved
            gp = drive.glue_plan(x, fs, ch, hits)
            Gg = drive.glue_gain_db(gp, n, fs)
            yg = gl.apply(x, fs, gp)
            rows.append(gain_row("glue", ch, f0, Gg, fs, yg, fs, {
                "pred_db": pred_ripple(f0, t["glue_attack_ms"], t["glue_release_ms"] / 2.0, float(-Gg.min())),
                "att_ms": t["glue_attack_ms"], "rel_ms": t["glue_release_ms"],
                "threshold_db": gp.threshold_db, "net_db": gp.budget["net_db"]}))
            # (e) limiter, 3 dB over the ceiling, as planned and with the raw preset release
            variants = [("limiter", False)]
            if t["limiter_release_raw_ms"] != t["limiter_release_ms"]:
                variants.append(("limiter(raw rel)", True))
            for label, raw in variants:
                lp = drive.limiter_plan(ch, input_gain_db=8.7, clip=False, budget=False, release_raw=raw)
                r = drive.limiter_run(xl, fs, lp, keep8=False)
                rows.append(gain_row(label, ch, f0, GDB(r["g8"]), r["fs8"], r["y"], fs, {
                    "pred_db": pred_ripple(f0, lp.attack_ms, lp.release_ms, r["info"]["max_gain_reduction_db"]),
                    "att_ms": lp.attack_ms, "rel_ms": lp.release_ms,
                    "relimit": len(r["info"]["relimit_passes"]), "trim_db": r["info"]["trim_db"]}))
            print(f"h1 f0={f0} {ch} done in {time.time() - t0:.0f} s", flush=True)
    save_json("h1.json", rows)
    md = "# H1: steady low-frequency sines (S1, -6 dBFS), sub band, per stage x character\n\n"
    md += "ripple_db = p-p of the applied gain; pred_db = architect's formula; vel in dB/ms; frac_gt = share of the gain's AC energy above f; line_1000_db = sideband a 1 kHz gain line would put on a partial (re carrier); thd/h2/h3/nonharm re fundamental; env_pp_db = p-p of the output's analytic envelope.\n\n"
    for stage in sorted({r["stage"] for r in rows}, key=lambda s: [r["stage"] for r in rows].index(s)):
        md += f"## {stage}\n\n" + table([_r(r) for r in rows if r["stage"] == stage], H1_COLS) + "\n\n"
    write_md("h1.md", md)
    print(md)
    return rows


# ---------------------------------------------------------------- H2

def _interp_refs(v, n, fs):
    from scipy.interpolate import CubicSpline, PchipInterpolator
    from scipy.signal import resample_poly
    k = np.arange(v.size, dtype=np.float64)
    t = np.clip(control_coord(np.arange(n), fs), 0.0, v.size - 1.0)
    out = {"pchip": PchipInterpolator(k, v)(t), "cubic": CubicSpline(k, v)(t)}
    up = int(round(fs / CONTROL_RATE_HZ))
    if abs(up - fs / CONTROL_RATE_HZ) < 1e-9:
        bl = resample_poly(v - v[0], up, 1, padtype="line") + v[0]
        out["bandlimited"] = bl[:n] if bl.size >= n else np.pad(bl, (0, n - bl.size), mode="edge")
    return out


def exp_h2(fs, chars, seconds=20.0):
    rows, rows5 = [], []
    x, bpm = signals.drum_loop(fs, beats=int(seconds * 2))
    n = x.shape[0]
    hits = signals.hits_for(x, fs, bpm)
    for ch in ("moderate", "aggressive"):
        p = drive.mb_realise(x, fs, drive.mb_plan(ch), hits)
        for k in range(4):
            v = p.gr_db[k][0].values
            if v.size == 1:
                continue
            for name, G in _interp_refs(v, n, fs).items():
                r = {"signal": "S3", "char": ch, "env": f"comp band{k} M", "interp": name, "gr_max_db": float(-G.min())}
                r.update(measure.velocity(G, fs))
                r.update(measure.gain_spectrum(10.0 ** (G / 20.0), fs))
                rows.append(r)
        if p.ceiling_gr_db is not None:
            for k in range(5):
                v = p.ceiling_gr_db[k][0].values
                if v.size == 1:
                    continue
                for name, G in _interp_refs(v, n, fs).items():
                    r = {"signal": "S3", "char": ch, "env": f"ceiling band{k} M (look {p.lookahead_ms()[k]} ms)",
                         "interp": name, "gr_max_db": float(-G.min())}
                    r.update(measure.velocity(G, fs))
                    r.update(measure.gain_spectrum(10.0 ** (G / 20.0), fs))
                    rows.append(r)
        # glue on S3
        gp = drive.glue_plan(x, fs, ch, hits)
        for name, G in _interp_refs(gp.gain_db.values, n, fs).items():
            r = {"signal": "S3", "char": ch, "env": "glue", "interp": name, "gr_max_db": float(-G.min())}
            r.update(measure.velocity(G, fs))
            r.update(measure.gain_spectrum(10.0 ** (G / 20.0), fs))
            rows.append(r)
        print(f"h2 S3 {ch} done", flush=True)
    # a 2 ms-lookahead ceiling on S3 band 3 (1600-6400), 6 dB cut
    for ch in ("moderate",):
        pc = drive.ceiling_only_plan(x, fs, 6.0, 3, [2.0] * 5, 50.0)
        v = pc.ceiling_gr_db[3][0].values
        for name, G in _interp_refs(v, n, fs).items():
            r = {"signal": "S3", "char": ch, "env": "ceiling band3 M 6 dB, look 2 ms, rel 50", "interp": name,
                 "gr_max_db": float(-G.min())}
            r.update(measure.velocity(G, fs))
            r.update(measure.gain_spectrum(10.0 ** (G / 20.0), fs))
            rows.append(r)
    # S5 probe: sidebands on the tone alone through each stage's gain, and in/out on the full probe
    x5, bursts = signals.probe_1k(fs, seconds)
    n5 = x5.shape[0]
    tone = signals.sine(1000.0, fs, seconds, -20.0)
    hits5 = signals.hits_for(x5, fs, 120.0)
    sb_in = measure.sidebands(x5, fs)
    rows5.append({"stage": "input", "char": "-", **{"full_" + k: v for k, v in sb_in.items()}})
    for ch in chars:
        p = drive.mb_realise(x5, fs, drive.mb_plan(ch), hits5)
        G2 = drive.mb_total_gain_db(p, 2, 0, n5, fs)
        yt = tone * (10.0 ** (G2 / 20.0))[:, None]
        y5 = mb.apply(x5, fs, p)
        r = {"stage": "multiband band2", "char": ch, "gr_max_db": float(-G2.min() + G2.max())}
        r.update({"tone_" + k: v for k, v in measure.sidebands(yt, fs).items()})
        r.update({"full_" + k: v for k, v in measure.sidebands(y5, fs).items()})
        r.update(measure.velocity(G2, fs))
        rows5.append(r)
        gp = drive.glue_plan(x5, fs, ch, hits5)
        Gg = drive.glue_gain_db(gp, n5, fs)
        yt = tone * (10.0 ** (Gg / 20.0))[:, None]
        y5 = gl.apply(x5, fs, gp)
        r = {"stage": "glue", "char": ch, "gr_max_db": float(-Gg.min())}
        r.update({"tone_" + k: v for k, v in measure.sidebands(yt, fs).items()})
        r.update({"full_" + k: v for k, v in measure.sidebands(y5, fs).items()})
        r.update(measure.velocity(Gg, fs))
        rows5.append(r)
        lp = drive.limiter_plan(ch, input_gain_db=6.0, clip=False, budget=False)
        rl = drive.limiter_run(x5, fs, lp, keep8=False)
        yt = drive.apply_gain_at_8x(tone * 10.0 ** (6.0 / 20.0), fs, rl["g8"])
        r = {"stage": "limiter (no clip)", "char": ch, "gr_max_db": rl["info"]["max_gain_reduction_db"]}
        r.update({"tone_" + k: v for k, v in measure.sidebands(yt, fs).items()})
        r.update({"full_" + k: v for k, v in measure.sidebands(rl["y"], fs).items()})
        r.update(measure.velocity(GDB(rl["g8"]), rl["fs8"]))
        rows5.append(r)
        print(f"h2 S5 {ch} done", flush=True)
    save_json("h2.json", {"interp": rows, "s5": rows5})
    cols = ["signal", "char", "env", "interp", "gr_max_db", "vel_max", "vel_p999", "acc_max", "frac_gt_100",
            "frac_gt_500", "line_1000_db", "line_2000_db", "line_3000_db"]
    cols5 = ["stage", "char", "gr_max_db", "vel_max", "tone_sb_db", "tone_sb_near200_db", "tone_sb_far_db",
             "tone_sb_1k_db", "full_sb_db", "full_sb_1k_db"]
    md = ("# H2: PCHIP across 1 ms knots\n\nInterpolation of the same control values: pchip (the stage), "
          "cubic (C2 spline), bandlimited (resample_poly). frac_gt_500 is the share of the gain's AC energy above "
          "the control Nyquist (pure interpolation artefact); line_f_db = sideband level a gain line at f would put "
          "on a partial, re carrier.\n\n")
    md += table([_r(r, 3) for r in rows], cols, {"frac_gt_100": ".2e", "frac_gt_500": ".2e"}) + "\n\n"
    md += ("## S5 probe (1 kHz at -20 dBFS + 5 ms bursts at -3 dBFS every 500 ms)\n\ntone_* = the tone alone "
           "through the stage's gain (pure modulation products); full_* = the whole probe in/out. sb = energy "
           "20 Hz..2 kHz either side of 1 kHz (excl. +-5 Hz) re carrier; sb_1k = within +-20 Hz of 0 and 2 kHz.\n\n")
    md += table([_r(r, 1) for r in rows5], cols5) + "\n"
    write_md("h2.md", md)
    print(md)


# ---------------------------------------------------------------- H3

def exp_h3(fs, chars, seconds=20.0):
    rows, rows_s1 = [], []
    x, bpm = signals.drum_loop(fs, beats=int(seconds * 2))
    n = x.shape[0]
    hits = signals.hits_for(x, fs, bpm)
    for ch in chars:
        p = drive.mb_realise(x, fs, drive.mb_plan(ch, profile="live"), hits)
        assert p.ms_link
        nc = control_length(n, fs)
        for k in range(5):
            Gb = drive.mb_total_gain_db(p, k, 0, n, fs)
            Gf = fixes.linked_gain_f3(p, k, n, fs)
            rm = np.clip(mb._ctrl(p.gr_db[k][0], nc), -6, 0) + (0 if p.ceiling_gr_db is None else mb._ctrl(p.ceiling_gr_db[k][0], nc))
            rs = np.clip(mb._ctrl(p.gr_db[k][1], nc), -6, 0) + (0 if p.ceiling_gr_db is None else mb._ctrl(p.ceiling_gr_db[k][1], nc))
            cross = int(np.count_nonzero(np.diff(np.sign(rm - rs)) != 0))
            vb, vf = measure.velocity(Gb, fs), measure.velocity(Gf, fs)
            sb, sf = measure.gain_spectrum(10 ** (Gb / 20), fs), measure.gain_spectrum(10 ** (Gf / 20), fs)
            rows.append({"char": ch, "band": k, "gr_max_db": float(-Gb.min() + Gb.max()), "crossings": cross,
                         "max_diff_db": float(np.max(np.abs(Gb - Gf))),
                         "acc_max_base": vb["acc_max"], "acc_max_f3": vf["acc_max"],
                         "vel_max_base": vb["vel_max"], "vel_max_f3": vf["vel_max"],
                         "frac500_base": sb["frac_gt_500"], "frac500_f3": sf["frac_gt_500"],
                         "line1k_base": sb["line_1000_db"], "line1k_f3": sf["line_1000_db"]})
        yb = mb.apply(x, fs, p)
        with fixes.f3():
            yf = mb.apply(x, fs, p)
        d = yb - yf
        rows.append({"char": ch, "band": "output diff", "max_diff_db": float(20 * np.log10(np.max(np.abs(d)) / np.max(np.abs(yb)))),
                     "diff_gt500_db": measure.band_energy_db(d, fs, 500, 20000) - measure.band_energy_db(yb, fs, 20, 20000),
                     "diff_total_db": measure.band_energy_db(d, fs, 20, 20000) - measure.band_energy_db(yb, fs, 20, 20000)})
        print(f"h3 {ch} done", flush=True)
    # steady state: a 40 Hz sine with R 6 dB below L (M and S both present), live profile
    x1 = signals.sine(40.0, fs, 12.0, -6.0, stereo_offset_db=-6.0)
    n1 = x1.shape[0]
    h1 = signals.hits_for(x1, fs, 120.0)
    for ch in chars:
        p = drive.mb_realise(x1, fs, drive.mb_plan(ch, profile="live"), h1)
        Gb = drive.mb_total_gain_db(p, 0, 0, n1, fs)
        Gf = fixes.linked_gain_f3(p, 0, n1, fs)
        yb = mb.apply(x1, fs, p)
        with fixes.f3():
            yf = mb.apply(x1, fs, p)
        hb, hf = measure.harmonics(yb, fs, 40.0, nh=50), measure.harmonics(yf, fs, 40.0, nh=50)
        rows_s1.append({"char": ch, "gr_M_db": float(-np.clip(p.gr_db[0][0].values, -6, 0).min()),
                        "gr_S_db": float(-np.clip(p.gr_db[0][1].values, -6, 0).min()),
                        "ripple_base": measure.ripple_pp_db(Gb, fs), "ripple_f3": measure.ripple_pp_db(Gf, fs),
                        "acc_base": measure.velocity(Gb, fs)["acc_max"], "acc_f3": measure.velocity(Gf, fs)["acc_max"],
                        "thd_base": hb["thd_db"], "thd_f3": hf["thd_db"], "h3_base": hb["h3_db"], "h3_f3": hf["h3_db"]})
    save_json("h3.json", {"s3": rows, "s1": rows_s1})
    md = "# H3: ms_link (live profile): np.minimum of two PCHIP curves at audio rate vs the minimum at the control rate (F3)\n\n"
    md += table([_r(r, 4) for r in rows], ["char", "band", "gr_max_db", "crossings", "max_diff_db", "vel_max_base", "vel_max_f3",
                                           "acc_max_base", "acc_max_f3", "frac500_base", "frac500_f3", "line1k_base", "line1k_f3",
                                           "diff_gt500_db", "diff_total_db"], {"frac500_base": ".2e", "frac500_f3": ".2e"}) + "\n\n"
    md += "## S1 40 Hz, R = L - 6 dB (M and S both compressed), live profile\n\n"
    md += table([_r(r, 3) for r in rows_s1], ["char", "gr_M_db", "gr_S_db", "ripple_base", "ripple_f3", "acc_base", "acc_f3",
                                              "thd_base", "thd_f3", "h3_base", "h3_f3"]) + "\n"
    write_md("h3.md", md)
    print(md)


# ---------------------------------------------------------------- H4

def _glue_metrics(name, ch, variant, x, on, G, y, fs, sub):
    r = {"signal": name, "char": ch, "variant": variant, "gr_max_db": float(-G.min())}
    r.update(measure.velocity(G, fs))
    r.update(measure.kick_look(G, fs, on))
    ys = sub * (10.0 ** (G / 20.0))[:, None]
    r["sub_splatter_db"] = measure.band_energy_db(ys, fs, 100, 1000) - measure.band_energy_db(ys, fs, 20, 60)
    r["sub_h3region_db"] = measure.band_energy_db(ys, fs, 110, 130) - measure.band_energy_db(ys, fs, 20, 60)
    r["sub_h2region_db"] = measure.band_energy_db(ys, fs, 70, 90) - measure.band_energy_db(ys, fs, 20, 60)
    r["full_100_1000_in_db"] = measure.band_energy_db(x, fs, 100, 1000)
    r["full_100_1000_out_db"] = measure.band_energy_db(y, fs, 100, 1000)
    r["onset_100_1000_in_db"] = measure.window_energy_db(x, fs, on, 2, 20, 100, 1000)
    r["onset_100_1000_out_db"] = measure.window_energy_db(y, fs, on, 2, 20, 100, 1000)
    r["onset_1k_5k_in_db"] = measure.window_energy_db(x, fs, on, 2, 20, 1000, 5000)
    r["onset_1k_5k_out_db"] = measure.window_energy_db(y, fs, on, 2, 20, 1000, 5000)
    r["thirds_in"] = measure.third_octaves(x, fs)
    r["thirds_out"] = measure.third_octaves(y, fs)
    return r


def exp_h4(fs, chars, seconds=20.0):
    rows = []
    x2, on2 = signals.sine_plus_kick(fs, seconds)
    x7, on7 = signals.kick_loop(fs, seconds)
    sub = signals.sine(40.0, fs, seconds, -12.0)
    n = x2.shape[0]
    hits2, hits7 = signals.hits_for(x2, fs, 128.0), signals.hits_for(x7, fs, 128.0)
    for ch in chars:
        for name, x, on, hits in (("S2", x2, on2, hits2), ("S7", x7, on7, hits7)):
            gp = drive.glue_plan(x, fs, ch, hits)
            G = drive.glue_gain_db(gp, n, fs)
            y = gl.apply(x, fs, gp)
            rows.append(_glue_metrics(name, ch, "base", x, on, G, y, fs, sub) | {"threshold_db": gp.threshold_db, "net_db": gp.budget["net_db"]})
            if name == "S2" and ch in ("fast", "aggressive", "moderate"):
                for variant, kw in (("F4a attack>=3ms", dict(attack_floor_ms=3.0)),
                                    ("F4b hold 25ms", dict(hold_ms=25.0)),
                                    ("F4a+b", dict(attack_floor_ms=3.0, hold_ms=25.0))):
                    with fixes.f4(**kw):
                        gpv = drive.glue_plan(x, fs, ch, hits)
                    Gv = drive.glue_gain_db(gpv, n, fs)
                    yv = gl.apply(x, fs, gpv)
                    rows.append(_glue_metrics(name, ch, variant, x, on, Gv, yv, fs, sub) | {"net_db": gpv.budget["net_db"]})
                ysp = fixes.glue_apply_split(x, fs, gp, 100.0)
                r = _glue_metrics(name, ch, "F4c split 100 Hz", x, on, G, ysp, fs, sub)
                ys = fixes.glue_apply_split(sub, fs, gp, 100.0)       # the sub alone through the split glue
                r["sub_splatter_db"] = measure.band_energy_db(ys, fs, 100, 1000) - measure.band_energy_db(ys, fs, 20, 60)
                r["sub_h3region_db"] = measure.band_energy_db(ys, fs, 110, 130) - measure.band_energy_db(ys, fs, 20, 60)
                r["sub_h2region_db"] = measure.band_energy_db(ys, fs, 70, 90) - measure.band_energy_db(ys, fs, 20, 60)
                rows.append(r)
        print(f"h4 {ch} done", flush=True)
    save_json("h4.json", rows)
    cols = ["signal", "char", "variant", "gr_max_db", "net_db", "vel_max", "vel_p999", "depth_db_med", "t_min_ms_med",
            "drop_1ms_db_med", "drop_1ms_db_max", "drop_2ms_db_max", "drop_5ms_db_max", "rec63_ms_med",
            "sub_splatter_db", "sub_h2region_db", "sub_h3region_db", "onset_100_1000_in_db", "onset_100_1000_out_db",
            "onset_1k_5k_in_db", "onset_1k_5k_out_db"]
    md = ("# H4: glue on S2 (40 Hz sine -12 dBFS + kick at 128 BPM) and S7 (kick only)\n\n"
          "drop_Nms = largest gain fall over any N ms around an onset; sub_splatter = energy of (sub x gain) in "
          "100..1000 Hz re its 20..60 Hz energy (the sub alone has none); onset_* = median band energy in the 2 ms "
          "before to 20 ms after each onset, input vs output.\n\n")
    md += table([_r(r) for r in rows], cols) + "\n"
    write_md("h4.md", md)
    print(md)


# ---------------------------------------------------------------- H5

def exp_h5(fs, seconds=12.0):
    from eluate.plan import Envelope, Notch, NotchPlan
    from eluate.stages import notch
    rows = []
    x7, on = signals.kick_loop(fs, seconds, kick_dbfs=-12.0)
    n = x7.shape[0]
    t = np.arange(n) / fs
    tones = {500.0: -12.0, 580.0: -14.0, 660.0: -16.0, 3150.0: -18.0}
    tone = np.zeros((n, 2))
    for f0, lv in tones.items():
        tone += (10 ** (lv / 20) * np.sin(2 * np.pi * f0 * t))[:, None]
    x = x7 + tone
    nc = control_length(n, fs)
    tc = np.arange(nc) / CONTROL_RATE_HZ

    def smooth_env(depth_db, spans, ramp_s=0.2):
        g = np.zeros(nc)
        for a, b in spans:
            w = np.clip((tc - a) / ramp_s, 0, 1) * np.clip((b - tc) / ramp_s, 0, 1)
            w = 0.5 * (1 - np.cos(np.pi * np.clip(w, 0, 1)))
            g = np.maximum(g, w)
        return Envelope(-depth_db * g)

    def planner_env(excess_db, spans, hop_s=0.1):
        tg = (np.arange(int(seconds / hop_s)) + 0.5) * hop_s
        ex = np.zeros(tg.size)
        for a, b in spans:
            ex = np.maximum(ex, excess_db * np.clip(np.minimum(tg - a, b - tg) / 1.5, 0, 1))
        return Envelope(planner._notch_envelope(ex, hop_s, 1.0, n, fs))

    spans = [(1.0, 5.0), (7.0, 10.5)]
    for style, masked in (("smooth", True), ("planner", True), ("smooth", False), ("planner", False)):
        if style == "smooth":
            envs = [smooth_env(6.0, spans), smooth_env(5.0, spans), smooth_env(6.0, spans), smooth_env(4.0, [(0.5, 11.0)])]
        else:
            envs = [planner_env(18.0, spans), planner_env(16.0, spans), planner_env(18.0, spans), planner_env(14.0, [(0.5, 11.0)])]
        notches = [Notch(500.0, 8.0, envs[0]), Notch(580.0, 8.0, envs[1]), Notch(660.0, 8.0, envs[2]), Notch(3150.0, 8.0, envs[3])]
        mask = notch.kick_mask(on, n / fs)
        plan = NotchPlan(notches, kick_mask=mask if masked else None)
        y = notch.apply(x, fs, plan)
        style = style + ("+mask" if masked else "")
        keep = 1.0 - np.clip(mask.at(n, fs), 0, 1) if masked else np.ones(n)
        keep_ctrl = 1.0 - np.clip(notch._ctrl_values(mask, nc), 0, 1) if masked else None
        soss, d_ctrl, f0s, ds = [], [], [], []
        for nt in notches:
            g_db = np.clip(nt.gain_db.at(n, fs), -6.0, 0.0)
            d = keep * (1.0 - 10.0 ** (g_db / 20.0))
            ds.append(d)
            soss.append(notch.notch_sos(nt, fs))
            f0s.append(nt.f0_hz)
            d_ctrl.append(1.0 - 10.0 ** (np.clip(notch._ctrl_values(nt.gain_db, nc), -6.0, 0.0) / 20.0))
        s_ctrl = notch.composite_scale(np.stack(d_ctrl, axis=1), keep_ctrl, soss, f0s, fs)
        from scipy.interpolate import PchipInterpolator
        s = PchipInterpolator(np.arange(nc, dtype=np.float64), s_ctrl)(np.clip(control_coord(np.arange(n), fs), 0, nc - 1))
        for i, nt in enumerate(notches):
            gain_f0 = 1.0 - s * ds[i]          # the notch gain at f0 (B = 1 there)
            Gdb = 20 * np.log10(gain_f0)
            r = {"style": style, "notch": nt.f0_hz, "gr_max_db": float(-Gdb.min()), "min_s": float(s.min())}
            r.update(measure.velocity(Gdb, fs))
            r.update({k: v for k, v in measure.gain_spectrum(gain_f0, fs).items() if k.startswith(("frac", "line_1000", "line_2000"))})
            tone_i = (10 ** (tones[nt.f0_hz] / 20) * np.sin(2 * np.pi * nt.f0_hz * t))
            yt = np.stack([tone_i * gain_f0] * 2, axis=1)
            r.update({"tone_" + k: v for k, v in measure.sidebands(yt, fs, fc=nt.f0_hz).items()})
            r.update({"in_" + k: v for k, v in measure.sidebands(x, fs, fc=nt.f0_hz).items()})
            r.update({"out_" + k: v for k, v in measure.sidebands(y, fs, fc=nt.f0_hz).items()})
            # count control-rate steps of the envelope (0.5 dB quantisation)
            dv = np.diff(nt.gain_db.values)
            r["ctrl_steps"] = int(np.count_nonzero(np.abs(dv) > 1e-6))
            r["ctrl_step_max_db"] = float(np.max(np.abs(dv)))
            rows.append(r)
        rs = {"style": style, "notch": "s(t)", "min_s": float(s.min()), "gr_max_db": float(-20 * np.log10(s.min()))}
        rs.update(measure.velocity(20 * np.log10(s), fs))
        rows.append(rs)
        print(f"h5 {style} done", flush=True)
    save_json("h5.json", rows)
    cols = ["style", "notch", "gr_max_db", "min_s", "ctrl_steps", "ctrl_step_max_db", "vel_max", "vel_p999", "acc_max",
            "frac_gt_100", "frac_gt_500", "line_1000_db", "tone_sb_db", "tone_sb_near200_db", "tone_sb_far_db",
            "in_sb_db", "out_sb_db"]
    md = ("# H5: notch d(t) = keep(t) depth(t) and the composite s(t)\n\nThree notches at 500/580/660 Hz (Q 8, the "
          "composite clamp engages) and one at 3150 Hz, on four tones + a kick loop (kick mask on). 'smooth' = "
          "raised-cosine envelopes; 'planner' = planner._notch_envelope (0.5 dB quantised). Velocity of the notch gain "
          "at f0 in dB/ms; tone_sb = modulation sidebands on the tone alone re carrier.\n\n")
    md += table([_r(r, 3) for r in rows], cols, {"frac_gt_100": ".2e", "frac_gt_500": ".2e"}) + "\n"
    write_md("h5.md", md)
    print(md)


# ---------------------------------------------------------------- H6

def _span_checks(r):
    """Span clipping vs the curve applied to every sample (must be identical: the curve is
    the identity below the knee), and the band-limit window edges."""
    u, uc, ubl, spans = r["u_pre"], r["uc"], r["u_bl"], r["spans"]
    if not spans:
        return {"n_spans": 0}
    L = lim.OVERSAMPLE
    m = lim.BL_MARGIN * L
    n = u.shape[0]
    ch = np.max(np.abs(uc - u), axis=1)
    bl = np.max(np.abs(ubl - u), axis=1)
    t, w, rr = lim.whole_threshold(r["plan"])
    src = u if r["u_hi"] is None else r["u_hi"]
    full = u + (lim._knee_clip(src, t, w, rr) - src)
    span_vs_full = float(np.max(np.abs(full - uc)))
    edge_bl, maxc, lens = [], [], []
    for a0, a1 in spans:
        pk = ch[a0:a1].max()
        if pk <= 0:
            continue
        maxc.append(pk)
        s0 = max(0, ((a0 - m) // L) * L)
        s1 = min(n, a1 + m)
        s1 = s0 + ((s1 - s0) // L) * L
        edge_bl.append(max(bl[s0], bl[s1 - 1]) / pk)
        lens.append((a1 - a0) / L)
    return {"n_spans": len(spans), "span_ms_med": float(np.median(lens)) * 1000 / r["fs8"] * L,
            "span_ms_max": float(np.max(lens)) * 1000 / r["fs8"] * L,
            "span_vs_full": span_vs_full, "edge_bl_rel": float(np.max(edge_bl)),
            "change_peak_db": float(20 * np.log10(np.max(maxc))),
            "raw_change_gt_nyq": measure.above_nyquist_fraction(uc - u, r["fs8"], r["fs8"] / L),
            "bl_change_gt_nyq": measure.above_nyquist_fraction(ubl - u, r["fs8"], r["fs8"] / L)}


def exp_h6(fs, chars, seconds=8.0):
    rows, rows_s = [], []
    x6 = signals.pink_bursts(fs, seconds)
    hits6 = signals.hits_for(x6, fs, 120.0)
    for ch in chars:
        lp = drive.limiter_plan(ch, input_gain_db=6.0, clip=True)
        lp = drive.limiter_solve(x6, fs, lp, hits6, 6.0)
        r = drive.limiter_run(x6, fs, lp)
        row = {"signal": "S6", "char": ch, "knee_db": lp.clip_knee_db, "clip_thr_db": lp.clip_threshold_db,
               "gain_db": lp.input_gain_db, "clip_red_db": r["info"]["clip_max_reduction_db"],
               "lim_gr_db": r["info"]["max_gain_reduction_db"], "relimit": len(r["info"]["relimit_passes"]),
               "relimit_max_db": max([p["max_reduction_db"] for p in r["info"]["relimit_passes"]], default=0.0),
               "trim_db": r["info"]["trim_db"], "tp8": r["info"]["true_peak_8x_dbtp"], "tp32": r["info"]["true_peak_32x_dbtp"]}
        row.update(_span_checks(r))
        row.update({"g8_" + k: v for k, v in measure.velocity(GDB(r["g8"]), r["fs8"]).items()})
        # the re-limit pass gain (None when the stage under test returns no separate trace)
        row["g2_min_db"] = None if r["g2"] is None else float(20 * np.log10(r["g2"].min()))
        rows.append(row)
        print(f"h6 S6 {ch} done", flush=True)
    # sines above the split: 997 Hz and 199 Hz, 3 dB over; THD of the clipper alone and of the whole stage
    for f0 in (997.0, 199.0):
        xs = signals.sine(f0, fs, seconds, -6.0)
        for ch in chars + ("hard knee", "no sample cap"):
            base = "aggressive" if ch in ("hard knee", "no sample cap") else ch
            lp = drive.limiter_plan(base, input_gain_db=8.7, clip=True, budget=False,
                                    knee_override=0.0 if ch == "hard knee" else None,
                                    sample_cap=ch != "no sample cap")
            lp = drive.clip_threshold_direct(xs, fs, lp, 1.0)
            r = drive.limiter_run(xs, fs, lp)
            from eluate.stages import resample
            y_clip = resample.decimate(r["u_bl"], lim.OVERSAMPLE)
            hc = measure.harmonics(y_clip, fs, f0, nh=min(200, int(20000 / f0)))
            hy = measure.harmonics(r["y"], fs, f0, nh=min(200, int(20000 / f0)))
            row = {"f0": f0, "char": ch, "knee_db": lp.clip_knee_db, "clip_thr_db": lp.clip_threshold_db,
                   "clip_red_db": r["info"]["clip_max_reduction_db"], "lim_gr_db": r["info"]["max_gain_reduction_db"],
                   "relimit": len(r["info"]["relimit_passes"]), "trim_db": r["info"]["trim_db"],
                   "clip_thd_db": hc["thd_db"], "clip_h2_db": hc["h2_db"], "clip_h3_db": hc["h3_db"],
                   "clip_nonharm_db": hc["nonharm_db"], "clip_lines": hc["nonharm_lines"],
                   "out_thd_db": hy["thd_db"], "out_h3_db": hy["h3_db"], "out_nonharm_db": hy["nonharm_db"],
                   "out_lines": hy["nonharm_lines"]}
            row.update(_span_checks(r))
            rows_s.append(row)
        # THD vs level at a fixed threshold (solved at 3 dB over), aggressive knee 0.5: the
        # threshold follows the input gain by design, so the signal level is varied instead
        lp0 = drive.clip_threshold_direct(xs, fs, drive.limiter_plan("aggressive", input_gain_db=8.7, clip=True, budget=False), 1.0)
        for over in (0.5, 1.0, 2.0, 3.0, 4.0, 6.0):
            lp = lp0
            r = drive.limiter_run(xs * 10.0 ** ((over - 3.0) / 20.0), fs, lp, keep8=True)
            from eluate.stages import resample
            y_clip = resample.decimate(r["u_bl"], lim.OVERSAMPLE)
            hc = measure.harmonics(y_clip, fs, f0, nh=min(200, int(20000 / f0)))
            hy = measure.harmonics(r["y"], fs, f0, nh=min(200, int(20000 / f0)))
            rows_s.append({"f0": f0, "char": f"aggr, {over:g} dB over", "knee_db": lp.clip_knee_db,
                           "clip_thr_db": lp.clip_threshold_db, "clip_red_db": r["info"]["clip_max_reduction_db"],
                           "lim_gr_db": r["info"]["max_gain_reduction_db"], "relimit": len(r["info"]["relimit_passes"]),
                           "trim_db": r["info"]["trim_db"], "clip_thd_db": hc["thd_db"], "clip_h2_db": hc["h2_db"],
                           "clip_h3_db": hc["h3_db"], "clip_nonharm_db": hc["nonharm_db"], "out_thd_db": hy["thd_db"],
                           "out_h3_db": hy["h3_db"], "out_nonharm_db": hy["nonharm_db"], **_span_checks(r)})
        print(f"h6 sine {f0} done", flush=True)
    save_json("h6.json", {"s6": rows, "sines": rows_s})
    cols = ["signal", "char", "knee_db", "clip_thr_db", "gain_db", "clip_red_db", "lim_gr_db", "relimit", "relimit_max_db",
            "trim_db", "tp8", "tp32", "n_spans", "span_ms_med", "span_ms_max", "span_vs_full", "edge_bl_rel",
            "raw_change_gt_nyq", "bl_change_gt_nyq", "g8_vel_max", "g2_min_db"]
    cols_s = ["f0", "char", "knee_db", "clip_red_db", "lim_gr_db", "relimit", "trim_db", "clip_thd_db", "clip_h2_db",
              "clip_h3_db", "clip_nonharm_db", "out_thd_db", "out_h3_db", "out_nonharm_db", "n_spans", "span_vs_full",
              "edge_bl_rel", "raw_change_gt_nyq", "bl_change_gt_nyq"]
    md = ("# H6: clipper and limiter\n\nspan_vs_full = max |span clipping - the curve on every sample| (must be 0); "
          "edge_bl_rel = |band-limited change| at the local window's "
          "edges re the peak change (truncation check); *_gt_nyq = energy fraction of the change above the base "
          "Nyquist (8x domain), raw and after band_limit_change.\n\n## S6 pink bursts (as planned, +6 dB requested)\n\n")
    md += table([_r(r, 4) for r in rows], cols, {"raw_change_gt_nyq": ".2e", "bl_change_gt_nyq": ".2e", "edge_bl_rel": ".2e"}) + "\n\n"
    md += "## Sines above the split, 3 dB over (clipper 1 dB target, then the limiter); clip_* = after the clipper and band-limit only\n\n"
    md += table([_r(r, 4) for r in rows_s], cols_s, {"raw_change_gt_nyq": ".2e", "bl_change_gt_nyq": ".2e", "edge_bl_rel": ".2e"}) + "\n"
    write_md("h6.md", md)
    print(md)


# ---------------------------------------------------------------- H7

def exp_h7(seconds=12.0):
    rows = []
    for fs in (48000, 44100):
        x = signals.sine(40.0, fs, seconds, -6.0)
        n = x.shape[0]
        hits = signals.hits_for(x, fs, 120.0)
        for ch in ("moderate", "aggressive"):
            p = drive.mb_realise(x, fs, drive.mb_plan(ch), hits)
            G = drive.mb_comp_gain_db(p, 0, 0, n, fs)
            y = mb.apply(x, fs, p)
            rows.append(gain_row("mb_comp", ch, 40.0, G, fs, y, fs, {"fs": fs, "threshold_db": p.budget["result"][0][0]["threshold_db"]}))
            pc = drive.ceiling_only_plan(x, fs, 6.0, 0, drive.timing(ch)["band_lookahead_ms"], 50.0)
            Gc = drive.mb_ceil_gain_db(pc, 0, 0, n, fs)
            rows.append(gain_row("mb_ceiling(rel50)", ch, 40.0, Gc, fs, mb.apply(x, fs, pc), fs, {"fs": fs}))
            xl = signals.sine(40.0, fs, 6.0, -6.0)
            lp = drive.limiter_plan(ch, input_gain_db=8.7, clip=False, budget=False)
            r = drive.limiter_run(xl, fs, lp, keep8=False)
            rows.append(gain_row("limiter", ch, 40.0, GDB(r["g8"]), r["fs8"], r["y"], fs, {"fs": fs}))
        # control_coord checks
        from fractions import Fraction
        idx = np.arange(n)
        c = control_coord(idx, fs)
        k = np.floor(c).astype(np.int64)
        lens = np.bincount(k)
        exact = idx[idx % (fs // 100) == 0] if fs == 44100 else idx[idx % 48 == 0]
        ce = control_coord(exact, fs)
        err_ulp = np.max(np.abs(ce - np.round(ce)) / np.spacing(np.maximum(ce, 1.0)))
        samp = idx[::997]
        worst = max(abs(float(Fraction(int(i)) * 1000 / fs) - float(control_coord(i, fs))) / max(float(np.spacing(control_coord(i, fs))), 1e-300) for i in samp)
        rows.append({"stage": "control_coord", "char": "-", "fs": fs, "seg_lengths": sorted(set(lens[:-1].tolist())),
                     "exact_boundary_err_ulp": float(err_ulp), "worst_err_ulp": float(worst)})
        print(f"h7 fs={fs} done", flush=True)
    save_json("h7.json", rows)
    md = "# H7: 48 kHz vs 44.1 kHz (S1 40 Hz)\n\n"
    md += table([_r(r, 3) for r in rows if r["stage"] != "control_coord"], ["stage", "char", "fs", "gr_max_db", "ripple_db", "vel_max", "frac_gt_500", "line_1000_db", "thd_db", "h2_db", "h3_db", "nonharm_db"], {"frac_gt_500": ".2e"}) + "\n\n"
    md += table([r for r in rows if r["stage"] == "control_coord"], ["fs", "seg_lengths", "exact_boundary_err_ulp", "worst_err_ulp"]) + "\n"
    write_md("h7.md", md)
    print(md)


# ---------------------------------------------------------------- fixes

HOLD_MS = [25.0, 10.0, 2.5, 1.0, 0.0]        # F1: one period of the band's lower edge (sub >= 25 ms)
WIN_MS = [25.0, 10.0, 3.0, 2.0, 2.0]         # F2: trailing detector window


def _chain(x, fs, ch, hits_beats, loud, variant, req_db, lim_hold_ms=25.0, profile="electronic"):
    """multiband -> glue -> speclim -> clipper + limiter, as realise does, with the frozen loud mask."""
    from eluate.stages import resample  # noqa: F401
    ctx = {"base": contextlib.nullcontext(), "F1": fixes.f1(HOLD_MS), "F2": fixes.f2(WIN_MS)}[variant]
    out = {"variant": variant}
    with ctx:
        h = envelope.hits_for(x, fs, hits_beats, loud=loud)
        p = drive.mb_realise(x, fs, drive.mb_plan(ch, profile), h)
        y = mb.apply(x, fs, p)
        rows = p.budget["result"]
        out["mb"] = [{"band": k, "comp_db": rows[k][0]["comp_db"], "comp_outlier_db": rows[k][0]["comp_outlier_db"],
                      "limit_db": rows[k][0]["limit_db"], "limit_outlier_db": rows[k][0]["limit_outlier_db"],
                      "threshold_db": rows[k][0]["threshold_db"], "ceiling_db": rows[k][0]["ceiling_db"]} for k in range(5)]
        out["mb_warnings"] = len(p.budget["warnings"])
        h = envelope.hits_for(y, fs, hits_beats, loud=loud)
        gp = drive.glue_plan(y, fs, ch, h, profile)
        y = gl.apply(y, fs, gp)
        out["glue"] = {"net_db": gp.budget["net_db"], "net_outlier_db": gp.budget["net_outlier_db"], "threshold_db": gp.threshold_db}
        h = envelope.hits_for(y, fs, hits_beats, loud=loud)
        sp = drive.speclim_plan(y, fs, ch, h, profile)
        if sp is not None:
            y = mb.apply(y, fs, sp)
            out["speclim"] = [{"band": k, "cut_M": sp.budget["result"][k][0]["cut_db"], "cut_S": sp.budget["result"][k][1]["cut_db"]} for k in range(5)]
            out["speclim_max_cut_db"] = max(float(-e.values.min()) for pair in sp.ceiling_gr_db for e in pair)
        else:
            out["speclim"], out["speclim_max_cut_db"] = None, 0.0
    out["pre_limiter_rms_db"] = drive.rms_meter_db(y, fs)
    out["pre_limiter_peak_db"] = float(20 * np.log10(np.max(np.abs(y))))
    h = envelope.hits_for(y, fs, hits_beats, loud=loud)
    lp = drive.limiter_plan(ch, profile, clip=True, budget=True)
    lp = drive.limiter_solve(y, fs, lp, h, req_db)
    smooth = fixes.limiter_smoother(lim_hold_ms, fs * lim.OVERSAMPLE) if variant == "F1" else None
    r = drive.limiter_run(y, fs, lp, smooth=smooth, keep8=False)
    out["limiter"] = {"requested_db": req_db, "input_gain_db": lp.input_gain_db,
                      "clip_thr_db": lp.clip_threshold_db, "clip_avg_db": lp.budget["clipper"]["peak_gr_db"],
                      "clip_outlier_db": lp.budget["clipper"]["outlier_gr_db"],
                      "planned_gr_db": lp.budget["limiter"].get("planned_gr_db"),
                      "planned_outlier_db": lp.budget["limiter"].get("planned_outlier_gr_db"),
                      "gr_max_db": r["info"]["max_gain_reduction_db"], "relimit": len(r["info"]["relimit_passes"]),
                      "trim_db": r["info"]["trim_db"], "clamped": bool(lp.budget["limiter"].get("warnings"))}
    out["out_rms_db"] = drive.rms_meter_db(r["y"], fs)
    out["out_crest_db"] = drive.crest_db(r["y"], fs)
    # the loudness cost at equal gain: the same requested gain without the budget clamp
    from dataclasses import replace
    lp_eq = replace(lp, input_gain_db=req_db, max_gr_db=None, max_outlier_gr_db=None)
    r2 = drive.limiter_run(y, fs, lp_eq, smooth=smooth, keep8=False)
    out["eqgain_rms_db"] = drive.rms_meter_db(r2["y"], fs)
    out["eqgain_gr_max_db"] = r2["info"]["max_gain_reduction_db"]
    out["eqgain_relimit"] = len(r2["info"]["relimit_passes"])
    return out


def exp_fixes(fs, chars, seconds=16.0):
    results = []
    x3, bpm3 = signals.drum_loop(fs, beats=int(seconds * 2))
    x2, _ = signals.sine_plus_kick(fs, seconds)
    cases = [("S3", x3, 120.0, chars), ("S2", x2, 128.0, [c for c in chars if c in ("fast", "aggressive")])]
    for name, x, bpm, chs in cases:
        beats = np.arange(60.0 / bpm, x.shape[0] / fs, 60.0 / bpm)
        loud = envelope.hits_for(x, fs, beats)["loud"]
        for ch in chs:
            # the requested limiter gain: the baseline chain's pre-limiter peak + 3 dB over the ceiling
            req = None
            for variant in ("base", "F1", "F2"):
                t0 = time.time()
                if req is None:
                    # one probe pass for the request (equal plan settings for every variant)
                    probe = _chain(x, fs, ch, beats, loud, "base", 0.0)
                    req = planner.CEILING_DBTP - probe["pre_limiter_peak_db"] + 3.0
                o = _chain(x, fs, ch, beats, loud, variant, req)
                o.update(signal=name, char=ch)
                results.append(o)
                print(f"fixes {name} {ch} {variant}: rms {o['out_rms_db']:.2f} (eq gain {o['eqgain_rms_db']:.2f}) "
                      f"gain {o['limiter']['input_gain_db']:.2f} lim GR {o['limiter']['gr_max_db']:.2f} "
                      f"[{time.time() - t0:.0f} s]", flush=True)
    # S1 ripple with the fixes (40 Hz, aggressive and fast)
    s1 = []
    for f0 in (40.0, 30.0):
        x = signals.sine(f0, fs, 12.0, -6.0)
        n = x.shape[0]
        hits = signals.hits_for(x, fs, 120.0)
        xl = signals.sine(f0, fs, 6.0, -6.0)
        for ch in ("fast", "aggressive"):
            t = drive.timing(ch)
            for variant in ("base", "F1", "F2"):
                ctx = {"base": contextlib.nullcontext(), "F1": fixes.f1(HOLD_MS), "F2": fixes.f2(WIN_MS)}[variant]
                with ctx:
                    p = drive.mb_realise(x, fs, drive.mb_plan(ch), hits)
                    G = drive.mb_comp_gain_db(p, 0, 0, n, fs)
                    y = mb.apply(x, fs, p)
                    s1.append(gain_row(f"mb_comp/{variant}", ch, f0, G, fs, y, fs, {"threshold_db": p.budget["result"][0][0]["threshold_db"], "comp_db": p.budget["result"][0][0]["comp_db"]}))
                    pc = drive.ceiling_only_plan(x, fs, 6.0, 0, t["band_lookahead_ms"], 50.0)
                    Gc = drive.mb_ceil_gain_db(pc, 0, 0, n, fs)
                    s1.append(gain_row(f"mb_ceiling(rel50)/{variant}", ch, f0, Gc, fs, mb.apply(x, fs, pc), fs))
                if variant != "F2":
                    for hold in ((25.0,) if variant == "F1" else (0.0,)):
                        lp = drive.limiter_plan(ch, input_gain_db=8.7, clip=False, budget=False)
                        sm = fixes.limiter_smoother(hold, fs * lim.OVERSAMPLE) if variant == "F1" else None
                        r = drive.limiter_run(xl, fs, lp, smooth=sm, keep8=False)
                        s1.append(gain_row(f"limiter/{variant}", ch, f0, GDB(r["g8"]), r["fs8"], r["y"], fs, {"hold_ms": hold, "relimit": len(r["info"]["relimit_passes"])}))
                    if variant == "F1":
                        for hold in (12.5, 33.0):
                            lp = drive.limiter_plan(ch, input_gain_db=8.7, clip=False, budget=False)
                            r = drive.limiter_run(xl, fs, lp, smooth=fixes.limiter_smoother(hold, fs * lim.OVERSAMPLE), keep8=False)
                            s1.append(gain_row(f"limiter/F1", ch, f0, GDB(r["g8"]), r["fs8"], r["y"], fs, {"hold_ms": hold, "relimit": len(r["info"]["relimit_passes"])}))
            print(f"fixes S1 {f0} {ch} done", flush=True)
    save_json("fixes.json", {"chain": results, "s1": s1})
    md = "# Fixes: loudness and GR cost (chain multiband -> glue -> speclim -> clipper + limiter, equal requested gain)\n\n"
    rows = []
    for o in results:
        rows.append({"signal": o["signal"], "char": o["char"], "variant": o["variant"],
                     "mb0_comp": o["mb"][0]["comp_db"], "mb0_out": o["mb"][0]["comp_outlier_db"], "mb0_thr": o["mb"][0]["threshold_db"],
                     "mb1_comp": o["mb"][1]["comp_db"], "mb1_out": o["mb"][1]["comp_outlier_db"],
                     "mb_limit_max": max(b["limit_db"] for b in o["mb"]), "glue_net": o["glue"]["net_db"], "glue_out": o["glue"]["net_outlier_db"],
                     "speclim_max": o["speclim_max_cut_db"], "pre_rms": o["pre_limiter_rms_db"], "req_db": o["limiter"]["requested_db"],
                     "gain_db": o["limiter"]["input_gain_db"], "clamped": o["limiter"]["clamped"], "clip_avg": o["limiter"]["clip_avg_db"],
                     "lim_gr": o["limiter"]["gr_max_db"], "relimit": o["limiter"]["relimit"], "out_rms": o["out_rms_db"],
                     "out_crest": o["out_crest_db"], "eqgain_rms": o["eqgain_rms_db"], "eqgain_gr": o["eqgain_gr_max_db"]})
    md += table([_r(r) for r in rows], list(rows[0].keys())) + "\n\n## S1 ripple / THD with the fixes\n\n"
    md += table([_r(r) for r in s1], ["stage", "char", "f0", "hold_ms", "gr_max_db", "ripple_db", "vel_max", "thd_db", "h3_db", "nonharm_db", "comp_db", "threshold_db", "relimit"]) + "\n"
    write_md("fixes.md", md)
    print(md)


def exp_mbkick(fs, chars, seconds=12.0):
    """Deliverable 2e for the multiband and speclim: the sub / low band gains around kicks on
    S2 and S7, and the output's 100..1000 Hz / 1..5 kHz onset energy vs the input."""
    rows = []
    x2, on2 = signals.sine_plus_kick(fs, seconds)
    x7, on7 = signals.kick_loop(fs, seconds)
    sub = signals.sine(40.0, fs, seconds, -12.0)
    n = x2.shape[0]

    def splat(G):
        ys = sub * (10 ** (G / 20))[:, None]
        return {"sub_splatter_db": measure.band_energy_db(ys, fs, 100, 1000) - measure.band_energy_db(ys, fs, 20, 60),
                "sub_h3region_db": measure.band_energy_db(ys, fs, 110, 130) - measure.band_energy_db(ys, fs, 20, 60)}

    def onset(x, y, on):
        return {"onset_100_1000_in_db": measure.window_energy_db(x, fs, on, 2, 20, 100, 1000),
                "onset_100_1000_out_db": measure.window_energy_db(y, fs, on, 2, 20, 100, 1000),
                "onset_1k_5k_in_db": measure.window_energy_db(x, fs, on, 2, 20, 1000, 5000),
                "onset_1k_5k_out_db": measure.window_energy_db(y, fs, on, 2, 20, 1000, 5000)}

    for ch in chars:
        for name, x, on in (("S2", x2, on2), ("S7", x7, on7)):
            hits = signals.hits_for(x, fs, 128.0)
            p = drive.mb_realise(x, fs, drive.mb_plan(ch), hits)
            y = mb.apply(x, fs, p)
            for k in (0, 1):
                G = drive.mb_total_gain_db(p, k, 0, n, fs)
                G = G - G.max()
                res = p.budget["result"][k][0]
                r = {"signal": name, "char": ch, "stage": f"multiband band{k} M", "gr_max_db": float(-G.min()),
                     "comp_db": res["comp_db"], "limit_db": res["limit_db"], "ceiling": res["ceiling_db"] is not None}
                r.update(measure.velocity(G, fs))
                r.update(measure.kick_look(G, fs, on))
                if k == 0:
                    r.update(splat(G))
                rows.append(r)
            rows.append({"signal": name, "char": ch, "stage": "multiband output", **onset(x, y, on)})
            sp = drive.speclim_plan(y, fs, ch, signals.hits_for(y, fs, 128.0))
            if sp is not None:
                ysp = mb.apply(y, fs, sp)
                for k in (0, 1):
                    Gs = drive.mb_ceil_gain_db(sp, k, 0, n, fs)
                    r = {"signal": name, "char": ch, "stage": f"speclim band{k} M", "gr_max_db": float(-Gs.min()),
                         "cut_db": sp.budget["result"][k][0]["cut_db"]}
                    r.update(measure.velocity(Gs, fs))
                    r.update(measure.kick_look(Gs, fs, on))
                    if k == 0:
                        r.update(splat(Gs))
                    rows.append(r)
                rows.append({"signal": name, "char": ch, "stage": "speclim output (after multiband)", **onset(y, ysp, on)})
            else:
                rows.append({"signal": name, "char": ch, "stage": "speclim: none needed"})
        print(f"mbkick {ch} done", flush=True)
    save_json("mbkick.json", rows)
    cols = ["signal", "char", "stage", "gr_max_db", "comp_db", "limit_db", "cut_db", "vel_max", "vel_p999", "depth_db_med",
            "t_min_ms_med", "drop_1ms_db_max", "drop_2ms_db_max", "drop_5ms_db_max", "rec63_ms_med", "sub_splatter_db",
            "sub_h3region_db", "onset_100_1000_in_db", "onset_100_1000_out_db", "onset_1k_5k_in_db", "onset_1k_5k_out_db"]
    md = "# Multiband and speclim around kicks (S2: 40 Hz sub + kick; S7: kick only), budget-solved as planned\n\n"
    md += table([_r(r) for r in rows], cols) + "\n"
    write_md("mbkick.md", md)
    print(md)


def exp_shape(fs):
    """The sub-band gain over one 40 Hz period (aggressive, fast): detector, required gain,
    the control-rate envelope and the audio-rate gain's per-period extremes; the limiter's
    8x gain over one period."""
    x = signals.sine(40.0, fs, 6.0, -6.0)
    n = x.shape[0]
    hits = signals.hits_for(x, fs, 120.0)
    out = {}
    lines = ["# Gain shape over one 40 Hz period (S1 -6 dBFS)\n"]
    for ch in ("aggressive", "fast"):
        p = drive.mb_realise(x, fs, drive.mb_plan(ch), hits)
        b = drive.ms_bands(x, fs)[0, :, 0]
        nc = control_length(n, fs)
        det = mb.detector(b, fs, nc, CONTROL_RATE_HZ)
        thr = p.budget["result"][0][0]["threshold_db"]
        rat = p.budget["result"][0][0]["ratio"]
        knee = drive.timing(ch)["mb_knee_db"]
        with np.errstate(divide="ignore"):
            g_req = mb.knee_gr_db(20 * np.log10(det), thr, rat, knee)
        env = p.gr_db[0][0].values
        k0 = 3000
        rows = [{"t_ms": k - k0, "det_dbfs": float(20 * np.log10(det[k])), "g_req_db": float(g_req[k]), "env_db": float(env[k])}
                for k in range(k0, k0 + 26)]
        G = drive.mb_comp_gain_db(p, 0, 0, n, fs)
        seg = G[int(3.0 * fs):int(3.0 * fs) + 1200]
        out[ch] = {"threshold_db": thr, "rows": rows, "audio_min": float(seg.min()), "audio_max": float(seg.max())}
        lines.append(f"\n## multiband sub band, {ch}: threshold {thr:.2f} dBFS, knee {knee} dB; audio-rate gain over one period {seg.min():.3f} .. {seg.max():.3f} dB\n")
        lines.append(table([_r(r, 3) for r in rows], ["t_ms", "det_dbfs", "g_req_db", "env_db"]))
    xl = signals.sine(40.0, fs, 3.0, -6.0)
    for raw in (False, True):
        lp = drive.limiter_plan("aggressive", input_gain_db=8.7, clip=False, budget=False, release_raw=raw)
        r = drive.limiter_run(xl, fs, lp, keep8=False)
        g = GDB(r["g8"])
        fs8 = r["fs8"]
        a = int(2.0 * fs8)
        seg = g[a:a + int(0.025 * fs8)]
        step = int(0.001 * fs8)
        rows = [{"t_ms": i, "g8_db": float(seg[i * step])} for i in range(25)]
        out[f"limiter_rel{lp.release_ms:g}"] = {"rows": rows, "min": float(seg.min()), "max": float(seg.max())}
        lines.append(f"\n## limiter aggressive 0.5 ms / {lp.release_ms:g} ms, 3 dB over: 8x gain over one period {seg.min():.3f} .. {seg.max():.3f} dB\n")
        lines.append(", ".join(f"{rr['t_ms']}:{rr['g8_db']:.2f}" for rr in rows) + "\n")
    save_json("shape.json", out)
    write_md("shape.md", "\n".join(lines))
    print("\n".join(lines))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["h1", "h2", "h3", "h4", "h5", "h6", "h7", "fixes", "mbkick", "shape", "all"])
    ap.add_argument("--fs", type=int, default=48000)
    ap.add_argument("--chars", default=",".join(CHARS))
    ap.add_argument("--freqs", default="30,40,60")
    ap.add_argument("--seconds", type=float, default=None)
    a = ap.parse_args()
    chars = tuple(a.chars.split(","))
    freqs = [float(f) for f in a.freqs.split(",")]
    t0 = time.time()
    if a.exp in ("h1", "all"):
        exp_h1(a.fs, chars, freqs, **({"seconds": a.seconds} if a.seconds else {}))
    if a.exp in ("h2", "all"):
        exp_h2(a.fs, chars, **({"seconds": a.seconds} if a.seconds else {}))
    if a.exp in ("h3", "all"):
        exp_h3(a.fs, chars, **({"seconds": a.seconds} if a.seconds else {}))
    if a.exp in ("h4", "all"):
        exp_h4(a.fs, chars, **({"seconds": a.seconds} if a.seconds else {}))
    if a.exp in ("h5", "all"):
        exp_h5(a.fs, **({"seconds": a.seconds} if a.seconds else {}))
    if a.exp in ("h6", "all"):
        exp_h6(a.fs, chars, **({"seconds": a.seconds} if a.seconds else {}))
    if a.exp in ("h7", "all"):
        exp_h7()
    if a.exp in ("fixes", "all"):
        exp_fixes(a.fs, chars, **({"seconds": a.seconds} if a.seconds else {}))
    if a.exp in ("mbkick", "all"):
        exp_mbkick(a.fs, chars, **({"seconds": a.seconds} if a.seconds else {}))
    if a.exp in ("shape",):
        exp_shape(a.fs)
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
