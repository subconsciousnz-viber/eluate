"""Plain-language process log of a finished master: out/process-log.md and .html.

Contract:
- `write(out) -> Path`: reads what `master` saved in out/ (report.json, solve.json,
  source.json, plan.json with plan.npz, limiter_gr.npz; run.json when present) and
  writes out/process-log.md and out/process-log.html (the same content). No audio is
  read and nothing is re-rendered.
- Content, in order: the settings used; the result in a few sentences; every stage of
  the chain in signal order with what it measured, what it decided and why, and what held
  it back; the renders the solver made; the sections, with each section's level against
  the loudest section before and after and the average gain reduction each dynamics
  stage applied in it, so a change in contrast between sections can be traced to the
  stage that caused it, and (solve.json made since 2026-10-21) the measured change after
  each stage of the chain (solver section_change_by_stage); what Preserve mix does in
  this build and what its check would have read on this render; the noise floor before and after (when the report has it);
  the flags; a glossary.
- Average gain reduction per section: the mean, over the section, of each stage's gain
  curve in dB (multiband compressor and band limiter averaged over the bands' mid and
  side channels, the glue's wet-path curve times its mix, the frequency-dependent
  limiter averaged like the multiband, and the clipper with the limiter from
  limiter_gr.npz). A mean of dB is not a level change, so the "explained by gain
  reduction" column is an estimate; what it does not explain comes from the EQ,
  saturation and the sections' different spectra.
- Section thresholds (S9): a section is called out as having lost contrast when it rose
  more than evaluate.FLAG_SECTION_LU (3 LU) against the loudest section, the report's
  and ui-data's rule; sections more than FLAG_SECTION_IGNORE_LU below it are not judged.
  The Preserve paragraph's pass / fail is a different number on purpose: it states what
  Preserve's own constraint (targets.section_tol_lu, 1 LU either way) would read, since
  Preserve keeps the mix's balance in both directions.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

import numpy as np

from eluate import evaluate, noise
from eluate.plan import INPUTEQ_HS_MID_MAX_DB as HS_MID_MAX
from eluate.plan import INPUTEQ_HS_SIDE_MAX_DB as HS_SIDE_MAX
from eluate.plan import INPUTEQ_MAX_DB as EQ_MAX
from eluate.plan import Plan


def _f(v, d=1, sign=False):
    if v is None or not math.isfinite(float(v)):
        return "n/a"
    s = f"{float(v):+.{d}f}" if sign else f"{float(v):.{d}f}"
    return s.replace("-", "\u2212")


def _mean_gr(env, i0, i1) -> float:
    """Mean gain reduction (positive dB) of a gain curve in dB over control samples [i0, i1)."""
    v = np.asarray(env.values)
    if v.size == 1:
        return float(max(0.0, -v[0]))
    seg = v[max(0, i0):max(i0 + 1, min(i1, v.size))]
    return float(np.mean(np.maximum(0.0, -seg))) if seg.size else 0.0


def _bands_mean(envs, t0, t1) -> float:
    vals = [_mean_gr(e, int(t0 * e.rate_hz), int(t1 * e.rate_hz)) for row in envs or [] for e in row]
    return float(np.mean(vals)) if vals else 0.0


def _section_gr(plan: Plan, lim_t, lim_gr, t0, t1) -> dict:
    out = {}
    if plan.multiband is not None:
        out["multiband"] = _bands_mean(plan.multiband.gr_db, t0, t1)
        out["band limiter"] = _bands_mean(plan.multiband.ceiling_gr_db, t0, t1)
    if plan.glue is not None:
        g = plan.glue.gain_db
        out["glue"] = _mean_gr(g, int(t0 * g.rate_hz), int(t1 * g.rate_hz)) * float(plan.glue.mix)
    if plan.speclim is not None and plan.speclim.ceiling_gr_db is not None:
        out["frequency limiter"] = _bands_mean(plan.speclim.ceiling_gr_db, t0, t1)
    if lim_t is not None and lim_t.size:
        m = (lim_t >= t0) & (lim_t < t1)
        out["clipper + limiter"] = float(np.mean(-lim_gr[m])) if np.any(m) else 0.0
    return out


# ------------------------------------------------------------------ blocks

class Doc:
    def __init__(self):
        self.blocks = []

    def h(self, level, text):
        self.blocks.append(("h", level, text))

    def p(self, text):
        self.blocks.append(("p", text))

    def ul(self, items):
        items = [i for i in items if i]
        if items:
            self.blocks.append(("ul", items))

    def table(self, head, rows):
        if rows:
            self.blocks.append(("table", head, rows))

    def md(self) -> str:
        L = []
        for b in self.blocks:
            if b[0] == "h":
                L += ["#" * b[1] + " " + b[2], ""]
            elif b[0] == "p":
                L += [b[1], ""]
            elif b[0] == "ul":
                L += [f"- {i}" for i in b[1]] + [""]
            else:
                L += ["| " + " | ".join(b[1]) + " |", "|" + "---|" * len(b[1])]
                L += ["| " + " | ".join(str(c) for c in r) + " |" for r in b[2]] + [""]
        return "\n".join(L)

    def html(self, title) -> str:
        e = html.escape
        B = []
        for b in self.blocks:
            if b[0] == "h":
                B.append(f"<h{b[1]}>{e(b[2])}</h{b[1]}>")
            elif b[0] == "p":
                B.append(f"<p>{e(b[1])}</p>")
            elif b[0] == "ul":
                B.append("<ul>" + "".join(f"<li>{e(i)}</li>" for i in b[1]) + "</ul>")
            else:
                head = "".join(f"<th>{e(c)}</th>" for c in b[1])
                rows = "".join("<tr>" + "".join(f"<td>{e(str(c))}</td>" for c in r) + "</tr>"
                               for r in b[2])
                B.append(f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>")
        css = ("body{background:#23303a;color:#dce6ea;font:15px/1.55 'Segoe UI',system-ui,sans-serif;"
               "max-width:58rem;margin:2rem auto;padding:0 1.5rem;font-variant-numeric:tabular-nums}"
               "h1{font-size:1.6rem;margin:0 0 1rem}h2{font-size:1.2rem;margin:2.2rem 0 .6rem;"
               "border-top:1px solid #3a4a56;padding-top:1.2rem}h3{font-size:1rem;margin:1.4rem 0 .4rem}"
               "p,li{max-width:46rem}table{border-collapse:collapse;margin:.6rem 0 1rem}"
               "th,td{padding:.25rem .8rem .25rem 0;text-align:left;border-bottom:1px solid #3a4a56}"
               "th{color:#9aabb6;font-weight:500}")
        return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{e(title)}</title>"
                f"<style>{css}</style></head><body>{''.join(B)}</body></html>\n")


# ------------------------------------------------------------------ content

MOVES = {"planned": "the planned settings",
         "true-peak ceiling": "the limiter ceiling lowered by the true-peak overshoot",
         "tone rotation": "the input EQ re-fitted to correct the remaining slope error",
         "loudness": "the limiter gain re-solved for the loudness target"}


def _settings(d: Doc, rep, src, run):
    c = (rep.get("planner") or {}).get("controls") or (run or {}).get("controls") or {}
    ch = (rep.get("planner") or {}).get("character") or {}
    lo = rep.get("loudness") or {}
    d.h(2, "Settings used")
    pr = (rep.get("planner") or {}).get("profile") or {}
    d.ul([
        f"Profile: {pr.get('name', 'electronic')}" + ("" if pr.get("calibrated", True) else
            " (phase 1: the live profile defaults are uncalibrated; they are the architect's "
            "numbers until live reference masters are measured)") + ".",
        f"Loudness target: {_f(lo.get('target_rms_dbfs'))} dB RMS on the 300 ms meter (AES17, "
        "read over the loudest passage).",
        f"Tilt: {_f(c.get('tilt_db', 0.0), 1, True)} dB/oct around the -4.5 dB/oct reference, so the "
        f"target slope is {_f(rep['targets']['slope_db_oct'], 2)} dB/oct.",
        f"Preserve mix: {'on' if c.get('preserve_mix') else 'off'}.",
        f"Tempo: {_f(ch.get('bpm'), 1)} BPM ({ch.get('bpm_source', 'detected')}); timing preset "
        f"{ch.get('preset', 'n/a')} ({ch.get('reason', '')}).",
        f"Delivery: {'streaming (-1 dBTP, 16-bit with dither)' if c.get('streaming') else 'master (-0.3 dBTP, 24-bit)'}.",
        f"Source: {src.get('path')}.",
    ])


def _result(d: Doc, rep, rep_tilt):
    lo, tg = rep["loudness"], rep["targets"]
    reached, target = rep["after"]["rms_meter_dbfs"], tg["rms_dbfs"]
    lim = next((w for w in rep.get("warnings") or [] if w.get("stage") == "limiter"), None)
    d.h(2, "Result")
    s = (f"Loudness reached {_f(reached)} dB against the {_f(target)} dB target "
         f"(premaster {_f(lo.get('before_rms_dbfs'))} dB). ")
    hist = (rep.get("solver") or {}).get("history") or []
    flat = any(h.get("reason") == "section contrast" for h in hist)
    if target - reached > tg["rms_tol_db"]:
        if lim is not None:
            s += (f"It is {_f(target - reached)} dB short because the final limiter was held at its "
                  f"{'average' if lim.get('bound') == 'average' else 'outlier'} budget: going "
                  "louder would have meant more limiting on the typical loud hit than the budget allows.")
        elif flat:
            s += (f"It is {_f(target - reached)} dB short because a louder render flattened the "
                  f"contrast between sections past {_f(evaluate.FLAG_SECTION_LU)} LU and was not kept.")
        else:
            s += f"It is {_f(target - reached)} dB short."
    else:
        s += "That is within the tolerance."
    d.p(s)
    sl, st = rep_tilt["after"]["stereo"], tg["slope_db_oct"]
    s = (f"Tone: the spectrum's slope went from {_f(rep_tilt['before']['stereo'], 2)} to {_f(sl, 2)} dB/oct "
         f"against a {_f(st, 2)} dB/oct target. ")
    if abs(sl - st) > tg["slope_tol"]:
        s += (f"It is {_f(abs(sl - st), 2)} dB/oct from the target ({_f(abs(sl - st) - tg['slope_tol'], 2)} "
              "beyond the tolerance); see the input EQ below for which sections reached their limits.")
    else:
        s += "That is within the tolerance."
    d.p(s)
    crest = (f"crest {_f(rep['before']['crest_db'])} to {_f(rep['after']['crest_db'])} dB, "
             if rep["after"].get("crest_db") is not None else "")
    d.p(f"Crest (peak against the loud-passage RMS) and loudness range: {crest}PLR "
        f"{_f(rep['before']['plr_db'])} to {_f(rep['after']['plr_db'])} dB, loudness range {_f(rep['before']['lra_lu'])} to "
        f"{_f(rep['after']['lra_lu'])} LU. True peak {_f(rep['after']['true_peak_dbtp'], 2)} dBTP.")


def _chain(d: Doc, rep):
    ch, pl, bud = rep["chain"], rep["planner"], (rep.get("budget") or {}).get("stages") or {}
    d.h(2, "What each stage did, in signal order")

    d.h(3, "1. Input level")
    i = ch.get("input") or {}
    d.p(f"The premaster's loudest section measured {_f(i.get('measured_dbfs'))} dBFS (1 s RMS). It was "
        f"moved by {_f(i.get('gain_db'), 1, True)} dB to the {_f(i.get('r_ref_dbfs'))} dBFS reference, "
        "so every rule after this works on the same scale whatever the premaster's level.")

    d.h(3, "2. Input EQ")
    eq = pl.get("inputeq") or {}
    if ch.get("inputeq"):
        d.p(f"Static EQ fitted to move the spectrum towards the target slope. Measured slope "
            f"{_f(eq.get('measured_slope'), 2)} dB/oct, target {_f(eq.get('target_slope'), 2)}. "
            f"Limits: each section within {_f(EQ_MAX)} dB, the presence and air shelves up to "
            f"+{_f(HS_MID_MAX)} dB (mid) and +{_f(HS_SIDE_MAX)} dB (side), each boost no more than "
            "the deficit measured in its own range and within the noise headroom. "
            "Sections used:")
        d.ul(list(ch["inputeq"].get("text") or []))
        at = eq.get("at_bound")
        if at is None:                       # a record from before the presence / air split
            sec = (ch["inputeq"].get("sections") or {})
            at = [k.replace("_", " ") for k, mx in (("high_shelf", HS_MID_MAX),
                                                    ("high_shelf_side", HS_SIDE_MAX))
                  if (sec.get(k) or {}).get("gain_db", 0) >= mx - 1e-6]
        if at:
            d.p("At their limits: " + "; ".join(at)
                + ". The EQ could not move the tone further within its range.")
        head = (eq.get("top_caps") or {}).get("noise_headroom_db")
        if head is not None:
            d.p(f"Noise headroom for the top shelves: {_f(head)} dB (the premaster's quiet-passage "
                "floor plus the master's gain, against the noise flag).")
        if eq.get("low_mid_shape"):
            d.p(f"Low-mid excess above its neighbours: {_f(eq.get('low_mid_excess_db'))} dB, "
                f"handled by the {eq.get('low_mid_shape')}.")
    else:
        d.p("Off (Preserve mix keeps the mix's tonal balance).")

    d.h(3, "3. Resonance notches")
    rows = [[_f(n["f0_hz"], 0) + " Hz", n.get("channel"), "yes" if n.get("rendered") else "no",
             n.get("reason") or ""] for n in rep.get("notches") or []]
    if rows:
        d.p("Persistent narrow peaks found in the premaster, and whether each was cut:")
        d.table(["frequency", "channel", "cut", "why"], rows)
    else:
        d.p("No persistent resonances found.")

    d.h(3, "4. Side low end")
    s = ch.get("sidestrip") or {}
    d.p(f"The side channel was high-passed below {_f(s.get('fc_hz'), 0)} Hz, so the low end is mono. "
        "The mid is not touched.")

    mb = ch.get("multiband") or {}
    d.h(3, "5. Multiband compression (mid and side, per band)")
    xo = pl.get("crossovers") or {}
    d.p(f"Crossovers {', '.join(_f(f, 0) for f in mb.get('crossovers_hz') or [])} Hz. {xo.get('reason', '')}. "
        "Each band's ratio comes from its crest (a spikier band gets a gentler ratio, 1.3:1 to 2:1). "
        "Each threshold is solved so the typical loud hit gets the band's budget of gain reduction "
        "as a 300 ms meter reads it (2.5 dB, 3 dB where peaks got past), and makeup gives it back, "
        "so the band's level is unchanged on the typical hit and only its peaks are reduced. A fast "
        "band limiter catches peaks the compressor misses, up to 3 dB.")
    rows = []
    names = ["band 1", "band 2", "band 3", "band 4", "band 5"]
    for k, row in enumerate(((bud.get("multiband") or {}).get("bands")) or []):
        for c, e in zip(("mid", "side"), row):
            st = e.get("status") or ""
            if e.get("gated"):
                st = "not compressed (near silent)"
            elif st == "over":
                st = f"largest above its {_f(e.get('comp_cap_db'))} dB reference"
            rows.append([names[k] if k < 5 else f"band {k + 1}", c, _f(e.get("ratio"), 2),
                         _f(e.get("comp_db")), _f(e.get("comp_outlier_db")), _f(e.get("limit_outlier_db")),
                         _f(e.get("makeup_db"), 1, True), st])
    d.table(["band", "channel", "ratio", "typical hit GR dB", "largest GR dB", "band limiter max dB",
             "makeup dB", "status"], rows)

    d.h(3, "6. Saturation")
    sat = (pl.get("saturation") or {}).get("bands") or []
    items = []
    for b in sat:
        dm, ds, dd = b.get("drive_m_db", 0), b.get("drive_s_db", 0), b.get("deficit_db")
        if (b.get("rule") or "").startswith("sub"):
            why = "never drives the sub"
        elif dd is None:
            why = "no deficit measurable in this band"
        else:
            why = f"sits {_f(abs(dd))} dB {'below' if dd > 0 else 'above'} the target line"
        items.append(f"band {b['band'] + 1}: drive {_f(dm)} dB mid / {_f(ds)} dB side ({why})")
    if items:
        d.p("Gentle saturation adds harmonics where a band is duller than the target slope (drive = the "
            "band's deficit, at most 2 dB); bands on or above the line get none.")
        d.ul(items)
    else:
        d.p("Off.")

    g, gb = ch.get("glue") or {}, bud.get("glue") or {}
    d.h(3, "7. Glue compressor")
    if g:
        d.p(f"A fast FET-style bus compressor ({_f(g.get('ratio'))}:1, attack {_f(g.get('attack_ms'))} ms, "
            f"release {_f(g.get('release_ms'), 0)} ms, {g.get('mode')}). Threshold solved for "
            f"{_f(gb.get('net_target_db'))} dB of gain reduction on the typical loud hit; it took "
            f"{_f(gb.get('net_db'))} dB, at most {_f(gb.get('net_outlier_db'))} dB.")

    sp = ch.get("speclim") or {}
    d.h(3, "8. Frequency-dependent limiter")
    if sp.get("bands"):
        items = []
        for k, row in enumerate(sp["bands"]):
            for c, e in zip(("mid", "side"), row):
                if e.get("gated") and e.get("peak_db") is not None:
                    items.append(f"band {k + 1} {c}: gated (near silent), left alone")
                elif (e.get("cut_db") or 0) > 0.05:
                    items.append(f"band {k + 1} {c}: up to {_f(e['cut_db'])} dB (typical hit "
                                 f"{_f(e.get('typical_db'))} dBFS, loudest peak {_f(e.get('peak_db'))})")
        from eluate.evaluate import allow_text
        d.p(f"Peaks more than {allow_text(sp.get('allow_db'))} above a band's typical loud hit are limited in "
            f"that band only, by at most {_f(sp.get('max_cut_db'))} dB, so one band's stray peaks do not "
            "pull the whole mix down at the final limiter.")
        d.ul(items or ["No band had peaks that far above its typical hit."])
    else:
        d.p("Off.")

    cl, cb = ch.get("clipper") or {}, bud.get("clipper") or {}
    d.h(3, "9. Clipper")
    where = (f"the {cl.get('mode', 'whole signal')}" if not cb.get("split_hz") else
             f"the signal above {_f(cb['split_hz'], 0)} Hz (the sub passes to the limiter untouched)")
    d.p(f"Soft clipping on {where}, solved for {_f(cb.get('target_db'))} dB "
        f"off the typical loud hit's peak (it took {_f(cb.get('planned_db'))} dB); no sample loses more "
        f"than {_f(cb.get('outlier_cap_db'))} dB (largest: {_f(cb.get('outlier_db'))} dB).")

    lm, lb = ch.get("limiter") or {}, bud.get("limiter") or {}
    d.h(3, "10. Final limiter")
    s = (f"To reach the loudness target the limiter was asked for {_f(lm.get('requested_gain_db'))} dB of "
         f"gain and given {_f(lm.get('drive_db'))} dB, into a {_f(lm.get('ceiling_dbtp'), 2)} dBTP ceiling. "
         f"Its budget is {_f(lb.get('budget_db'))} dB of gain reduction on the typical loud hit and "
         f"{_f(lb.get('outlier_cap_db'))} dB on the loudest moment; it used {_f(lb.get('planned_db'))} and "
         f"{_f(lb.get('outlier_db'))} dB. ")
    if (lm.get("requested_gain_db") or 0) - (lm.get("drive_db") or 0) > 0.05:
        s += ("The gain was held back by the " + ("typical-hit budget" if lb.get("bound") == "average"
                                                    else "loudest-moment cap") + ".")
    d.p(s)


def _renders(d: Doc, solve):
    d.h(2, "Renders")
    d.p("The engine renders, measures and adjusts. Each render, and whether it was kept:")
    items, last = [], None
    for h in solve.get("history") or []:
        if "move" in h and h.get("move"):
            last = (f"Render {h['render']}: {MOVES.get(h['move'], h['move'])}. Loudness "
                    f"{_f(h.get('rms_dbfs'))} dB, slope {_f(h.get('slope'), 2)} dB/oct, true peak "
                    f"{_f(h.get('true_peak'), 2)} dBTP")
            items.append(last)
        elif "kept" in h:
            items[-1] = items[-1] + ". Kept."
        elif "reverted" in h and h.get("reason") == "section contrast":
            items[-1] = items[-1] + (
                f". Not kept: it flattened the contrast between sections (largest rise "
                f"{_f(h.get('contrast_loss_before_lu'))} to {_f(h.get('contrast_loss_after_lu'))} LU, "
                f"past {_f(evaluate.FLAG_SECTION_LU)} LU).")
        elif "reverted" in h:
            items[-1] = items[-1] + ". Not kept: it made the overall result worse."
    d.ul(items)


def _sections(d: Doc, rep, plan, lim_t, lim_gr):
    secs = rep.get("sections") or []
    d.h(2, "Sections and the contrast between them")
    if not secs or plan is None:
        d.p("No sections were found.")
        return None
    measured = [s for s in secs if s["before_lufs"] is not None and s["after_lufs"] is not None]
    if not measured:
        d.p("No section has a measurable loudness.")
        return None
    ref = max(measured, key=lambda s: s["before_lufs"])
    d.p(f"Levels are LUFS relative to the loudest section of the premaster ({ref['label']} at "
        f"{_f(ref['start'], 0)} s). \"Change\" is how far each section moved against that section: "
        "positive means it now sits closer to it (less contrast), negative means further away.")
    rows, stage_rows, changes, worst = [], [], [], 0.0
    gr_ref = _section_gr(plan, lim_t, lim_gr, ref["start"], ref["end"])
    stages = list(gr_ref)
    for s in secs:
        live = s in measured and s["before_lufs"] > ref["before_lufs"] - evaluate.FLAG_SECTION_IGNORE_LU
        bef = None if s["before_lufs"] is None else s["before_lufs"] - ref["before_lufs"]
        aft = None if s["after_lufs"] is None else s["after_lufs"] - ref["after_lufs"]
        change = aft - bef if live else None
        gr = _section_gr(plan, lim_t, lim_gr, s["start"], s["end"])
        tot = sum(gr[k] for k in stages)
        tot_ref = sum(gr_ref[k] for k in stages)
        explained = tot_ref - tot
        when = f"{_f(s['start'], 0)} to {_f(s['end'], 0)} s"
        if live:
            worst = max(worst, abs(change))
            changes.append((s, change))
            rows.append([s["label"], when, _f(bef, 1, True), _f(aft, 1, True), _f(change, 1, True),
                         _f(explained, 1, True)])
        else:
            rows.append([s["label"], when, _f(bef, 1, True), _f(aft, 1, True), "not judged", ""])
        stage_rows.append([s["label"], when] + [_f(gr[k]) for k in stages] + [_f(tot)])
    d.table(["section", "time", "premaster", "master", "change", "explained by gain reduction"], rows)
    d.p("\"Explained by gain reduction\" is the difference in total average gain reduction between the "
        "loudest section and this one: a section that was compressed and limited less than the loudest "
        "section rises against it by about that much. The rest of the change comes from the EQ, "
        "saturation and the sections' different spectra.")
    d.h(3, "Average gain reduction per section, dB")
    d.table(["section", "time"] + stages + ["total"], stage_rows)
    moved = [(s, c) for s, c in changes if c > evaluate.FLAG_SECTION_LU]
    if moved and stages:
        biggest = max(stages, key=lambda k: gr_ref[k])
        s, c = max(moved, key=lambda sc: sc[1])
        gr = _section_gr(plan, lim_t, lim_gr, s["start"], s["end"])
        d.p(f"{len(moved)} section(s) rose more than {_f(evaluate.FLAG_SECTION_LU)} LU against the loudest "
            f"section. The loudest section got {_f(sum(gr_ref[k] for k in stages))} dB of gain reduction on "
            f"average, most of it from the {biggest} ({_f(gr_ref[biggest])} dB); the {s['label']} at "
            f"{_f(s['start'], 0)} s, which rose most ({_f(c, 1, True)} LU), got "
            f"{_f(sum(gr[k] for k in stages))} dB. Stages that work in proportion to level take more from "
            "loud sections than quiet ones, which is what narrows the contrast.")
    _stage_contrast(d, rep, secs, ref)
    return worst


def _stage_contrast(d: Doc, rep, secs, ref):
    """The measured change of each section against the loudest after every stage (solver
    section_change_by_stage), cumulative from the premaster, and each stage's own share."""
    by = ((rep.get("solver") or {}).get("reported") or {}).get("section_change_by_stage")
    if not by:
        return
    d.h(3, "Contrast change after each stage, LU")
    d.p("Measured on the audio after each stage: each section's level against the loudest section, "
        "minus the premaster's (positive = the section rose toward the loudest, less contrast). "
        "Each column includes every stage before it; the last row is each stage's own share for the "
        "section that rose most.")
    names = [b["stage"] for b in by]
    rows, judged = [], []
    for i, s in enumerate(secs):
        vals = [b["change_lu"][i] if i < len(b["change_lu"]) else None for b in by]
        if all(v is None for v in vals) or s is ref:
            continue
        judged.append((s, vals))
        rows.append([s["label"], f"{_f(s['start'], 0)} s"] + [_f(v, 1, True) for v in vals])
    if not rows:
        return
    last = [v for _, v in judged if v[-1] is not None]
    if last:
        s, vals = max(judged, key=lambda sv: -math.inf if sv[1][-1] is None else sv[1][-1])
        prev, share = 0.0, []
        for v in vals:
            share.append(None if v is None else v - prev)
            prev = prev if v is None else v
        rows.append([f"share ({s['label']}, {_f(s['start'], 0)} s)", ""] + [_f(v, 1, True) for v in share])
    d.table(["section", "start"] + names, rows)


def _preserve(d: Doc, rep, worst):
    c = (rep.get("planner") or {}).get("controls") or {}
    tol = (rep.get("targets") or {}).get("section_tol_lu", 1.0)
    d.h(2, "Preserve mix: what it does in this build")
    d.ul([
        "Turns off the input EQ and the saturation, so the mix's tonal balance is kept (the side makeup "
        "of the multiband stays).",
        f"Replaces the slope target with a check that the spectrum's shape stays within "
        f"{_f(rep['targets'].get('ltas_tol_db', 1.0))} dB of the premaster's in every third-octave band.",
        f"Adds a check that each section's level against the loudest section stays within "
        f"{_f(tol)} LU of the premaster's, either way.",
        "Both checks are measured and reported as misses, and both count in the score the engine "
        "uses to keep or undo a render. With Preserve on or off, a louder render that flattens the "
        f"contrast between sections past {_f(evaluate.FLAG_SECTION_LU)} LU is undone once the loudness "
        "is inside the accepted range. The dynamics stages are planned the same way with Preserve on "
        "or off.",
    ])
    if worst is not None:
        d.p(f"On this render (Preserve {'on' if c.get('preserve_mix') else 'off'}) the largest section move "
            f"was {_f(worst)} LU, so the section check would "
            f"{'pass' if worst <= tol else 'fail'} (limit {_f(tol)} LU).")


def _noise(d: Doc, rep):
    nz = rep.get("noise")
    if not nz:
        return
    d.h(2, "Noise floor")
    spans = nz.get("quiet_spans") or []
    if not spans:
        d.p("The track has no passages quiet enough to measure a noise floor in (more than "
            f"{noise.QUIET_BELOW_DB:g} dB below its loud level); in the rest of the track the music "
            "masks it.")
        return
    where = ", ".join(f"{_f(a, 1)} to {_f(b, 1)} s" for a, b in spans[:8]) + \
        (f" and {len(spans) - 8} more" if len(spans) > 8 else "")
    d.p(f"The background in the quiet passages (more than {noise.QUIET_BELOW_DB:g} dB below the "
        f"track's loud level), from {noise.FLOOR_BANDS_HZ[0] / 1000:g} kHz up, weighted with the BS.468 noise curve, which counts the high register (hiss) most. "
        f"Measured over {_f(nz.get('quiet_s'), 1)} s: {where}. Premaster {_f(nz['before_db'])} dB, "
        f"master {_f(nz['after_db'])} dB; the threshold is {_f(nz['threshold_db'])} dB. Loudest band "
        f"after the master: around {_f(nz.get('loudest_band_hz'), 0)} Hz. A quiet musical tail (a "
        "reverb or a fade) reads the same as noise here, so listen to those passages to tell which.")
    st = nz.get("status")
    d.p({"exposed": "The master lifted the floor past the threshold: noise that sat below it in the "
                    "premaster is now above it. Noise reduction is planned as an opt-in option; it is "
                    "not in this build.",
         "present": "The floor is above the threshold in the premaster already, and the master keeps "
                    "it there or raises it.",
         None: "Below the threshold after the master; nothing to do."}[st])


def _flags(d: Doc, rep):
    d.h(2, "Warnings and flags")
    items = [evaluate.warning_text(w, rep) for w in rep.get("warnings") or []]
    items += rep.get("flags") or []
    d.ul(items or ["None."])


def _glossary(d: Doc):
    d.h(2, "Words used here")
    d.ul([
        "Meter reading: RMS over 300 ms, AES17 (a full-scale sine reads 0 dB), read at the loudest "
        "passage (the 95th percentile of the readings). The same as the Ozone RMS meter at 300 ms.",
        "Typical loud hit: the median, over the beats of the loud section, of each beat's largest "
        "gain reduction. Budgets are set on this, not on the single largest moment.",
        "Largest / loudest moment: the single biggest reading over the whole track.",
        "Mid and side: the sum (L+R) and the difference (L-R); the engine processes them separately.",
        "LU: loudness units, the same size as dB, on the EBU R128 loudness scale.",
        "Crest / PLR: how far peaks sit above the loud-passage loudness. Lower means denser.",
        "Slope: the straight line fitted to the spectrum from 100 Hz to 10 kHz, in dB per octave.",
    ])


def write(out) -> Path:
    out = Path(out)
    rep = json.loads((out / "report.json").read_text(encoding="utf-8"))
    src = json.loads((out / "source.json").read_text(encoding="utf-8"))
    solve = json.loads((out / "solve.json").read_text(encoding="utf-8")) \
        if (out / "solve.json").exists() else {}
    run = json.loads((out / "run.json").read_text(encoding="utf-8")) if (out / "run.json").exists() else None
    plan = Plan.load(out / "plan.json") if (out / "plan.json").exists() else None
    lim_t = lim_gr = None
    if (out / "limiter_gr.npz").exists():
        z = np.load(out / "limiter_gr.npz")
        lim_t, lim_gr = np.asarray(z["t"]), np.asarray(z["gr_db"])

    d = Doc()
    title = f"Process log: {Path(src['path']).stem}"
    d.h(1, title)
    d.p("What the engine measured, decided and did on this render, in the order it happened. "
        "report.md has the full numbers.")
    _settings(d, rep, src, run)
    _result(d, rep, rep["tilt"])
    _chain(d, rep)
    _renders(d, solve)
    worst = _sections(d, rep, plan, lim_t, lim_gr)
    _preserve(d, rep, worst)
    _noise(d, rep)
    _flags(d, rep)
    _glossary(d)

    md = out / "process-log.md"
    md.write_text(d.md(), encoding="utf-8", newline="\n")
    (out / "process-log.html").write_text(d.html(title), encoding="utf-8", newline="\n")
    return md


if __name__ == "__main__":
    import sys
    print(write(sys.argv[1]))
