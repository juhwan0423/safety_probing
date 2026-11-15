"""Thesis figure: how early can the future be predicted on ForkSafe-CF v2? (fixed split, held-out pairs)

Observation truncated to the first 24/36/47/56/65 frames (1.0-2.71 s); the steering pulse starts at t = 2.0 s and the
future targets are the same for every observation length. Lines: video only / video + planned command; horizontal
references: planned command only and the train-mean baseline. Also prints the lead time before the tip-over event.
Run from analysis_thesis/:  PYTHONPATH=.:v2 ../external/LTX-2/.venv/bin/python v2/make_fig_v2_early.py p5
"""
import json, os, sys
import numpy as np
import thesis_style as ts
from thesis_style import plt

name = sys.argv[1] if len(sys.argv) > 1 else "p5"
ts.apply()
FRAMES = [24, 36, 47, 56, 65]; T = {f: f / 24.0 for f in FRAMES}
load = lambda n: json.load(open(f"out/v2_{n}.json")) if os.path.exists(f"out/v2_{n}.json") else None
VID = {f: load(f"{name}_obs{f}") for f in FRAMES}; CMD = {f: load(f"{name}_obs{f}_cmd") for f in FRAMES}; ONLY = load(f"{name}_cmdonly")
base = VID[65]["train_mean_baseline"]
panels = [("tip_auroc", "tip-over AUROC", None, (0.7, 1.005)), ("roll_mae_deg", "future roll MAE (deg)", "roll_mae_deg", (0, None)), ("mass_mae_kg", "payload mass MAE (kg)", "mass_mae_kg", (0, None))]
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
for a, (key, title, bkey, ylim) in zip(axes, panels):
    for res, col, mk, lab in ((VID, ts.GEN, "s", "video only"), (CMD, ts.PRED, "o", "video + planned command")):
        x = [T[f] for f in FRAMES]; y = np.array([res[f][key]["point"] for f in FRAMES]); lo = np.array([res[f][key]["ci_low"] for f in FRAMES]); hi = np.array([res[f][key]["ci_high"] for f in FRAMES])
        a.fill_between(x, lo, hi, color=col, alpha=0.13, lw=0); a.plot(x, y, "-", color=col, marker=mk, ms=3.8, label=lab)
    if key != "mass_mae_kg": a.axhline(ONLY[key]["point"], color=ts.BASE, ls="--", lw=1.0, label="planned command only")
    else: a.plot([], [], color=ts.BASE, ls="--", lw=1.0, label="planned command only")
    if bkey: a.axhline(base[bkey], color=ts.INK, ls=":", lw=0.9, label="train-mean baseline")
    else: a.plot([], [], color=ts.INK, ls=":", lw=0.9, label="train-mean baseline")
    a.axvline(2.0, color=ts.TIP, lw=0.8, alpha=0.7); a.set_xlabel("observed video length (s)"); a.set_title(title, fontsize=9); a.set_xticks([1.0, 1.5, 2.0, 2.5]); a.set_ylim(*ylim)
axes[0].text(2.03, 0.705, "steering\nstarts", fontsize=7.5, color=ts.TIP, va="bottom")
handles, labels = axes[0].get_legend_handles_labels()
fig.tight_layout(w_pad=1.0, rect=(0, 0, 1, 0.9)); fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8.5, handlelength=1.6, columnspacing=1.3, bbox_to_anchor=(0.5, 1.0))
ts.save(fig, f"fig_v2_early_prediction_{name}")
rows = [json.loads(l) for l in open(f"../artifacts/v2/{name}/manifests/canonical_runs.jsonl")]
ev = np.array([r["label_evidence"]["event_time_s"] for r in rows if r["safety_outcome"] == "observed_tip"], float)      # sustained support loss
for f in FRAMES:
    lead = ev - T[f]
    print(f"obs {T[f]:.2f} s: video AUROC {VID[f]['tip_auroc']['point']:.3f} roll {VID[f]['roll_mae_deg']['point']:.1f} | +cmd AUROC {CMD[f]['tip_auroc']['point']:.3f} roll {CMD[f]['roll_mae_deg']['point']:.1f} | lead before sustained support loss: median {np.median(lead):.2f} s, 5% {np.quantile(lead, .05):.2f} s (n={len(ev)})")
print("command only: AUROC %.3f roll %.2f" % (ONLY["tip_auroc"]["point"], ONLY["roll_mae_deg"]["point"]))
