"""Thesis figure: what fixes the outcome, and how much of it the video recovers (ForkSafe-CF v2, pair-grouped 5-fold CV).

Grey bars: a small network that is GIVEN the planned command and, step by step, the true payload mass and the true
lateral CoM offset (payload_value_v2.py; no video).  Blue/orange bars: readouts of the observed video with and without
the planned command (psq_v2_cv.py).  Panels: future roll error, tip-over AUROC, accuracy on outcome-discordant pairs.
Run from analysis_thesis/:  PYTHONPATH=.:v2 ../external/LTX-2/.venv/bin/python v2/make_fig_v2_information.py p5
"""
import json, sys
import numpy as np
import thesis_style as ts
from thesis_style import plt

name = sys.argv[1] if len(sys.argv) > 1 else "p5"
ts.apply()
PV = json.load(open(f"out/v2_payload_value_{name}_cv.json"))
VID, VC = json.load(open(f"out/v2cv_{name}.json")), json.load(open(f"out/v2cv_{name}_cmd.json"))
rows = [("cmd", PV["command"], ts.BASE), ("cmd\n+ mass", PV["command+mass"], ts.BASE), ("cmd\n+ mass\n+ CoM", PV["command+mass+com"], ts.BASE),
        ("video", VID, ts.GEN), ("video\n+ cmd", VC, ts.PRED)]
panels = [("roll_mae_deg", "future roll MAE (deg)", (0, 8.6)), ("tip_auroc", "tip-over AUROC", (0.9, 1.008)),
          ("outcome_flip", "pairs with the same command and\ndifferent outcomes: accuracy", (0.4, 1.06))]
fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.0))
for a, (key, title, ylim) in zip(axes, panels):
    x = np.array([0, 1, 2, 3.4, 4.4])
    v = np.array([r[key]["point"] for _, r, _ in rows]); lo = np.array([r[key]["ci_low"] for _, r, _ in rows]); hi = np.array([r[key]["ci_high"] for _, r, _ in rows])
    a.set_ylim(*ylim)
    a.bar(x, v, 0.72, color=[c for _, _, c in rows], yerr=[v - lo, hi - v], error_kw=dict(lw=0.7, capsize=1.8, ecolor=ts.INK))
    for xi, vi, hi_i in zip(x, v, hi):
        a.text(xi, hi_i + 0.015 * (ylim[1] - ylim[0]), f"{vi:.2f}" if key != "roll_mae_deg" else f"{vi:.1f}", ha="center", va="bottom", fontsize=8)
    a.set_xticks(x); a.set_xticklabels([n for n, _, _ in rows], fontsize=8); a.set_title(title, fontsize=9)
    if key == "outcome_flip": a.axhline(0.5, color=ts.INK, ls="--", lw=0.8)
    a.axvline(2.7, color=ts.BASE, lw=0.6, ls=":")
from matplotlib.patches import Patch
handles = [Patch(color=ts.BASE, label="true values given to the predictor (no video)"), Patch(color=ts.GEN, label="read from the video"), Patch(color=ts.PRED, label="video + planned command")]
fig.tight_layout(w_pad=1.0, rect=(0, 0, 1, 0.91)); fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, 1.0), handlelength=1.2, columnspacing=1.4)
ts.save(fig, f"fig_v2_information_{name}")
print("pairs with different outcomes:", VID["n_outcome_flip_pairs"], "| train-mean roll baseline", round(VID["train_mean_baseline"]["roll_mae_deg"], 2))
