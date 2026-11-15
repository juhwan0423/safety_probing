"""Thesis figure: what fixes the outcome, and how much of it the video recovers (ForkSafe-CF v2, pair-grouped 5-fold CV).

Grey bars: a small network that is GIVEN the planned command and, step by step, the true payload mass and the true
lateral CoM offset (payload_value_v2.py; no video).  Orange: the video readout alone, end to end (psq_v2_cv.py).
Teal: the two-stage predictor, planned command + the payload mass read from the video, with the mass taken from the
first 1.96 s (before the steering starts) or from the full 2.71 s observation (plugin_mass_v2.py).
Panels: future roll error, tip-over AUROC, and accuracy on pairs that share the command but end differently, split by
what differs within the pair (payload mass / lateral CoM offset).
Run from analysis_thesis/:  PYTHONPATH=.:v2 ../external/LTX-2/.venv/bin/python v2/make_fig_v2_information.py p5
"""
import json, sys
import numpy as np
import thesis_style as ts
from thesis_style import plt
from matplotlib.patches import Patch

name = sys.argv[1] if len(sys.argv) > 1 else "p5"
ts.apply()
PV = json.load(open(f"out/v2_payload_value_{name}_cv.json")); VID = json.load(open(f"out/v2cv_{name}.json")); PL = json.load(open(f"out/v2_plugin_mass_{name}.json"))
EARLY = "#5FB3A8"
rows = [("cmd", PV["command"], ts.BASE), ("cmd\n+mass", PV["command+mass"], ts.BASE), ("cmd\n+mass\n+CoM", PV["command+mass+com"], ts.BASE),
        ("video\nonly", VID, ts.GEN), ("video\nmass\n1.96 s", PL[f"video mass ({name}_obs47)"], EARLY), ("video\nmass\n2.71 s", PL[f"video mass ({name})"], ts.REAL)]
x = np.array([0, 1, 2, 3.4, 4.4, 5.4]); cols = [c for _, _, c in rows]
fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.25), gridspec_kw=dict(width_ratios=[1, 1, 1.35]))
for a, (key, title, ylim) in zip(axes[:2], (("roll_mae_deg", "future roll MAE (deg)", (0, 8.6)), ("tip_auroc", "tip-over AUROC", (0.9, 1.008)))):
    v = np.array([r[key]["point"] for _, r, _ in rows]); lo = np.array([r[key]["ci_low"] for _, r, _ in rows]); hi = np.array([r[key]["ci_high"] for _, r, _ in rows])
    a.set_ylim(*ylim); a.bar(x, v, 0.72, color=cols, yerr=[v - lo, hi - v], error_kw=dict(lw=0.7, capsize=1.8, ecolor=ts.INK))
    for xi, vi, hi_i in zip(x, v, hi): a.text(xi, hi_i + 0.015 * (ylim[1] - ylim[0]), f"{vi:.3f}" if key == "tip_auroc" else f"{vi:.1f}", ha="center", va="bottom", fontsize=7.5)
    a.set_xticks(x); a.set_xticklabels([n for n, _, _ in rows], fontsize=7.4, linespacing=0.95); a.set_title(title, fontsize=9); a.axvline(2.7, color=ts.BASE, lw=0.6, ls=":")
a = axes[2]; a.set_ylim(0, 1.42); a.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); w = 0.36
for off, key, hatch, alpha in ((-w / 2, "outcome_flip_mass", None, 1.0), (w / 2, "outcome_flip_com", "////", 0.45)):
    v = np.array([r[key]["point"] for _, r, _ in rows]); lo = np.array([r[key]["ci_low"] for _, r, _ in rows]); hi = np.array([r[key]["ci_high"] for _, r, _ in rows])
    a.bar(x + off, v, w, color=cols, alpha=alpha, hatch=hatch, edgecolor="white" if hatch else None, yerr=[v - lo, hi - v], error_kw=dict(lw=0.6, capsize=1.4, ecolor=ts.INK))
    for xi, vi, hi_i in zip(x + off, v, hi):
        if hatch: a.text(xi, 0.03, f"{vi:.2f}", ha="center", va="bottom", fontsize=6.8, rotation=90)        # CoM pairs: value inside the bar
        else: a.text(xi - 0.05, hi_i + 0.02, f"{vi:.2f}", ha="center", va="bottom", fontsize=6.8)
a.axhline(0.5, color=ts.INK, ls="--", lw=0.8); a.set_xticks(x); a.set_xticklabels([n for n, _, _ in rows], fontsize=7.4, linespacing=0.95); a.axvline(2.7, color=ts.BASE, lw=0.6, ls=":", ymax=1.1 / 1.42)
a.set_title("same command, different outcome:\nwhich run tips?", fontsize=9)
a.legend(handles=[Patch(facecolor=ts.INK, label=f"pairs differing in mass ({VID['n_outcome_flip_mass']})"), Patch(facecolor=ts.INK, alpha=0.45, hatch="////", edgecolor="white", label=f"pairs differing in CoM offset ({VID['n_outcome_flip_com']})")],
         fontsize=7, loc="upper left", handlelength=1.2, borderaxespad=0.2)
handles = [Patch(color=ts.BASE, label="true values given to the predictor (no video)"), Patch(color=ts.GEN, label="video only"), Patch(color=EARLY, label="command + mass read from the video before the turn"), Patch(color=ts.REAL, label="same, full observation")]
fig.tight_layout(w_pad=0.9, rect=(0, 0, 1, 0.9)); fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=7.8, bbox_to_anchor=(0.5, 1.0), handlelength=1.1, columnspacing=1.1)
ts.save(fig, f"fig_v2_information_{name}")
print("pairs with different outcomes:", VID["n_outcome_flip_pairs"], "| train-mean roll baseline", round(VID["train_mean_baseline"]["roll_mae_deg"], 2))
for n, r, _ in rows: print(n.replace("\n", " "), {k: round(r[k]["point"], 3) for k in ("roll_mae_deg", "tip_auroc", "outcome_flip_mass", "outcome_flip_com")})
