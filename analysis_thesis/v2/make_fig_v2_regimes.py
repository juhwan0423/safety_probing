"""Thesis figure: how much the video adds depends on how much the payload decides (ForkSafe-CF v2, pair-grouped CV).

Two regimes: the full command range (drive command 15-25, set p5) and the boundary band (18-22, set band5), where
the planned command alone is a weak predictor. Bars per regime:
  planned command only | video only | planned command + mass read from the video (two-stage) |
  planned command + true mass | planned command + true mass + true lateral CoM offset
Panels: tip-over AUROC, future roll MAE, accuracy on pairs that differ in mass and end differently.
Run from analysis_thesis/:  PYTHONPATH=.:v2 ../external/LTX-2/.venv/bin/python v2/make_fig_v2_regimes.py p5 band5
"""
import json, sys
import numpy as np
import thesis_style as ts
from thesis_style import plt
from matplotlib.patches import Patch

sets = sys.argv[1:] or ["p5", "band5"]; LABEL = {"p5": "full range\n(drive 15-25)", "band5": "boundary band\n(drive 18-22)", "all10": "both"}
ts.apply()
ROWS = [("planned command only", ts.BASE, lambda PV, V, PL, n: PV["command"]), ("video only", ts.GEN, lambda PV, V, PL, n: V),
        ("command + mass read from the video", ts.REAL, lambda PV, V, PL, n: PL[f"video mass ({n})"]), ("command + true mass", "#5C6B7A", lambda PV, V, PL, n: PV["command+mass"]),
        ("command + true mass + true CoM offset", ts.INK, lambda PV, V, PL, n: PV["command+mass+com"])]
DATA = {n: (json.load(open(f"out/v2_payload_value_{n}_cv.json")), json.load(open(f"out/v2cv_{n}.json")), json.load(open(f"out/v2_plugin_mass_{n}.json"))) for n in sets}
panels = [("tip_auroc", "tip-over AUROC", (0.7, 1.05), "{:.3f}"), ("roll_mae_deg", "future roll MAE (deg)", (0, 9.4), "{:.1f}"), ("outcome_flip_mass", "pairs differing in mass:\nwhich run tips?", (0.4, 1.09), "{:.2f}")]
fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.1)); w = 0.16
for a, (key, title, ylim, fmt) in zip(axes, panels):
    for gi, n in enumerate(sets):
        PV, V, PL = DATA[n]
        for bi, (lab, col, get) in enumerate(ROWS):
            r = get(PV, V, PL, n)[key]; x = gi + (bi - 2) * w
            a.bar(x, r["point"], w * 0.92, color=col, yerr=[[r["point"] - r["ci_low"]], [r["ci_high"] - r["point"]]], error_kw=dict(lw=0.6, capsize=1.3, ecolor=ts.INK if col != ts.INK else ts.BASE))
            a.text(x, max(r["ci_high"], ylim[0]) + 0.012 * (ylim[1] - ylim[0]), fmt.format(r["point"]), ha="center", va="bottom", fontsize=5.6, rotation=90)
    a.set_xticks(range(len(sets))); a.set_xticklabels([LABEL.get(n, n) for n in sets], fontsize=8); a.set_ylim(*ylim); a.set_title(title, fontsize=9)
    if key == "outcome_flip_mass": a.axhline(0.5, color=ts.INK, ls="--", lw=0.7)
    if key != "roll_mae_deg": a.set_yticks([t for t in a.get_yticks() if t <= 1.0001])
fig.tight_layout(w_pad=1.0, rect=(0, 0, 1, 0.86)); fig.legend(handles=[Patch(color=c, label=l) for l, c, _ in ROWS], loc="upper center", ncol=3, fontsize=7.6, bbox_to_anchor=(0.5, 1.0), handlelength=1.1, columnspacing=1.2)
ts.save(fig, "fig_v2_regimes_" + "_".join(sets))
for n in sets:
    PV, V, PL = DATA[n]; print(n, "| mass pairs", V.get("n_outcome_flip_mass"), "CoM pairs", V.get("n_outcome_flip_com"))
    for lab, _, get in ROWS:
        r = get(PV, V, PL, n); print(f"   {lab:40s} AUROC {r['tip_auroc']['point']:.3f} roll {r['roll_mae_deg']['point']:.2f} mass pairs {r['outcome_flip_mass']['point']:.3f}")
