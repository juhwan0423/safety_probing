"""Thesis figure: the ForkSafe-CF v2 dataset and what decides tip-over in it.

(a) planned command plane (drive command x steering command), outcome per run: a band of mixed outcomes
(b) inside that band: payload mass x lateral CoM offset towards the outside of the turn
(c) tip-over rate against the drive command for light / medium / heavy payloads
Needs out/mechanics_true_v2_<tag>.json (mechanics_true.py with the v2 manifest; true CoM, correct outside direction).
Run from analysis_thesis/:  PYTHONPATH=.:v2 ../external/LTX-2/.venv/bin/python v2/make_fig_v2_dataset.py p5
"""
import json, sys
import numpy as np
import thesis_style as ts
from thesis_style import plt

tag = sys.argv[1] if len(sys.argv) > 1 else "p5"
ts.apply()
R = json.load(open(f"out/mechanics_true_v2_{tag}.json"))
A = lambda k: np.array([r[k] for r in R], dtype=float)
definite = np.array([r["outcome"] != "right_censored" for r in R]); tip = np.array([r["outcome"] == "observed_tip" for r in R])
drive, steer, mass, off_out = np.abs(A("drive_cmd")), np.abs(A("steer_cmd")), A("mass"), A("offset_out") * 100.0      # cm, towards the outside
band = definite & (drive >= 18.0) & (drive <= 22.0)
print(f"runs {len(R)}, definite {definite.sum()}, tip rate {tip[definite].mean():.2f} | band 18-22: {band.sum()} runs, tip rate {tip[band].mean():.2f}")

fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.75))
for sel, col, mk, lab in ((definite & ~tip, ts.REC, "o", "recovered"), (definite & tip, ts.TIP, "^", "tip-over")):
    ax[0].scatter(drive[sel], steer[sel], s=7, color=col, alpha=0.55, marker=mk, linewidths=0, label=lab)
ax[0].axvspan(18, 22, color="#E9EEF2", lw=0, zorder=0)
ax[0].set_xlabel("drive command"); ax[0].set_ylabel("steering command"); ax[0].set_title("planned manoeuvre", fontsize=9); ts.panel(ax[0], "a")
for sel, col, mk in ((band & ~tip, ts.REC, "o"), (band & tip, ts.TIP, "^")):
    ax[1].scatter(mass[sel], off_out[sel], s=9, color=col, alpha=0.65, marker=mk, linewidths=0)
ax[1].set_xlabel("payload mass (kg)"); ax[1].set_ylabel("payload CoM offset towards\nthe outside of the turn (cm)"); ax[1].set_title("payload, drive command 18-22", fontsize=9); ts.panel(ax[1], "b")
edges = np.arange(15, 25.01, 1.0); centres = 0.5 * (edges[:-1] + edges[1:])
for (lo, hi), col, lab in (((240, 400), "#9DB7D1", "240-400 kg"), ((400, 560), "#4F7FAE", "400-560 kg"), ((560, 720), ts.PRED, "560-720 kg")):
    rate = [tip[definite & (mass >= lo) & (mass < hi) & (drive >= a) & (drive < b)].mean() if (definite & (mass >= lo) & (mass < hi) & (drive >= a) & (drive < b)).sum() >= 4 else np.nan for a, b in zip(edges[:-1], edges[1:])]
    ax[2].plot(centres, rate, "-o", color=col, ms=3.2, label=lab)
ax[2].axvspan(18, 22, color="#E9EEF2", lw=0, zorder=0)
ax[2].set_xlabel("drive command"); ax[2].set_ylabel("tip-over rate"); ax[2].set_title("tip-over rate by payload mass", fontsize=9); ax[2].legend(fontsize=7.5, loc="lower right"); ts.panel(ax[2], "c")
handles, labels = ax[0].get_legend_handles_labels()
fig.tight_layout(w_pad=1.0, rect=(0, 0, 1, 0.92)); fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=9, bbox_to_anchor=(0.3, 1.0), handletextpad=0.2)
ts.save(fig, f"fig_v2_dataset_{tag}")
