"""Paper-split comparison (prediction from the observation / real-future readout / generated future) on the
true-CoM targets. Reads out/truecom_table1.json (truecom_eval.py). The paper split shares pre-turn segments and
replicas between train and test, so this figure reproduces the paper's comparison on corrected labels.
Run:  PYTHONPATH=. ../external/LTX-2/.venv/bin/python make_fig_table1.py
"""
import json
import numpy as np
import thesis_style as ts
from thesis_style import plt

ts.apply()
T = json.load(open("out/truecom_table1.json"))["truecom"]; base = T["train_mean"]
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
metrics = [("mass_mae_kg", "payload\nmass"), ("com_mae_m", "lateral\nCoM"), ("roll_mae_deg", "future\nroll"), ("zmp_mae_m", "min. ZMP\nmargin")]
pm = [("mass_direction", "mass\ndirection"), ("com_direction", "CoM\ndirection"), ("gold_zmp_ordering", "ZMP\nordering")]
for j, (name, col, lab) in enumerate((("vjepa_prefix", ts.PRED, "prediction from the observed video"), ("ltx_oracle", ts.REAL, "readout of the real future"), ("ltx_generated", ts.GEN, "readout of the generated future"))):
    v = [T[name][m]["point"] / base[m] for m, _ in metrics]; lo = [T[name][m]["ci_low"] / base[m] for m, _ in metrics]; hi = [T[name][m]["ci_high"] / base[m] for m, _ in metrics]
    xs = np.arange(4) + (j - 1) * 0.26; ax[0].bar(xs, v, width=0.24, color=col, label=lab); ax[0].errorbar(xs, v, yerr=[np.subtract(v, lo), np.subtract(hi, v)], fmt="none", ecolor=ts.INK, elinewidth=0.7, capsize=1.5)
    v = [T[name][m]["point"] for m, _ in pm]; lo = [T[name][m]["ci_low"] for m, _ in pm]; hi = [T[name][m]["ci_high"] for m, _ in pm]
    xs = np.arange(3) + (j - 1) * 0.26; ax[1].bar(xs, v, width=0.24, color=col); ax[1].errorbar(xs, v, yerr=[np.subtract(v, lo), np.subtract(hi, v)], fmt="none", ecolor=ts.INK, elinewidth=0.7, capsize=1.5)
ax[0].axhline(1.0, color=ts.INK, ls="--", lw=0.9); ax[0].text(3.42, 1.02, "train-mean baseline", fontsize=8, ha="right", va="bottom")
ax[0].set_xticks(range(4)); ax[0].set_xticklabels([l for _, l in metrics]); ax[0].set_ylabel("error relative to the baseline"); ax[0].set_ylim(0, 1.6); ts.panel(ax[0], "a")
ax[1].axhline(0.5, color=ts.INK, ls="--", lw=0.9); ax[1].text(-0.42, 0.51, "chance", fontsize=8, ha="left", va="bottom")
ax[1].set_xticks(range(3)); ax[1].set_xticklabels([l for _, l in pm]); ax[1].set_ylabel("pair accuracy"); ax[1].set_ylim(0, 1.08); ts.panel(ax[1], "b")
handles, labels = ax[0].get_legend_handles_labels()
fig.tight_layout(w_pad=1.6, rect=(0, 0, 1, 0.9)); fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, 1.0), handlelength=1.2, columnspacing=1.2)
ts.save(fig, "fig_paper_split_truecom")
