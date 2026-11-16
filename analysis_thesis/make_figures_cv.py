"""Thesis figures for the physically disjoint evaluations on ForkSafe-CF v1 (psq_cv.py outputs).

fig_generalization_v1 : error relative to the train-mean baseline and pair/tip metrics under three protocols
                        (replica-shared paper split, unseen payload-manoeuvre combination, unseen manoeuvre)
fig_unseen_manoeuvre_v1 : leave-one-manoeuvre-out results on the 3 x 3 speed-steering grid
Run:  PYTHONPATH=. ../external/LTX-2/.venv/bin/python make_figures_cv.py
"""
import json
import numpy as np
import thesis_style as ts
from thesis_style import plt

ts.apply()
import os
if os.path.exists("out/truecom_table1.json") and "truecom" in json.load(open("out/truecom_table1.json")):
    T1 = json.load(open("out/truecom_table1.json"))["truecom"]            # targets from the true multi-body CoM / ZMP
else:
    T1 = json.load(open("out/comfix_table1.json"))["comfix_med5"]
PAPER, PAPER_BASE = T1["vjepa_prefix"], T1["train_mean"]
first = lambda *names: json.load(open(next(f"out/cv_{n}.json" for n in names if os.path.exists(f"out/cv_{n}.json"))))
# config5 and logo9 share pre-turn segments between train and test (see psq_cv.py); shown as the reference protocols.
C5, L9 = first("config5_full_tc_obs65", "config5_full"), first("logo9_full_tc_obs65", "logo9_full")
MOS = json.load(open("out/cv_mos5_full_tc_obs65.json")) if os.path.exists("out/cv_mos5_full_tc_obs65.json") else None
pt = lambda d, k: d[k]["point"]
ci = lambda d, k: (d[k]["point"] - d[k]["ci_low"], d[k]["ci_high"] - d[k]["point"])

# ---------------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})
protocols = [("same configurations\n(replica split)", PAPER, {"mass": PAPER_BASE["mass_mae_kg"], "com": PAPER_BASE["com_mae_m"], "roll": PAPER_BASE["roll_mae_deg"], "zmp": PAPER_BASE["zmp_mae_m"]}, "#9DB7D1"),
             ("unseen payload-manoeuvre\ncombination", C5, C5["train_mean_baseline"], "#4F7FAE"),
             ("unseen manoeuvre", L9, L9["train_mean_baseline"], ts.PRED)]
errs = [("mass_mae_kg", "mass", "payload\nmass"), ("com_mae_m", "com", "lateral\nCoM"), ("roll_mae_deg", "roll", "future\nroll"), ("zmp_mae_m", "zmp", "min. ZMP\nmargin")]
w = 0.26
for j, (lab, res, base, col) in enumerate(protocols):
    x = np.arange(len(errs)) + (j - 1) * w
    v = np.array([pt(res, k) / base[b] for k, b, _ in errs]); e = np.array([[ci(res, k)[0] / base[b], ci(res, k)[1] / base[b]] for k, b, _ in errs]).T
    ax[0].bar(x, v, w * 0.92, color=col, label=lab.replace("\n", " "), yerr=e, error_kw=dict(lw=0.7, capsize=1.5, ecolor=ts.INK))
ax[0].axhline(1.0, color=ts.INK, ls="--", lw=0.9); ax[0].text(3.42, 1.02, "train-mean baseline", fontsize=8, ha="right", va="bottom")
ax[0].set_xticks(range(len(errs))); ax[0].set_xticklabels([e[2] for e in errs]); ax[0].set_ylabel("error relative to the baseline"); ax[0].set_ylim(0, 1.12)
ts.panel(ax[0], "a")
mets = [("tip_auroc", "tip-over\nAUROC"), ("mass_direction", "mass\ndirection"), ("com_direction", "CoM\ndirection"), ("gold_zmp_ordering", "ZMP\nordering")]
for j, (lab, res, base, col) in enumerate(protocols):
    x = np.arange(len(mets)) + (j - 1) * w
    v = np.array([pt(res, k) for k, _ in mets]); e = np.array([ci(res, k) for k, _ in mets]).T
    ax[1].bar(x, v, w * 0.92, color=col, yerr=e, error_kw=dict(lw=0.7, capsize=1.5, ecolor=ts.INK))
ax[1].axhline(0.5, color=ts.INK, ls="--", lw=0.9); ax[1].text(-0.42, 0.51, "chance", fontsize=8, ha="left", va="bottom")
ax[1].set_xticks(range(len(mets))); ax[1].set_xticklabels([m[1] for m in mets]); ax[1].set_ylabel("accuracy / AUROC"); ax[1].set_ylim(0.3, 1.03); ts.panel(ax[1], "b")
handles, labels = ax[0].get_legend_handles_labels()
fig.tight_layout(w_pad=1.6, rect=(0, 0, 1, 0.9)); fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9, handlelength=1.2, columnspacing=1.6, bbox_to_anchor=(0.5, 1.0))
ts.save(fig, "fig_generalization_v1")

# ---------------------------------------------------------------------------------------------------------------
speeds, steers = ["slow", "medium", "fast"], ["wide", "medium", "tight"]
panels = [("roll_mae_deg", "future roll MAE (deg)", "{:.1f}", L9["train_mean_baseline"]["roll"]), ("mass_mae_kg", "payload mass MAE (kg)", "{:.0f}", L9["train_mean_baseline"]["mass"]),
          ("com_direction", "lateral-CoM direction accuracy", "{:.2f}", None), ("tip_auroc", "tip-over AUROC", "{:.2f}", None)]
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.25))
for a, (key, title, fmt, base) in zip(axes, panels):
    M = np.array([[L9["per_fold"][f"{sp}_{st}"].get(key, np.nan) for st in steers] for sp in speeds], float)
    if base is not None:
        im = a.imshow(M, cmap="Blues", vmin=0, vmax=base, origin="lower")
    else:
        im = a.imshow(M, cmap="Blues_r", vmin=0.5, vmax=1.0, origin="lower")
    for i in range(3):
        for j in range(3):
            if np.isnan(M[i, j]):
                a.text(j, i, "one\nclass", ha="center", va="center", fontsize=7, color=ts.BASE)
            else:
                dark = (M[i, j] / base > 0.55) if base is not None else (M[i, j] < 0.72)
                a.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=9, color="white" if dark else ts.INK)
    a.set_xticks(range(3)); a.set_xticklabels(steers); a.set_yticks(range(3)); a.set_yticklabels(speeds if a is axes[0] else [])
    a.set_xlabel("steering"); a.set_title(title, fontsize=9)
    for s in a.spines.values(): s.set_visible(False)
    a.tick_params(length=0)
axes[0].set_ylabel("speed of the held-out manoeuvre")
fig.tight_layout(w_pad=0.6); ts.save(fig, "fig_unseen_manoeuvre_v1")
print("baseline (train mean): roll %.1f deg, mass %.0f kg" % (L9["train_mean_baseline"]["roll"], L9["train_mean_baseline"]["mass"]))
