"""Thesis figure: how early can the future be predicted? (ForkSafe-CF v1, 5-fold CV over physical configurations)

Protocol mos5: folds keep every route that shares a pre-turn segment together (clean; see psq_cv.py).
Observation truncated to the first 24/36/47/56/65 frames (1.0-2.71 s); the steering pulse starts at t = 2.0 s.
Future targets are identical for every observation length (roll over 2.71-5.0 s, minimum ZMP margin, tip-over).
Run:  PYTHONPATH=. ../external/LTX-2/.venv/bin/python make_figures_early.py
"""
import json, os
import numpy as np
import thesis_style as ts
from thesis_style import plt

ts.apply()
FRAMES = [24, 36, 47, 56, 65]
T = {f: f / 24.0 for f in FRAMES}
load = lambda n: json.load(open(f"out/cv_{n}.json")) if os.path.exists(f"out/cv_{n}.json") else None
# results on the true-CoM targets (suffix _tc_)
VID = {f: load(f"mos5_full_tc_obs{f}") for f in FRAMES}
CMD = {f: load(f"mos5_full_cmd_tc_obs{f}") for f in FRAMES}
ONLY = load("mos5_command_only_tc_obs65")
BASE = (ONLY or next(v for v in VID.values() if v))["train_mean_baseline"]
D = json.load(open("out/dense_table.json"))
t_loss = np.array([r["t_support_loss"] for r in D if r["outcome"] == "observed_tip" and r["t_support_loss"] is not None])

panels = [("tip_auroc", "tip-over AUROC", None, None), ("roll_mae_deg", "future roll MAE (deg)", "roll", None),
          ("mass_mae_kg", "payload mass MAE (kg)", "mass", None), ("com_direction", "lateral-CoM direction accuracy", None, (0.3, 1.0))]
fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.5))
for a, (key, ylab, base_key, ylim) in zip(axes, panels):
    for res, col, mk, lab in ((VID, ts.GEN, "s", "video only"), (CMD, ts.PRED, "o", "video + planned command")):
        fs = [f for f in FRAMES if res[f] is not None]
        if not fs: continue
        x = [T[f] for f in fs]; y = np.array([res[f][key]["point"] for f in fs])
        lo = np.array([res[f][key]["ci_low"] for f in fs]); hi = np.array([res[f][key]["ci_high"] for f in fs])
        a.fill_between(x, lo, hi, color=col, alpha=0.13, lw=0); a.plot(x, y, "-", color=col, marker=mk, ms=3.8, label=lab)
    if ONLY is not None:
        a.axhline(ONLY[key]["point"], color=ts.BASE, ls="--", lw=1.0, label="planned command only")
    if base_key is not None:
        a.axhline(BASE[base_key], color=ts.INK, ls=":", lw=0.9, label="train-mean baseline")
    if key == "com_direction":
        a.axhline(0.5, color=ts.INK, ls=":", lw=0.9)
    a.axvline(2.0, color=ts.TIP, lw=0.8, alpha=0.7)
    a.set_xlabel("observed video length (s)"); a.set_title(ylab, fontsize=9); a.set_xticks([1.0, 1.5, 2.0, 2.5])
    if ylim: a.set_ylim(*ylim)
    elif key == "tip_auroc": a.set_ylim(min(0.86, a.get_ylim()[0]), 1.005)
    else: a.set_ylim(0, None)
axes[0].text(2.03, axes[0].get_ylim()[0] + 0.003, "steering\nstarts", fontsize=7.5, color=ts.TIP, va="bottom")
handles, labels = axes[1].get_legend_handles_labels()
fig.tight_layout(w_pad=0.9, rect=(0, 0, 1, 0.9)); fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9, handlelength=1.6, columnspacing=1.4, bbox_to_anchor=(0.5, 1.0))
ts.save(fig, "fig_early_prediction_v1")
for f in FRAMES:
    lead = t_loss - T[f]
    row = [f"obs {T[f]:.2f}s: lead median {np.median(lead):.2f}s min {lead.min():.2f}s"]
    for nm, res in (("video", VID), ("video+cmd", CMD)):
        if res[f]: row.append(f"{nm}: AUROC {res[f]['tip_auroc']['point']:.3f} roll {res[f]['roll_mae_deg']['point']:.2f} mass {res[f]['mass_mae_kg']['point']:.1f} com {res[f]['com_mae_m']['point']:.3f} comdir {res[f]['com_direction']['point']:.3f}")
    print(" | ".join(row))
if ONLY: print("command only: AUROC %.3f roll %.2f" % (ONLY["tip_auroc"]["point"], ONLY["roll_mae_deg"]["point"]))
