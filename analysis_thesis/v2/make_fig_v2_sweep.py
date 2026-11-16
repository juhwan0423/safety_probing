"""Thesis figure: the tip-over boundary on the ForkSafe-CF v2 sweep and where the video places a payload on it.

(a) mass map: outcome of every sweep run in the (drive command, payload mass) plane; line = boundary of the two-stage
    predictor given the true mass (trained on other data), dashed = the boundary when the mass is not known
(b) CoM map: outcome in the (drive command, lateral CoM offset towards the outside of the turn) plane at 480 kg
(c) critical drive command per payload: true (sweep), predictor with the true mass, and predictor with the mass read
    from the video before the turn (mean and standard deviation over the runs of that payload)
Needs out/v2_sweep_eval{tag}.npz/.json (sweep_eval_v2.py).
Run from analysis_thesis/:  PYTHONPATH=.:v2 ../external/LTX-2/.venv/bin/python v2/make_fig_v2_sweep.py [tag]
"""
import json, sys
import numpy as np
import thesis_style as ts
from thesis_style import plt

tag = sys.argv[1] if len(sys.argv) > 1 else ""
ts.apply()
Z = np.load(f"out/v2_sweep_eval{tag}.npz", allow_pickle=True); J = json.load(open(f"out/v2_sweep_eval{tag}.json"))
mass, offset, drive, tip, known, massmap, commap, rep = (Z[k] for k in ("mass", "offset", "drive", "tip", "known", "massmap", "commap", "rep"))
GM, GD, pmap = Z["GM"], Z["GD"], Z["pmap"]
fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.9))
sel = massmap & known & ~rep
for s, col, mk, lab in ((sel & ~tip, ts.REC, "o", "recovered"), (sel & tip, ts.TIP, "^", "tip-over")):
    ax[0].scatter(drive[s], mass[s], s=13, color=col, marker=mk, linewidths=0, label=lab, zorder=3)
ax[0].contour(GD, GM, pmap, levels=[0.5], colors=[ts.INK], linewidths=1.2)
crit_nomass = float(Z["crit_cmdonly"]) if "crit_cmdonly" in Z.files and np.isfinite(Z["crit_cmdonly"]) else None      # the command-only readout: one critical command for every payload
if crit_nomass is not None: ax[0].axvline(crit_nomass, color=ts.BASE, ls="--", lw=1.0)
ax[0].plot([], [], color=ts.INK, lw=1.2, label="predicted, mass known"); ax[0].plot([], [], color=ts.BASE, ls="--", lw=1.0, label="predicted, mass unknown")
ax[0].set_xlim(15.6, 24.4); ax[0].set_ylim(212, 748); ax[0].set_xlabel("drive command"); ax[0].set_ylabel("payload mass (kg)"); ax[0].set_title("outcome by payload mass", fontsize=9); ts.panel(ax[0], "a")
sel = commap & known
for s, col, mk in ((sel & ~tip, ts.REC, "o"), (sel & tip, ts.TIP, "^")):
    ax[1].scatter(drive[s], 90.0 * offset[s], s=13, color=col, marker=mk, linewidths=0, zorder=3)      # left turn: +offset is the outside; physical shift = 0.9 x configured
ax[1].set_xlabel("drive command"); ax[1].set_ylabel("payload CoM offset towards\nthe outside of the turn (cm)"); ax[1].set_title("outcome by CoM offset (480 kg)", fontsize=9); ts.panel(ax[1], "b")
levels = np.array(sorted(float(k) for k in J["critical_drive_by_mass"])); true_c = np.array([J["critical_drive_by_mass"][str(int(m))] for m in levels])
ax[2].plot(levels, true_c, "-", color=ts.INK, lw=1.3, marker="s", ms=3.5, label="true (sweep)")
ax[2].plot(levels, [J["critical_drive_model_true_mass"][str(int(m))] for m in levels], "-", color=ts.PRED, lw=1.1, label="predicted, true mass")
key = "crit_pred|mass from video 1.96 s"
if key in Z.files:
    mm = mass[massmap]; pc = Z[key]; mean = np.array([pc[np.abs(mm - m) < 1].mean() for m in levels]); sd = np.array([pc[np.abs(mm - m) < 1].std() for m in levels])
    ax[2].errorbar(levels, mean, yerr=sd, fmt="o", color=ts.GEN, ms=3.5, lw=0.9, capsize=2, label="predicted, mass from video\nbefore the turn")
if crit_nomass is not None: ax[2].axhline(crit_nomass, color=ts.BASE, ls="--", lw=1.0, label="predicted, mass unknown")
ax[2].set_xlabel("payload mass (kg)"); ax[2].set_ylabel("critical drive command"); ax[2].set_title("where tip-over starts", fontsize=9); ax[2].legend(fontsize=7, loc="best"); ts.panel(ax[2], "c")
handles, labels = ax[0].get_legend_handles_labels()
fig.tight_layout(w_pad=0.9, rect=(0, 0, 1, 0.91)); fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8, bbox_to_anchor=(0.42, 1.0), handletextpad=0.3, columnspacing=1.2)
ts.save(fig, f"fig_v2_sweep_boundary{tag}")
