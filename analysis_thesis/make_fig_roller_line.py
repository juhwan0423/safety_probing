"""Thesis figure: roll-over about the line of the outer fork rollers.

(a) top view of the contact points in the chassis frame, the system CoM for payloads of 240-720 kg, and the two
    candidate tip lines: the edge of the support triangle (outer front roller - rear wheel) and the outer roller line
(b, c) turn severity |v_x omega_z| during the steering pulse against the threshold g (b - y_c) / h for every run of
    v1 and v2; the diagonal is "severity = threshold" (no fitted constant)
Needs out/rollover_criterion.npz/.json (rollover_criterion.py).
Run from analysis_thesis/:  PYTHONPATH=. ../external/LTX-2/.venv/bin/python make_fig_roller_line.py
"""
import json
import numpy as np
import thesis_style as ts
from thesis_style import plt
import truecom as tc

ts.apply()
Z = np.load("out/rollover_criterion.npz"); J = json.load(open("out/rollover_criterion.json"))
fig, ax = plt.subplots(1, 3, figsize=(8.0, 3.0), gridspec_kw=dict(width_ratios=[1.15, 1, 1]))
a = ax[0]
REAR, FL, FR, BL, BR = (0.58, 0.008), (-0.91, 0.499), (-0.91, -0.494), (-0.717, 0.501), (-0.717, -0.5)
a.plot([REAR[0], FL[0], FR[0], REAR[0]], [REAR[1], FL[1], FR[1], REAR[1]], "--", color=ts.BASE, lw=0.9, label="support triangle")
a.plot([-1.05, 0.7], [0.5, 0.5], "-", color=ts.TIP, lw=1.6, label="outer roller line (left turn)")
for p in (FL, FR, BL, BR): a.plot(*p, "s", color=ts.INK, ms=4.5)
a.plot(*REAR, "o", color=ts.INK, ms=6)
masses = np.arange(240, 721, 60); coms = np.array([tc.com_body(float(m), np.zeros(3)) for m in masses]); cols = plt.cm.viridis(np.linspace(0.15, 0.9, len(masses)))
a.scatter(coms[:, 0], coms[:, 1], c=cols, s=18, zorder=4, edgecolors="none")
a.annotate("", xy=(coms[4, 0], 0.5), xytext=(coms[4, 0], coms[4, 1]), arrowprops=dict(arrowstyle="<->", color=ts.TIP, lw=0.9)); a.text(coms[4, 0] + 0.03, 0.25, r"$b-y_c$", color=ts.TIP, fontsize=8)
a.text(coms[0, 0] + 0.03, -0.10, "240 kg", fontsize=7); a.text(coms[-1, 0] - 0.33, -0.10, "720 kg", fontsize=7)
a.text(REAR[0] - 0.02, -0.13, "rear\nwheel", fontsize=7, ha="center", va="top"); a.text(-0.81, -0.62, "fork rollers", fontsize=7, ha="center")
a.annotate("travel", xy=(-1.0, -0.82), xytext=(-0.45, -0.82), arrowprops=dict(arrowstyle="->", color=ts.INK, lw=0.8), fontsize=7, va="center")
a.set_aspect("equal"); a.set_xlim(-1.15, 0.8); a.set_ylim(-0.95, 0.95); a.set_xlabel("chassis x (m)"); a.set_ylabel("chassis y (m)"); a.set_title("contact points and system CoM", fontsize=9)
a.legend(fontsize=6.8, loc="upper right", handlelength=1.4); ts.panel(a, "a")
for a, name, letter in ((ax[1], "v1", "b"), (ax[2], "v2", "c")):
    thr, sev, tip = Z[f"{name}|roller"], Z[f"{name}|severity_pulse"], Z[f"{name}|tip"].astype(bool); o = J[name]["pulse_2.0-3.0s"]
    for s, col, mk, lab in ((~tip, ts.REC, "o", "recovered"), (tip, ts.TIP, "^", "tip-over")): a.scatter(thr[s], sev[s], s=6, color=col, marker=mk, alpha=0.5, linewidths=0, label=lab)
    lo, hi = 2.3, 3.9; a.plot([lo, hi], [lo, hi], "-", color=ts.INK, lw=1.0)
    a.set_xlim(lo, hi); a.set_ylim(1.2, 5.6); a.set_xlabel(r"threshold $g\,(b-y_c)/h$ (m/s$^2$)"); a.set_ylabel(r"turn severity $|v_x\,\omega_z|$ (m/s$^2$)" if letter == "b" else "")
    a.set_title(f"{name}: {J[name]['n']} runs", fontsize=9); ts.panel(a, letter)
    a.text(0.04, 0.96, f"AUROC {o['over_roller_line_threshold']['point']:.3f}\naccuracy {100 * o['accuracy_at_ratio_1']:.0f}%", transform=a.transAxes, fontsize=7.5, va="top")
handles, labels = ax[1].get_legend_handles_labels()
fig.tight_layout(w_pad=0.8, rect=(0, 0, 1, 0.93)); fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=8.5, bbox_to_anchor=(0.68, 1.0), handletextpad=0.2, markerscale=2.0)
ts.save(fig, "fig_mechanics_roller_line")
