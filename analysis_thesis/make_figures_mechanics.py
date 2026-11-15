"""Mechanics figures with the true system CoM (truecom.py / mechanics_true.py).

fig_mechanics_stability_triangle : (a) support points, tipping axis and the true CoM for the five payload masses and
                                   three lateral payload offsets, to scale (chassis frame, top view)
                                   (b) kinematic severity of the turn against the static rollover threshold g d / h
fig_label_validation_zmp_cop     : (a) one run: lateral position of the wheel-load CoP, the logged ZMP and the true-CoM ZMP
                                   (b) |ZMP - CoP| over all runs for the logged and the true-CoM ZMP
Run:  PYTHONPATH=. ../external/LTX-2/.venv/bin/python make_figures_mechanics.py
"""
import json, sys
from pathlib import Path
import numpy as np
import thesis_style as ts
from thesis_style import plt
import comfix as cf, truecom as tc, com_identify as ci

sys.path.insert(0, "/home/jude/CWM_paper/Safety_Probing_paper/scripts")
import build_statistics as bs

ts.apply()
R = json.load(open("out/mechanics_true_v1.json"))
A = lambda k: np.array([r[k] for r in R], dtype=float)
definite = np.array([r["outcome"] != "right_censored" for r in R]); tip = np.array([r["outcome"] == "observed_tip" for r in R])

# ---------------------------------------------------------------------------------------------------------------
FL, FR, BL, BR, BW = (-0.910, 0.499), (-0.910, -0.494), (-0.718, 0.499), (-0.718, -0.494), (0.580, 0.008)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.25, 1]})
a = ax[0]
hull = np.array([FL, BL, BW, BR, FR, FL])
a.fill(hull[:, 0], hull[:, 1], color="#EEF2F6", zorder=0); a.plot(hull[:, 0], hull[:, 1], color=ts.BASE, lw=0.9, zorder=1)
a.plot([BL[0], BW[0]], [BL[1], BW[1]], color=ts.TIP, lw=2.0, zorder=2)
a.text(0.02, 0.36, "tipping axis\n(left turn)", color=ts.TIP, fontsize=8, ha="left", va="center")
for (x, y), lab, dx, dy, ha in ((FL, "front rollers", 0.0, 0.085, "left"), (BW, "drive and\nsteer wheel", 0.0, -0.14, "center")):
    a.text(x + dx, y + dy, lab, fontsize=8, ha=ha, va="center")
for (x, y) in (FL, FR, BW): a.plot([x], [y], "s", color=ts.INK, ms=5, zorder=3)
for (x, y) in (BL, BR): a.plot([x], [y], "s", color=ts.INK, ms=3.5, mfc="white", zorder=3)
masses = [240, 360, 480, 600, 720]; shades = plt.cm.Blues(np.linspace(0.45, 0.95, len(masses)))
for m, col in zip(masses, shades):
    pts = np.array([tc.com_body(m, [0, o, 0])[:2] for o in (-0.25, 0.0, 0.25)])
    a.plot(pts[:, 0], pts[:, 1], "-", color=col, lw=1.0, zorder=4); a.plot(pts[:, 0], pts[:, 1], "o", color=col, ms=3.6, zorder=5, label=f"{m} kg")
c = tc.com_body(480, [0, 0, 0])[:2]
e = np.array(BW) - np.array(BL); e /= np.linalg.norm(e); foot = np.array(BL) + e * np.dot(c - np.array(BL), e)
a.annotate("", xy=foot, xytext=c, arrowprops=dict(arrowstyle="<->", lw=0.9, color=ts.INK), zorder=6)
a.text((c[0] + foot[0]) / 2 + 0.03, (c[1] + foot[1]) / 2, "$d$", fontsize=10, va="center")
a.plot([tc.C_VEHICLE[0]], [tc.C_VEHICLE[1]], "x", color=ts.INK, ms=6, zorder=5); a.text(tc.C_VEHICLE[0], tc.C_VEHICLE[1] - 0.09, "empty\nvehicle", fontsize=8, ha="center", va="top")
a.annotate("travel", xy=(-1.28, 0.0), xytext=(-1.02, 0.0), fontsize=8, va="center", ha="left", arrowprops=dict(arrowstyle="->", lw=0.9, color=ts.INK))
a.set_aspect("equal"); a.set_xlim(-1.35, 0.95); a.set_ylim(-0.66, 0.70); a.set_xlabel("chassis $x$ (m)"); a.set_ylabel("chassis $y$ (m)")
a.legend(title="system centre of mass for a payload of", fontsize=8, title_fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=5, handletextpad=0.1, columnspacing=0.8); ts.panel(a, "a")
b = ax[1]
for sel, col, mk, lab in ((definite & ~tip, ts.REC, "o", "recovered"), (definite & tip, ts.TIP, "^", "tip-over")):
    b.scatter(A("a_crit")[sel], A("a_kin")[sel], s=8, color=col, alpha=0.55, marker=mk, linewidths=0, label=lab)
ratio = A("a_kin") / A("a_crit"); y = tip[definite].astype(int)
cand = np.unique(np.round(ratio[definite], 3)); acc = [np.mean((ratio[definite] > c0) == (y == 1)) for c0 in cand]; k = float(cand[int(np.argmax(acc))])
xs = np.array([1.4, 2.9]); b.plot(xs, k * xs, color=ts.INK, lw=0.9, ls="--")
b.text(2.9, 0.75, f"dashed: demand = {k:.2f} $\\times$ threshold", fontsize=8, ha="right", va="bottom")
b.set_xlabel("static rollover threshold $g\\,d/h$ (m/s$^2$)"); b.set_ylabel("peak speed $\\times$ yaw rate (m/s$^2$)"); b.set_ylim(0.6, None); b.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.1); ts.panel(b, "b")
fig.tight_layout(w_pad=1.6); ts.save(fig, "fig_mechanics_stability_triangle")
auc = bs.auc(y, ratio[definite])
print(f"panel b: AUROC of demand / threshold {auc:.3f}; best ratio threshold {k:.2f} -> accuracy {max(acc):.3f}; AUROC of demand alone {bs.auc(y, A('a_kin')[definite]):.3f}")
for m in masses:
    c0 = tc.com_body(m, [0, 0, 0]); cp = tc.com_body(m, [0, 0.25, 0]); print(f"  {m} kg: CoM x {c0[0]:+.3f} h {c0[2]:.3f}; lateral shift for a configured 0.25 m offset {cp[1] - c0[1]:+.3f} m")

# ---------------------------------------------------------------------------------------------------------------
man = cf.load_manifest()
pick = next((p, r) for p, r in man.items() if r["speed_label"] == "slow" and r["steering_label"] == "medium" and r["turn_direction"] == "left"
            and float(r["load_mass_kg"]) == 480.0 and float(r["load_com_offset_local_xyz_m"][1]) == 0.25)
rows = cf.read_rows(Path(pick[0])); S = ci.series(rows); mass, off = 480.0, np.array(pick[1]["load_com_offset_local_xyz_m"], float)
T = tc.compute(rows, mass, off); C = cf.correct(rows, mass, off); t = S["t"]
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.6), gridspec_kw={"width_ratios": [1, 1, 0.8]})
loaded = (S["N"][:, 3] > 300) & (S["N"][:, 4] > 300) & (S["N"][:, 0] > 300) & (t > 0.3)
for j, (axis, lab) in enumerate(((0, "longitudinal position $x$ (m)"), (1, "lateral position $y$ (m)"))):
    cop = ci.running_median(S["cop"][:, axis], 5)
    zl = ci.running_median(np.nan_to_num(ci.body_frame(C["zmp_rep"], S["origin"], S["yaw"])[:, axis]), 5)
    zt = ci.running_median(np.nan_to_num(ci.body_frame(T["zmp"], S["origin"], S["yaw"])[:, axis]), 5)
    w = loaded & (t < 2.9)
    ax[j].plot(t[w], cop[w], color=ts.INK, lw=1.6, label="centre of pressure (wheel loads)")
    ax[j].plot(t[w], zl[w], color=ts.GEN, lw=1.1, label="ZMP, logged CoM")
    ax[j].plot(t[w], zt[w], color=ts.PRED, lw=1.1, label="ZMP, true CoM")
    ax[j].axvline(2.0, color=ts.BASE, lw=0.8, ls=":"); ax[j].set_xlabel("time (s)"); ax[j].set_title(lab, fontsize=9)
ax[1].text(2.03, ax[1].get_ylim()[0] + 0.01, "steering\nstarts", fontsize=7.5, color=ts.BASE, va="bottom"); ts.panel(ax[0], "a"); ts.panel(ax[1], "b")
E = {n: np.array([r["err"][n] for r in R if n in r["err"]]) for n in ("logged", "true")}
pos = [0, 1, 2.6, 3.6]; data = [E["logged"][:, 0], E["true"][:, 0], E["logged"][:, 1], E["true"][:, 1]]; cols = [ts.GEN, ts.PRED, ts.GEN, ts.PRED]
bp = ax[2].boxplot(data, positions=pos, widths=0.7, patch_artist=True, showfliers=False, medianprops=dict(color=ts.INK, lw=1.0), whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7))
for patch, col in zip(bp["boxes"], cols): patch.set_facecolor(col); patch.set_alpha(0.75); patch.set_linewidth(0.6)
ax[2].set_xticks([0.5, 3.1]); ax[2].set_xticklabels(["longitudinal", "lateral"]); ax[2].set_ylabel("|ZMP $-$ CoP| per run (m)"); ax[2].set_title("all 810 runs", fontsize=9); ts.panel(ax[2], "c")
handles, labels = ax[0].get_legend_handles_labels()
fig.tight_layout(w_pad=1.0, rect=(0, 0, 1, 0.9)); fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 1.0), handlelength=1.6, columnspacing=1.4)
ts.save(fig, "fig_label_validation_zmp_cop")
for n in ("logged", "true"): print(f"|ZMP-CoP| {n}: longitudinal median {np.median(E[n][:, 0]):.3f} lateral {np.median(E[n][:, 1]):.3f}")
