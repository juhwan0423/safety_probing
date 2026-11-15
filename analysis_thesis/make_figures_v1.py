"""Draft thesis figures from the analyses that are already valid on ForkSafe-CF v1. Run with a Python that has matplotlib."""
import json, sys, numpy as np
import thesis_style as ts
from thesis_style import plt
ts.apply()
T_OBS = 65 / 24

# ---- Fig. physics: stability arm and static threshold -------------------------------------------------------
R = json.load(open("out/rollover_physics.json"))["runs"]
A = lambda k: np.array([r[k] for r in R])
out_off = np.where(A("turn") == "left", 1.0, -1.0) * A("com")
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
masses = sorted(set(A("mass"))); shades = plt.cm.Blues(np.linspace(0.45, 0.95, len(masses)))
for m, col in zip(masses, shades):
    sel = A("mass") == m; xs = sorted(set(out_off[sel]))
    ax[0].plot(xs, [A("d")[sel & (out_off == x)].mean() for x in xs], "-o", color=col, ms=3.5, label=f"{m:.0f} kg")
ax[0].set_xlabel("payload lateral CoM offset towards the inside of the turn (m)"); ax[0].set_ylabel("stability arm $d$ (m)")
ax[0].legend(title="payload mass", ncol=2, title_fontsize=9); ax[0].set_ylim(0, 0.24); ts.panel(ax[0], "a")
for oc, col, lab, mk in (("confirmed_recovered", ts.REC, "recovered", "o"), ("observed_tip", ts.TIP, "tip-over", "^")):
    sel = A("outcome") == oc
    ax[1].scatter(A("a_crit")[sel], A("a_peak")[sel], s=9, color=col, alpha=0.55, marker=mk, label=lab, linewidths=0)
lim = [0, 7.5]; ax[1].plot(lim, lim, color=ts.INK, lw=0.9, ls="--"); ax[1].annotate("static limit", xy=(1.25, 1.25), xytext=(1.45, 0.35), fontsize=9, arrowprops=dict(arrowstyle="-", lw=0.6, color=ts.INK))
ax[1].set_xlim(0, 2.2); ax[1].set_ylim(0, 7.5); ax[1].set_xlabel("static rollover threshold $g\\,d/h$ (m/s$^2$)"); ax[1].set_ylabel("peak lateral demand (m/s$^2$)")
ax[1].legend(loc="upper left"); ts.panel(ax[1], "b"); fig.tight_layout(w_pad=2.0); ts.save(fig, "fig_physics_stability_arm")

# ---- Fig. lead time ----------------------------------------------------------------------------------------
D = json.load(open("out/dense_table.json")); tip = [r for r in D if r["outcome"] == "observed_tip"]
sl = np.array([r["t_support_loss"] - T_OBS for r in tip if r["t_support_loss"] is not None]); rt = np.array([r["t_roll_tip"] - T_OBS for r in tip if r["t_roll_tip"] is not None])
fig, ax = plt.subplots(figsize=(4.6, 2.7)); bins = np.arange(0, 2.35, 0.1)
ax.hist(sl, bins=bins, color=ts.PRED, alpha=0.85, label=f"sustained support loss (n={len(sl)})"); ax.hist(rt, bins=bins, color=ts.GEN, alpha=0.6, label=f"roll exceeds 20$^\\circ$ (n={len(rt)})")
for v, col in ((np.median(sl), ts.PRED), (np.median(rt), ts.GEN)): ax.axvline(v, color=col, ls="--", lw=1.0)
ax.set_xlabel("time after the end of the observation (s)"); ax.set_ylabel("tip-over conditions"); ax.legend(); ts.save(fig, "fig_lead_time_v1")

# ---- Fig. matched-pair motion difference ---------------------------------------------------------------------
from common import Context
c = Context("artifacts/targets/safety_query_full_comfix_med5.pt"); traj = np.load("out/obs_trajectory.npy")
index = {cid: i for i, cid in enumerate(c.ids)}; runs = {(str(r["pair_id"]), str(r["run_label"])): str(r["condition_id"]) for r in c.run_rows}
diff = {"load_mass": [], "com_shift": []}
for p in c.pair_rows:
    a, b = index[runs[(str(p["pair_id"]), "A")]], index[runs[(str(p["pair_id"]), "B")]]
    diff[p["intervention_type"]].append(np.hypot(traj[a, :, 0] - traj[b, :, 0], traj[a, :, 1] - traj[b, :, 1]).max())
fig, ax = plt.subplots(figsize=(4.6, 2.7)); bins = np.logspace(-3, 0.2, 36)
ax.hist(diff["load_mass"], bins=bins, color=ts.PRED, alpha=0.8, label=f"mass pairs (median {np.median(diff['load_mass']):.2f} m)")
ax.hist(diff["com_shift"], bins=bins, color=ts.GEN, alpha=0.65, label=f"lateral-CoM pairs (median {np.median(diff['com_shift']):.2f} m)")
ax.axvline(1 / 32, color=ts.INK, ls=":", lw=1.0); ax.text(1 / 32 * 0.92, ax.get_ylim()[1] * 0.55, "1 px at the\nmodel input", fontsize=8, va="top", ha="right")
ax.set_xscale("log"); ax.set_xlabel("largest position difference within the observation window (m)"); ax.set_ylabel("matched pairs"); ax.legend(loc="upper left")
ts.save(fig, "fig_pair_motion_difference")

# ---- Fig. held-out payload level -----------------------------------------------------------------------------
t = c.tnp; fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw={"width_ratios": [1.5, 1]})
for j, (lvl, col) in enumerate(((360, ts.REAL), (480, ts.PRED), (600, ts.GEN))):
    z = np.load(f"out/heldout_pred_mass{lvl}.npz", allow_pickle=True); held = z["split"] == "heldout"
    ax[0].hist(z["static"][:, held, 0].ravel(), bins=np.arange(180, 800, 20), color=col, alpha=0.55, label=f"held-out {lvl} kg"); ax[0].axvline(lvl, color=col, lw=1.3)
for lv in (240, 360, 480, 600, 720): ax[0].plot([lv], [0], marker="|", color=ts.INK, ms=8)
ax[0].set_xlabel("predicted payload mass for conditions of the unseen level (kg)"); ax[0].set_ylabel("predictions (3 seeds)"); ax[0].legend(); ts.panel(ax[0], "a")
H = {k: json.load(open(f"out/heldout_{k}.json")) for k in ("mass360", "mass480", "mass600")}
K = json.load(open("out/kinematics_heldout_load_mass_kg_480.json"))
names = ["trained\nlevels", "360", "480", "600", "480, true\nkinematics"]
vals = [np.mean([H[k]["indist"]["com_direction"]["point"] for k in H]), H["mass360"]["all"]["com_direction"]["point"], H["mass480"]["all"]["com_direction"]["point"], H["mass600"]["all"]["com_direction"]["point"], K["kin+traj/mlp"]["com_direction"]["point"]]
ax[1].bar(range(5), vals, color=[ts.PRED, ts.PRED, ts.PRED, ts.PRED, ts.BASE], alpha=0.9, width=0.65); ax[1].axhline(0.5, color=ts.INK, ls="--", lw=0.9); ax[1].text(4.45, 0.51, "chance", fontsize=8, ha="right")
for i, v in enumerate(vals): ax[1].text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=8)
ax[1].set_xticks(range(5)); ax[1].set_xticklabels(names, fontsize=8); ax[1].set_ylim(0, 1.08); ax[1].set_ylabel("lateral-CoM direction accuracy"); ax[1].set_xlabel("held-out mass level (kg)"); ts.panel(ax[1], "b")
fig.tight_layout(w_pad=1.6); ts.save(fig, "fig_heldout_mass_level_v1")

# ---- Fig. label validation against contact forces ------------------------------------------------------------
V = json.load(open("out/comfix_verify.json"))["per_run"]; x = np.array([p["expected"] for p in V]); nz = x != 0
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
for a_, key, title in ((ax[0], "r0_med", "logged ZMP"), (ax[1], "r1_med", "corrected ZMP")):
    y = np.array([p[key] for p in V]); s, b = np.polyfit(x[nz], y[nz], 1)
    a_.scatter(x, y, s=5, color=ts.PRED if key == "r1_med" else ts.GEN, alpha=0.5, linewidths=0); xx = np.array([-0.09, 0.09]); a_.plot(xx, s * xx + b, color=ts.INK, lw=1.0)
    a_.set_title(f"{title}: slope {s:.2f}"); a_.set_xlabel("missing CoM shift $(m/M)\\,y_{\\mathrm{load}}$ (m)"); a_.set_ylim(-0.2, 0.2)
ax[0].set_ylabel("centre of pressure $-$ ZMP, lateral (m)"); ts.panel(ax[0], "a"); ts.panel(ax[1], "b"); fig.tight_layout(); ts.save(fig, "fig_label_validation_cop")

# ---- Fig. Table 1 under corrected labels ---------------------------------------------------------------------
T = json.load(open("out/comfix_table1.json"))["comfix_med5"]; base = T["train_mean"]
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
metrics = [("mass_mae_kg", "Mass"), ("com_mae_m", "Lateral CoM"), ("roll_mae_deg", "Future roll"), ("zmp_mae_m", "ZMP margin")]
for j, (name, col, lab) in enumerate((("vjepa_prefix", ts.PRED, "prediction from observation"), ("ltx_oracle", ts.REAL, "real future (readout ceiling)"), ("ltx_generated", ts.GEN, "generated future"))):
    v = [T[name][m]["point"] / base[m] for m, _ in metrics]; lo = [T[name][m]["ci_low"] / base[m] for m, _ in metrics]; hi = [T[name][m]["ci_high"] / base[m] for m, _ in metrics]
    xs = np.arange(4) + (j - 1) * 0.26; ax[0].bar(xs, v, width=0.24, color=col, label=lab); ax[0].errorbar(xs, v, yerr=[np.subtract(v, lo), np.subtract(hi, v)], fmt="none", ecolor=ts.INK, elinewidth=0.7, capsize=1.5)
    pm = [("mass_direction", "Mass\ndirection"), ("com_direction", "Lateral-CoM\ndirection"), ("gold_zmp_ordering", "ZMP-margin\nordering")]
    v = [T[name][m]["point"] for m, _ in pm]; lo = [T[name][m]["ci_low"] for m, _ in pm]; hi = [T[name][m]["ci_high"] for m, _ in pm]
    xs = np.arange(3) + (j - 1) * 0.26; ax[1].bar(xs, v, width=0.24, color=col); ax[1].errorbar(xs, v, yerr=[np.subtract(v, lo), np.subtract(hi, v)], fmt="none", ecolor=ts.INK, elinewidth=0.7, capsize=1.5)
ax[0].axhline(1.0, color=ts.INK, ls="--", lw=0.9); ax[0].set_xticks(range(4)); ax[0].set_xticklabels([l for _, l in metrics]); ax[0].set_ylabel("MAE / train-mean MAE"); ax[0].legend(loc="upper left", fontsize=8); ax[0].set_ylim(0, 1.75); ts.panel(ax[0], "a")
ax[1].axhline(0.5, color=ts.INK, ls="--", lw=0.9); ax[1].set_xticks(range(3)); ax[1].set_xticklabels([l for _, l in pm], fontsize=8.5); ax[1].set_ylabel("pair accuracy"); ax[1].set_ylim(0, 1.08); ts.panel(ax[1], "b")
fig.tight_layout(w_pad=1.6); ts.save(fig, "fig_table1_corrected_labels_v1")
