"""Thesis figures for ForkSafe-CF v2 (continuous payload and manoeuvre parameters).

fig_v2_identification : predicted vs true payload mass, lateral CoM, peak future roll and minimum ZMP margin (held-out pairs)
fig_v2_learning_curve : error vs number of training pairs (written when learning-curve runs exist)
Run from analysis_thesis/:
  PYTHONPATH=.:v2 ../external/LTX-2/.venv/bin/python v2/make_figures_v2.py --name p5 [--kin p5] [--curve p5_n1 p5_n2 ...]
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thesis_style as ts
from thesis_style import plt
from common_v2 import ContextV2

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--data", help="artifacts/v2/<dir> (default: derived from --name)")
ap.add_argument("--targets", default="targets_med5.pt"); ap.add_argument("--suffix", default="")
ap.add_argument("--curve", nargs="*", default=[]); ap.add_argument("--curve-cmd", nargs="*", default=[])
args = ap.parse_args()
ts.apply()
D = args.data or f"artifacts/v2/{args.name.split('_')[0]}"
c = ContextV2(f"{D}/{args.targets}", f"{D}/manifests")
z = np.load(f"out/v2_pred_{args.name}.npz", allow_pickle=True)
res = json.load(open(f"out/v2_{args.name}.json"))
te = np.flatnonzero(z["split"] == "test"); t = c.tnp
tip = t["future_tip"][te].astype(bool); tipm = t["future_tip_mask"][te].astype(bool)
col = np.where(tip, ts.TIP, ts.REC)

def scatter(ax, y, p, unit, title, lim=None, stats=None):
    ax.scatter(y[~tip], p[~tip], s=11, color=ts.REC, alpha=0.75, linewidths=0, label="recovered")
    ax.scatter(y[tip], p[tip], s=13, color=ts.TIP, alpha=0.75, linewidths=0, marker="^", label="tip-over")
    lo, hi = lim if lim else (min(y.min(), p.min()), max(y.max(), p.max()))
    pad = 0.04 * (hi - lo); ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=ts.INK, lw=0.8, ls="--")
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"true ({unit})"); ax.set_title(title, fontsize=9)
    r2 = 1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2)
    ax.text(0.04, 0.96, (stats + "\n" if stats else "") + f"$R^2$ = {r2:.2f}", transform=ax.transAxes, va="top", fontsize=8.5)
    return r2

fig, ax = plt.subplots(1, 4, figsize=(7.4, 2.35))
P = {k: z[k][:, te].mean(0) for k in ("static", "future_state", "future_summary")}
r = {}
r["mass"] = scatter(ax[0], t["static"][te, 0], P["static"][:, 0], "kg", "payload mass", (240, 720), f"MAE {res['mass_mae_kg']['point']:.0f} kg")
r["com"] = scatter(ax[1], t["static"][te, 1] * 100, P["static"][:, 1] * 100, "cm", "payload lateral CoM offset", (-25, 25), f"MAE {res['com_mae_m']['point'] * 100:.1f} cm")
fm = t["future_state_mask"][te, :, 0].astype(bool)
true_roll = np.where(fm, np.abs(t["future_state"][te, :, 0]), 0).max(1); pred_roll = np.abs(P["future_state"][:, :, 0]).max(1)
r["roll"] = scatter(ax[2], true_roll, pred_roll, "deg", "peak future roll", None, f"MAE {res['roll_mae_deg']['point']:.1f} deg")
sm = t["future_summary_mask"][te, 0].astype(bool)
r["zmp"] = scatter(ax[3], t["future_summary"][te, 0][sm] * 100, P["future_summary"][:, 0][sm] * 100, "cm", "minimum ZMP margin", None, f"MAE {res['zmp_mae_m']['point'] * 100:.1f} cm") if False else None
y, p = t["future_summary"][te, 0] * 100, P["future_summary"][:, 0] * 100
ax[3].scatter(y[sm & ~tip], p[sm & ~tip], s=11, color=ts.REC, alpha=0.75, linewidths=0); ax[3].scatter(y[sm & tip], p[sm & tip], s=13, color=ts.TIP, alpha=0.75, linewidths=0, marker="^")
lo, hi = min(y[sm].min(), p[sm].min()), max(y[sm].max(), p[sm].max()); pad = 0.04 * (hi - lo)
ax[3].plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=ts.INK, lw=0.8, ls="--"); ax[3].set_xlim(lo - pad, hi + pad); ax[3].set_ylim(lo - pad, hi + pad); ax[3].set_aspect("equal", adjustable="box")
r["zmp"] = 1 - np.sum((p[sm] - y[sm]) ** 2) / np.sum((y[sm] - y[sm].mean()) ** 2)
ax[3].set_xlabel("true (cm)"); ax[3].set_title("minimum ZMP margin", fontsize=9); ax[3].text(0.04, 0.96, f"MAE {res['zmp_mae_m']['point'] * 100:.1f} cm\n$R^2$ = {r['zmp']:.2f}", transform=ax[3].transAxes, va="top", fontsize=8.5)
ax[0].set_ylabel("predicted from the observed video")
handles, labels = ax[0].get_legend_handles_labels()
fig.tight_layout(w_pad=0.5, rect=(0, 0, 1, 0.92)); fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 1.0), handletextpad=0.2, columnspacing=1.5)
ts.save(fig, f"fig_v2_identification{args.suffix}")
auc = res.get("tip_auroc", {}).get("point", float("nan"))
print(f"{args.name}: test n={len(te)} tip rate {tip[tipm].mean():.2f} AUROC {auc:.3f} | R2 mass {r['mass']:.3f} com {r['com']:.3f} peak roll {r['roll']:.3f} zmp {r['zmp']:.3f}")
print("baseline", res["train_mean_baseline"], "counts", res["counts"])

if args.curve:
    fig, ax = plt.subplots(1, 4, figsize=(7.4, 2.3))
    keys = [("mass_mae_kg", "payload mass MAE (kg)", 1), ("com_mae_m", "lateral CoM MAE (cm)", 100), ("roll_mae_deg", "future roll MAE (deg)", 1), ("tip_auroc", "tip-over AUROC", 1)]
    for names, colr, mk, lab in ((args.curve, ts.GEN, "s", "video only"), (args.curve_cmd, ts.PRED, "o", "video + planned command")):
        R = [json.load(open(f"out/v2_{n}.json")) for n in names if os.path.exists(f"out/v2_{n}.json")]
        if not R: continue
        x = [q["counts"]["train"] // 2 for q in R]
        for a, (k, lab_y, s) in zip(ax, keys):
            yv = np.array([q[k]["point"] * s for q in R]); lo = np.array([q[k]["ci_low"] * s for q in R]); hi = np.array([q[k]["ci_high"] * s for q in R])
            a.fill_between(x, lo, hi, color=colr, alpha=0.13, lw=0); a.plot(x, yv, "-", color=colr, marker=mk, ms=3.8, label=lab)
            a.set_title(lab_y, fontsize=9); a.set_xlabel("training pairs")
    base = res["train_mean_baseline"]
    for a, (k, _, s) in zip(ax[:3], keys[:3]):
        a.axhline(base[k] * s, color=ts.INK, ls=":", lw=0.9, label="train-mean baseline"); a.set_ylim(0, None)
    handles, labels = ax[0].get_legend_handles_labels()
    fig.tight_layout(w_pad=0.9, rect=(0, 0, 1, 0.9)); fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    ts.save(fig, f"fig_v2_learning_curve{args.suffix}")
