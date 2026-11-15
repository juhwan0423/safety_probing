"""Thesis figure: what decides tip-over in ForkSafe-CF v2 (continuous payload and manoeuvre parameters)?

(a) planned command plane (speed command x steering command): outcomes overlap in a band -> the command alone cannot decide
(b) inside that band: payload mass x lateral CoM offset towards the outside of the turn -> the hidden payload decides
(c) tip-over AUROC of logistic models with command only / command + true payload (5-fold), i.e. the head-room for payload identification
Needs out/rollover_physics_v2_<tag>.json (rollover_physics.py with COMFIX_MANIFEST / ROLLOVER_OUT).
Run from analysis_thesis/:  PYTHONPATH=.:v2 ../external/LTX-2/.venv/bin/python v2/make_fig_v2_dataset.py p5
"""
import json, sys
import numpy as np
import thesis_style as ts
from thesis_style import plt
sys.path.insert(0, "/home/jude/CWM_paper/Safety_Probing_paper/scripts")
import build_statistics as bs

tag = sys.argv[1] if len(sys.argv) > 1 else "p5"
ts.apply()
R = json.load(open(f"out/rollover_physics_v2_{tag}.json"))["runs"]
A = lambda k: np.array([r[k] for r in R])
definite = A("outcome") != "right_censored"; tip = A("outcome") == "observed_tip"
drive, steer = np.abs(A("drive_cmd").astype(float)), np.abs(A("steer_cmd").astype(float)); sgn = np.sign(A("steer_cmd").astype(float))
mass, com_out = A("mass"), -sgn * A("com")            # lateral CoM offset towards the outside of the turn (left turn: outside = -y)
# sign check: a CoM shifted towards the outside must shorten the stability arm d
slope = np.polyfit(com_out, A("d"), 1)[0]
if slope > 0: com_out = -com_out
print(f"runs {len(R)}, definite {definite.sum()}, tip rate {tip[definite].mean():.2f}; d vs outward CoM slope {np.polyfit(com_out, A('d'), 1)[0]:+.2f}")

def cv_logit(X, y, folds=5, seed=0):
    X = (X - X.mean(0)) / (X.std(0) + 1e-9); X = np.c_[X, np.ones(len(X))]
    idx = np.random.default_rng(seed).permutation(len(y)); score = np.zeros(len(y))
    for f in range(folds):
        te = idx[f::folds]; tr = np.setdiff1d(idx, te); w = np.zeros(X.shape[1])
        for _ in range(4000):
            p = 1 / (1 + np.exp(-X[tr] @ w)); w -= 0.2 * (X[tr].T @ (p - y[tr]) / len(tr) + 1e-3 * w)
        score[te] = X[te] @ w
    return score

y = tip[definite].astype(int)
Xc = np.c_[drive, steer, drive * steer, drive ** 2][definite]
Xp = np.c_[Xc, mass[definite], com_out[definite], (mass * com_out)[definite], A("d")[definite]]
s_cmd, s_pay = cv_logit(Xc, y), cv_logit(Xp, y)
auc_cmd, auc_pay = bs.auc(y, s_cmd), bs.auc(y, s_pay)
p_cmd = 1 / (1 + np.exp(-s_cmd)); band = (p_cmd > 0.2) & (p_cmd < 0.8)
print(f"AUROC command only {auc_cmd:.3f} | command + true payload {auc_pay:.3f} | undecided band (0.2 < p < 0.8): {band.sum()} runs ({band.mean():.0%}), AUROC there cmd {bs.auc(y[band], s_cmd[band]):.3f} payload {bs.auc(y[band], s_pay[band]):.3f}")

fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.7), gridspec_kw={"width_ratios": [1, 1, 0.62]})
for sel, col, mk, lab in ((definite & ~tip, ts.REC, "o", "recovered"), (definite & tip, ts.TIP, "^", "tip-over")):
    ax[0].scatter(drive[sel], steer[sel], s=9, color=col, alpha=0.6, marker=mk, linewidths=0, label=lab)
ax[0].set_xlabel("drive command magnitude"); ax[0].set_ylabel("steering command magnitude"); ax[0].set_title("planned manoeuvre", fontsize=9); ts.panel(ax[0], "a")
di = np.flatnonzero(definite)[band]
for sel, col, mk in ((~tip[di], ts.REC, "o"), (tip[di], ts.TIP, "^")):
    ax[1].scatter(mass[di][sel], com_out[di][sel] * 100, s=11, color=col, alpha=0.7, marker=mk, linewidths=0)
ax[1].set_xlabel("payload mass (kg)"); ax[1].set_ylabel("lateral CoM offset towards\nthe outside of the turn (cm)"); ax[1].set_title("hidden payload, undecided manoeuvres", fontsize=9); ts.panel(ax[1], "b")
ax[2].bar([0, 1], [auc_cmd, auc_pay], color=[ts.BASE, ts.PRED], width=0.62)
for x, v in ((0, auc_cmd), (1, auc_pay)): ax[2].text(x, v + 0.008, f"{v:.2f}", ha="center", fontsize=9)
ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(["command\nonly", "command +\ntrue payload"]); ax[2].set_ylim(0.5, 1.03); ax[2].set_ylabel("tip-over AUROC"); ax[2].set_title("what decides the outcome", fontsize=9); ts.panel(ax[2], "c")
handles, labels = ax[0].get_legend_handles_labels()
fig.tight_layout(w_pad=1.0, rect=(0, 0, 1, 0.92)); fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 1.0), handletextpad=0.2)
ts.save(fig, f"fig_v2_dataset_{tag}")
json.dump({"n_runs": len(R), "n_definite": int(definite.sum()), "tip_rate": float(tip[definite].mean()), "auroc_command_only": float(auc_cmd), "auroc_command_plus_payload": float(auc_pay),
           "band_runs": int(band.sum()), "band_auroc_command_only": float(bs.auc(y[band], s_cmd[band])), "band_auroc_command_plus_payload": float(bs.auc(y[band], s_pay[band]))},
          open(f"out/v2_dataset_summary_{tag}.json", "w"), indent=1)
