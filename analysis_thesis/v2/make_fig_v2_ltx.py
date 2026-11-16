"""Thesis figure: audit of generated futures on ForkSafe-CF v2 (held-out test pairs, fixed split).

Bars: prediction from the observed video (V-JEPA readout, psq_v2.py) | readout of the REAL future (LTX latents) |
the same readout applied to GENERATED futures (LTX continuations, mean over rollouts) | train-mean baseline.
Panels: payload mass MAE, future roll MAE, tip-over AUROC, pairwise mass direction.
Needs out/v2ltx_<name>.json (ltx_v2.py) and out/v2_<name>.json.
Run from analysis_thesis/:  PYTHONPATH=.:v2 ../external/LTX-2/.venv/bin/python v2/make_fig_v2_ltx.py p5
"""
import json, sys
import numpy as np
import thesis_style as ts
from thesis_style import plt

name = sys.argv[1] if len(sys.argv) > 1 else "p5"
ts.apply()
L = json.load(open(f"out/v2ltx_{name}.json")); V = json.load(open(f"out/v2_{name}.json")); base = L["train_mean_baseline"]
rows = [("observed\nvideo", V, ts.PRED), ("real\nfuture", L["real_future"], ts.REAL), ("generated\nfuture", L["generated_future"], ts.GEN)]
panels = [("mass_mae_kg", "payload mass MAE (kg)", base["mass_mae_kg"], "{:.0f}"), ("roll_mae_deg", "future roll MAE (deg)", base["roll_mae_deg"], "{:.1f}"),
          ("tip_auroc", "tip-over AUROC", None, "{:.3f}"), ("mass_direction", "pairs differing in mass:\nwhich payload is heavier?", 0.5, "{:.2f}")]
fig, axes = plt.subplots(1, 4, figsize=(8.4, 2.9))
for a, (key, title, ref, fmt) in zip(axes, panels):
    v = np.array([r[key]["point"] for _, r, _ in rows]); lo = np.array([r[key]["ci_low"] for _, r, _ in rows]); hi = np.array([r[key]["ci_high"] for _, r, _ in rows]); x = np.arange(len(rows))
    a.bar(x, v, 0.66, color=[c for _, _, c in rows], yerr=[v - lo, hi - v], error_kw=dict(lw=0.7, capsize=1.8, ecolor=ts.INK))
    top = max(hi.max(), ref or 0)
    for xi, vi, hi_i in zip(x, v, hi): a.text(xi, hi_i + 0.02 * top, fmt.format(vi), ha="center", va="bottom", fontsize=7.5)
    if ref is not None: a.axhline(ref, color=ts.INK, ls=":" if key != "mass_direction" else "--", lw=0.9)
    a.set_xticks(x); a.set_xticklabels([n for n, _, _ in rows], fontsize=7.6); a.set_title(title, fontsize=9)
    a.set_ylim(0.5 if key == "tip_auroc" else (0.4 if key == "mass_direction" else 0), None if key in ("mass_mae_kg", "roll_mae_deg") else 1.08)
axes[0].plot([], [], color=ts.INK, ls=":", lw=0.9, label="train-mean baseline"); axes[0].legend(fontsize=7, loc="center left", bbox_to_anchor=(0.0, 0.62))
fig.tight_layout(w_pad=0.9); ts.save(fig, f"fig_v2_generated_audit_{name}")
for n, r, _ in rows: print(n.replace("\n", " "), {k: round(r[k]["point"], 3) for k in ("mass_mae_kg", "roll_mae_deg", "tip_auroc", "mass_direction", "com_direction", "gold_zmp_ordering") if k in r})
print("baseline", {k: round(v, 2) for k, v in base.items()}, "| rollout seeds", L["rollout_seeds"], "| counts", L["counts"])
