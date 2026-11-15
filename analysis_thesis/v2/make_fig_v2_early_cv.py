"""Thesis figure: how early can the future be predicted on ForkSafe-CF v2? (pair-grouped 5-fold CV, all runs out of fold)

Observation truncated to 1.0 / 1.5 / 1.96 / 2.33 / 2.71 s; the steering pulse starts at 2.0 s; the future targets are
the same for every length. Lines:
  video only                      the readout alone (psq_v2_cv.py)
  command + mass from the video   two-stage predictor with the mass read from the truncated video (plugin_mass_v2.py)
References: planned command only, and command + true mass (payload_value_v2.py).
Panels: tip-over AUROC, future roll MAE, accuracy on pairs that differ in mass and end differently.
Run from analysis_thesis/:  PYTHONPATH=.:v2 ../external/LTX-2/.venv/bin/python v2/make_fig_v2_early_cv.py p5
"""
import json, sys
import numpy as np
import thesis_style as ts
from thesis_style import plt

name = sys.argv[1] if len(sys.argv) > 1 else "p5"
ts.apply()
FRAMES = [24, 36, 47, 56, 65]; T = np.array([f / 24.0 for f in FRAMES]); tag = lambda f: name if f == 65 else f"{name}_obs{f}"
VID = [json.load(open(f"out/v2cv_{tag(f)}.json")) for f in FRAMES]; PL = json.load(open(f"out/v2_plugin_mass_{name}.json")); PV = json.load(open(f"out/v2_payload_value_{name}_cv.json"))
TWO = [PL[f"video mass ({tag(f)})"] for f in FRAMES]
panels = [("tip_auroc", "tip-over AUROC", (0.5, 1.005)), ("roll_mae_deg", "future roll MAE (deg)", (0, None)), ("outcome_flip_mass", f"pairs differing in mass ({VID[-1]['n_outcome_flip_mass']}):\nwhich run tips?", (0.4, 1.03))]
fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9))
for a, (key, title, ylim) in zip(axes, panels):
    for res, col, mk, lab in ((VID, ts.GEN, "s", "video only"), (TWO, ts.REAL, "o", "planned command + mass read from the video")):
        y = np.array([r[key]["point"] for r in res]); lo = np.array([r[key]["ci_low"] for r in res]); hi = np.array([r[key]["ci_high"] for r in res])
        a.fill_between(T, lo, hi, color=col, alpha=0.13, lw=0); a.plot(T, y, "-", color=col, marker=mk, ms=3.8, label=lab)
    a.axhline(PV["command"][key]["point"], color=ts.BASE, ls="--", lw=1.0, label="planned command only")
    a.axhline(PV["command+mass"][key]["point"], color=ts.INK, ls=":", lw=1.0, label="planned command + true mass")
    a.axvline(2.0, color=ts.TIP, lw=0.8, alpha=0.7); a.set_xlabel("observed video length (s)"); a.set_title(title, fontsize=9); a.set_xticks([1.0, 1.5, 2.0, 2.5]); a.set_ylim(*ylim)
axes[0].text(2.03, 0.52, "steering\nstarts", fontsize=7.5, color=ts.TIP, va="bottom")
handles, labels = axes[0].get_legend_handles_labels()
fig.tight_layout(w_pad=1.0, rect=(0, 0, 1, 0.9)); fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=7.8, handlelength=1.6, columnspacing=1.2, bbox_to_anchor=(0.5, 1.0))
ts.save(fig, f"fig_v2_early_prediction_cv_{name}")
for f, v, w in zip(FRAMES, VID, TWO):
    g = lambda r, k: f"{r[k]['point']:.3f}"
    print(f"obs {f / 24:.2f} s | video only: AUROC {g(v, 'tip_auroc')} roll {g(v, 'roll_mae_deg')} mass MAE {g(v, 'mass_mae_kg')} mass pairs {g(v, 'outcome_flip_mass')} | two-stage: AUROC {g(w, 'tip_auroc')} roll {g(w, 'roll_mae_deg')} mass pairs {g(w, 'outcome_flip_mass')}")
print("command only:", {k: round(PV['command'][k]['point'], 3) for k in ('tip_auroc', 'roll_mae_deg', 'outcome_flip_mass')}, "| command + true mass:", {k: round(PV['command+mass'][k]['point'], 3) for k in ('tip_auroc', 'roll_mae_deg', 'outcome_flip_mass')})
