"""Qualitative roll forecasts on ForkSafe-CF v2: |roll| predicted from the 2.7 s observation vs the simulator, for runs
predicted out of fold (pair-grouped cross-validation, psq_v2_cv.py). Every run has its own command and payload.
Picks runs at fixed quantiles of the per-run error (best to worst) for tip-over and recovered runs, so the gallery
shows typical and bad cases rather than hand-picked ones.
Run from analysis_thesis/:  PYTHONPATH=.:v2 ../external/LTX-2/.venv/bin/python v2/make_fig_v2_gallery.py p5
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thesis_style as ts
from thesis_style import plt
from common_v2 import ContextV2

name = sys.argv[1] if len(sys.argv) > 1 else "p5"
D = f"artifacts/v2/{name.split('_')[0]}"
ts.apply()
c = ContextV2(f"{D}/targets_truecom.pt", f"{D}/manifests"); t = c.tnp
z = np.load(f"out/v2cv_pred_{name}.npz", allow_pickle=True)
pred = np.abs(z["future_state"][:, :, :, 0].mean(0)); true = np.abs(t["future_state"][:, :, 0]); obs = np.abs(t["observed_state"][:, :, 0])
mask = t["future_state_mask"][:, :, 0].astype(bool)
to, tf = np.array(c.targets["contract"]["observed_times_s"]), np.array(c.targets["contract"]["future_times_s"])
err = np.array([np.abs(pred[n] - true[n])[mask[n]].mean() if mask[n].any() else np.nan for n in range(len(c.ids))])
tipm = t["future_tip_mask"].astype(bool); tip = t["future_tip"] == 1
pick = []
for sel in (tipm & tip, tipm & ~tip):
    idx = np.flatnonzero(sel & np.isfinite(err)); order = idx[np.argsort(err[idx])]
    pick += [int(order[int(q * (len(order) - 1))]) for q in (0.1, 0.5, 0.8, 0.95)]
fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.6), sharex=True)
for ax, n, q in zip(axes.ravel(), pick, ["10%", "median", "80%", "95%"] * 2):
    m = mask[n]
    ax.axvspan(0, 65 / 24, color="#E9EEF2", lw=0)
    ax.plot(to, obs[n], color=ts.INK, lw=1.0)
    ax.plot(tf[m], true[n][m], "-o", color=ts.REAL, ms=2.5, label="simulator")
    ax.plot(tf, pred[n], "--s", color=ts.PRED, ms=2.5, label="predicted from the video")
    r = c.man[c.ids[n]]
    ax.set_title(f"{r['load_mass_kg']:.0f} kg, drive {abs(r['drive_velocity_command']):.1f}, steer {abs(r['steering_position_command']):.2f}\nerror {err[n]:.1f} deg ({q} of runs)", fontsize=7.5)
    ax.set_ylim(-1, max(24, float(np.nanmax(np.r_[true[n][m], pred[n]])) * 1.1)); ax.tick_params(labelsize=7.5)
for ax in axes[1]: ax.set_xlabel("time (s)")
axes[0, 0].set_ylabel("tip-over runs\n|roll| (deg)"); axes[1, 0].set_ylabel("recovered runs\n|roll| (deg)")
axes[0, 0].legend(fontsize=7, loc="upper left")
fig.tight_layout(h_pad=0.8, w_pad=0.6); ts.save(fig, f"fig_v2_roll_forecast_gallery_{name}")
print(f"{name}: runs {len(c.ids)}; per-run roll error median {np.nanmedian(err):.2f} deg, mean {np.nanmean(err):.2f}; tip-over runs median {np.nanmedian(err[tipm & tip]):.2f}, recovered {np.nanmedian(err[tipm & ~tip]):.2f}")
