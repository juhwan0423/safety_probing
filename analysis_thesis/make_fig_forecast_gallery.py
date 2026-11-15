"""Qualitative forecasts: future roll predicted from the 2.7 s observation vs simulator ground truth, for held-out test
conditions whose physical configuration never appears in the training split."""
import json, collections, numpy as np, torch
import thesis_style as ts
from thesis_style import plt
from common import Context, ROOT, bs
ts.apply()
c = Context("artifacts/targets/safety_query_full_comfix_med5.pt"); t = c.tnp
man = {str(r["condition_id"]): r for r in c.canonical_rows}
cfg = [(float(man[i]["load_mass_kg"]), float(man[i]["load_com_offset_local_xyz_m"][1]), man[i]["route_id"]) for i in c.ids]
by = collections.defaultdict(list)
for n, k in enumerate(cfg): by[k].append(n)
unseen = np.array([c.split[n] == "test" and not any(c.split[j] == "train" for j in by[cfg[n]]) for n in range(len(cfg))])
P = []
for s in bs.QUERY_SEEDS:
    ck = torch.load(ROOT / f"outputs/safety_query/full_comfix_med5/seed_{s:04d}.pt", map_location="cpu", weights_only=False)
    P.append(ck["predictions_physical"]["future_state"].numpy()[:, :, 0])
P = np.stack(P); pred = np.abs(P.mean(0)); true = np.abs(t["future_state"][:, :, 0]); obs = np.abs(t["observed_state"][:, :, 0])
to, tf = np.array(c.targets["contract"]["observed_times_s"]), np.array(c.targets["contract"]["future_times_s"])
idx = np.flatnonzero(unseen & t["future_tip_mask"].astype(bool)); tip = t["future_tip"][idx] == 1
print("unseen-configuration test conditions:", len(idx), "tip", int(tip.sum()))
rng = np.random.default_rng(3)
pick = list(rng.choice(idx[tip], 4, replace=False)) + list(rng.choice(idx[~tip], 4, replace=False))
fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.6), sharex=True)
for ax, n in zip(axes.ravel(), pick):
    m = t["future_state_mask"][n, :, 0]
    ax.axvspan(0, 65 / 24, color="#E9EEF2", lw=0)
    ax.plot(to, obs[n], color=ts.INK, lw=1.0)
    ax.plot(tf[m], true[n][m], "-o", color=ts.REAL, ms=2.5, label="simulator")
    ax.plot(tf, pred[n], "--s", color=ts.PRED, ms=2.5, label="predicted from observation")
    r = man[c.ids[n]]; ev = r.get("failure_time_s")
    if ev: ax.axvline(ev, color=ts.TIP, lw=0.8, ls=":")
    ax.set_title(f"{r['load_mass_kg']:.0f} kg, CoM {r['load_com_offset_local_xyz_m'][1]:+.2f} m\n{r['speed_label']} / {r['steering_label']} / {r['turn_direction']}", fontsize=7.5)
    ax.set_ylim(-1, max(24, float(np.nanmax(true[n][m])) * 1.1 if m.any() else 24)); ax.tick_params(labelsize=7.5)
for ax in axes[1]: ax.set_xlabel("time (s)")
for ax in axes[:, 0]: ax.set_ylabel("|roll| (deg)")
axes[0, 0].legend(fontsize=7, loc="upper left")
fig.tight_layout(h_pad=0.8, w_pad=0.6); ts.save(fig, "fig_roll_forecast_gallery_v1")
err = np.abs(P[:, idx] - t["future_state"][idx, :, 0][None]); msk = t["future_state_mask"][idx, :, 0]
print("future roll MAE on these conditions: %.2f deg | tip AUROC: %.3f" % (float(np.mean([e[msk].mean() for e in err])), float(np.mean([bs.auc((t['future_tip'][idx] == 1).astype(int), torch.load(ROOT / f'outputs/safety_query/full_comfix_med5/seed_{s:04d}.pt', map_location='cpu', weights_only=False)['predictions_normalized']['future_tip_logits'].numpy()[idx]) for s in bs.QUERY_SEEDS]))))
