import json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T_OBS = 65 / 24.0
d = json.load(open("out/dense_table.json"))
tip = [r for r in d if r["outcome"] == "observed_tip"]
res = {}
for name, rows in (("all", tip), ("test", [r for r in tip if r["split"] == "test"])):
    sl = np.array([r["t_support_loss"] - T_OBS for r in rows if r["t_support_loss"] is not None])
    rt = np.array([r["t_roll_tip"] - T_OBS for r in rows if r["t_roll_tip"] is not None])
    q = lambda x: dict(n=int(len(x)), min=float(x.min()), p05=float(np.quantile(x, .05)), median=float(np.median(x)), mean=float(x.mean()), p95=float(np.quantile(x, .95)), max=float(x.max()))
    res[name] = dict(support_loss=q(sl), roll20=q(rt))
    print(name, json.dumps(res[name]))
print("roll at obs end, max over tip conds:", max(r["roll_obs_max"] for r in tip))
by = {}
for r in tip:
    if r["t_support_loss"] is not None: by.setdefault(r["speed_label"], []).append(r["t_support_loss"] - T_OBS)
print({k: (len(v), round(float(np.median(v)), 2), round(float(np.min(v)), 2)) for k, v in by.items()})
res["by_speed"] = {k: dict(n=len(v), median=float(np.median(v)), min=float(np.min(v))) for k, v in by.items()}
json.dump(res, open("out/lead_time.json", "w"), indent=1)
sl = np.array([r["t_support_loss"] - T_OBS for r in tip if r["t_support_loss"] is not None])
rt = np.array([r["t_roll_tip"] - T_OBS for r in tip if r["t_roll_tip"] is not None])
fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=200)
bins = np.arange(0, 2.35, 0.1)
ax.hist(sl, bins=bins, color="#2a78d6", alpha=.85, label=f"support loss (n={len(sl)})")
ax.hist(rt, bins=bins, color="#d6542a", alpha=.6, label=f"roll > 20 deg (n={len(rt)})")
ax.axvline(np.median(sl), color="#2a78d6", ls="--", lw=1); ax.axvline(np.median(rt), color="#d6542a", ls="--", lw=1)
ax.set_xlabel("time after end of observation (s)"); ax.set_ylabel("tip-over conditions"); ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig("out/lead_time_hist.png")
