"""Independent check of the logged system CoM: centre of pressure from contact forces (static, early rows)."""
import json, csv, numpy as np, collections
from common import Context
c = Context(); rows = {str(r["condition_id"]): r for r in c.canonical_rows}
B = ["back_wheel", "roller_back_left", "roller_back_right", "roller_front_left", "roller_front_right"]
f = lambda x: float(x) if x not in ("", "nan", "None") else np.nan
out = []
rng = np.random.default_rng(0)
for cid in rng.choice(c.ids, 240, replace=False):
    r = rows[cid]; L = []
    with open(r["dense_log_path"]) as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            if i >= 30: break
            L.append(row)
    res = []
    for row in L[8:30]:
        F = np.array([f(row[b + "_force_z"]) for b in B]); P = np.array([[f(row[b + "_contact_pos_x"]), f(row[b + "_contact_pos_y"])] for b in B])
        ok = np.isfinite(F) & np.isfinite(P).all(1) & (np.abs(F) > 1e-6)
        if ok.sum() < 3: continue
        cop = (np.abs(F[ok])[:, None] * P[ok]).sum(0) / np.abs(F[ok]).sum()
        yaw = np.radians(f(row["yaw_deg"])); d = cop - np.array([f(row["system_com_x"]), f(row["system_com_y"])])
        lat = -np.sin(yaw) * d[0] + np.cos(yaw) * d[1]                      # CoP minus logged CoM, chassis-lateral
        res.append((lat, np.abs(F[ok]).sum()))
    if res:
        a = np.array(res); out.append((float(r["load_mass_kg"]), float(r["load_com_offset_local_xyz_m"][1]), a[:, 0].mean(), a[:, 1].mean()))
o = np.array(out); print("n", len(o), "mean total |Fz| / (M g):", np.mean(o[:, 3] / ((1590.82 + o[:, 0]) * 9.81)).round(3))
g = collections.defaultdict(list)
for m, y, lat, _ in o: g[(m, y)].append(lat)
print("mass  offset  CoP-loggedCoM lateral (m)   expected (m/M)*offset")
for k in sorted(g): print(f"{k[0]:5.0f} {k[1]:+.2f}   {np.mean(g[k]):+.4f} ± {np.std(g[k]):.4f}   {k[0] / (1590.82 + k[0]) * k[1]:+.4f}   n={len(g[k])}")
json.dump(out, open("out/cop_check.json", "w"))
