import json, numpy as np
from common import bs
d = json.load(open("out/dense_table.json"))
A = lambda k: np.array([r[k] for r in d], dtype=float)
split = np.array([r["split"] for r in d]); tip = A("tip"); ok = tip >= 0
m, yl = A("load_mass"), A("load_com_y"); M, cy, cz, cx = A("sys_mass"), A("com_y0"), A("com_z0"), A("com_x0")
tr = (split == "train")
print("vehicle mass:", np.unique(np.round(M - m, 2))[:5])
# rigid-body composition fits on train
for name, tgt, X in (("M*cy", M * cy, np.c_[np.ones_like(m), m, m * yl]), ("M*cz", M * cz, np.c_[np.ones_like(m), m]), ("M*cx", M*cx, np.c_[np.ones_like(m), m])):
    w, *_ = np.linalg.lstsq(X[tr], tgt[tr], rcond=None); print(name, w, "max resid/M", np.abs(X @ w - tgt).max() / M.mean())
turn = np.array([1.0 if r["turn"] == "right" else -1.0 for r in d])
print("corr(load_com_y, com_y0)", np.corrcoef(yl, cy)[0, 1], "com_y0 range", cy.min(), cy.max(), "cz range", cz.min(), cz.max())
g = 9.81
routes = sorted({r["route"] for r in d}); print(len(routes), "routes")
for dem_name in ("ay_obs_max", "ay_full_max", "speed_obs_end", "wz_obs_end"):
    a = np.abs(A(dem_name))
    for sgn in (+1, -1):
        s = cz * a / g + sgn * turn * cy
        print(dem_name, sgn, "train AUROC", round(bs.auc(tip[tr & ok].astype(int), s[tr & ok]), 3), "test", round(bs.auc(tip[(split == "test") & ok].astype(int), s[(split == "test") & ok]), 3),
              "| demand only test", round(bs.auc(tip[(split == "test") & ok].astype(int), (a)[(split == "test") & ok]), 3))
# per-route spread of obs-window demand
import collections
by = collections.defaultdict(list)
for r in d: by[r["route"]].append((r["ay_obs_max"], r["ay_full_max"], r["tip"], r["split"]))
for k in routes[:6]: v = np.array([(x[0], x[1], x[2]) for x in by[k]]); print(k, len(v), "ay_obs", v[:, 0].min().round(2), v[:, 0].max().round(2), "ay_full(rec)", np.round(v[v[:, 2] == 0, 1].mean() if (v[:, 2] == 0).any() else np.nan, 2), "tip rate", v[:, 2].clip(0).mean().round(2))
print({k: sorted({x[3] for x in by[k]}) for k in routes})
