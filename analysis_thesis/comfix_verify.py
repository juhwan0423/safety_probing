import json, collections, numpy as np
from pathlib import Path
import comfix as cf

man = cf.load_manifest()
rep = collections.defaultdict(float)          # max abs replication errors
per_run = []
for path, row in man.items():
    rows = cf.read_rows(Path(path))
    m, off = float(row["load_mass_kg"]), np.asarray(row["load_com_offset_local_xyz_m"], float)
    C = cf.correct(rows, m, off)
    k = np.arange(len(rows)) >= 1
    def mx(name, a, b, sel):
        d = np.abs(a - b)[sel]
        d = d[np.isfinite(d)]
        if d.size: rep[name] = max(rep[name], float(d.max()))
    mx("vel", C["v_rep"], C["v"], k); mx("acc", C["a_rep"], C["a"], k)
    mx("zmp", C["zmp_rep"], C["zmp"], np.isfinite(C["zmp"]).all(1))
    mx("zmp_margin", C["zmp_margin_rep"], C["zmp_margin"], np.isfinite(C["zmp_margin"]))
    mx("com_margin", C["com_margin_rep"], C["com_margin"], np.isfinite(C["com_margin"]))
    ok = (np.isfinite(C["zmp"]).all(1) & np.isfinite(C["cop"]).all(1) & (C["n_force"] >= 3) & (np.abs(C["roll"]) < 5.0)
          & (C["t"] > 0.3) & (np.abs(C["fz"] / (C["M"] * cf.G) - 1.0) < 0.2))
    if ok.sum() < 20: continue
    r0 = cf.lateral(C["yaw"], C["cop"] - C["zmp"])[ok]; r1 = cf.lateral(C["yaw"], C["cop"] - C["zmp2"])[ok]
    dl = cf.lateral(C["yaw"], C["delta"][:, :2])[ok]
    per_run.append(dict(mass=m, off=float(off[1]), n=int(ok.sum()), r0_mean=float(r0.mean()), r1_mean=float(r1.mean()),
                        r0_rms=float(np.sqrt((r0 ** 2).mean())), r1_rms=float(np.sqrt((r1 ** 2).mean())),
                        r0_med=float(np.median(r0)), r1_med=float(np.median(r1)), delta_lat=float(dl.mean()),
                        expected=m / (1590.82 + m) * float(off[1])))
print("replication max abs error:", {k: f"{v:.2e}" for k, v in rep.items()})
P = per_run; print("runs used", len(P))
A = lambda k: np.array([p[k] for p in P])
print("delta lateral vs expected (m/M)*offset: max abs diff", np.abs(A("delta_lat") - A("expected")).max())
print("\nmass offset |  CoP-ZMP lateral median:  logged   corrected | n")
cells = collections.defaultdict(list)
for p in P: cells[(p["mass"], p["off"])].append(p)
for key in sorted(cells):
    c = cells[key]; print(f"{key[0]:5.0f} {key[1]:+.2f} | {np.median([x['r0_med'] for x in c]):+.4f}  {np.median([x['r1_med'] for x in c]):+.4f} | {len(c)}")
nz = A("off") != 0
for name, key in (("logged", "r0_med"), ("corrected", "r1_med")):
    y = A(key); x = A("expected")
    slope, icpt = np.polyfit(x[nz], y[nz], 1)
    print(f"{name:9s}: slope of residual on expected shift = {slope:+.3f}, intercept {icpt:+.4f}; |median residual| over offset runs: {np.mean(np.abs(y[nz])):.4f}, zero-offset runs: {np.mean(np.abs(y[~nz])):.4f}")
print("RMS residual (offset runs): logged", float(np.sqrt(np.mean(A('r0_rms')[nz] ** 2))), "corrected", float(np.sqrt(np.mean(A('r1_rms')[nz] ** 2))))
json.dump(dict(replication=dict(rep), per_run=P), open("out/comfix_verify.json", "w"))
