import json, sys, numpy as np
from pathlib import Path
import comfix as cf
man = cf.load_manifest()
want = sys.argv[1] if len(sys.argv) > 1 else "confirmed_recovered"
# pick a mid-speed recovered run with zero CoM offset
cands = sorted([(p, r) for p, r in man.items() if r["safety_outcome"] == want and r["load_com_offset_local_xyz_m"][1] == 0.0 and r["load_mass_kg"] == 480.0], key=lambda x: -x[1]["max_roll_deg"])
path, row = cands[0]; print(row["run_id"], "max roll", round(row["max_roll_deg"], 2), "n cands", len(cands))
rows = cf.read_rows(Path(path)); C = cf.correct(rows, 480.0, np.zeros(3)); polys = cf.polygons(rows)
valid = np.array([str(r.get("outward_lateral_margin_valid", "")).lower() in ("true", "1") for r in rows])
F = np.array([[cf._f(r.get(b + "_force_z")) for b in cf.CONTACT_BODIES] for r in rows])
a_lat = cf.lateral(C["yaw"], C["a"][:, :2]); v = np.hypot(*np.array([[cf._f(r["linear_vel_x"]), cf._f(r["linear_vel_y"])] for r in rows]).T)
wz = np.array([cf._f(r["angular_vel_z"]) for r in rows])
print("polygon example (world):", polys[150]); print("com xy", C["p"][150, :2].round(3), "zmp", C["zmp"][150].round(3), "cop", C["cop"][150].round(3), "dir", C["dir"][150].round(3), "fork", C["fork"][150].round(3), "yaw deg", np.degrees(C["yaw"][150]).round(1))
print("   t   v    wz   v*wz  a_lat(fd)  a_z(fd)  roll  nC  valid | Fz: back  rbl   rbr   rfl   rfr | zmp_margin  cop_margin  com_margin")
for i in range(150, 360, 6):
    cm = cf.directional_support_margin(C["cop"][i], polys[i], C["dir"][i]) if polys[i] is not None and np.all(np.isfinite(C["dir"][i])) else None
    print(f"{C['t'][i]:5.2f} {v[i]:4.2f} {wz[i]:+5.2f} {v[i]*wz[i]:+5.2f}  {a_lat[i]:+8.2f} {C['a'][i,2]:+8.2f} {C['roll'][i]:+6.2f} {int(C['contacts'][i]) if np.isfinite(C['contacts'][i]) else -1:2d} {int(valid[i])}     | " + " ".join(f"{x:6.0f}" for x in F[i]) + f" | {C['zmp_margin'][i]:+8.3f}  {('%+8.3f' % cm) if cm is not None else '   None '}  {C['com_margin'][i]:+8.3f}")
fut = valid & (C["t"] > 65 / 24)
print("future valid rows", fut.sum(), "| a_lat(fd) std", np.std(a_lat[fut]).round(2), "median", np.median(a_lat[fut]).round(2), "| v*wz median", np.median((v * wz)[fut]).round(2))
print("zmp margin future: min", np.nanmin(C["zmp_margin"][fut]).round(3), "median", np.nanmedian(C["zmp_margin"][fut]).round(3), "max", np.nanmax(C["zmp_margin"][fut]).round(3))
