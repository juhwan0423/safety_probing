"""Compare three definitions of the future minimum outward stability margin per run:
 A  logged   : ZMP from finite-difference CoM acceleration (paper label)
 A' corrected: same formula, CoM corrected for the payload offset
 B  CoP      : centre of pressure from contact forces (force-based ZMP), same polygon / outward direction
"""
import json, collections, numpy as np
from pathlib import Path
import comfix as cf
T_OBS = 65 / 24.0
man = cf.load_manifest(); out = []
for path, row in man.items():
    rows = cf.read_rows(Path(path))
    m, off = float(row["load_mass_kg"]), np.asarray(row["load_com_offset_local_xyz_m"], float)
    C = cf.correct(rows, m, off); polys = cf.polygons(rows)
    valid = np.array([str(r.get("outward_lateral_margin_valid", "")).lower() in ("true", "1") for r in rows])
    fut = valid & (C["t"] > T_OBS)
    cop_margin = np.full(len(rows), np.nan)
    for i in np.flatnonzero(fut):
        if polys[i] is None or not np.all(np.isfinite(C["cop"][i])) or C["n_force"][i] < 2: continue
        v = cf.directional_support_margin(C["cop"][i], polys[i], C["dir"][i])
        if v is not None: cop_margin[i] = v
    def mn(x):
        x = x[fut]; x = x[np.isfinite(x)]
        return float(x.min()) if x.size else None
    def mn_med5(x):                      # min after a 5-sample running median (spike-robust)
        idx = np.flatnonzero(fut & np.isfinite(x))
        if idx.size < 5: return None
        y = x[idx]; return float(min(np.median(y[j:j + 5]) for j in range(len(y) - 4)))
    out.append(dict(cid=row["condition_id"], outcome=row["safety_outcome"], mass=m, off=float(off[1]), n_valid=int(fut.sum()),
                    A=mn(C["zmp_margin"]), A2=mn(C["zmp_margin2"]), B=mn(cop_margin),
                    A_med5=mn_med5(C["zmp_margin"]), A2_med5=mn_med5(C["zmp_margin2"]), B_med5=mn_med5(cop_margin),
                    spread_A=float(np.nanstd(np.diff(C["zmp_margin"][fut]))) if fut.sum() > 3 else None,
                    spread_B=float(np.nanstd(np.diff(cop_margin[fut]))) if fut.sum() > 3 else None))
json.dump(out, open("out/zmp_label_compare.json", "w"))
ok = [o for o in out if None not in (o["A"], o["A2"], o["B"], o["A_med5"], o["B_med5"])]
print("runs with all labels:", len(ok), "of", len(out))
for oc in ("confirmed_recovered", "observed_tip"):
    s = [o for o in ok if o["outcome"] == oc]
    q = lambda k: np.round(np.quantile([o[k] for o in s], [0.05, 0.5, 0.95]), 3)
    print(f"\n{oc} (n={len(s)})  5/50/95% of future min outward margin (m)")
    for k, name in (("A", "A  logged accel-ZMP"), ("A2", "A' corrected accel-ZMP"), ("A_med5", "A  logged, 5-median"), ("A2_med5", "A' corrected, 5-median"), ("B", "B  contact-force CoP"), ("B_med5", "B  CoP, 5-median")):
        print(f"   {name:26s}", q(k), " frac<0:", round(float(np.mean([o[k] < 0 for o in s])), 3))
from scipy.stats import spearmanr
A = lambda k: np.array([o[k] for o in ok])
for a, b in (("A", "B"), ("A2", "B"), ("A_med5", "B_med5"), ("A2_med5", "B_med5"), ("A", "A2")):
    print(f"spearman {a:8s} vs {b:8s}: {spearmanr(A(a), A(b)).statistic:.3f}   mean abs diff {np.mean(np.abs(A(a) - A(b))):.3f}")
tip = np.array([o["outcome"] == "observed_tip" for o in ok])
import sys; sys.path.insert(0, "/home/jude/CWM_paper/Safety_Probing_paper/scripts"); import build_statistics as bs
for k in ("A", "A2", "A_med5", "A2_med5", "B", "B_med5"):
    print(f"AUROC of (-min margin) for tip vs recovered, {k:8s}: {bs.auc(tip.astype(int), -A(k)):.3f}")
print("step-to-step jitter (std of diff, m): accel-ZMP", np.nanmedian([o["spread_A"] for o in out if o["spread_A"] is not None]).round(4), " CoP", np.nanmedian([o["spread_B"] for o in out if o["spread_B"] is not None]).round(4))
