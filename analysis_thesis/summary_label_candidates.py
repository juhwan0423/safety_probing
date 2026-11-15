"""Candidate future stability summaries from the corrected ZMP margin; how well does each separate tip vs recovered?"""
import json, numpy as np, sys
from pathlib import Path
import comfix as cf
sys.path.insert(0, "/home/jude/CWM_paper/Safety_Probing_paper/scripts"); import build_statistics as bs
T_OBS, DT = 65 / 24.0, 1 / 72.0
man = cf.load_manifest(); R = []
for path, row in man.items():
    if row["safety_outcome"] == "right_censored": continue
    rows = cf.read_rows(Path(path)); m, off = float(row["load_mass_kg"]), np.asarray(row["load_com_offset_local_xyz_m"], float)
    C = cf.correct(rows, m, off)
    valid = np.array([str(r.get("outward_lateral_margin_valid", "")).lower() in ("true", "1") for r in rows])
    fut = valid & (C["t"] > T_OBS) & np.isfinite(C["zmp_margin2"])
    if fut.sum() < 5: continue
    raw = C["zmp_margin2"]; med = cf._running_median(raw); com = C["com_margin2"]
    R.append(dict(tip=int(row["safety_outcome"] == "observed_tip"), n=int(fut.sum()),
                  min_raw=float(raw[fut].min()), min_med5=float(med[fut].min()), p10=float(np.quantile(raw[fut], 0.10)),
                  mean=float(raw[fut].mean()), impulse=float(np.clip(-med[fut], 0, None).sum() * DT), impulse_raw=float(np.clip(-raw[fut], 0, None).sum() * DT),
                  neg_duration=float((med[fut] < 0).sum() * DT), min_com=float(com[fut].min()), max_roll=float(row["max_roll_deg"]),
                  speed=row["speed_label"]))
json.dump(R, open("out/summary_label_candidates.json", "w"))
tip = np.array([r["tip"] for r in R]); print("runs", len(R), "tip", tip.sum())
for k, sign in (("min_raw", -1), ("min_med5", -1), ("p10", -1), ("mean", -1), ("impulse", 1), ("impulse_raw", 1), ("neg_duration", 1), ("min_com", -1)):
    x = np.array([r[k] for r in R])
    a_all = bs.auc(tip, sign * x)
    sel = np.array([r["speed"] == "medium" for r in R]); a_med = bs.auc(tip[sel], sign * x[sel])
    print(f"{k:13s} AUROC tip vs recovered: all {a_all:.3f} | medium-speed only {a_med:.3f} | recovered median {np.median(x[tip == 0]):+.3f}, tip median {np.median(x[tip == 1]):+.3f}")
