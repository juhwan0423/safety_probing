"""Roll-over criterion about the line of the outer fork rollers (static stability factor, no fitted constant).

During the turn the inner front roller and then the rear wheel unload, and the vehicle is carried by the two rollers
of the outer fork (front roller at x = -0.91 m, back roller at x = -0.72 m, both at |y| = b = 0.50 m in the chassis
frame; measured from the logged contact positions). Tipping about that line starts when the lateral acceleration
exceeds
        a_crit = g (b - y_c) / h
with y_c the lateral position of the true system CoM towards the outside of the turn and h its height.
The demand is the turn severity |v_x * omega_z| (chassis speed times yaw rate, 5-sample mean), taken as its peak
  - during the steering pulse only, 2.0-3.0 s (before any support-loss event, which starts at 3.01 s at the earliest)
  - over 2.0-3.6 s (the definition used in mechanics_true.py)
Compared with the severity alone and with the earlier threshold from the support triangle (arm to the edge between the
outer front roller and the rear wheel), whose value rises with payload mass, opposite to the simulated trend.
Both the severity and the payload are privileged quantities: this is a mechanics check, not a prediction.
Run from analysis_thesis/:  PYTHONPATH=. ../.venv/bin/python rollover_criterion.py
"""
import csv, json, os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
G, B, T_TURN = 9.81, 0.50, 2.0
SETS = [("v1", ROOT / "artifacts/current/manifests/canonical_runs.jsonl", HERE / "out/mechanics_true_v1.json"),
        ("v2", ROOT / "artifacts/v2/p5/manifests/canonical_runs.jsonl", HERE / "out/mechanics_true_v2_p5.json"),
        ("v2 band", ROOT / "artifacts/v2/band5/manifests/canonical_runs.jsonl", HERE / "out/mechanics_true_v2_band5.json"),
        ("v2 sweep", ROOT / "artifacts/v2/sweep/manifests/canonical_runs.jsonl", HERE / "out/mechanics_true_v2_sweep.json")]


def severity(path):
    with open(path) as fh: rows = list(csv.DictReader(fh))
    f = lambda k: np.array([float(r[k]) if r[k] not in ("", "nan", "NaN", "None") else np.nan for r in rows])
    t, vx, vy, wz, yaw = f("sim_time"), f("linear_vel_x"), f("linear_vel_y"), f("angular_vel_z"), np.radians(f("yaw_deg"))
    kin = np.abs((np.cos(yaw) * vx + np.sin(yaw) * vy) * wz)
    sm = np.convolve(np.pad(np.nan_to_num(kin), 2, mode="edge"), np.ones(5) / 5, mode="valid")
    return float(sm[(t >= T_TURN) & (t <= T_TURN + 1.0)].max()), float(sm[(t >= T_TURN) & (t <= T_TURN + 1.6)].max())


def auc(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float); r = s.argsort().argsort() + 1.0
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True); r = (np.bincount(inv, weights=r) / cnt)[inv]
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_ci(y, s, rng, n=2000):
    idx = np.arange(len(y)); vals = []
    for _ in range(n):
        k = rng.choice(idx, len(idx)); yy = y[k]
        if yy.all() or not yy.any(): continue
        vals.append(auc(yy, s[k]))
    return dict(point=auc(y, s), ci_low=float(np.quantile(vals, .025)), ci_high=float(np.quantile(vals, .975)))


def best_threshold(y, s):
    th = np.sort(s); acc = [((s > x) == y).mean() for x in th]; i = int(np.argmax(acc))
    return float(th[i]), float(acc[i])


res, arrays = {}, {}
for name, manifest, mech in SETS:
    if not (manifest.exists() and mech.exists()): print("skip", name); continue
    dense = {json.loads(l)["condition_id"]: json.loads(l)["dense_log_path"] for l in open(manifest)}
    R = [r for r in json.load(open(mech)) if r["outcome"] != "right_censored"]
    sev = np.array([severity(dense[r["cid"]]) for r in R]); tip = np.array([r["outcome"] == "observed_tip" for r in R])
    h = np.array([r["h"] for r in R]); y_out = np.array([r["com_body"][1] * r["out_y"] for r in R]); mass = np.array([r["mass"] for r in R]); tri = np.array([r["a_crit"] for r in R])
    roller = G * (B - y_out) / h; rng = np.random.default_rng(0); out = {"n": int(len(R)), "tip_rate": float(tip.mean()), "threshold_range": [float(roller.min()), float(roller.max())], "y_out_range": [float(y_out.min()), float(y_out.max())]}
    for wname, s in (("pulse_2.0-3.0s", sev[:, 0]), ("window_2.0-3.6s", sev[:, 1])):
        ratio = s / roller; bt, ba = best_threshold(tip, ratio)
        out[wname] = dict(severity_alone=auc_ci(tip, s, rng), over_triangle_threshold=auc_ci(tip, s / tri, rng), over_roller_line_threshold=auc_ci(tip, ratio, rng),
                          over_roller_line_no_lateral_com=auc(tip, s / (G * B / h)), accuracy_at_ratio_1=float(((ratio > 1.0) == tip).mean()), best_ratio=bt, best_accuracy=ba)
        o = out[wname]; g = lambda d: f"{d['point']:.3f}[{d['ci_low']:.3f},{d['ci_high']:.3f}]"
        print(f"{name:9s} n={len(R)} {wname:16s} AUROC severity {g(o['severity_alone'])} | / triangle {g(o['over_triangle_threshold'])} | / roller line {g(o['over_roller_line_threshold'])} "
              f"(without lateral CoM {o['over_roller_line_no_lateral_com']:.3f}) | accuracy at ratio 1.0: {o['accuracy_at_ratio_1']:.3f} | best ratio {bt:.3f} ({ba:.3f})", flush=True)
    res[name] = out; arrays[name] = dict(severity_pulse=sev[:, 0], severity_window=sev[:, 1], roller=roller, triangle=tri, tip=tip, mass=mass, y_out=y_out, h=h)
json.dump(res, open(HERE / "out/rollover_criterion.json", "w"), indent=1)
np.savez(HERE / "out/rollover_criterion.npz", **{f"{n}|{k}": v for n, d in arrays.items() for k, v in d.items()})
