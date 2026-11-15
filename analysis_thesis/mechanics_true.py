"""Tip-over mechanics with the true system CoM (truecom.py), per run:

  validation   |ZMP - CoP| while every load-bearing wheel is in contact, for the logged, comfix and true-CoM ZMP
  statics      stability arm d (CoM to the support edge on the outside of the turn, just before steering),
               CoM height h, static rollover threshold a_crit = g d / h
  demand       lateral CoM acceleration towards the inside of the turn (true CoM, smoothed): peak, time above a_crit
  margin       minimum outward ZMP margin after the observation window, for the three ZMP definitions
Run:  PYTHONPATH=. ../.venv/bin/python mechanics_true.py [--manifest path] [--out out/mechanics_true_v1.json]
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

import numpy as np

import comfix as cf, truecom as tc, com_identify as ci, rollover_physics as rp

sys.path.insert(0, "/home/jude/CWM_paper/Safety_Probing_paper/scripts")
import build_statistics as bs  # noqa: E402

G, T_TURN, T_OBS = 9.81, 2.0, 65 / 24


def one(path: Path, row: dict) -> dict | None:
    rows = cf.read_rows(path)
    mass, offset = float(row["load_mass_kg"]), np.asarray(row["load_com_offset_local_xyz_m"], float)
    T = tc.compute(rows, mass, offset); C = cf.correct(rows, mass, offset); S = ci.series(rows)
    t = T["t"]; yaw = T["yaw"]
    # --- validation against the centre of pressure
    w = (t > 0.3) & (t < 3.2) & np.isfinite(S["cop"][:, 0]) & (np.abs(S["roll"]) < 1.0) & (S["N"][:, 3] > 300) & (S["N"][:, 4] > 300) & (S["N"][:, 0] > 300)
    err = {}
    if w.sum() >= 30:
        cx, cy = ci.running_median(S["cop"][:, 0], 5), ci.running_median(S["cop"][:, 1], 5)
        for name, z in (("logged", C["zmp_rep"]), ("comfix", C["zmp2"]), ("true", T["zmp"])):
            zb = ci.body_frame(z, S["origin"], yaw); zx, zy = ci.running_median(np.nan_to_num(zb[:, 0]), 5), ci.running_median(np.nan_to_num(zb[:, 1]), 5)
            err[name] = [float(np.nanmedian(np.abs(zx[w] - cx[w]))), float(np.nanmedian(np.abs(zy[w] - cy[w]))), float(np.nanmedian(zx[w] - cx[w])), float(np.nanmedian(zy[w] - cy[w]))]
    # --- statics just before steering
    ref = int(np.argmin(np.abs(t - (T_TURN - 0.1))))
    poly = T["polys"][ref]
    if poly is None or len(poly) < 3:
        return None
    poly_b = rp.to_body(np.asarray(poly, float), T["fork"][ref], yaw[ref]); com_b = T["com_body"][ref]
    col = lambda k: np.array([cf._f(r[k]) for r in rows])
    vx, vy, wz = col("linear_vel_x"), col("linear_vel_y"), col("angular_vel_z")
    vb_x = np.cos(yaw) * vx + np.sin(yaw) * vy
    turn = (t >= T_TURN) & (t < T_TURN + 1.0)
    sign = np.sign(np.nanmedian(wz[turn])); travel = np.sign(np.nanmedian(vb_x[(t > T_TURN - 1.0) & (t < T_TURN + 1.0)]))
    if sign == 0 or travel == 0:
        return None
    out_y = -sign * travel                                           # outside of the turn along the chassis y axis
    d, normal = rp.ray_edge(com_b[:2], poly_b, np.array([0.0, out_y]))
    if d is None:
        return None
    h = float(T["com"][ref, 2])
    # --- lateral demand: true CoM acceleration towards the inside of the turn, chassis frame, light smoothing
    a = T["acc"]; ab_y = -np.sin(yaw) * a[:, 0] + np.cos(yaw) * a[:, 1]; ab_x = np.cos(yaw) * a[:, 0] + np.sin(yaw) * a[:, 1]
    a_edge = -(ab_x * normal[0] + ab_y * normal[1])                   # inertial load pushes along +normal when the CoM accelerates along -normal
    dem = ci.running_mean(ci.running_median(np.nan_to_num(a_edge), 5), 9)
    k5 = np.flatnonzero(np.abs(T["roll"]) > 5.0); t_end = t[k5[0]] if len(k5) else 1e9
    win = (t >= T_TURN) & (t <= T_TURN + 1.6) & (t < t_end)
    a_crit = G * d / h
    kin = np.abs(vb_x * wz)                                          # kinematic severity of the manoeuvre: chassis speed x yaw rate
    a_kin = float(np.nanmax(ci.running_mean(np.nan_to_num(kin), 5)[(t >= T_TURN) & (t <= T_TURN + 1.6)]))
    v0 = float(np.nanmedian(np.abs(vb_x[(t > T_TURN - 0.3) & (t < T_TURN)])))
    fut = t > T_OBS
    mm = lambda v: float(np.nanmin(v[fut])) if np.isfinite(v[fut]).any() else float("nan")
    med = lambda v: ci.running_median(np.where(np.isfinite(v), v, np.nan), 5)
    lifted = (S["N"][:, 3] < 1.0) | (S["N"][:, 4] < 1.0)              # a front roller carries no load
    first_lift = float(t[(t >= T_TURN) & lifted][0]) if ((t >= T_TURN) & lifted).any() else None
    return dict(cid=row["condition_id"], split=row["splits"]["main"], outcome=row["safety_outcome"], mass=mass, offset=float(offset[1]), turn=row["turn_direction"],
                speed=row.get("speed_label"), steer=row.get("steering_label"), drive_cmd=row.get("drive_velocity_command"), steer_cmd=row.get("steering_position_command"),
                out_y=float(out_y), com_body=[float(v) for v in com_b], d=float(d), h=h, a_crit=float(a_crit), offset_out=float(offset[1] * out_y * tc.LOAD_SCALE[1]),
                a_peak=float(np.max(dem[win])) if win.any() else float("nan"), over_s=float(np.sum((dem > a_crit) & win) / 72.0), err=err, a_kin=a_kin, v0=v0,
                min_margin_true=mm(med(T["zmp_margin"])), min_margin_logged=mm(med(C["zmp_margin_rep"])), min_margin_comfix=mm(med(C["zmp_margin2"])),
                min_com_margin_true=mm(T["com_margin"]), first_lift_s=first_lift, max_abs_roll=float(np.nanmax(np.abs(T["roll"]))))


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest"); ap.add_argument("--out", default="out/mechanics_true_v1.json")
    args = ap.parse_args()
    if args.manifest: os.environ["COMFIX_MANIFEST"] = args.manifest
    man = cf.load_manifest(); R = []
    for path, row in man.items():
        r = one(Path(path), row)
        if r is not None: R.append(r)
    json.dump(R, open(args.out, "w"))
    A = lambda k: np.array([r[k] for r in R], dtype=float)
    definite = np.array([r["outcome"] != "right_censored" for r in R]); tip = np.array([r["outcome"] == "observed_tip" for r in R]).astype(int)
    print(f"runs {len(R)} of {len(man)} | definite {definite.sum()} | tip rate {tip[definite].mean():.3f}")
    for name in ("logged", "comfix", "true"):
        e = np.array([r["err"][name] for r in R if name in r["err"]])
        print(f"|ZMP - CoP| in full contact, {name:7s}: longitudinal median {np.median(e[:, 0]):.3f} m, lateral median {np.median(e[:, 1]):.3f} m (bias {np.median(e[:, 2]):+.3f}, {np.median(e[:, 3]):+.3f}), n={len(e)}")
    q = lambda k: np.round(np.nanquantile(A(k), [.05, .5, .95]), 3)
    print("stability arm d (m)", q("d"), "| CoM height h (m)", q("h"), "| static threshold g d / h (m/s^2)", q("a_crit"), "| peak lateral demand (m/s^2)", q("a_peak"))
    print("fraction of runs whose peak demand exceeds the static threshold: %.3f | a front roller unloads during the turn: %.3f" % (np.nanmean(A("a_peak") > A("a_crit")), np.mean([r["first_lift_s"] is not None for r in R])))
    for name, v in (("speed at the start of the turn", A("v0")), ("kinematic severity v*omega", A("a_kin")), ("v*omega / static threshold", A("a_kin") / A("a_crit")),
                    ("peak CoM lateral acceleration", A("a_peak")), ("peak CoM accel / static threshold", A("a_peak") / A("a_crit")), ("time above static threshold", A("over_s")),
                    ("- min ZMP margin, true CoM", -A("min_margin_true")), ("- min ZMP margin, logged", -A("min_margin_logged")), ("- min ZMP margin, comfix", -A("min_margin_comfix"))):
        ok = definite & np.isfinite(v); print(f"AUROC for tip-over | {name:36s} {bs.auc(tip[ok], v[ok]):.3f}")
    ok = np.isfinite(A("min_margin_true")) & np.isfinite(A("min_margin_comfix"))
    print("min ZMP margin: true 5/50/95%%", q("min_margin_true"), "| comfix", q("min_margin_comfix"), "| Pearson r(true, comfix) = %.3f, r(true, logged) = %.3f" % (
        np.corrcoef(A("min_margin_true")[ok], A("min_margin_comfix")[ok])[0, 1], np.corrcoef(A("min_margin_true")[ok], A("min_margin_logged")[ok])[0, 1]))
    masses = sorted(set(np.round(A("mass"), -1))) if len(set(A("mass"))) > 12 else sorted(set(A("mass")))
    if len(masses) <= 6:
        for m in masses:
            s = A("mass") == m
            print(f"   mass {m:4.0f} kg: CoM x {np.median([r['com_body'][0] for r, k in zip(R, s) if k]):+.3f} h {np.median(A('h')[s]):.3f} d {np.median(A('d')[s]):.3f} a_crit {np.median(A('a_crit')[s]):.2f} peak demand {np.nanmedian(A('a_peak')[s]):.2f} tip rate {tip[s & definite].mean():.3f}")
        for o in sorted(set(np.round(A("offset_out"), 3))):
            s = np.round(A("offset_out"), 3) == o
            print(f"   payload CoM {o:+.3f} m towards the outside: d {np.median(A('d')[s]):.3f} a_crit {np.median(A('a_crit')[s]):.2f} tip rate {tip[s & definite].mean():.3f}")


if __name__ == "__main__":
    main()
