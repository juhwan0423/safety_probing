"""Mechanics of the benchmark: static rollover threshold and a one-parameter rocking (rigid-body) model.

Per run (corrected CoM, see comfix.py), in the chassis frame just before the steering pulse:
  tipping axis   = support-polygon edge hit by the ray from the system CoM towards the outside of the turn
  d              = perpendicular distance from the CoM to that edge,  h = CoM height
  a_perp(t)      = v(t)|omega(t)| projected on the edge normal (lateral inertial load about the tipping axis)
Static criterion : wheels on the inner side unload when  a_perp * h > g * d.
Rocking model    : rho^2 * phi'' = a_perp (h cos phi + d sin phi) - g (d cos phi - h sin phi),  phi >= 0,
                   with rho^2 = k^2 + h^2 + d^2 and a single radius of gyration k fitted on the training split.
The measured v(t), omega(t) of the same run drive the model, so this is an explanatory model (privileged future),
not a predictor. Read-only on the dataset.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

import comfix as cf

sys.path.insert(0, "/home/jude/CWM_paper/Safety_Probing_paper/scripts")
import build_statistics as bs  # noqa: E402

G, DT, T_TURN = 9.81, 1.0 / 72.0, 2.0


def ray_edge(point: np.ndarray, polygon: np.ndarray, direction: np.ndarray):
    """Edge of a convex polygon hit by the ray point + s*direction; returns (inward distance, outward unit normal)."""
    area2 = float(np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1) - polygon[:, 1] * np.roll(polygon[:, 0], -1)))
    if area2 < 0:
        polygon = polygon[::-1]
    best = None
    for i in range(len(polygon)):
        a, b = polygon[i], polygon[(i + 1) % len(polygon)]
        e = b - a
        den = direction[0] * e[1] - direction[1] * e[0]
        if abs(den) < 1e-12:
            continue
        w = a - point
        s = (w[0] * e[1] - w[1] * e[0]) / den            # along the ray
        u = (w[0] * direction[1] - w[1] * direction[0]) / den  # along the edge
        if s > 0 and -1e-9 <= u <= 1 + 1e-9 and (best is None or s < best[0]):
            length = float(np.linalg.norm(e))
            normal = np.array([e[1], -e[0]]) / length
            dist = float((e[0] * (point[1] - a[1]) - e[1] * (point[0] - a[0])) / length)
            best = (s, dist, normal)
    return (None, None) if best is None else (best[1], best[2])


def to_body(xy: np.ndarray, origin: np.ndarray, yaw: float) -> np.ndarray:
    d = xy - origin
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], -1)


def run_features(path: Path, row: dict) -> dict | None:
    rows = cf.read_rows(path)
    mass, offset = float(row["load_mass_kg"]), np.asarray(row["load_com_offset_local_xyz_m"], float)
    C = cf.correct(rows, mass, offset)
    polys = cf.polygons(rows)
    t = C["t"]
    ref = int(np.argmin(np.abs(t - (T_TURN - 0.1))))
    if polys[ref] is None or len(polys[ref]) < 4:
        return None
    poly_b = to_body(np.asarray(polys[ref], float), C["fork"][ref], C["yaw"][ref])
    com_b = to_body(C["p2"][ref, :2], C["fork"][ref], C["yaw"][ref])
    h = float(C["p2"][ref, 2])
    vx = np.array([cf._f(r["linear_vel_x"]) for r in rows]); vy = np.array([cf._f(r["linear_vel_y"]) for r in rows])
    wz = np.array([cf._f(r["angular_vel_z"]) for r in rows])
    speed = np.hypot(vx, vy)
    turn = t >= T_TURN
    sign = np.sign(np.nanmedian(wz[turn & (t < T_TURN + 1.0)]))      # +1: left turn (yaw rate > 0)
    if sign == 0:
        return None
    # Outside of the turn in the chassis frame. The centripetal acceleration is omega x v, so with yaw rate wz and
    # body-frame longitudinal velocity vb_x its lateral component is wz * vb_x; the outside is the opposite side.
    # (This forklift travels along -x of its root frame, so the outside of a left turn is +y. An earlier version
    #  assumed forward motion along +x and measured the arm to the inner edge; fixed 2026-11-16.)
    vb_x = np.cos(C["yaw"]) * vx + np.sin(C["yaw"]) * vy
    travel = np.sign(np.nanmedian(vb_x[(t > T_TURN - 1.0) & (t < T_TURN + 1.0)]))
    if travel == 0:
        return None
    outward_b = np.array([0.0, -sign * travel])
    # guard: must agree with the outward direction logged by the simulator during the turn
    ox = np.array([cf._f(r["outward_direction_x"]) for r in rows]); oy = np.array([cf._f(r["outward_direction_y"]) for r in rows])
    ob_y = -np.sin(C["yaw"]) * ox + np.cos(C["yaw"]) * oy
    logged = turn & (t < T_TURN + 1.0) & np.isfinite(ob_y) & (np.hypot(ox, oy) > 0.5)
    agrees = bool(np.sign(np.nanmedian(ob_y[logged])) == outward_b[1]) if logged.any() else None
    d, normal = ray_edge(com_b, poly_b, outward_b)
    if d is None:
        return None
    a_out = speed * np.abs(wz) * (np.sign(wz) == sign)               # centripetal demand while turning the commanded way
    a_perp = a_out * float(outward_b @ normal)
    a_perp[~turn] = 0.0
    roll_out = -sign * C["roll"]
    return dict(t=t, a_perp=a_perp, d=float(d), h=h, cos_edge=float(outward_b @ normal), roll=C["roll"], roll_out=roll_out, outward_agrees=agrees, out_sign=float(outward_b[1]), com_out=float(com_b[1] * outward_b[1]),
                a_peak=float(np.nanmax(a_perp)), a_crit=float(G * d / h),
                exceed_s=float(np.sum(a_perp * h > G * d) * DT), max_abs_roll=float(np.nanmax(np.abs(C["roll"]))))


def rocking(a_perp: np.ndarray, d: float, h: float, k: float) -> float:
    """Max roll angle (deg) of a rigid body rocking about the tipping edge under a_perp(t)."""
    rho2 = k * k + h * h + d * d
    phi = om = peak = 0.0
    for a in a_perp:
        torque = a * (h * np.cos(phi) + d * np.sin(phi)) - G * (d * np.cos(phi) - h * np.sin(phi))
        if phi <= 0.0 and torque <= 0.0:
            phi = om = 0.0
            continue
        om += torque / rho2 * DT
        phi += om * DT
        if phi <= 0.0:
            phi = om = 0.0
        peak = max(peak, phi)
        if phi > np.pi / 2:
            return 90.0
    return float(np.degrees(peak))


def main() -> None:
    man = cf.load_manifest()
    R = []
    for path, row in man.items():
        f = run_features(Path(path), row)
        if f is None:
            continue
        R.append(dict(cid=row["condition_id"], split=row["splits"]["main"], outcome=row["safety_outcome"], mass=float(row["load_mass_kg"]),
                      com=float(row["load_com_offset_local_xyz_m"][1]), turn=row["turn_direction"], speed=row.get("speed_label"), steer=row.get("steering_label"),
                      drive_cmd=row.get("drive_velocity_command"), steer_cmd=row.get("steering_position_command"), **f))
    print("runs", len(R), "of", len(man))
    ag = [r["outward_agrees"] for r in R if r["outward_agrees"] is not None]
    print("outward direction agrees with the simulator log in %d of %d runs" % (sum(ag), len(ag)))
    A = lambda k: np.array([r[k] for r in R])
    definite = np.array([r["outcome"] != "right_censored" for r in R]); tip = np.array([r["outcome"] == "observed_tip" for r in R]).astype(int)
    print("d (m): 5/50/95%%  %s | h (m): %s | static threshold a_crit (m/s^2): %s" % (np.round(np.quantile(A("d"), [.05, .5, .95]), 3), np.round(np.quantile(A("h"), [.05, .5, .95]), 3), np.round(np.quantile(A("a_crit"), [.05, .5, .95]), 2)))
    print("peak lateral demand a_perp (m/s^2): 5/50/95%% %s | fraction of runs exceeding the static threshold: %.3f" % (np.round(np.quantile(A("a_peak"), [.05, .5, .95]), 2), float(np.mean(A("a_peak") > A("a_crit")))))
    # outward-side CoM offset effect on d
    out_off = A("out_sign") * A("com")                                  # payload lateral CoM offset towards the outside of the turn
    for m in sorted(set(A("mass"))):
        sel = A("mass") == m
        slope = np.polyfit(out_off[sel], A("d")[sel], 1)
        print(f"   mass {m:5.0f} kg: d = {slope[1]:.3f} + ({slope[0]:+.3f}) * signed CoM offset;  h = {A('h')[sel].mean():.3f} m")
    ratio = A("a_peak") / A("a_crit")
    print("AUROC for tip-over | static ratio a_peak/a_crit: %.3f | exceedance time: %.3f | peak demand alone: %.3f" % (
        bs.auc(tip[definite], ratio[definite]), bs.auc(tip[definite], A("exceed_s")[definite]), bs.auc(tip[definite], A("a_peak")[definite])))
    train = (A("split") == "train") & definite
    best = None
    for k in np.arange(0.0, 2.01, 0.1):
        phi = np.array([rocking(r["a_perp"], r["d"], r["h"], k) for r in R])
        auc_tr = bs.auc(tip[train], phi[train])
        err = np.median(np.abs(np.log10(np.clip(phi[train], 0.05, 90)) - np.log10(np.clip(A("max_abs_roll")[train], 0.05, 90))))
        if best is None or (auc_tr, -err) > (best[1], -best[2]):
            best = (k, auc_tr, err, phi)
        print(f"   k={k:.1f} m: train AUROC {auc_tr:.3f}, median |log10 roll error| {err:.3f}")
    k, _, _, phi = best
    test = (A("split") == "test") & definite
    acc = lambda sel: float(np.mean((phi[sel] >= 20.0) == (tip[sel] == 1)))
    print(f"rocking model, k={k:.1f} m (fitted on train): AUROC train {bs.auc(tip[train], phi[train]):.3f} / test {bs.auc(tip[test], phi[test]):.3f} / all {bs.auc(tip[definite], phi[definite]):.3f}"
          f" | accuracy with the 20 deg rule: train {acc(train):.3f} test {acc(test):.3f}")
    rec = definite & (tip == 0)
    print("recovered runs: predicted vs simulated max roll (deg) median %.2f vs %.2f, Spearman-like rank corr %.3f" % (
        np.median(phi[rec]), np.median(A("max_abs_roll")[rec]), np.corrcoef(np.argsort(np.argsort(phi[rec])), np.argsort(np.argsort(A("max_abs_roll")[rec])))[0, 1]))
    out = [dict(cid=r["cid"], split=r["split"], outcome=r["outcome"], mass=r["mass"], com=r["com"], turn=r["turn"], speed=r["speed"], steer=r["steer"], drive_cmd=r["drive_cmd"], steer_cmd=r["steer_cmd"], out_sign=r["out_sign"], com_out=r["com_out"], d=r["d"], h=r["h"],
                a_peak=r["a_peak"], a_crit=r["a_crit"], exceed_s=r["exceed_s"], max_abs_roll=r["max_abs_roll"], rocking_max_roll=float(p)) for r, p in zip(R, phi)]
    json.dump(dict(k=float(k), runs=out), open(os.environ.get("ROLLOVER_OUT", "out/rollover_physics.json"), "w"))   # ROLLOVER_OUT: other datasets (v2)


if __name__ == "__main__":
    main()
