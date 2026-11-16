"""Identify the true system centre of mass of each run from wheel loads (centre of pressure), independent of the logged CoM.

Why: the simulator logger composes the system CoM from PhysX link masses paired with USD prim paths in *traversal order*
(state_reader._read_articulation_mass_components falls back to that order when the articulation view has no body names;
the log's source string says "+body_path_traversal_order"). If the two orders differ, masses are attached to the wrong
link poses and the logged CoM / kinematic ZMP are biased. Wheel loads do not depend on that code path.

Model (rigid vehicle + fixed payload, flat ground, before any wheel lifts):
    CoP_x(t) = x_c - (h / g) * a_x(t)        CoP_y(t) = y_c - (h / g) * a_y(t)        (chassis frame)
with CoP from the normal forces and contact points of the five contact bodies and a(t) the chassis acceleration.
Per run we estimate (x_c, y_c) and h by least squares over the pre-steering window.
Run:  PYTHONPATH=. ../.venv/bin/python com_identify.py [--manifest path] [--out out/com_identified_v1.json]
"""
from __future__ import annotations

import argparse, json, os
from pathlib import Path

import numpy as np

import comfix as cf

G = 9.81
BODIES = cf.CONTACT_BODIES


def running_median(x: np.ndarray, k: int = 9) -> np.ndarray:
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(xp, k), axis=1)


def running_mean(x: np.ndarray, k: int = 9) -> np.ndarray:
    pad = k // 2
    return np.convolve(np.pad(x, pad, mode="edge"), np.ones(k) / k, mode="valid")


def body_frame(xy: np.ndarray, origin: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    d = xy - origin
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], 1)


def series(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    f = lambda k: np.array([cf._f(r[k]) for r in rows])
    t = f("sim_time"); yaw = np.radians(f("yaw_deg")); origin = np.stack([f("forklift_pos_x"), f("forklift_pos_y")], 1)
    N = np.stack([np.nan_to_num(f(b + "_force_z")) for b in BODIES], 1)                       # [T, 5]
    P = np.stack([body_frame(np.stack([f(b + "_contact_pos_x"), f(b + "_contact_pos_y")], 1), origin, yaw) for b in BODIES], 1)   # [T, 5, 2]
    P = np.where(np.isfinite(P), P, 0.0)
    tot = N.sum(1)
    cop = (N[:, :, None] * P).sum(1) / np.where(tot > 1.0, tot, np.nan)[:, None]
    vx, vy = f("linear_vel_x"), f("linear_vel_y")
    dt = np.gradient(t)
    ax_w, ay_w = np.gradient(running_mean(vx, 5)) / dt, np.gradient(running_mean(vy, 5)) / dt
    a_b = np.stack([np.cos(yaw) * ax_w + np.sin(yaw) * ay_w, -np.sin(yaw) * ax_w + np.cos(yaw) * ay_w], 1)
    return dict(t=t, yaw=yaw, origin=origin, N=N, P=P, tot=tot, cop=cop, a_b=a_b, roll=f("roll_deg"), wz=f("angular_vel_z"),
                vb_x=np.cos(yaw) * vx + np.sin(yaw) * vy, M=f("system_mass_kg"))


def identify(S: dict[str, np.ndarray], t0: float = 0.25, t1: float = 1.95) -> dict | None:
    t = S["t"]
    w = (t >= t0) & (t <= t1) & np.isfinite(S["cop"][:, 0]) & (np.abs(S["roll"]) < 1.0)
    if w.sum() < 40:
        return None
    cx, cy = running_median(S["cop"][:, 0]), running_median(S["cop"][:, 1])
    ax, ay = running_mean(S["a_b"][:, 0], 9), running_mean(S["a_b"][:, 1], 9)
    # x: CoP_x = x_c - (h/g) a_x  -> slope gives h when the longitudinal acceleration varies enough
    A = np.c_[np.ones(w.sum()), -ax[w] / G]
    (x_c, h), *_ = np.linalg.lstsq(A, cx[w], rcond=None)
    res = cx[w] - A @ np.array([x_c, h])
    y_c = float(np.median(cy[w] + (h / G) * ay[w]))
    return dict(x_c=float(x_c), y_c=y_c, h=float(h), rms_x=float(np.sqrt(np.mean(res ** 2))), ax_span=float(np.ptp(ax[w])), weight_N=float(np.median(S["tot"][w])), n=int(w.sum()))


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest"); ap.add_argument("--out", default="out/com_identified_v1.json")
    args = ap.parse_args()
    if args.manifest: os.environ["COMFIX_MANIFEST"] = args.manifest
    man = cf.load_manifest()
    out = {}
    for path, row in man.items():
        rows = cf.read_rows(Path(path)); S = series(rows); r = identify(S)
        if r is None: continue
        A = cf.arrays(rows); k = int(np.argmin(np.abs(S["t"] - 1.0)))
        logged = body_frame(A["p"][k:k + 1, :2], S["origin"][k:k + 1], S["yaw"][k:k + 1])[0]
        r.update(mass=float(row["load_mass_kg"]), offset=float(row["load_com_offset_local_xyz_m"][1]), system_mass=float(S["M"][k]),
                 logged_x=float(logged[0]), logged_y=float(logged[1]), logged_h=float(A["p"][k, 2]))
        out[row["condition_id"]] = r
    R = list(out.values()); V = lambda k: np.array([r[k] for r in R])
    print(f"runs {len(R)} of {len(man)} | weight check: median total normal force / (M g) = {np.median(V('weight_N') / (V('system_mass') * G)):.4f}")
    print("identified from wheel loads : x_c 5/50/95%% %s | y_c %s | h %s" % tuple(np.round(np.quantile(V(k), [.05, .5, .95]), 3) for k in ("x_c", "y_c", "h")))
    print("logged by the simulator     : x   5/50/95%% %s | y   %s | h %s" % tuple(np.round(np.quantile(V(k), [.05, .5, .95]), 3) for k in ("logged_x", "logged_y", "logged_h")))
    m, off, M = V("mass"), V("offset"), V("system_mass"); Mv = float(np.median(M - m))
    for name, key in (("identified", ("x_c", "y_c", "h")), ("logged", ("logged_x", "logged_y", "logged_h"))):
        # x_c (Mv + m) = Mv x_v + m x_p ; h (Mv + m) = Mv z_v + m z_p ; y_c = y0 + (m / M) * k * offset
        B = np.c_[np.full_like(m, Mv), m]
        xv, xp = np.linalg.lstsq(B, V(key[0]) * (Mv + m), rcond=None)[0]; zv, zp = np.linalg.lstsq(B, V(key[2]) * (Mv + m), rcond=None)[0]
        ky = np.linalg.lstsq(np.c_[np.ones_like(m), (m / M) * off], V(key[1]), rcond=None)[0]
        print(f"{name:10s}: vehicle CoM x {xv:+.3f} z {zv:.3f} | payload CoM x {xp:+.3f} z {zp:.3f} | lateral: y0 {ky[0]:+.3f}, gain on (m/M)*offset {ky[1]:.2f} (1.00 = rigid payload offset)")
    for mm in sorted(set(np.round(m, -1)))[:6] if len(set(np.round(m))) > 12 else sorted(set(m)):
        s = (np.round(m, -1) == mm) if len(set(np.round(m))) > 12 else (m == mm)
        print(f"   mass {mm:5.0f}: identified x_c {np.median(V('x_c')[s]):+.3f} h {np.median(V('h')[s]):.3f} (rms {np.median(V('rms_x')[s]):.3f}) | logged x {np.median(V('logged_x')[s]):+.3f} h {np.median(V('logged_h')[s]):.3f}")
    json.dump(dict(vehicle_mass=Mv, runs=out), open(args.out, "w"))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
