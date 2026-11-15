"""Rigid-body system CoM, ZMP and outward stability margins from ground-truth mass properties.

Background (found 2026-11-16): the dataset logger pairs PhysX link masses (articulation link order) with link poses
taken in USD *traversal* order (state_reader._read_articulation_mass_components, fallback branch; the logged source
string contains "+body_path_traversal_order"). The two orders differ, so e.g. the payload mass is placed at the rear
wheel swivel and the chassis mass at the mast. The logged system CoM is therefore about 0.5 m too far back and its
height falls with payload mass instead of rising; the kinematic ZMP and the CoM/ZMP margins inherit the error.

Ground truth from sim_probe/probe_mass_properties.py (out/sim_probe_mass_properties.json):
    vehicle without payload : 1590.82 kg, CoM (0.0595, -0.0005, 1.3336) m in the chassis ("body") frame
    payload                 : cube centre at (-1.863, -0.0122, 2.5115) m; the configured local CoM offset is scaled by
                              the cube scale (1.2, 0.9, 0.9), i.e. a nominal 0.25 m lateral offset is 0.225 m
    mast ("lift") + payload ride on a prismatic joint (joint 0) and move a few cm vertically
The vehicle is treated as rigid apart from that joint:  c_body = (M_v c_v + m (p_L + S * offset)) / (M_v + m).

API
    com_body(mass, offset)               -> true CoM in the chassis frame
    compute(rows, mass, offset)          -> dict with the true CoM trajectory, ZMP, margins and the contact-force CoP
    corrected_rows(path, med5=True)      -> dense rows with corrected CoM / ZMP / margin columns (drop-in for the target builder)
Read-only on the dataset.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import comfix as cf

G = 9.81
HERE = Path(__file__).resolve().parent
M_VEHICLE, C_VEHICLE = 1590.82, np.array([0.0595, -0.0005, 1.3336])
P_LOAD, LOAD_SCALE, M_LIFT = np.array([-1.863, -0.0122, 2.5115]), np.array([1.2, 0.9, 0.9]), 465.33
# vehicle links (mass, CoM in the chassis frame, rides on the lift joint); replaced by the probe values when available
LINKS: list[tuple[float, np.ndarray, bool]] = [(1100.78, np.array([0.319, -0.004, 0.728]), False), (465.33, np.array([-0.563, 0.007, 2.83]), True),
                                                (24.71, np.array([0.19, 0.0, 0.14]), False)]


def _load_probe() -> None:
    """Replace the constants above by the exact probe values when the probe output is available."""
    global M_VEHICLE, C_VEHICLE, P_LOAD, M_LIFT
    path = HERE / "out" / "sim_probe_mass_properties.json"
    if not path.exists():
        return
    r = json.load(open(path))[0]
    o = np.array(r["root_origin"]); ex = np.array(r["root_x_axis_point"]) - o; ey = np.array(r["root_y_axis_point"]) - o
    tb = lambda p: np.array([np.dot(np.array(p) - o, ex), np.dot(np.array(p) - o, ey), float(p[2])])
    links = r["truth_system_com"]["per_link"]
    veh = [(l["mass"], tb(l["world_com"])) for l in links if "forklift_load" not in l["path"]]
    M_VEHICLE = float(sum(m for m, _ in veh)); C_VEHICLE = sum(m * c for m, c in veh) / M_VEHICLE
    P_LOAD = tb([l for l in links if "forklift_load" in l["path"]][0]["world_origin"])
    M_LIFT = float([l["mass"] for l in links if l["path"].endswith("/lift")][0])
    LINKS.clear()
    LINKS.extend((float(l["mass"]), tb(l["world_com"]), l["path"].endswith("/lift")) for l in links if "forklift_load" not in l["path"])


_load_probe()


def com_body(mass: float, offset_local: np.ndarray) -> np.ndarray:
    """True system CoM in the chassis frame for a payload of `mass` with the configured local CoM offset."""
    p = P_LOAD + LOAD_SCALE * np.asarray(offset_local, dtype=float)
    return (M_VEHICLE * C_VEHICLE + mass * p) / (M_VEHICLE + mass)


def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    pad = k // 2
    kernel = np.ones(k) / k
    return np.stack([np.convolve(np.pad(x[:, j], pad, mode="edge"), kernel, mode="valid") for j in range(x.shape[1])], 1)


def compute(rows: list[dict[str, str]], mass: float, offset_local: np.ndarray, smooth: int = 5) -> dict[str, np.ndarray]:
    col = lambda name: np.array([cf._f(r.get(name)) for r in rows])
    t = col("sim_time"); n = len(rows)
    r = np.stack([col(f"forklift_pos_{a}") for a in "xyz"], 1)
    q = np.stack([col(f"forklift_quat_{a}") for a in "xyzw"], 1)
    lift = np.array([(json.loads(row["joint_positions"])[0] if row.get("joint_positions") else math.nan) for row in rows])
    lift = np.where(np.isfinite(lift), lift, lift[np.isfinite(lift)][0] if np.isfinite(lift).any() else 0.0)
    M = M_VEHICLE + mass
    cb = np.tile(com_body(mass, offset_local), (n, 1))
    cb[:, 2] += (M_LIFT + mass) / M * (lift - lift[0])                 # mast + payload ride on the lift joint
    c = r + cf.quat_rotate(q, cb)                                       # world frame
    cs = _smooth(c, smooth)
    dt = np.gradient(t)
    v = np.gradient(cs, axis=0) / dt[:, None]
    a = np.gradient(v, axis=0) / dt[:, None]
    # Multi-body ZMP: every link is a point mass at its own CoM, moving rigidly with the chassis (the mast and payload
    # also follow the lift joint). This keeps the angular-momentum term that a single point mass at the system CoM
    # drops; with the payload 2.5 m high and 1.9 m ahead of the chassis origin, yaw acceleration alone shifts the
    # ZMP sideways by more than 0.1 m.
    #   x_zmp = sum m_i ((az_i + g) x_i - ax_i z_i) / sum m_i (az_i + g)      (same for y)
    dz = lift - lift[0]
    parts = [(m, cl, rides) for m, cl, rides in LINKS] + [(mass, P_LOAD + LOAD_SCALE * np.asarray(offset_local, dtype=float), True)]
    num = np.zeros((n, 2)); den = np.zeros(n)
    for m_i, c_i, rides in parts:
        local = np.tile(c_i, (n, 1))
        if rides:
            local[:, 2] += dz
        p_i = r + cf.quat_rotate(q, local)
        a_i = np.gradient(np.gradient(_smooth(p_i, smooth), axis=0) / dt[:, None], axis=0) / dt[:, None]
        w_i = m_i * (a_i[:, 2] + G)
        num += w_i[:, None] * p_i[:, :2] - m_i * a_i[:, :2] * p_i[:, 2:3]
        den += w_i
    zmp = np.full((n, 2), np.nan); good = np.abs(den) >= cf.MIN_ZMP_DENOMINATOR * M
    zmp[good] = num[good] / den[good, None]
    # single point mass at the system CoM, kept for comparison
    den1 = G + a[:, 2]
    zmp_point = np.full((n, 2), np.nan); good1 = np.abs(den1) >= cf.MIN_ZMP_DENOMINATOR
    zmp_point[good1] = c[good1, :2] - c[good1, 2:3] * a[good1, :2] / den1[good1, None]
    A = cf.arrays(rows); polys = cf.polygons(rows)
    zmp_margin, com_margin = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        d = A["dir"][i]
        if polys[i] is None or not np.all(np.isfinite(d)):
            continue
        for arr, point in ((zmp_margin, zmp[i]), (com_margin, c[i, :2])):
            value = cf.directional_support_margin(point, polys[i], d)
            if value is not None:
                arr[i] = value
    return dict(t=t, com=c, com_body=cb, vel=v, acc=a, zmp=zmp, zmp_point=zmp_point, zmp_margin=zmp_margin, com_margin=com_margin, cop=A["cop"], fz=A["fz"],
                yaw=A["yaw"], roll=A["roll"], fork=A["fork"], dir=A["dir"], logged_com=A["p"], logged_zmp=A["zmp"], logged_zmp_margin=A["zmp_margin"],
                logged_com_margin=A["com_margin"], polys=polys, M=np.full(n, M))


def corrected_rows(path: Path, med5: bool = True) -> list[dict[str, str]]:
    """Dense rows whose system CoM, ZMP and outward-margin columns come from the true rigid-body CoM.
    med5: centred 5-sample running median on the ZMP position and ZMP margin (same de-spiking as comfix)."""
    rows = cf.read_rows(path)
    mass, offset = cf.run_info(path)
    T = compute(rows, mass, offset)
    zmp, zmp_margin = T["zmp"].copy(), T["zmp_margin"].copy()
    if med5:
        zmp = np.stack([cf._running_median(zmp[:, 0]), cf._running_median(zmp[:, 1])], 1)
        zmp_margin = cf._running_median(zmp_margin)
    for i, row in enumerate(rows):
        row["system_com_x"], row["system_com_y"], row["system_com_z"] = (cf._fmt(x) for x in T["com"][i])
        row["system_mass_kg"] = cf._fmt(float(T["M"][i]))
        if row.get("zmp_x", "") != "":
            row["zmp_x"], row["zmp_y"] = cf._fmt(zmp[i, 0]), cf._fmt(zmp[i, 1])
        if row.get("zmp_outward_lateral_margin_m", "") != "":
            row["zmp_outward_lateral_margin_m"] = cf._fmt(zmp_margin[i])
        if row.get("com_outward_lateral_margin_m", "") != "":
            row["com_outward_lateral_margin_m"] = cf._fmt(T["com_margin"][i])
    return rows
