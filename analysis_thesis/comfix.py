"""Payload-CoM-offset correction of the logged system CoM / ZMP / stability margins.

Finding: the simulator logger's system CoM ignores the payload's lateral CoM offset
(the physics uses it; the log does not). This module
  1. replicates the logger's ZMP and directional-margin formulas and checks the replica
     against the logged values (so the only thing we change is the CoM input),
  2. adds the missing rigid-body term   delta = (m_load / M_total) * R(load quat) * offset_local,
  3. recomputes ZMP and outward margins from the corrected CoM,
  4. validates logged vs corrected ZMP against the centre of pressure from contact forces.

Read-only on the dataset. `corrected_rows(path)` returns dense-log rows with corrected
columns for the target builder.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

G = 9.81
MIN_ZMP_DENOMINATOR = 1.0e-6
CONTACT_BODIES = ("back_wheel", "roller_back_left", "roller_back_right", "roller_front_left", "roller_front_right")
ROOT = Path("/home/jude/isaac_ws/safety_probing")


def _f(value) -> float:
    try:
        if value is None or value == "":
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _fmt(value: float) -> str:
    return "" if (value is None or not math.isfinite(value)) else repr(float(value))


def quat_rotate(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) v by unit quaternion(s) q = (x, y, z, w)."""
    q = np.asarray(q_xyzw, dtype=float)
    u, w = q[..., :3], q[..., 3:4]
    v = np.broadcast_to(np.asarray(v, dtype=float), u.shape)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


# ---- verbatim logic of forklift_dataset/state_reader.py::_directional_support_margin ----
def directional_support_margin(point_xy, polygon_xy, direction_xy):
    if direction_xy is None or polygon_xy is None or len(polygon_xy) < 3:
        return None
    point = np.asarray(point_xy[:2], dtype=float)
    polygon = np.asarray(polygon_xy, dtype=float)[:, :2]
    direction = np.asarray(direction_xy[:2], dtype=float)
    if not np.all(np.isfinite(point)) or not np.all(np.isfinite(polygon)) or not np.all(np.isfinite(direction)):
        return None
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        return None
    direction = direction / norm
    signed_area_twice = float(np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1) - polygon[:, 1] * np.roll(polygon[:, 0], -1)))
    if signed_area_twice < 0.0:
        polygon = polygon[::-1]
    best_alignment, best = 0.0, None
    for index in range(len(polygon)):
        start, end = polygon[index], polygon[(index + 1) % len(polygon)]
        edge = end - start
        length = float(np.linalg.norm(edge))
        if length <= 0.0:
            continue
        outward_normal = np.array([edge[1], -edge[0]], dtype=float) / length
        alignment = float(np.dot(outward_normal, direction))
        if alignment <= best_alignment:
            continue
        offset = point - start
        best_alignment = alignment
        best = float((edge[0] * offset[1] - edge[1] * offset[0]) / length)
    return best


def load_manifest() -> dict[str, dict]:
    import os
    path = os.environ.get("COMFIX_MANIFEST") or str(ROOT / "artifacts/current/manifests/canonical_runs.jsonl")
    rows = [json.loads(line) for line in open(path) if line.strip()]
    return {str(Path(r["dense_log_path"]).resolve()): r for r in rows}


_MANIFEST: dict[str, dict] | None = None


def run_info(path: Path) -> tuple[float, np.ndarray]:
    global _MANIFEST
    if _MANIFEST is None:
        _MANIFEST = load_manifest()
    row = _MANIFEST[str(Path(path).resolve())]
    return float(row["load_mass_kg"]), np.asarray(row["load_com_offset_local_xyz_m"], dtype=float)


def read_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: float(row["sim_time"]))
    return rows


def arrays(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    col = lambda name: np.array([_f(r.get(name)) for r in rows])
    vec = lambda prefix: np.stack([col(f"{prefix}_{a}") for a in "xyz"], 1)
    out = {
        "t": col("sim_time"), "p": vec("system_com"), "v": vec("system_com_vel"), "a": vec("system_com_acc"),
        "M": col("system_mass_kg"), "zmp": np.stack([col("zmp_x"), col("zmp_y")], 1),
        "q": np.stack([col(f"load_quat_{a}") for a in "xyzw"], 1),
        "yaw": np.radians(col("yaw_deg")), "roll": col("roll_deg"),
        "fork": np.stack([col("forklift_pos_x"), col("forklift_pos_y")], 1),
        "dir": np.stack([col("outward_direction_x"), col("outward_direction_y")], 1),
        "zmp_margin": col("zmp_outward_lateral_margin_m"), "com_margin": col("com_outward_lateral_margin_m"),
        "contacts": col("wheel_contact_count"),
    }
    force = np.stack([col(f"{b}_force_z") for b in CONTACT_BODIES], 1)
    pos = np.stack([np.stack([col(f"{b}_contact_pos_x"), col(f"{b}_contact_pos_y")], 1) for b in CONTACT_BODIES], 1)
    ok = np.isfinite(force) & np.isfinite(pos).all(-1) & (np.abs(force) > 1e-6)
    w = np.where(ok, np.abs(force), 0.0)
    total = w.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cop = (w[..., None] * np.where(ok[..., None], pos, 0.0)).sum(1) / total[:, None]
    out["cop"], out["fz"], out["n_force"] = cop, total, ok.sum(1)
    return out


def polygons(rows: list[dict[str, str]]):
    result = []
    for r in rows:
        text = r.get("support_polygon_xy") or ""
        try:
            poly = json.loads(text) if text else None
        except json.JSONDecodeError:
            poly = None
        result.append(poly if poly and len(poly) >= 3 else None)
    return result


def correct(rows: list[dict[str, str]], load_mass: float, offset_local: np.ndarray) -> dict[str, np.ndarray]:
    """Corrected CoM, ZMP and directional margins (plus replicas of the logged ones)."""
    A = arrays(rows)
    n = len(rows)
    dt = np.diff(A["t"], prepend=A["t"][0])
    delta = (load_mass / A["M"])[:, None] * quat_rotate(A["q"], offset_local)          # world frame
    dv = np.zeros_like(delta)
    dv[1:] = (delta[1:] - delta[:-1]) / dt[1:, None]
    da = np.zeros_like(delta)
    da[1:] = (dv[1:] - dv[:-1]) / dt[1:, None]
    p2, a2 = A["p"] + delta, A["a"] + da

    def zmp_of(p, a):
        den = G + a[:, 2]
        z = np.full((n, 2), np.nan)
        good = np.abs(den) >= MIN_ZMP_DENOMINATOR
        z[good] = p[good, :2] - p[good, 2:3] * a[good, :2] / den[good, None]
        return z

    zmp_rep, zmp2 = zmp_of(A["p"], A["a"]), zmp_of(p2, a2)
    polys = polygons(rows)
    margins = {k: np.full(n, np.nan) for k in ("zmp_rep", "com_rep", "zmp2", "com2")}
    for i in range(n):
        d = A["dir"][i]
        if polys[i] is None or not np.all(np.isfinite(d)):
            continue
        for key, point in (("zmp_rep", zmp_rep[i]), ("com_rep", A["p"][i, :2]), ("zmp2", zmp2[i]), ("com2", p2[i, :2])):
            value = directional_support_margin(point, polys[i], d)
            if value is not None:
                margins[key][i] = value
    # finite-difference replica of the logger (checks the differencing scheme)
    v_rep = np.zeros_like(A["p"])
    v_rep[1:] = (A["p"][1:] - A["p"][:-1]) / dt[1:, None]
    a_rep = np.zeros_like(A["p"])
    a_rep[1:] = (A["v"][1:] - A["v"][:-1]) / dt[1:, None]
    A.update(delta=delta, p2=p2, a2=a2, zmp_rep=zmp_rep, zmp2=zmp2, v_rep=v_rep, a_rep=a_rep,
             zmp_margin_rep=margins["zmp_rep"], com_margin_rep=margins["com_rep"],
             zmp_margin2=margins["zmp2"], com_margin2=margins["com2"])
    return A


def _running_median(x: np.ndarray, window: int = 5) -> np.ndarray:
    """Centred running median over finite neighbours (spike-robust; ~0.07 s at 72 Hz)."""
    half = window // 2
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        if not math.isfinite(x[i]):
            continue
        seg = x[max(0, i - half): i + half + 1]
        out[i] = float(np.median(seg[np.isfinite(seg)]))
    return out


def corrected_rows(path: Path, med5: bool = False) -> list[dict[str, str]]:
    """Dense rows with corrected CoM/ZMP/margin columns (drop-in for the target builder).

    med5=True additionally replaces the finite-difference ZMP position and ZMP margin by a
    centred 5-sample running median (label de-spiking); the CoM margin is geometric and untouched.
    """
    rows = read_rows(path)
    mass, offset = run_info(path)
    if not np.any(offset) and not med5:
        return rows
    C = correct(rows, mass, offset)
    zmp, zmp_margin = C["zmp2"].copy(), C["zmp_margin2"].copy()
    if med5:
        zmp = np.stack([_running_median(zmp[:, 0]), _running_median(zmp[:, 1])], 1)
        zmp_margin = _running_median(zmp_margin)
    for i, row in enumerate(rows):
        if math.isfinite(C["p"][i, 0]):
            row["system_com_x"], row["system_com_y"], row["system_com_z"] = (_fmt(x) for x in C["p2"][i])
        if row.get("zmp_x", "") != "":
            row["zmp_x"], row["zmp_y"] = _fmt(zmp[i, 0]), _fmt(zmp[i, 1])
        if row.get("zmp_outward_lateral_margin_m", "") != "":
            row["zmp_outward_lateral_margin_m"] = _fmt(zmp_margin[i])
        if row.get("com_outward_lateral_margin_m", "") != "":
            row["com_outward_lateral_margin_m"] = _fmt(C["com_margin2"][i])
    return rows


def lateral(yaw: np.ndarray, d: np.ndarray) -> np.ndarray:
    return -np.sin(yaw) * d[:, 0] + np.cos(yaw) * d[:, 1]
