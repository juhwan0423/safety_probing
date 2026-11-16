"""Per-condition physical table from dense logs: event times, initial inertial
parameters, lateral-acceleration demand in the observation window and over the full run."""
import json, csv, math, numpy as np
from common import Context
c = Context()
rows = {str(r["condition_id"]): r for r in c.canonical_rows}
T_OBS = 65 / 24.0
out = []
def f(x):
    try: return float(x)
    except Exception: return float("nan")
for i, cid in enumerate(c.ids):
    r = rows[cid]
    L = list(csv.DictReader(open(r["dense_log_path"])))
    t = np.array([f(x["sim_time"]) for x in L])
    yaw = np.radians(np.array([f(x["yaw_deg"]) for x in L]))
    vx, vy = np.array([f(x["linear_vel_x"]) for x in L]), np.array([f(x["linear_vel_y"]) for x in L])
    wz = np.array([f(x["angular_vel_z"]) for x in L])
    speed = np.hypot(vx, vy)
    ay = speed * wz                                   # centripetal demand v*omega (signed)
    # system CoM in chassis-yaw-local frame
    dx = np.array([f(x["system_com_x"]) - f(x["forklift_pos_x"]) for x in L])
    dy = np.array([f(x["system_com_y"]) - f(x["forklift_pos_y"]) for x in L])
    cl_x = np.cos(yaw) * dx + np.sin(yaw) * dy
    cl_y = -np.sin(yaw) * dx + np.cos(yaw) * dy
    cz = np.array([f(x["system_com_z"]) for x in L])
    obs = t <= T_OBS
    k0 = 5                                            # settled early sample
    out.append(dict(
        condition_id=cid, split=str(c.split[i]), component=int(c.all_clusters[i]),
        outcome=r["safety_outcome"], tip=int(r["future_tip_label"]) if r["future_tip_label"] is not None else -1,
        route=r["route_id"], speed_label=r["speed_label"], steering_label=r["steering_label"], turn=r["turn_direction"],
        load_mass=float(r["load_mass_kg"]), load_com_y=float(r["load_com_offset_local_xyz_m"][1]),
        t_support_loss=r["dense_summary"]["posthoc_support_loss_time_s"], t_roll_tip=r["dense_summary"]["posthoc_roll_tip_time_s"],
        failure_time=r["failure_time_s"],
        sys_mass=f(L[k0]["system_mass_kg"]), com_x0=float(cl_x[k0]), com_y0=float(cl_y[k0]), com_z0=float(cz[k0]),
        zmp_margin0=f(L[k0]["zmp_outward_lateral_margin_m"]), com_margin0=f(L[k0]["com_outward_lateral_margin_m"]),
        com_support_margin0=f(L[k0]["com_support_margin_m"]),
        speed_obs_end=float(speed[obs][-1]), wz_obs_end=float(wz[obs][-1]),
        ay_obs_max=float(np.abs(ay[obs]).max()), ay_full_max=float(np.abs(ay).max()),
        speed_max=float(speed.max()), roll_obs_max=float(np.abs([f(x["roll_deg"]) for x in np.array(L)[obs]]).max()),
        min_zmp_margin_obs=float(np.nanmin([f(x["zmp_outward_lateral_margin_m"]) if x["outward_lateral_margin_valid"] in ("True","true","1") else np.nan for x in np.array(L)[obs]] or [np.nan])),
        support_polygon0=L[k0]["support_polygon_xy"],
    ))
json.dump(out, open("out/dense_table.json", "w"))
print(len(out)); print({k: out[0][k] for k in out[0]})
