"""How large is the observable motion difference between the two members of a matched pair during the observation window?"""
import json, numpy as np
from common import Context
c = Context("artifacts/targets/safety_query_full_comfix_med5.pt")
traj = np.load("out/obs_trajectory.npy")                        # [N,10,3] dx, dy (m), dyaw (rad) relative to first sample
t = c.tnp; ch = c.targets["channels"]["state"]; obs = t["observed_state"]
index = {cid: i for i, cid in enumerate(c.ids)}
runs = {(str(r["pair_id"]), str(r["run_label"])): str(r["condition_id"]) for r in c.run_rows}
PX_PER_M = 32.0                                                    # approx. scale at the V-JEPA input (384 px crop)
out = {}
for kind in ("load_mass", "com_shift"):
    rows = []
    for p in c.pair_rows:
        if p["intervention_type"] != kind: continue
        a, b = index[runs[(str(p["pair_id"]), "A")]], index[runs[(str(p["pair_id"]), "B")]]
        dpos = np.hypot(traj[a, :, 0] - traj[b, :, 0], traj[a, :, 1] - traj[b, :, 1])
        dyaw = np.degrees(np.abs(traj[a, :, 2] - traj[b, :, 2]))
        dv = np.abs(obs[a, :, ch.index("linear_vel_body_x")] - obs[b, :, ch.index("linear_vel_body_x")])
        droll = np.abs(obs[a, :, ch.index("roll_deg")] - obs[b, :, ch.index("roll_deg")])
        rows.append((dpos.max(), dpos[-1], dyaw.max(), dv.max(), droll.max(), dpos[4]))
    r = np.array(rows); q = lambda j: np.round(np.quantile(r[:, j], [0.1, 0.5, 0.9]), 4)
    out[kind] = dict(n=len(r), pos_max_m=q(0).tolist(), yaw_max_deg=q(2).tolist(), speed_max=q(3).tolist(), roll_max_deg=q(4).tolist())
    print(f"{kind:10s} n={len(r)}  10/50/90% over pairs of the max difference within 0-2.71 s:")
    print(f"   position {q(0)} m  (~{np.round(q(0) * PX_PER_M, 2)} px at model input) | at 1.2 s {q(5)} m")
    print(f"   yaw {q(2)} deg | forward speed {q(3)} m/s | roll {q(4)} deg")
json.dump(out, open("out/pair_effect_size.json", "w"), indent=1)
