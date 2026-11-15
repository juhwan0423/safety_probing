"""No-video control: can ground-truth kinematics of the observation window (0-2.71 s) identify the hidden
payload state and predict the future? Tells us whether the information is in the motion at all.

Feature sets (all from simulator logs, sampled at the 10 observation tubelet times):
  speed      : forward speed only
  body_kin   : roll, pitch, body angular velocity (3), body-frame linear velocity (3), IMU linear acceleration (3)
  traj       : planar pose relative to the first sample (dx, dy, dyaw)
  kin+traj   : both
Same split, seeds, early stopping on validation, and paper metrics as the video readouts. CPU only.
"""
import json, numpy as np, torch, torch.nn as nn
from pathlib import Path
from common import Context, ROOT, fmt
import comfix as cf

torch.set_num_threads(6)
SEEDS = (17, 29, 43)
c = Context("artifacts/targets/safety_query_full_comfix_med5.pt")
t = c.tnp; ch = c.targets["channels"]["state"]
obs = t["observed_state"].astype(np.float32); obs_mask = t["observed_state_mask"]
kin_names = ["roll_deg", "pitch_deg", "imu_ang_vel_x", "imu_ang_vel_y", "imu_ang_vel_z", "linear_vel_body_x", "linear_vel_body_y",
             "linear_vel_body_z", "imu_lin_acc_x", "imu_lin_acc_y", "imu_lin_acc_z"]
ki = [ch.index(n) for n in kin_names]
body_kin = np.where(obs_mask[:, :, ki], obs[:, :, ki], 0.0).reshape(len(c.ids), -1)
speed = obs[:, :, ch.index("linear_vel_body_x")]

traj_path = Path("out/obs_trajectory.npy")
if traj_path.exists():
    traj = np.load(traj_path)
else:
    man = {str(r["condition_id"]): r for r in c.canonical_rows}
    times = np.array(c.targets["contract"]["observed_times_s"])
    traj = np.zeros((len(c.ids), len(times), 3), np.float32)
    for n, cid in enumerate(c.ids):
        rows = cf.read_rows(Path(man[cid]["dense_log_path"]))
        tt = np.array([float(r["sim_time"]) for r in rows]); k = np.abs(tt[None] - times[:, None]).argmin(1)
        x = np.array([float(rows[i]["forklift_pos_x"]) for i in k]); y = np.array([float(rows[i]["forklift_pos_y"]) for i in k])
        yaw = np.radians(np.array([float(rows[i]["yaw_deg"]) for i in k])); yaw = np.unwrap(yaw)
        dx, dy = x - x[0], y - y[0]; c0, s0 = np.cos(yaw[0]), np.sin(yaw[0])
        traj[n, :, 0] = c0 * dx + s0 * dy; traj[n, :, 1] = -s0 * dx + c0 * dy; traj[n, :, 2] = yaw - yaw[0]
    np.save(traj_path, traj)
traj_flat = traj.reshape(len(c.ids), -1)
FEATURES = {"speed": speed, "body_kin": body_kin, "traj": traj_flat, "kin+traj": np.concatenate([body_kin, traj_flat], 1)}

Y = np.concatenate([t["static"], t["future_state"][:, :, 0], t["future_summary"][:, :1]], 1).astype(np.float32)
M = np.concatenate([t["static_mask"], t["future_state_mask"][:, :, 0], t["future_summary_mask"][:, :1]], 1).astype(bool)
ymu = np.array([Y[c.train, j][M[c.train, j]].mean() for j in range(Y.shape[1])], np.float32)
ysd = np.array([Y[c.train, j][M[c.train, j]].std() + 1e-6 for j in range(Y.shape[1])], np.float32)
Yn = torch.tensor((Y - ymu) / ysd); Mt = torch.tensor(M)
tip = torch.tensor(t["future_tip"]).float(); tipm = torch.tensor(t["future_tip_mask"]).bool()
NOUT = Y.shape[1] + 1
tr, va, te = torch.tensor(c.train), torch.tensor(c.val), torch.tensor(c.test)

def loss_fn(out, idx):
    reg = ((out[:, :-1] - Yn[idx]) ** 2)[Mt[idx]].mean(); m = tipm[idx]
    return reg + nn.functional.binary_cross_entropy_with_logits(out[:, -1][m], tip[idx][m])

def run(X, seed, kind):
    torch.manual_seed(seed)
    d = X.shape[1]
    model = nn.Linear(d, NOUT) if kind == "linear" else nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, NOUT))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    best, state, bad = 1e9, None, 0
    for ep in range(2000):
        model.train(); perm = tr[torch.randperm(len(tr))]
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]; opt.zero_grad(); loss_fn(model(X[idx]), idx).backward(); opt.step()
        model.eval()
        with torch.no_grad(): v = loss_fn(model(X[va]), va).item()
        if v < best - 1e-4: best, bad, state = v, 0, {k: w.clone() for k, w in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 60: break
    model.load_state_dict(state); model.eval()
    with torch.no_grad(): out = model(X[te]).numpy()
    phys = out[:, :-1] * ysd + ymu
    return {"static": phys[:, :2], "future_state": phys[:, 2:12, None], "future_summary": phys[:, 12:13], "tip_score": out[:, -1]}

results = {}
for name, feat in FEATURES.items():
    X = torch.tensor(feat, dtype=torch.float32)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; X = (X - mu) / sd
    for kind in ("linear", "mlp"):
        preds = [run(X, s, kind) for s in SEEDS]
        pred = {k: np.stack([p[k] for p in preds]) for k in preds[0]}
        r = c.evaluate(pred, bootstrap=2000); results[f"{name}/{kind}"] = r
        print(f"{name:9s} {kind:6s} dim={feat.shape[1]:3d} {fmt(r)}", flush=True)
json.dump(results, open("out/kinematics_control.json", "w"), indent=1)
