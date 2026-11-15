"""No-video ceiling on ForkSafe-CF v2: ground-truth kinematics of the observation window -> payload state and future."""
import argparse, csv, json, sys, numpy as np, torch, torch.nn as nn
from pathlib import Path
from common_v2 import ContextV2, ROOT, bs, fmt
torch.set_num_threads(4)
ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--targets", required=True); ap.add_argument("--manifest-dir", required=True)
ap.add_argument("--train-parts", nargs="*"); ap.add_argument("--out", default="analysis_thesis/out")
args = ap.parse_args()
c = ContextV2(args.targets, args.manifest_dir); t = c.tnp; ch = c.targets["channels"]["state"]
obs = t["observed_state"].astype(np.float32); om = t["observed_state_mask"]
kin = ["roll_deg", "pitch_deg", "imu_ang_vel_x", "imu_ang_vel_y", "imu_ang_vel_z", "linear_vel_body_x", "linear_vel_body_y", "linear_vel_body_z", "imu_lin_acc_x", "imu_lin_acc_y", "imu_lin_acc_z"]
ki = [ch.index(n) for n in kin]; body = np.where(om[:, :, ki], obs[:, :, ki], 0.0).reshape(len(c.ids), -1)
times = np.array(c.targets["contract"]["observed_times_s"]); traj = np.zeros((len(c.ids), len(times), 3), np.float32)
for n, cid in enumerate(c.ids):
    with open(c.man[cid]["dense_log_path"]) as fh: rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: float(r["sim_time"])); tt = np.array([float(r["sim_time"]) for r in rows]); k = np.abs(tt[None] - times[:, None]).argmin(1)
    x = np.array([float(rows[i]["forklift_pos_x"]) for i in k]); y = np.array([float(rows[i]["forklift_pos_y"]) for i in k]); yaw = np.unwrap(np.radians([float(rows[i]["yaw_deg"]) for i in k]))
    dx, dy = x - x[0], y - y[0]; c0, s0 = np.cos(yaw[0]), np.sin(yaw[0]); traj[n, :, 0] = c0 * dx + s0 * dy; traj[n, :, 1] = -s0 * dx + c0 * dy; traj[n, :, 2] = yaw - yaw[0]
FEATS = {"kin+traj": np.concatenate([body, traj.reshape(len(c.ids), -1)], 1), "speed": obs[:, :, ch.index("linear_vel_body_x")]}
split = c.split.copy()
if args.train_parts: split = np.where((split == "train") & ~np.isin(c.part, args.train_parts), "unused", split)
if not (split == "val").any():        # tiny smoke sets only: carve validation/test pairs out of train
    order = np.argsort(c.cluster, kind="stable"); n = len(order); split[order[: int(.6 * n)]] = "train"; split[order[int(.6 * n): int(.8 * n)]] = "val"; split[order[int(.8 * n):]] = "test"
tr, va, te = (torch.tensor(np.flatnonzero(split == s)) for s in ("train", "val", "test")); trn = tr.numpy()
Y = np.concatenate([t["static"], t["future_state"][:, :, 0], t["future_summary"][:, :1]], 1).astype(np.float32)
M = np.concatenate([t["static_mask"], t["future_state_mask"][:, :, 0], t["future_summary_mask"][:, :1]], 1).astype(bool)
ymu = np.array([Y[trn, j][M[trn, j]].mean() for j in range(Y.shape[1])], np.float32); ysd = np.array([Y[trn, j][M[trn, j]].std() + 1e-6 for j in range(Y.shape[1])], np.float32)
Yn, Mt = torch.tensor((Y - ymu) / ysd), torch.tensor(M); tip = torch.tensor(t["future_tip"]).float(); tipm = torch.tensor(t["future_tip_mask"]).bool(); NOUT = Y.shape[1] + 1
def loss_fn(out, i):
    m = tipm[i]; return ((out[:, :-1] - Yn[i]) ** 2)[Mt[i]].mean() + nn.functional.binary_cross_entropy_with_logits(out[:, -1][m], tip[i][m])
def run(X, seed):
    torch.manual_seed(seed); model = nn.Sequential(nn.Linear(X.shape[1], 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, NOUT))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3); best, state, bad = 1e9, None, 0
    for ep in range(3000):
        model.train(); perm = tr[torch.randperm(len(tr))]
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]; opt.zero_grad(); loss_fn(model(X[b]), b).backward(); opt.step()
        model.eval()
        with torch.no_grad(): v = loss_fn(model(X[va]), va).item()
        if v < best - 1e-4: best, bad, state = v, 0, {k: w.clone() for k, w in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 80: break
    model.load_state_dict(state); model.eval()
    with torch.no_grad(): out = model(X[te]).numpy()
    phys = out[:, :-1] * ysd + ymu
    return {"static": phys[:, :2], "future_state": phys[:, 2:12, None], "future_summary": phys[:, 12:13], "tip_score": out[:, -1]}
res = {}
for name, feat in FEATS.items():
    X = torch.tensor(feat, dtype=torch.float32); mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; X = (X - mu) / sd
    preds = [run(X, s) for s in (17, 29, 43)]; pred = {k: np.stack([p[k] for p in preds]) for k in preds[0]}
    r = c.evaluate(pred, te.numpy(), bootstrap=2000); res[name] = r
    print(f"V2KIN {args.name} {name:9s} {fmt(r)}", flush=True)
json.dump(res, open(ROOT / args.out / f"v2_kin_{args.name}.json", "w"), indent=1)
