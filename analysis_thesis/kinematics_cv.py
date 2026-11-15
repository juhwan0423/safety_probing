"""No-video ceiling on ForkSafe-CF v1 under the clean protocol (mos5): ground-truth kinematics of the observation
window -> payload state and future. Same folds as psq_cv.py --protocol mos5 (every route that shares a pre-turn
segment is in one fold), 3 seeds, true-CoM targets. CPU only.

Feature sets: speed (forward speed at the 10 observation times) and kin+traj (roll, pitch, body rates, body-frame
velocity, IMU acceleration, planar pose relative to the first sample).
Run:  PYTHONPATH=. ../.venv/bin/python kinematics_cv.py [--frames 65]
"""
import argparse, json
import numpy as np, torch, torch.nn as nn
from common import Context, bs, physical_config, evaluate_any, fmt_any

ap = argparse.ArgumentParser(); ap.add_argument("--frames", type=int, default=65); ap.add_argument("--protocol", default="mos5", choices=("mos5", "config5"))
args = ap.parse_args()
torch.set_num_threads(4)
c = Context(f"artifacts/targets/thesis_v1_obs{args.frames}_truecom.pt")
t = c.tnp; ch = c.targets["channels"]["state"]
obs = t["observed_state"].astype(np.float32); om = t["observed_state_mask"]
kin = ["roll_deg", "pitch_deg", "imu_ang_vel_x", "imu_ang_vel_y", "imu_ang_vel_z", "linear_vel_body_x", "linear_vel_body_y", "linear_vel_body_z", "imu_lin_acc_x", "imu_lin_acc_y", "imu_lin_acc_z"]
ki = [ch.index(n) for n in kin]; body = np.where(om[:, :, ki], obs[:, :, ki], 0.0).reshape(len(c.ids), -1)
feats = {"speed": obs[:, :, ch.index("linear_vel_body_x")]}
if args.frames == 65:
    traj = np.load("out/obs_trajectory.npy").reshape(len(c.ids), -1); feats["kin+traj"] = np.concatenate([body, traj], 1)
else:
    feats["body_kin"] = body
cfg, group, man = physical_config(c)
speed = np.array([man[i]["speed_label"] for i in c.ids])
mos = np.array([f"{k[0]:.0f}|{k[1]:+.2f}|{sp}" for k, sp in zip(cfg, speed)])
rs = np.random.default_rng(20261116)
if args.protocol == "mos5":
    groups = sorted(set(mos)); perm = rs.permutation(len(groups)); fold_of = {g: int(j % 5) for g, j in zip(np.array(groups)[perm], range(len(groups)))}
    fold = np.array([fold_of[g] for g in mos]); unit = np.array([groups.index(g) for g in mos]); clusters = mos
else:
    ids = {k: j for j, k in enumerate(sorted(set(cfg)))}; unit = np.array([ids[k] for k in cfg]); perm = rs.permutation(len(ids)); fc = np.empty(len(ids), int); fc[perm] = np.arange(len(ids)) % 5
    fold = fc[unit]; clusters = np.array([k[2] for k in cfg])
Y = np.concatenate([t["static"], t["future_state"][:, :, 0], t["future_summary"][:, :1]], 1).astype(np.float32)
M = np.concatenate([t["static_mask"], t["future_state_mask"][:, :, 0], t["future_summary_mask"][:, :1]], 1).astype(bool)
tip = torch.tensor(t["future_tip"]).float(); tipm = torch.tensor(t["future_tip_mask"]).bool(); NOUT = Y.shape[1] + 1; N = len(c.ids)
res = {}
for name, F in feats.items():
    OOF = {"static": np.zeros((3, N, 2), np.float32), "future_state": np.zeros((3, N, 10, 1), np.float32), "future_summary": np.zeros((3, N, 1), np.float32), "tip_score": np.zeros((3, N), np.float32)}
    for k in range(5):
        test = fold == k; rest = np.unique(unit[~test]); val_units = np.random.default_rng(1000 + k).choice(rest, size=max(1, int(round(0.15 * len(rest)))), replace=False)
        va_mask = np.isin(unit, val_units) & ~test; tr_mask = ~test & ~va_mask
        tr, va, te = (torch.tensor(np.flatnonzero(m)) for m in (tr_mask, va_mask, test)); trn = tr.numpy()
        ymu = np.array([Y[trn, j][M[trn, j]].mean() for j in range(Y.shape[1])], np.float32); ysd = np.array([Y[trn, j][M[trn, j]].std() + 1e-6 for j in range(Y.shape[1])], np.float32)
        Yn, Mt = torch.tensor((Y - ymu) / ysd), torch.tensor(M)
        X = torch.tensor(F, dtype=torch.float32); mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; X = (X - mu) / sd

        def loss_fn(out, i):
            m = tipm[i]; return ((out[:, :-1] - Yn[i]) ** 2)[Mt[i]].mean() + nn.functional.binary_cross_entropy_with_logits(out[:, -1][m], tip[i][m])

        for si, seed in enumerate((17, 29, 43)):
            torch.manual_seed(seed)
            model = nn.Sequential(nn.Linear(X.shape[1], 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, NOUT))
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3); best, state, bad = 1e9, None, 0
            for ep in range(2000):
                model.train(); p = tr[torch.randperm(len(tr))]
                for i in range(0, len(p), 64):
                    b = p[i:i + 64]; opt.zero_grad(); loss_fn(model(X[b]), b).backward(); opt.step()
                model.eval()
                with torch.no_grad(): v = loss_fn(model(X[va]), va).item()
                if v < best - 1e-4: best, bad, state = v, 0, {kk: w.clone() for kk, w in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= 60: break
            model.load_state_dict(state); model.eval()
            with torch.no_grad(): out = model(X[te]).numpy()
            phys = out[:, :-1] * ysd + ymu; ten = te.numpy()
            OOF["static"][si, ten] = phys[:, :2]; OOF["future_state"][si, ten, :, 0] = phys[:, 2:12]; OOF["future_summary"][si, ten, 0] = phys[:, 12]; OOF["tip_score"][si, ten] = out[:, -1]
    r = evaluate_any(c, OOF, np.arange(N), clusters); res[name] = r
    np.savez(f"out/kincv_pred_{args.protocol}_{name.replace('+', '_')}_obs{args.frames}.npz", fold=fold, **OOF)
    print(f"KINCV {args.protocol} obs{args.frames} {name:9s} {fmt_any(r)}", flush=True)
json.dump(res, open(f"out/kincv_{args.protocol}_obs{args.frames}.json", "w"), indent=1)
