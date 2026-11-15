"""No-video kinematics readout under the same held-out-level protocol as psq_heldout.py.
If perfect kinematics cannot interpolate to an unseen payload level either, the limit is the coarse training grid, not the video representation."""
import sys, json, numpy as np, torch, torch.nn as nn
from common import Context, ROOT, bs, physical_config, evaluate_any, fmt_any
torch.set_num_threads(6)
hold_key, hold_value = sys.argv[1], sys.argv[2]
c = Context("artifacts/targets/safety_query_full_comfix_med5.pt")
cfg, group, man = physical_config(c)
t = c.tnp; ch = c.targets["channels"]["state"]; obs = t["observed_state"].astype(np.float32); om = t["observed_state_mask"]
kin = ["roll_deg", "pitch_deg", "imu_ang_vel_x", "imu_ang_vel_y", "imu_ang_vel_z", "linear_vel_body_x", "linear_vel_body_y", "linear_vel_body_z", "imu_lin_acc_x", "imu_lin_acc_y", "imu_lin_acc_z"]
ki = [ch.index(n) for n in kin]
body = np.where(om[:, :, ki], obs[:, :, ki], 0.0).reshape(len(c.ids), -1); traj = np.load("out/obs_trajectory.npy").reshape(len(c.ids), -1)
FEATS = {"kin+traj": np.concatenate([body, traj], 1), "traj": traj}
attr = {"load_mass_kg": np.array([k[0] for k in cfg]), "com": np.array([k[1] for k in cfg]), "speed_label": np.array([man[i]["speed_label"] for i in c.ids])}[hold_key]
held = attr == (float(hold_value) if attr.dtype.kind == "f" else hold_value)
main = c.split; tr = torch.tensor(np.flatnonzero((main == "train") & ~held)); va = torch.tensor(np.flatnonzero((main == "val") & ~held)); idx = np.flatnonzero(held)
Y = np.concatenate([t["static"], t["future_state"][:, :, 0], t["future_summary"][:, :1]], 1).astype(np.float32)
M = np.concatenate([t["static_mask"], t["future_state_mask"][:, :, 0], t["future_summary_mask"][:, :1]], 1).astype(bool)
trn = tr.numpy(); ymu = np.array([Y[trn, j][M[trn, j]].mean() for j in range(Y.shape[1])], np.float32); ysd = np.array([Y[trn, j][M[trn, j]].std() + 1e-6 for j in range(Y.shape[1])], np.float32)
Yn, Mt = torch.tensor((Y - ymu) / ysd), torch.tensor(M); tip = torch.tensor(t["future_tip"]).float(); tipm = torch.tensor(t["future_tip_mask"]).bool(); NOUT = Y.shape[1] + 1
def loss_fn(out, i):
    m = tipm[i]; return ((out[:, :-1] - Yn[i]) ** 2)[Mt[i]].mean() + nn.functional.binary_cross_entropy_with_logits(out[:, -1][m], tip[i][m])
def run(X, seed, kind):
    torch.manual_seed(seed); d = X.shape[1]
    model = nn.Linear(d, NOUT) if kind == "linear" else nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, NOUT))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3); best, state, bad = 1e9, None, 0
    for ep in range(2000):
        model.train(); perm = tr[torch.randperm(len(tr))]
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]; opt.zero_grad(); loss_fn(model(X[b]), b).backward(); opt.step()
        model.eval()
        with torch.no_grad(): v = loss_fn(model(X[va]), va).item()
        if v < best - 1e-4: best, bad, state = v, 0, {k: w.clone() for k, w in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 60: break
    model.load_state_dict(state); model.eval()
    with torch.no_grad(): out = model(X[torch.tensor(idx)]).numpy()
    phys = out[:, :-1] * ysd + ymu
    return {"static": phys[:, :2], "future_state": phys[:, 2:12, None], "future_summary": phys[:, 12:13], "tip_score": out[:, -1]}
res = {}
for name, feat in FEATS.items():
    X = torch.tensor(feat, dtype=torch.float32); mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; X = (X - mu) / sd
    for kind in ("linear", "mlp"):
        preds = [run(X, s, kind) for s in (17, 29, 43)]; pred = {k: np.stack([p[k] for p in preds]) for k in preds[0]}
        r = evaluate_any(c, pred, idx, np.array([k[2] for k in cfg]), bootstrap=1000)
        r["mass_bias"] = float((pred["static"][:, :, 0] - t["static"][idx, 0]).mean()); r["pred_mass_sd"] = float(pred["static"][:, :, 0].std())
        res[f"{name}/{kind}"] = r
        print(f"KIN-HELD {hold_key}={hold_value} {name:9s} {kind:6s} bias={r['mass_bias']:+.1f} {fmt_any(r)}", flush=True)
json.dump(res, open(f"out/kinematics_heldout_{hold_key}_{hold_value}.json", "w"), indent=1)
