"""Feasibility test of 'physical hypothesis testing' (analysis-by-synthesis) for the hidden payload state.

Forward surrogate  f(theta, command) -> observed motion (0-2.71 s),  theta = (payload mass, lateral CoM), learned from the training conditions.
Identification     theta_hat = argmin_theta || f(theta, command) - observed motion ||   (dense search over hypotheses).
Compared with direct regression  motion (+command) -> theta  under the same held-out-level protocol.
Observation space here is the ground-truth kinematics (no vision): the cheapest check of whether hypothesis testing
interpolates to a payload level that never appears in training, where direct regression does not.
"""
import sys, json, numpy as np, torch, torch.nn as nn
from common import Context, bs, physical_config
torch.set_num_threads(6)
hold_key, hold_value = sys.argv[1], float(sys.argv[2])
c = Context("artifacts/targets/safety_query_full_comfix_med5.pt"); cfg, group, man = physical_config(c)
t = c.tnp; ch = c.targets["channels"]["state"]; obs = t["observed_state"].astype(np.float32); om = t["observed_state_mask"]
kin = ["roll_deg", "pitch_deg", "imu_ang_vel_x", "imu_ang_vel_y", "imu_ang_vel_z", "linear_vel_body_x", "linear_vel_body_y", "linear_vel_body_z", "imu_lin_acc_x", "imu_lin_acc_y", "imu_lin_acc_z"]
ki = [ch.index(n) for n in kin]
Yraw = np.concatenate([np.where(om[:, :, ki], obs[:, :, ki], 0.0).reshape(len(c.ids), -1), np.load("out/obs_trajectory.npy").reshape(len(c.ids), -1)], 1).astype(np.float32)   # observed motion, 140-d
SPEED = {"slow": -15.0, "medium": -20.0, "fast": -25.0}; STEER = {"wide": 1.0, "medium": 1.25, "tight": 1.5}
cmd = np.array([[SPEED[man[i]["speed_label"]], (1.0 if man[i]["turn_direction"] == "left" else -1.0) * STEER[man[i]["steering_label"]]] for i in c.ids], np.float32)
theta = t["static"].astype(np.float32)                                       # [N,2] mass (kg), CoM (m)
attr = theta[:, 0] if hold_key == "mass" else theta[:, 1]
held = np.isclose(attr, hold_value); main = c.split
tr, va, te = np.flatnonzero((main == "train") & ~held), np.flatnonzero((main == "val") & ~held), np.flatnonzero(held)
ymu, ysd = Yraw[tr].mean(0), Yraw[tr].std(0) + 1e-6; Y = torch.tensor((Yraw - ymu) / ysd)
TH_MU, TH_SD = np.array([480.0, 0.0], np.float32), np.array([170.0, 0.2], np.float32); CM_MU, CM_SD = cmd[tr].mean(0), cmd[tr].std(0) + 1e-6
X = torch.tensor(np.concatenate([(theta - TH_MU) / TH_SD, (cmd - CM_MU) / CM_SD], 1))

def fit(make, inp, out, seed, epochs=4000, patience=150):
    torch.manual_seed(seed); model = make(); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4); best, state, bad = 1e9, None, 0
    tri, vai = torch.tensor(tr), torch.tensor(va)
    for ep in range(epochs):
        model.train(); perm = tri[torch.randperm(len(tri))]
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]; opt.zero_grad(); ((model(inp[b]) - out[b]) ** 2).mean().backward(); opt.step()
        model.eval()
        with torch.no_grad(): v = ((model(inp[vai]) - out[vai]) ** 2).mean().item()
        if v < best - 1e-5: best, bad, state = v, 0, {k: w.clone() for k, w in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(state); model.eval(); return model, best

mass_grid = np.arange(180.0, 781.0, 5.0, dtype=np.float32); com_grid = np.arange(-0.35, 0.351, 0.01, dtype=np.float32)
G = np.stack(np.meshgrid(mass_grid, com_grid, indexing="ij"), -1).reshape(-1, 2)                    # hypotheses
Gn = torch.tensor((G - TH_MU) / TH_SD)
res = {}; preds = {"hypothesis": [], "regression": [], "regression+cmd": []}
for seed in (17, 29, 43):
    fwd, vloss = fit(lambda: nn.Sequential(nn.Linear(4, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, Y.shape[1])), X, Y, seed)
    est = np.zeros((len(te), 2), np.float32)
    with torch.no_grad():
        for n, i in enumerate(te):
            ci = torch.tensor((cmd[i] - CM_MU) / CM_SD).expand(len(Gn), -1)
            err = ((fwd(torch.cat([Gn, ci], 1)) - Y[i]) ** 2).mean(1)
            est[n] = G[int(err.argmin())]
    preds["hypothesis"].append(est)
    Tn = torch.tensor((theta - TH_MU) / TH_SD)
    for name, inp in (("regression", Y), ("regression+cmd", torch.cat([Y, torch.tensor((cmd - CM_MU) / CM_SD)], 1))):
        reg, _ = fit(lambda: nn.Sequential(nn.Linear(inp.shape[1], 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 2)), inp, Tn, seed, epochs=3000, patience=100)
        with torch.no_grad(): preds[name].append(reg(inp[torch.tensor(te)]).numpy() * TH_SD + TH_MU)
    print(f"seed {seed}: forward-model val loss {vloss:.4f}", flush=True)
index = {cid: i for i, cid in enumerate(c.ids)}; pos = {int(g): j for j, g in enumerate(te)}
runs = {(str(r["pair_id"]), str(r["run_label"])): str(r["condition_id"]) for r in c.run_rows}
pairs = {"load_mass": [], "com_shift": []}
for row in c.pair_rows:
    a, b = pos.get(index[runs[(str(row["pair_id"]), "A")]]), pos.get(index[runs[(str(row["pair_id"]), "B")]])
    if a is not None and b is not None: pairs[row["intervention_type"]].append((a, b))
for name, p in preds.items():
    p = np.stack(p); tt = theta[te]
    r = dict(mass_mae=float(np.abs(p[:, :, 0] - tt[:, 0]).mean()), mass_bias=float((p[:, :, 0] - tt[:, 0]).mean()), com_mae=float(np.abs(p[:, :, 1] - tt[:, 1]).mean()), com_bias=float((p[:, :, 1] - tt[:, 1]).mean()))
    if pairs["com_shift"]: r["com_direction"] = float(bs.pair_correctness(p[:, :, 1], tt[:, 1], pairs["com_shift"]).mean())
    if pairs["load_mass"]: r["mass_direction"] = float(bs.pair_correctness(p[:, :, 0], tt[:, 0], pairs["load_mass"]).mean())
    res[name] = r
    print(f"HYP held-out {hold_key}={hold_value:g} n={len(te)} {name:15s} " + " ".join(f"{k}={v:+.3f}" if 'bias' in k else f"{k}={v:.3f}" for k, v in r.items()), flush=True)
json.dump(res, open(f"out/hypothesis_test_kin_{hold_key}_{hold_value:g}.json", "w"), indent=1)
