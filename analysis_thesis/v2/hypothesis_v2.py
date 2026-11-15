"""Physical-hypothesis testing on ForkSafe-CF v2 (continuous payload and manoeuvre parameters).

Forward model   f(theta, command) -> [observation code ; future outcome]      theta = (payload mass, lateral CoM)
Identification  theta_hat = argmin_theta || f_obs(theta, command) - observation code ||     (dense hypothesis search)
Prediction      future outcome = f_fut(theta_hat, command)
What-if         for a matched pair (A, B): predict B's future from A's identified payload plus the known intervention,
                f_fut(theta_hat_A + (theta_B - theta_A), command), and compare with what really happened in B.
Baselines       direct regression  observation code (+command) -> theta and -> future outcome.

--obs kin     observation code = ground-truth kinematics of the observation window (no vision; upper reference)
--obs latent  observation code = top-k principal components of the frozen V-JEPA spatial tokens (train-set PCA)
"""
import argparse, csv, json, numpy as np, torch, torch.nn as nn
from common_v2 import ContextV2, ROOT, bs

torch.set_num_threads(6)
ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--targets", required=True); ap.add_argument("--manifest-dir", required=True)
ap.add_argument("--obs", choices=("kin", "latent"), default="kin"); ap.add_argument("--features"); ap.add_argument("--components", type=int, default=128)
ap.add_argument("--out", default="analysis_thesis/out"); ap.add_argument("--seeds", default="17,29,43")
args = ap.parse_args()
c = ContextV2(args.targets, args.manifest_dir); t = c.tnp; N = len(c.ids)
split = c.split.copy()
if not (split == "val").any():      # tiny smoke sets only
    order = np.argsort(c.cluster, kind="stable"); split[order[: int(.6 * N)]] = "train"; split[order[int(.6 * N): int(.8 * N)]] = "val"; split[order[int(.8 * N):]] = "test"
tr, va, te = (np.flatnonzero(split == s) for s in ("train", "val", "test"))

# ---- observation code -------------------------------------------------------------------------------------------
if args.obs == "kin":
    ch = c.targets["channels"]["state"]; obs = t["observed_state"].astype(np.float32); om = t["observed_state_mask"]
    kin = ["roll_deg", "pitch_deg", "imu_ang_vel_x", "imu_ang_vel_y", "imu_ang_vel_z", "linear_vel_body_x", "linear_vel_body_y", "linear_vel_body_z", "imu_lin_acc_x", "imu_lin_acc_y", "imu_lin_acc_z"]
    ki = [ch.index(n) for n in kin]; body = np.where(om[:, :, ki], obs[:, :, ki], 0.0).reshape(N, -1)
    times = np.array(c.targets["contract"]["observed_times_s"]); traj = np.zeros((N, len(times), 3), np.float32)
    for n, cid in enumerate(c.ids):
        with open(c.man[cid]["dense_log_path"]) as fh: rows = list(csv.DictReader(fh))
        rows.sort(key=lambda r: float(r["sim_time"])); tt = np.array([float(r["sim_time"]) for r in rows]); k = np.abs(tt[None] - times[:, None]).argmin(1)
        x = np.array([float(rows[i]["forklift_pos_x"]) for i in k]); y = np.array([float(rows[i]["forklift_pos_y"]) for i in k]); yaw = np.unwrap(np.radians([float(rows[i]["yaw_deg"]) for i in k]))
        dx, dy = x - x[0], y - y[0]; c0, s0 = np.cos(yaw[0]), np.sin(yaw[0]); traj[n, :, 0] = c0 * dx + s0 * dy; traj[n, :, 1] = -s0 * dx + c0 * dy; traj[n, :, 2] = yaw - yaw[0]
    O = np.concatenate([body, traj.reshape(N, -1)], 1).astype(np.float32)
    O = (O - O[tr].mean(0)) / (O[tr].std(0) + 1e-6)
else:
    F = torch.load(ROOT / args.features, map_location="cpu", weights_only=False)
    assert [str(i) for i in F["contract"]["condition_ids"]] == c.ids
    X = F["features"].float().reshape(N, -1); X = X - X[tr].mean(0, keepdim=True)
    gram = X[tr] @ X[tr].T; evals, evecs = torch.linalg.eigh(gram.double()); k = min(args.components, len(tr) - 1)
    top = torch.argsort(evals, descending=True)[:k]; comps = (X[tr].T.double() @ evecs[:, top] / evals[top].clamp_min(1e-9).sqrt()).float()      # [D,k] orthonormal
    O = (X @ comps).numpy(); O = O / (O[tr].std() + 1e-6)                                         # keep the relative variance of the components
    print(f"latent code: {k} components explain {float(evals[top].sum() / evals.clamp_min(0).sum()):.3f} of the train variance", flush=True)

# ---- hypotheses, commands, future outcome ----------------------------------------------------------------------
theta = t["static"].astype(np.float32); TH_MU, TH_SD = theta[tr].mean(0), theta[tr].std(0) + 1e-6
cmd = np.array([[c.man[i]["drive_velocity_command"], c.man[i]["steering_position_command"]] for i in c.ids], np.float32); CM_MU, CM_SD = cmd[tr].mean(0), cmd[tr].std(0) + 1e-6
fut = np.concatenate([t["future_state"][:, :, 0], t["future_summary"][:, :1]], 1).astype(np.float32)          # roll x10, min ZMP margin
fmask = np.concatenate([t["future_state_mask"][:, :, 0], t["future_summary_mask"][:, :1]], 1).astype(bool)
FU_MU = np.array([fut[tr, j][fmask[tr, j]].mean() for j in range(fut.shape[1])], np.float32); FU_SD = np.array([fut[tr, j][fmask[tr, j]].std() + 1e-6 for j in range(fut.shape[1])], np.float32)
tip = t["future_tip"].astype(np.float32); tipm = t["future_tip_mask"].astype(bool)
Ot, Tn, Cn = torch.tensor(O), torch.tensor((theta - TH_MU) / TH_SD), torch.tensor((cmd - CM_MU) / CM_SD)
Fn, Fm, Tip, Tipm = torch.tensor((fut - FU_MU) / FU_SD), torch.tensor(fmask), torch.tensor(tip), torch.tensor(tipm)
DO, DF = O.shape[1], fut.shape[1]

def mlp(i, o): return nn.Sequential(nn.Linear(i, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, o))

def fut_loss(out, idx):          # out: [B, DF+1]
    m = Tipm[idx]
    return ((out[:, :DF] - Fn[idx]) ** 2)[Fm[idx]].mean() + nn.functional.binary_cross_entropy_with_logits(out[:, DF][m], Tip[idx][m])

def train(model, loss_of, seed, epochs=4000, patience=150):
    torch.manual_seed(seed); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4); best, state, bad = 1e9, None, 0
    tri, vai = torch.tensor(tr), torch.tensor(va)
    for ep in range(epochs):
        model.train(); perm = tri[torch.randperm(len(tri))]
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]; opt.zero_grad(); loss_of(model, b).backward(); opt.step()
        model.eval()
        with torch.no_grad(): v = loss_of(model, vai).item()
        if v < best - 1e-5: best, bad, state = v, 0, {k: w.clone() for k, w in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(state); model.eval(); return model

lo, hi = theta[tr].min(0), theta[tr].max(0); span = hi - lo
mass_grid = np.linspace(lo[0] - 0.1 * span[0], hi[0] + 0.1 * span[0], 97, dtype=np.float32); com_grid = np.linspace(lo[1] - 0.1 * span[1], hi[1] + 0.1 * span[1], 73, dtype=np.float32)
G = np.stack(np.meshgrid(mass_grid, com_grid, indexing="ij"), -1).reshape(-1, 2); Gn = torch.tensor((G - TH_MU) / TH_SD)
index = c.index; pairs = [(index[p["condition_id_A"]], index[p["condition_id_B"]], p["intervention_type"], p["quality_tier"]) for p in c.pair_rows if p["condition_id_A"] in index and p["condition_id_B"] in index]
test_set = set(te.tolist()); test_pairs = [p for p in pairs if p[0] in test_set and p[1] in test_set]

R = {k: [] for k in ("hyp_theta", "reg_theta", "hyp_fut", "reg_fut", "oracle_fut", "cf_fut", "cf_fut_oracle")}
for seed in [int(s) for s in args.seeds.split(",")]:
    fwd = train(mlp(4, DO + DF + 1), lambda m, b: ((m(torch.cat([Tn[b], Cn[b]], 1))[:, :DO] - Ot[b]) ** 2).mean() + fut_loss(m(torch.cat([Tn[b], Cn[b]], 1))[:, DO:], b), seed)
    reg = train(mlp(DO + 2, 2 + DF + 1), lambda m, b: ((m(torch.cat([Ot[b], Cn[b]], 1))[:, :2] - Tn[b]) ** 2).mean() + fut_loss(m(torch.cat([Ot[b], Cn[b]], 1))[:, 2:], b), seed)
    with torch.no_grad():
        est = np.zeros((N, 2), np.float32)
        for i in te:
            err = ((fwd(torch.cat([Gn, Cn[i].expand(len(Gn), -1)], 1))[:, :DO] - Ot[i]) ** 2).mean(1); est[i] = G[int(err.argmin())]
        estn = torch.tensor((est - TH_MU) / TH_SD)
        hyp_f = fwd(torch.cat([estn, Cn], 1))[:, DO:].numpy()                         # future from the identified hypothesis
        ora_f = fwd(torch.cat([Tn, Cn], 1))[:, DO:].numpy()                           # future from the true payload (model ceiling)
        r = reg(torch.cat([Ot, Cn], 1)).numpy()
        cf = np.full((N, DF + 1), np.nan, np.float32); cfo = cf.copy()
        for a, b, kind, tier in test_pairs:                                            # what-if: A observed, intervention known, B's future predicted
            for src, dst in ((a, b), (b, a)):
                th = est[src] + (theta[dst] - theta[src]); thn = torch.tensor((th - TH_MU) / TH_SD)[None]
                cf[dst] = fwd(torch.cat([thn, Cn[dst][None]], 1))[0, DO:].numpy(); cfo[dst] = ora_f[dst]
    R["hyp_theta"].append(est); R["reg_theta"].append(r[:, :2] * TH_SD + TH_MU); R["hyp_fut"].append(hyp_f); R["reg_fut"].append(r[:, 2:]); R["oracle_fut"].append(ora_f); R["cf_fut"].append(cf); R["cf_fut_oracle"].append(cfo)
    print(f"{args.name} seed {seed} done", flush=True)
R = {k: np.stack(v) for k, v in R.items()}

def theta_metrics(p):
    tt = theta[te]; q = p[:, te]; out = {}
    for j, nm in enumerate(("mass", "com")):
        m = q[:, :, j].mean(0); out[f"{nm}_mae"] = float(np.abs(q[:, :, j] - tt[:, j]).mean()); out[f"{nm}_r2"] = float(1 - ((m - tt[:, j]) ** 2).sum() / ((tt[:, j] - tt[:, j].mean()) ** 2).sum())
    pos = {int(g): j for j, g in enumerate(te)}
    for kind, j, nm in (("load_mass", 0, "mass_dir"), ("com_shift", 1, "com_dir")):
        pr = [(pos[a], pos[b]) for a, b, k2, _ in test_pairs if k2 == kind]
        if pr: out[nm] = float(bs.pair_correctness(q[:, :, j], tt[:, j], pr).mean())
    return out

def fut_metrics(f, idx):
    idx = np.asarray(idx); ok = np.isfinite(f[0, idx, 0]); idx = idx[ok]
    if len(idx) == 0: return {}
    phys = f[:, idx, :DF] * FU_SD + FU_MU; out = {"n": int(len(idx))}
    out["roll_mae"] = float(np.mean([bs.condition_mae(s[:, :10], fut[idx, :10], fmask[idx, :10]).mean() for s in phys]))
    zm = fmask[idx, 10]; out["zmp_mae"] = float(np.mean([np.abs(s[zm, 10] - fut[idx, 10][zm]).mean() for s in phys]))
    m = tipm[idx]; lab = tip[idx][m].astype(int)
    if len(np.unique(lab)) == 2:
        out["tip_auroc"] = float(np.mean([bs.auc(lab, s[idx, DF][m]) for s in f])); out["tip_acc"] = float(np.mean([((s[idx, DF][m] > 0) == (lab == 1)).mean() for s in f]))
    return out

res = {"counts": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te)), "test_pairs": len(test_pairs)},
       "identification": {"hypothesis": theta_metrics(R["hyp_theta"]), "regression": theta_metrics(R["reg_theta"])},
       "future": {"hypothesis": fut_metrics(R["hyp_fut"], te), "regression": fut_metrics(R["reg_fut"], te), "true_payload": fut_metrics(R["oracle_fut"], te)},
       "what_if": {"identified_payload": fut_metrics(R["cf_fut"], te), "true_payload": fut_metrics(R["cf_fut_oracle"], te)}}
gold = sorted({x for a, b, k2, tier in test_pairs if tier == "gold" for x in (a, b)})
res["what_if_outcome_flips"] = {"identified_payload": fut_metrics(R["cf_fut"], gold), "true_payload": fut_metrics(R["cf_fut_oracle"], gold)} if gold else {}
json.dump(res, open(ROOT / args.out / f"v2_hyp_{args.name}.json", "w"), indent=1)
np.savez(ROOT / args.out / f"v2_hyp_pred_{args.name}.npz", test=te, theta=theta, **R)
for sec in ("identification", "future", "what_if", "what_if_outcome_flips"):
    for k, v in res[sec].items(): print(f"HYPV2 {args.name} {sec:22s} {k:20s} " + " ".join(f"{a}={b:.3f}" if isinstance(b, float) else f"{a}={b}" for a, b in v.items()), flush=True)
