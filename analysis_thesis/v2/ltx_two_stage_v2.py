"""What does a planner lose if it takes the payload from a generated future? (ForkSafe-CF v2, held-out test pairs)

The second stage ([planned command, payload mass] -> future) is trained on the training pairs with the true mass and
applied to the test runs with the mass taken from
  true mass              upper bound
  observed video         V-JEPA readout of the 2.71 s observation            (psq_v2.py, out/v2_pred_<name>.npz)
  real future            LTX-latent readout of the real future               (ltx_v2.py, real-future branch)
  generated future       the same readout applied to LTX continuations       (ltx_v2.py, rollout branch)
  no mass                training-mean mass
Needs out/v2ltx_pred_<name>.npz (ltx_v2.py) and out/v2_pred_<name>.npz. CPU only.
Run from the safety_probing root:
  PYTHONPATH=analysis_thesis/v2:src .venv/bin/python analysis_thesis/v2/ltx_two_stage_v2.py --name p5
"""
import argparse, json
import numpy as np, torch, torch.nn as nn
from common_v2 import ContextV2, ROOT

torch.set_num_threads(4)
ap = argparse.ArgumentParser(); ap.add_argument("--name", default="p5"); ap.add_argument("--out", default="analysis_thesis/out"); args = ap.parse_args()
D = f"artifacts/v2/{args.name}"; c = ContextV2(f"{D}/targets_truecom.pt", f"{D}/manifests"); t = c.tnp; N = len(c.ids); OUT = ROOT / args.out
L = np.load(OUT / f"v2ltx_pred_{args.name}.npz", allow_pickle=True); assert [str(i) for i in L["ids"]] == c.ids
V = np.load(OUT / f"v2_pred_{args.name}.npz", allow_pickle=True); assert [str(i) for i in V["ids"]] == c.ids
te = L["gen_idx"].astype(int); assert set(c.split[te]) == {"test"}
mass = t["static"][:, 0].astype(np.float32); cmd = np.array([[c.man[i]["drive_velocity_command"], c.man[i]["steering_position_command"]] for i in c.ids], np.float32)
MASS_IN = {"true mass": mass[te], "observed video (V-JEPA)": V["static"][:, te, 0].mean(0), "real future (LTX latents)": L["real_static"][:, te, 0].mean(0),
           "generated future (LTX rollouts)": L["gen_static"][:, :, 0].mean(0), "no mass (train mean)": np.full(len(te), mass[c.split == "train"].mean(), np.float32)}
for k, v in MASS_IN.items(): print(f"mass input {k:32s} MAE {np.abs(v - mass[te]).mean():6.1f} kg | corr with the true mass {np.corrcoef(v, mass[te])[0, 1] if np.std(v) > 0 else float('nan'):+.3f}", flush=True)
feat = lambda cm, ms: np.c_[cm, np.abs(cm[:, 1]), np.sign(cm[:, 1]), ms].astype(np.float32)
Y = np.concatenate([t["future_state"][:, :, 0], t["future_summary"][:, :1]], 1).astype(np.float32); M = np.concatenate([t["future_state_mask"][:, :, 0], t["future_summary_mask"][:, :1]], 1).astype(bool)
tip_t = torch.tensor(t["future_tip"]).float(); tipm_t = torch.tensor(t["future_tip_mask"]).bool(); tipk, tipv = t["future_tip_mask"].astype(bool), t["future_tip"].astype(bool)
tr, va = torch.tensor(np.flatnonzero(c.split == "train")), torch.tensor(np.flatnonzero(c.split == "val")); trn = tr.numpy()
ymu = np.array([Y[trn, j][M[trn, j]].mean() for j in range(Y.shape[1])], np.float32); ysd = np.array([Y[trn, j][M[trn, j]].std() + 1e-6 for j in range(Y.shape[1])], np.float32)
Yn, Mt = torch.tensor((Y - ymu) / ysd), torch.tensor(M); F = feat(cmd, mass); mu, sd = F[trn].mean(0), F[trn].std(0) + 1e-6; X = torch.tensor((F - mu) / sd)
loss_fn = lambda out, i: ((out[:, :-1] - Yn[i]) ** 2)[Mt[i]].mean() + nn.functional.binary_cross_entropy_with_logits(out[:, -1][tipm_t[i]], tip_t[i][tipm_t[i]])
S = {k: {"roll": [], "zmp": [], "score": []} for k in MASS_IN}
for seed in (17, 29, 43):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(X.shape[1], 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, Y.shape[1] + 1))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3); best, state, bad = 1e9, None, 0
    for ep in range(3000):
        model.train(); p = tr[torch.randperm(len(tr))]
        for i in range(0, len(p), 64):
            b = p[i:i + 64]; opt.zero_grad(); loss_fn(model(X[b]), b).backward(); opt.step()
        model.eval()
        with torch.no_grad(): v = loss_fn(model(X[va]), va).item()
        if v < best - 1e-4: best, bad, state = v, 0, {k: w.clone() for k, w in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 80: break
    model.load_state_dict(state); model.eval()
    with torch.no_grad():
        for k, ms in MASS_IN.items():
            o = model(torch.tensor((feat(cmd[te], np.asarray(ms, np.float32)) - mu) / sd)).numpy(); phys = o[:, :-1] * ysd + ymu
            S[k]["roll"].append(phys[:, :10]); S[k]["zmp"].append(phys[:, 10]); S[k]["score"].append(o[:, -1])
res = {"n_test": int(len(te)), "mass_mae_kg": {k: float(np.abs(v - mass[te]).mean()) for k, v in MASS_IN.items()}}
for k in MASS_IN:
    full = {"static": np.repeat(t["static"][None, te], 3, 0).astype(np.float32), "future_state": np.stack(S[k]["roll"])[..., None], "future_summary": np.stack(S[k]["zmp"])[..., None], "tip_score": np.stack(S[k]["score"])}
    r = c.evaluate(full, te, bootstrap=3000); acc = float(((full["tip_score"].mean(0) > 0) == tipv[te])[tipk[te]].mean())
    res[k] = {kk: r.get(kk) for kk in ("roll_mae_deg", "tip_auroc", "outcome_flip", "outcome_flip_mass", "n_outcome_flip_mass")} | {"accuracy": acc}
    g = lambda d: "n/a" if d is None else f"{d['point']:.3f}[{d['ci_low']:.3f},{d['ci_high']:.3f}]"
    print(f"LTX2STAGE {args.name} {k:32s} roll {g(r['roll_mae_deg'])} deg | AUROC {g(r.get('tip_auroc'))} | accuracy {acc:.3f} | mass pairs {g(r.get('outcome_flip_mass'))} (n={r.get('n_outcome_flip_mass')})", flush=True)
json.dump(res, open(OUT / f"v2ltx_two_stage_{args.name}.json", "w"), indent=1)
