"""Evaluate models trained on one v2 set on a fully held-out set merged in with merge_sets.py (--unused <target>).

Reports, on the runs of the held-out set only:
  1. the video readouts trained on the source training pairs (psq_v2.py on the merged set): payload mass, future roll,
     tip-over AUROC and accuracy on pairs that share the command and end differently (by what differs in the pair)
  2. the two-stage predictor: [planned command, payload mass] -> future, trained on the SOURCE runs with the true mass,
     applied with the true mass, the mass read from the video (2.71 s and 1.96 s observation) or no mass
Run from the safety_probing root:
  PYTHONPATH=analysis_thesis/v2:src .venv/bin/python analysis_thesis/v2/transfer_eval_v2.py --set p5v1 --target v1grid
"""
import argparse, json
import numpy as np, torch, torch.nn as nn
from common_v2 import ContextV2, ROOT

torch.set_num_threads(4)
ap = argparse.ArgumentParser(); ap.add_argument("--set", required=True); ap.add_argument("--target", required=True); ap.add_argument("--out", default="analysis_thesis/out"); args = ap.parse_args()
c = ContextV2(f"artifacts/v2/{args.set}/targets_truecom.pt", f"artifacts/v2/{args.set}/manifests"); t = c.tnp; N = len(c.ids); OUT = ROOT / args.out
man = [c.man[i] for i in c.ids]; tgt = np.array([m.get("source_set") == args.target for m in man]); src = ~tgt; TG = np.flatnonzero(tgt)
mass = t["static"][:, 0].astype(np.float32); cmd = np.array([[m["drive_velocity_command"], m["steering_position_command"]] for m in man], np.float32)
tipk, tipv = t["future_tip_mask"].astype(bool), t["future_tip"].astype(bool)
print(f"{args.set}: source runs {src.sum()}, held-out {args.target} runs {tgt.sum()} (outcome known {tipk[tgt].sum()}, tip-over rate {tipv[tgt & tipk].mean():.2f})", flush=True)
g = lambda d: "n/a" if d is None else f"{d['point']:.3f}[{d['ci_low']:.3f},{d['ci_high']:.3f}]"
res = {"n_target": int(tgt.sum())}
READ = {"video 2.71 s": args.set, "video 2.71 s + command": f"{args.set}_cmd", "video 1.96 s": f"{args.set}_obs47", "video 1.96 s + command": f"{args.set}_obs47_cmd", "command only": f"{args.set}_cmdonly"}
PRED, res["readouts"] = {}, {}
for label, name in READ.items():
    f = OUT / f"v2_pred_{name}.npz"
    if not f.exists(): print("missing", f.name); continue
    P = np.load(f, allow_pickle=True); assert [str(i) for i in P["ids"]] == c.ids; PRED[label] = P
    r = c.evaluate({k: P[k][:, TG] for k in ("static", "future_state", "future_summary", "tip_score")}, TG, bootstrap=2000)
    acc = float(((P["tip_score"].mean(0)[TG] > 0) == tipv[TG])[tipk[TG]].mean())
    res["readouts"][label] = {k: r.get(k) for k in ("mass_mae_kg", "mass_r2", "roll_mae_deg", "tip_auroc", "mass_direction", "com_direction", "outcome_flip", "outcome_flip_mass", "outcome_flip_com", "n_outcome_flip_mass", "n_outcome_flip_com")} | {"accuracy": acc}
    print(f"TRANSFER {label:24s} mass MAE {g(r['mass_mae_kg'])} kg (R2 {r['mass_r2']:.2f}) | roll {g(r['roll_mae_deg'])} deg | AUROC {g(r.get('tip_auroc'))} acc {acc:.3f} | mass pairs {g(r.get('outcome_flip_mass'))} "
          f"(n={r.get('n_outcome_flip_mass')}) | CoM pairs {g(r.get('outcome_flip_com'))} (n={r.get('n_outcome_flip_com')})", flush=True)
# two-stage predictor trained on the source set with the true mass
feat = lambda cm, ms: np.c_[cm, np.abs(cm[:, 1]), np.sign(cm[:, 1]), ms].astype(np.float32)
Y = np.concatenate([t["future_state"][:, :, 0], t["future_summary"][:, :1]], 1).astype(np.float32); M = np.concatenate([t["future_state_mask"][:, :, 0], t["future_summary_mask"][:, :1]], 1).astype(bool)
tip_t = torch.tensor(t["future_tip"]).float(); tipm_t = torch.tensor(tipk)
pairs_src = np.unique(c.cluster[src]); val_pairs = np.random.default_rng(2026).choice(pairs_src, size=int(round(0.15 * len(pairs_src))), replace=False)
va_m = src & np.isin(c.cluster, val_pairs); tr_m = src & ~va_m; tr, va = torch.tensor(np.flatnonzero(tr_m)), torch.tensor(np.flatnonzero(va_m)); trn = tr.numpy()
ymu = np.array([Y[trn, j][M[trn, j]].mean() for j in range(Y.shape[1])], np.float32); ysd = np.array([Y[trn, j][M[trn, j]].std() + 1e-6 for j in range(Y.shape[1])], np.float32)
Yn, Mt = torch.tensor((Y - ymu) / ysd), torch.tensor(M); F = feat(cmd, mass); mu, sd = F[trn].mean(0), F[trn].std(0) + 1e-6; X = torch.tensor((F - mu) / sd)
loss_fn = lambda out, i: ((out[:, :-1] - Yn[i]) ** 2)[Mt[i]].mean() + nn.functional.binary_cross_entropy_with_logits(out[:, -1][tipm_t[i]], tip_t[i][tipm_t[i]])
MASS_IN = {"true mass": mass[TG], "no mass (train mean)": np.full(len(TG), mass[trn].mean(), np.float32)}
for label in ("video 1.96 s", "video 2.71 s"):
    if label in PRED: MASS_IN[f"mass from {label}"] = PRED[label]["static"][:, TG, 0].mean(0).astype(np.float32)
S2 = {k: {"roll": [], "zmp": [], "score": []} for k in MASS_IN}
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
            o = model(torch.tensor((feat(cmd[TG], ms) - mu) / sd)).numpy(); phys = o[:, :-1] * ysd + ymu
            S2[k]["roll"].append(phys[:, :10]); S2[k]["zmp"].append(phys[:, 10]); S2[k]["score"].append(o[:, -1])
res["two_stage"] = {}
for k in MASS_IN:
    full = {"static": np.repeat(t["static"][None, TG], 3, 0).astype(np.float32), "future_state": np.stack(S2[k]["roll"])[..., None], "future_summary": np.stack(S2[k]["zmp"])[..., None], "tip_score": np.stack(S2[k]["score"])}
    r = c.evaluate(full, TG, bootstrap=2000); acc = float(((full["tip_score"].mean(0) > 0) == tipv[TG])[tipk[TG]].mean())
    res["two_stage"][k] = {kk: r.get(kk) for kk in ("roll_mae_deg", "tip_auroc", "outcome_flip", "outcome_flip_mass", "outcome_flip_com", "n_outcome_flip_mass", "n_outcome_flip_com")} | {"accuracy": acc}
    print(f"TWOSTAGE {k:26s} roll {g(r['roll_mae_deg'])} deg | AUROC {g(r.get('tip_auroc'))} acc {acc:.3f} | mass pairs {g(r.get('outcome_flip_mass'))} | CoM pairs {g(r.get('outcome_flip_com'))}", flush=True)
json.dump(res, open(OUT / f"v2_transfer_{args.set}.json", "w"), indent=1); print("wrote", OUT / f"v2_transfer_{args.set}.json")
