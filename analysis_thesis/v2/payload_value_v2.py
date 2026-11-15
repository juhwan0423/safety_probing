"""Value of payload knowledge for predicting the future on ForkSafe-CF v2 (no video involved).

A small network maps  [planned command] (+ true payload mass) (+ true lateral CoM offset)  to the future roll trajectory,
the minimum ZMP margin and the tip-over logit. Comparing the four input sets shows how much of the outcome is fixed by
the manoeuvre, how much by the payload mass (which the video readout identifies) and how much by the lateral CoM
offset (which it does not). Same split, targets and metrics as psq_v2.py; CPU only.
Run from the safety_probing root:
  PYTHONPATH=analysis_thesis/v2:src .venv/bin/python analysis_thesis/v2/payload_value_v2.py --name p5 \
      --targets artifacts/v2/p5/targets_truecom.pt --manifest-dir artifacts/v2/p5/manifests
"""
import argparse, json
import numpy as np, torch, torch.nn as nn
from common_v2 import ContextV2, ROOT, fmt

torch.set_num_threads(4)
ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--targets", required=True); ap.add_argument("--manifest-dir", required=True)
ap.add_argument("--out", default="analysis_thesis/out"); ap.add_argument("--cv", type=int, default=0, help="pair-grouped folds (0 = fixed split)")
args = ap.parse_args()
c = ContextV2(args.targets, args.manifest_dir); t = c.tnp; N = len(c.ids)
cmd = np.array([[c.man[i]["drive_velocity_command"], c.man[i]["steering_position_command"]] for i in c.ids], np.float32)
feat_cmd = np.c_[cmd, np.abs(cmd[:, 1]), np.sign(cmd[:, 1])]
mass = t["static"][:, :1].astype(np.float32); com = t["static"][:, 1:2].astype(np.float32)
out_com = com * np.sign(cmd[:, 1:2])                       # the offset matters through its side relative to the turn
SETS = {"command": feat_cmd, "command+mass": np.c_[feat_cmd, mass], "command+com": np.c_[feat_cmd, com, out_com],
        "command+mass+com": np.c_[feat_cmd, mass, com, out_com, mass * out_com]}
Y = np.concatenate([t["future_state"][:, :, 0], t["future_summary"][:, :1]], 1).astype(np.float32)
M = np.concatenate([t["future_state_mask"][:, :, 0], t["future_summary_mask"][:, :1]], 1).astype(bool)
tip = torch.tensor(t["future_tip"]).float(); tipm = torch.tensor(t["future_tip_mask"]).bool(); NOUT = Y.shape[1] + 1


def fit_predict(F, tr, va, te, seed):
    trn = tr.numpy()
    ymu = np.array([Y[trn, j][M[trn, j]].mean() for j in range(Y.shape[1])], np.float32); ysd = np.array([Y[trn, j][M[trn, j]].std() + 1e-6 for j in range(Y.shape[1])], np.float32)
    Yn, Mt = torch.tensor((Y - ymu) / ysd), torch.tensor(M)
    X = torch.tensor(F, dtype=torch.float32); X = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-6)
    loss_fn = lambda out, i: ((out[:, :-1] - Yn[i]) ** 2)[Mt[i]].mean() + nn.functional.binary_cross_entropy_with_logits(out[:, -1][tipm[i]], tip[i][tipm[i]])
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(X.shape[1], 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, NOUT))
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
    with torch.no_grad(): out = model(X[te]).numpy()
    phys = out[:, :-1] * ysd + ymu
    return phys[:, :10], phys[:, 10], out[:, -1]


res = {}
if args.cv:
    n_pairs = int(c.cluster.max() + 1); perm = np.random.default_rng(20261116).permutation(n_pairs); fold_of = np.empty(n_pairs, int); fold_of[perm] = np.arange(n_pairs) % args.cv
    fold = fold_of[c.cluster]
for name, F in SETS.items():
    P = {"static": np.zeros((3, N, 2), np.float32), "future_state": np.zeros((3, N, 10, 1), np.float32), "future_summary": np.zeros((3, N, 1), np.float32), "tip_score": np.zeros((3, N), np.float32)}
    P["static"][:] = t["static"][None]                         # payload is an input here; static metrics are not meaningful
    if args.cv:
        for k in range(args.cv):
            test = fold == k; rest = np.unique(c.cluster[~test]); val_pairs = np.random.default_rng(1000 + k).choice(rest, size=max(1, int(round(0.15 * len(rest)))), replace=False)
            va_m = np.isin(c.cluster, val_pairs) & ~test; tr_m = ~test & ~va_m
            tr, va, te = (torch.tensor(np.flatnonzero(m)) for m in (tr_m, va_m, test))
            for si, seed in enumerate((17, 29, 43)):
                roll, zmp, score = fit_predict(F, tr, va, te, seed); ten = te.numpy()
                P["future_state"][si, ten, :, 0] = roll; P["future_summary"][si, ten, 0] = zmp; P["tip_score"][si, ten] = score
        idx = np.arange(N)
    else:
        tr, va, te = (torch.tensor(np.flatnonzero(c.split == s)) for s in ("train", "val", "test"))
        for si, seed in enumerate((17, 29, 43)):
            roll, zmp, score = fit_predict(F, tr, va, te, seed); ten = te.numpy()
            P["future_state"][si, ten, :, 0] = roll; P["future_summary"][si, ten, 0] = zmp; P["tip_score"][si, ten] = score
        idx = te.numpy()
    r = c.evaluate({k: v[:, idx] for k, v in P.items()}, idx, bootstrap=3000); res[name] = r
    f = lambda k: f"{r[k]['point']:.3f}[{r[k]['ci_low']:.3f},{r[k]['ci_high']:.3f}]"
    print(f"PAYVAL {args.name} {'cv' if args.cv else 'split'} {name:18s} n={len(idx)} roll_mae_deg={f('roll_mae_deg')} zmp_mae_m={f('zmp_mae_m')} tip_auroc={f('tip_auroc')} outcome_flip={f('outcome_flip') if 'outcome_flip' in r else 'n/a'} (pairs {r['n_outcome_flip_pairs']})", flush=True)
json.dump(res, open(ROOT / args.out / f"v2_payload_value_{args.name}{'_cv' if args.cv else ''}.json", "w"), indent=1)
