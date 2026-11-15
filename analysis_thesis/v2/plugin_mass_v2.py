"""Two-stage prediction on ForkSafe-CF v2: the video weighs the payload, a small model predicts the future.

Stage 1 (already run): the video readout estimates the payload mass (psq_v2_cv.py, out-of-fold estimates).
Stage 2 (here): a small network maps [planned command, payload mass] to the future roll trajectory, the minimum
ZMP margin and the tip-over logit. It is trained with the TRUE mass of the training pairs and tested, on pairs it
never saw, with
  true mass        : the upper bound for this input set (same as payload_value_v2.py "command+mass")
  video mass       : the out-of-fold estimate of a video readout (one column per --mass-from entry)
  train-mean mass  : no payload knowledge
The folds are those of the video readout, so the mass estimate of a test run comes from a readout that never saw its
pair and the stage-2 model never saw its outcome. CPU only.
Run from the safety_probing root:
  PYTHONPATH=analysis_thesis/v2:src .venv/bin/python analysis_thesis/v2/plugin_mass_v2.py --name p5 \
      --targets artifacts/v2/p5/targets_truecom.pt --manifest-dir artifacts/v2/p5/manifests --mass-from p5 p5_obs47
"""
import argparse, json
import numpy as np, torch, torch.nn as nn
from common_v2 import ContextV2, ROOT

torch.set_num_threads(4)
ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--targets", required=True); ap.add_argument("--manifest-dir", required=True)
ap.add_argument("--mass-from", nargs="+", required=True, help="names of v2cv_pred_<name>.npz files whose mass estimates are plugged in")
ap.add_argument("--out", default="analysis_thesis/out")
args = ap.parse_args()
c = ContextV2(args.targets, args.manifest_dir); t = c.tnp; N = len(c.ids)
cmd = np.array([[c.man[i]["drive_velocity_command"], c.man[i]["steering_position_command"]] for i in c.ids], np.float32)
feat_cmd = np.c_[cmd, np.abs(cmd[:, 1]), np.sign(cmd[:, 1])]
mass_true = t["static"][:, 0].astype(np.float32)
EST, fold = {}, None
for name in args.mass_from:
    P = np.load(ROOT / args.out / f"v2cv_pred_{name}.npz", allow_pickle=True); assert [str(i) for i in P["ids"]] == c.ids
    assert fold is None or np.array_equal(fold, P["fold"]), "mass estimates come from different fold assignments"
    fold = P["fold"]; EST[name] = P["static"][:, :, 0].mean(0).astype(np.float32)             # seed-mean out-of-fold estimate
    print(f"mass estimate from {name}: MAE {np.abs(EST[name] - mass_true).mean():.1f} kg", flush=True)
Y = np.concatenate([t["future_state"][:, :, 0], t["future_summary"][:, :1]], 1).astype(np.float32)
M = np.concatenate([t["future_state_mask"][:, :, 0], t["future_summary_mask"][:, :1]], 1).astype(bool)
tip = torch.tensor(t["future_tip"]).float(); tipm = torch.tensor(t["future_tip_mask"]).bool(); NOUT = Y.shape[1] + 1
SEEDS = (17, 29, 43); VARIANTS = ["true mass"] + [f"video mass ({n})" for n in args.mass_from] + ["train-mean mass"]
PRED = {v: {"static": np.repeat(t["static"][None], len(SEEDS), 0).astype(np.float32), "future_state": np.zeros((len(SEEDS), N, 10, 1), np.float32),
            "future_summary": np.zeros((len(SEEDS), N, 1), np.float32), "tip_score": np.zeros((len(SEEDS), N), np.float32)} for v in VARIANTS}
for k in sorted(set(fold.tolist())):
    test = fold == k; rest = np.unique(c.cluster[~test]); val_pairs = np.random.default_rng(1000 + k).choice(rest, size=max(1, int(round(0.15 * len(rest)))), replace=False)
    va_m = np.isin(c.cluster, val_pairs) & ~test; tr_m = ~test & ~va_m; tr, va, te = (torch.tensor(np.flatnonzero(m)) for m in (tr_m, va_m, test)); trn = tr.numpy(); ten = te.numpy()
    ymu = np.array([Y[trn, j][M[trn, j]].mean() for j in range(Y.shape[1])], np.float32); ysd = np.array([Y[trn, j][M[trn, j]].std() + 1e-6 for j in range(Y.shape[1])], np.float32)
    Yn, Mt = torch.tensor((Y - ymu) / ysd), torch.tensor(M)
    F = np.c_[feat_cmd, mass_true]; mu, sd = F[trn].mean(0), F[trn].std(0) + 1e-6
    X = torch.tensor((F - mu) / sd, dtype=torch.float32)
    TEST = {"true mass": X[te], "train-mean mass": torch.tensor((np.c_[feat_cmd[ten], np.full(len(ten), mass_true[trn].mean())] - mu) / sd, dtype=torch.float32)}
    for n in args.mass_from: TEST[f"video mass ({n})"] = torch.tensor((np.c_[feat_cmd[ten], EST[n][ten]] - mu) / sd, dtype=torch.float32)
    loss_fn = lambda out, i: ((out[:, :-1] - Yn[i]) ** 2)[Mt[i]].mean() + nn.functional.binary_cross_entropy_with_logits(out[:, -1][tipm[i]], tip[i][tipm[i]])
    for si, seed in enumerate(SEEDS):
        torch.manual_seed(seed)
        model = nn.Sequential(nn.Linear(X.shape[1], 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, NOUT))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3); best, state, bad = 1e9, None, 0
        for ep in range(3000):
            model.train(); p = tr[torch.randperm(len(tr))]
            for i in range(0, len(p), 64):
                b = p[i:i + 64]; opt.zero_grad(); loss_fn(model(X[b]), b).backward(); opt.step()
            model.eval()
            with torch.no_grad(): v = loss_fn(model(X[va]), va).item()
            if v < best - 1e-4: best, bad, state = v, 0, {kk: w.clone() for kk, w in model.state_dict().items()}
            else:
                bad += 1
                if bad >= 80: break
        model.load_state_dict(state); model.eval()
        for v, xt in TEST.items():
            with torch.no_grad(): out = model(xt).numpy()
            phys = out[:, :-1] * ysd + ymu
            PRED[v]["future_state"][si, ten, :, 0] = phys[:, :10]; PRED[v]["future_summary"][si, ten, 0] = phys[:, 10]; PRED[v]["tip_score"][si, ten] = out[:, -1]
    print(f"fold {k} done", flush=True)
res = {}; idx = np.arange(N)
for v in VARIANTS:
    r = c.evaluate(PRED[v], idx, bootstrap=3000); res[v] = r
    f = lambda key: f"{r[key]['point']:.3f}[{r[key]['ci_low']:.3f},{r[key]['ci_high']:.3f}]" if key in r else "n/a"
    print(f"PLUGIN {args.name} {v:26s} roll_mae_deg={f('roll_mae_deg')} zmp_mae_m={f('zmp_mae_m')} tip_auroc={f('tip_auroc')} flip_all={f('outcome_flip')} flip_mass={f('outcome_flip_mass')} "
          f"(n={r.get('n_outcome_flip_mass')}) flip_com={f('outcome_flip_com')} (n={r.get('n_outcome_flip_com')})", flush=True)
json.dump(res, open(ROOT / args.out / f"v2_plugin_mass_{args.name}.json", "w"), indent=1)
np.savez(ROOT / args.out / f"v2_plugin_mass_pred_{args.name}.npz", ids=np.array(c.ids), fold=fold, **{f"{v}|{k}": a for v, d in PRED.items() for k, a in d.items()})
