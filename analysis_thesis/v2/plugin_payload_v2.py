"""Two-stage prediction with BOTH payload quantities read from the video: mass and the lateral CoM offset.

Stage 1: a video readout trained with the offset towards the OUTSIDE of the planned turn as its lateral-CoM target
         (psq_v2_cv.py --com-target outside; out-of-fold estimates). The turn direction is known from the planned
         command, so the signed offset follows from the estimate.
Stage 2: a small network maps [planned command, payload mass, lateral CoM offset] to the future roll trajectory, the
         minimum ZMP margin and the tip-over logit; trained with the TRUE payload of the training pairs and tested with
           true mass + true offset        upper bound
           video mass + video offset      everything from the video
           video mass + true offset / true mass + video offset     which estimate limits the result
           video mass, offset set to zero  no offset knowledge
Folds are those of the video readout. CPU only.
Run from the safety_probing root:
  PYTHONPATH=analysis_thesis/v2:src .venv/bin/python analysis_thesis/v2/plugin_payload_v2.py --name p5 --pred p5_yout
"""
import argparse, json
import numpy as np, torch, torch.nn as nn
from common_v2 import ContextV2, ROOT

torch.set_num_threads(4)
ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--pred", required=True, help="v2cv_pred_<pred>.npz from a readout trained with --com-target outside"); ap.add_argument("--out", default="analysis_thesis/out")
args = ap.parse_args()
D = f"artifacts/v2/{args.name}"; c = ContextV2(f"{D}/targets_truecom.pt", f"{D}/manifests"); t = c.tnp; N = len(c.ids)
cmd = np.array([[c.man[i]["drive_velocity_command"], c.man[i]["steering_position_command"]] for i in c.ids], np.float32); sgn = np.sign(cmd[:, 1])
mass = t["static"][:, 0].astype(np.float32); yout = (t["static"][:, 1] * sgn).astype(np.float32)             # true offset towards the outside
P = np.load(ROOT / args.out / f"v2cv_pred_{args.pred}.npz", allow_pickle=True); assert [str(i) for i in P["ids"]] == c.ids
fold = P["fold"]; m_hat = P["static"][:, :, 0].mean(0).astype(np.float32); y_hat = P["static"][:, :, 1].mean(0).astype(np.float32)
r2 = lambda p, y: 1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
print(f"video estimates ({args.pred}): mass MAE {np.abs(m_hat - mass).mean():.1f} kg (R2 {r2(m_hat, mass):.3f}) | outside offset MAE {np.abs(y_hat - yout).mean():.4f} m (R2 {r2(y_hat, yout):.3f}, corr {np.corrcoef(y_hat, yout)[0, 1]:.3f})", flush=True)
feat = lambda m, y: np.c_[cmd, np.abs(cmd[:, 1]), sgn, m, y * sgn, y, m * y].astype(np.float32)     # as payload_value_v2.py "command+mass+com"
Y = np.concatenate([t["future_state"][:, :, 0], t["future_summary"][:, :1]], 1).astype(np.float32); M = np.concatenate([t["future_state_mask"][:, :, 0], t["future_summary_mask"][:, :1]], 1).astype(bool)
tip = torch.tensor(t["future_tip"]).float(); tipm = torch.tensor(t["future_tip_mask"]).bool(); SEEDS = (17, 29, 43)
INPUTS = {"true mass + true offset": (mass, yout), "video mass + video offset": (m_hat, y_hat), "video mass + true offset": (m_hat, yout), "true mass + video offset": (mass, y_hat),
          "video mass, no offset": (m_hat, np.zeros(N, np.float32)), "true mass, no offset": (mass, np.zeros(N, np.float32))}
PRED = {v: {"static": np.repeat(t["static"][None], len(SEEDS), 0).astype(np.float32), "future_state": np.zeros((len(SEEDS), N, 10, 1), np.float32),
            "future_summary": np.zeros((len(SEEDS), N, 1), np.float32), "tip_score": np.zeros((len(SEEDS), N), np.float32)} for v in INPUTS}
F_true = feat(mass, yout)
for k in sorted(set(fold.tolist())):
    test = fold == k; rest = np.unique(c.cluster[~test]); val_pairs = np.random.default_rng(1000 + k).choice(rest, size=max(1, int(round(0.15 * len(rest)))), replace=False)
    va_m = np.isin(c.cluster, val_pairs) & ~test; tr_m = ~test & ~va_m; tr, va = torch.tensor(np.flatnonzero(tr_m)), torch.tensor(np.flatnonzero(va_m)); trn = tr.numpy(); ten = np.flatnonzero(test)
    ymu = np.array([Y[trn, j][M[trn, j]].mean() for j in range(Y.shape[1])], np.float32); ysd = np.array([Y[trn, j][M[trn, j]].std() + 1e-6 for j in range(Y.shape[1])], np.float32)
    Yn, Mt = torch.tensor((Y - ymu) / ysd), torch.tensor(M); mu, sd = F_true[trn].mean(0), F_true[trn].std(0) + 1e-6; X = torch.tensor((F_true - mu) / sd)
    loss_fn = lambda out, i: ((out[:, :-1] - Yn[i]) ** 2)[Mt[i]].mean() + nn.functional.binary_cross_entropy_with_logits(out[:, -1][tipm[i]], tip[i][tipm[i]])
    for si, seed in enumerate(SEEDS):
        torch.manual_seed(seed)
        model = nn.Sequential(nn.Linear(X.shape[1], 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, Y.shape[1] + 1))
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
        for v, (mm, yy) in INPUTS.items():
            with torch.no_grad(): o = model(torch.tensor((feat(mm, yy)[ten] - mu) / sd)).numpy()
            phys = o[:, :-1] * ysd + ymu; PRED[v]["future_state"][si, ten, :, 0] = phys[:, :10]; PRED[v]["future_summary"][si, ten, 0] = phys[:, 10]; PRED[v]["tip_score"][si, ten] = o[:, -1]
res = {"mass_mae_kg": float(np.abs(m_hat - mass).mean()), "mass_r2": float(r2(m_hat, mass)), "outside_offset_mae_m": float(np.abs(y_hat - yout).mean()), "outside_offset_r2": float(r2(y_hat, yout))}
for v in INPUTS:
    r = c.evaluate(PRED[v], np.arange(N), bootstrap=3000); res[v] = r
    g = lambda key: f"{r[key]['point']:.3f}[{r[key]['ci_low']:.3f},{r[key]['ci_high']:.3f}]" if key in r else "n/a"
    print(f"PAYLOAD2 {args.name} {v:28s} roll_mae_deg={g('roll_mae_deg')} tip_auroc={g('tip_auroc')} flip_all={g('outcome_flip')} flip_mass={g('outcome_flip_mass')} (n={r.get('n_outcome_flip_mass')}) flip_com={g('outcome_flip_com')} (n={r.get('n_outcome_flip_com')})", flush=True)
json.dump(res, open(ROOT / args.out / f"v2_plugin_payload_{args.name}.json", "w"), indent=1)
