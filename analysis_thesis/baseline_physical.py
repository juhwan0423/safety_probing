"""Baseline probes with the full physical target set (thesis Table: PSQ vs simple probes).

Same frozen V-JEPA observation features as the paper baselines (spatial-mean per tubelet,
[N,10,1024]); same split, seeds, validation-based checkpoint selection and paper metrics.
Models never see generated data or test targets. CPU only.
"""
import json, sys, numpy as np, torch, torch.nn as nn
from common import Context, ROOT, fmt

torch.set_num_threads(8)
SEEDS = (17, 29, 43)
c = Context()
feat = torch.load(ROOT / "artifacts/features/vjepa_2_1_full_observation.pt", map_location="cpu", weights_only=False)
assert [str(i) for i in feat["contract"]["condition_ids"]] == c.ids
X = feat["features"].float()                      # [N,10,1024]
mu, sd = X[c.train].mean((0, 1)), X[c.train].std((0, 1)) + 1e-6
X = (X - mu) / sd

t = c.tnp
Y = np.concatenate([t["static"], t["future_state"][:, :, 0], t["future_summary"][:, :1]], 1).astype(np.float32)   # 2+10+1
M = np.concatenate([t["static_mask"], t["future_state_mask"][:, :, 0], t["future_summary_mask"][:, :1]], 1).astype(bool)
ymu = np.array([Y[c.train, j][M[c.train, j]].mean() for j in range(Y.shape[1])], dtype=np.float32)
ysd = np.array([Y[c.train, j][M[c.train, j]].std() + 1e-6 for j in range(Y.shape[1])], dtype=np.float32)
Yn = torch.tensor((Y - ymu) / ysd); Mt = torch.tensor(M)
tip = torch.tensor(t["future_tip"]).float(); tipm = torch.tensor(t["future_tip_mask"]).bool()
NOUT = Y.shape[1] + 1


class Lin(nn.Module):
    def __init__(s, mode):
        super().__init__(); s.mode = mode
        s.f = nn.Linear(1024 if mode == "mean" else 10240, NOUT)
    def forward(s, x):
        return s.f(x.mean(1) if s.mode == "mean" else x.flatten(1))

class MLP(nn.Module):
    def __init__(s):
        super().__init__()
        s.f = nn.Sequential(nn.Linear(1024, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, NOUT))
    def forward(s, x): return s.f(x.mean(1))

class TT(nn.Module):
    def __init__(s, d=256):
        super().__init__()
        s.inp = nn.Linear(1024, d); s.pos = nn.Parameter(torch.zeros(10, d))
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, 0.1, batch_first=True, norm_first=True), 2)
        s.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, NOUT))
    def forward(s, x): return s.out(s.enc(s.inp(x) + s.pos).mean(1))

MODELS = {
    "linear_global_mean": (lambda: Lin("mean"), 1e-3, 1e-1),
    "linear_temporal_concat": (lambda: Lin("cat"), 3e-4, 1e-1),
    "mlp_mean_pool": (MLP, 1e-3, 1e-2),
    "temporal_transformer": (TT, 3e-4, 1e-2),
}

def loss_fn(out, idx):
    reg = ((out[:, :-1] - Yn[idx]) ** 2)[Mt[idx]].mean()
    m = tipm[idx]
    bce = nn.functional.binary_cross_entropy_with_logits(out[:, -1][m], tip[idx][m])
    return reg + bce

def run(name, seed):
    make, lr, wd = MODELS[name]
    torch.manual_seed(seed); np.random.seed(seed)
    model = make(); opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best, best_state, bad = 1e9, None, 0
    tr = torch.tensor(c.train); va = torch.tensor(c.val)
    for ep in range(600):
        model.train(); perm = tr[torch.randperm(len(tr))]
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]; opt.zero_grad(); loss_fn(model(X[idx]), idx).backward(); opt.step()
        model.eval()
        with torch.no_grad(): v = loss_fn(model(X[va]), va).item()
        if v < best - 1e-4: best, bad, best_state = v, 0, {k: w.clone() for k, w in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 40: break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad(): out = model(X[torch.tensor(c.test)]).numpy()
    phys = out[:, :-1] * ysd + ymu
    return {"static": phys[:, :2], "future_state": phys[:, 2:12, None], "future_summary": phys[:, 12:13], "tip_score": out[:, -1]}, best, sum(p.numel() for p in model.parameters())

results = {}
for name in MODELS:
    preds, vals = [], []
    for s in SEEDS:
        p, v, n = run(name, s); preds.append(p); vals.append(v)
    pred = {k: np.stack([p[k] for p in preds]) for k in preds[0]}
    r = c.evaluate(pred); r["params"] = n; r["val_loss"] = vals
    results[name] = r
    print(name, n, fmt(r), flush=True)
    np.savez(f"out/baseline_pred_{name}.npz", **pred)
json.dump(results, open("out/baseline_physical.json", "w"), indent=1)
