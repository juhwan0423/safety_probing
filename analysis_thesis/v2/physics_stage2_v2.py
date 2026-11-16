"""Physics-based second stage on ForkSafe-CF v2: predicted turn severity over the roller-line threshold.

  severity law   s(u, m, delta) = (a + b u + e |delta|) * min(1, 1 - c (m - m0))
                 u drive command, m payload mass, delta steering command; target: peak |v_x omega_z| in 2.0-3.0 s.
                 Linear in the command, independent of the payload up to m0 and falling beyond it (heavier payloads
                 yaw more slowly). Fitted on the training pairs (grid over c, m0; least squares for a, b, e).
  threshold      a_crit(m, y) = g (b_r - y_c) / h, b_r = 0.50 m, CoM from the rigid-body model (truecom.com_body);
                 no fitted constant
  tip score      (i)  log(s / a_crit): tip-over predicted when s > a_crit
                 (ii) logistic model on [log(s / a_crit), m]: a critical ratio that depends linearly on the mass
                      (3 more coefficients, fitted on the training pairs)
The payload mass is the true one, the out-of-fold estimate of the video readout (2.71 s or 1.96 s observation) or the
training mean (no payload knowledge); the lateral CoM offset is unknown (y = 0) unless stated.
Pair-grouped 5-fold CV with the folds of the video readout; compare with the learned second stage in
plugin_mass_v2.py. CPU only.
Run from the safety_probing root:
  PYTHONPATH=analysis_thesis/v2:analysis_thesis:src .venv/bin/python analysis_thesis/v2/physics_stage2_v2.py --name p5
"""
import argparse, csv, json
import numpy as np, torch
from common_v2 import ContextV2, ROOT
import truecom as tc

G, B, T_TURN = 9.81, 0.50, 2.0
ap = argparse.ArgumentParser(); ap.add_argument("--name", default="p5"); ap.add_argument("--mass-from", nargs="*", default=None); ap.add_argument("--out", default="analysis_thesis/out"); args = ap.parse_args()
D = f"artifacts/v2/{args.name}"; c = ContextV2(f"{D}/targets_truecom.pt", f"{D}/manifests"); t = c.tnp; N = len(c.ids)
names = args.mass_from if args.mass_from is not None else [args.name, f"{args.name}_obs47"]
cmd = np.array([[c.man[i]["drive_velocity_command"], c.man[i]["steering_position_command"]] for i in c.ids], float); u, st = np.abs(cmd[:, 0]), np.abs(cmd[:, 1])
mass = t["static"][:, 0].astype(float); offset = np.array([c.man[i]["load_com_offset_local_xyz_m"][1] for i in c.ids], float)
tip, known = t["future_tip"].astype(bool), t["future_tip_mask"].astype(bool)


def severity(path):
    with open(path) as fh: rows = list(csv.DictReader(fh))
    f = lambda k: np.array([float(r[k]) if r[k] not in ("", "nan", "NaN", "None") else np.nan for r in rows])
    tt, vx, vy, wz, yaw = f("sim_time"), f("linear_vel_x"), f("linear_vel_y"), f("angular_vel_z"), np.radians(f("yaw_deg"))
    sm = np.convolve(np.pad(np.nan_to_num(np.abs((np.cos(yaw) * vx + np.sin(yaw) * vy) * wz)), 2, mode="edge"), np.ones(5) / 5, mode="valid")
    return float(sm[(tt >= T_TURN) & (tt <= T_TURN + 1.0)].max())


cache = ROOT / args.out / f"v2_{args.name}_severity_pulse.npy"
sev = np.load(cache) if cache.exists() and len(np.load(cache)) == N else np.array([severity(c.man[i]["dense_log_path"]) for i in c.ids]); np.save(cache, sev)


def threshold(m, y):
    """g (b_r - y_c) / h for payload mass m and configured lateral offset y towards the OUTSIDE of the turn (arrays)."""
    out = np.empty(len(m))
    for k, (mm, yy) in enumerate(zip(m, y)):
        cb = tc.com_body(float(mm), np.array([0.0, float(yy), 0.0])); out[k] = G * (B - cb[1]) / cb[2]
    return out


def fit_law(uu, mm, dd, ss):
    best = None
    for m0 in np.arange(300.0, 601.0, 10.0):
        for cc in np.arange(0.0004, 0.0031, 0.0001):
            f = np.minimum(1.0, 1 - cc * (mm - m0)); A = np.c_[f, uu * f, dd * f]; w, *_ = np.linalg.lstsq(A, ss, rcond=None); err = ((A @ w - ss) ** 2).sum()
            if best is None or err < best[0]: best = (err, np.array([w[0], w[1], w[2], cc, m0]))
    return best[1]


law = lambda p, uu, mm, dd: np.maximum((p[0] + p[1] * uu + p[2] * dd) * np.minimum(1.0, 1 - p[3] * (mm - p[4])), 1e-3)


def fit_logit(X, y):
    X = torch.tensor(X); y = torch.tensor(y.astype(float)); w = torch.zeros(X.shape[1] + 1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=300, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w[:-1] + w[-1], y); loss.backward(); return loss
    opt.step(closure); return w.detach().numpy()


y_out_cfg = offset * np.sign(cmd[:, 1])                     # left turn (steer > 0): +offset is towards the outside
EST, fold = {}, None
for n in names:
    P = np.load(ROOT / args.out / f"v2cv_pred_{n}.npz", allow_pickle=True); assert [str(i) for i in P["ids"]] == c.ids
    fold = P["fold"]; EST[n] = P["static"][:, :, 0].mean(0).astype(float)
INPUTS = {"true mass": (mass, np.zeros(N)), "true mass + true offset": (mass, y_out_cfg)}
for n in names: INPUTS[f"video mass ({n})"] = (EST[n], np.zeros(N))
INPUTS["no mass (train mean)"] = (None, np.zeros(N))
S1 = {v: np.zeros(N) for v in INPUTS}; S2 = {v: np.zeros(N) for v in INPUTS}; params, r2, crit = [], [], []
for k in sorted(set(fold.tolist())):
    tr, te = fold != k, fold == k
    p = fit_law(u[tr], mass[tr], st[tr], sev[tr]); params.append(p)
    pr = law(p, u[te], mass[te], st[te]); r2.append(1 - ((pr - sev[te]) ** 2).sum() / ((sev[te] - sev[te].mean()) ** 2).sum())
    kt = tr & known; lr_tr = np.log(law(p, u[kt], mass[kt], st[kt]) / threshold(mass[kt], np.zeros(kt.sum())))
    w = fit_logit(np.c_[lr_tr, (mass[kt] - 480.0) / 100.0], tip[kt]); crit.append(w)
    for v, (mm, yy) in INPUTS.items():
        m_te = np.full(te.sum(), mass[tr].mean()) if mm is None else np.clip(mm[te], 100, 900)
        lr = np.log(law(p, u[te], m_te, st[te]) / threshold(m_te, yy[te])); S1[v][te] = lr; S2[v][te] = w[0] * lr + w[1] * (m_te - 480.0) / 100.0 + w[2]
p_mean, w_mean = np.mean(params, 0), np.mean(crit, 0)
print(f"severity law (held-out folds): R2 {np.mean(r2):.3f}; s = ({p_mean[0]:.3f} + {p_mean[1]:.4f} u + {p_mean[2]:.3f} |steer|) * min(1, 1 - {p_mean[3]:.5f} (m - {p_mean[4]:.0f}))", flush=True)
print(f"critical ratio: log(s/a_crit) at p = 0.5 equals {-w_mean[2] / w_mean[0]:+.3f} at 480 kg and changes by {-w_mean[1] / w_mean[0]:+.3f} per 100 kg (ratio {np.exp(-w_mean[2] / w_mean[0]):.3f} at 480 kg)", flush=True)
res = {"severity_law_r2": float(np.mean(r2)), "severity_law_parameters_a_b_e_c_m0": [float(x) for x in p_mean], "critical_ratio_logit_w_logratio_w_mass100_bias": [float(x) for x in w_mean]}
for label, S in (("ratio vs 1", S1), ("fitted critical ratio", S2)):
    res[label] = {}
    for v, sc in S.items():
        pred = {"static": t["static"][None].astype(np.float32), "future_state": t["future_state"][None, :, :, :1], "future_summary": t["future_summary"][None], "tip_score": sc[None].astype(np.float32)}
        r = c.evaluate(pred, np.arange(N), bootstrap=2000); acc = float(((sc > 0) == tip)[known].mean())
        res[label][v] = dict(tip_auroc=r["tip_auroc"], accuracy=acc, outcome_flip_mass=r.get("outcome_flip_mass"), outcome_flip_com=r.get("outcome_flip_com"), outcome_flip=r.get("outcome_flip"))
        g = lambda d: "n/a" if d is None else f"{d['point']:.3f}[{d['ci_low']:.3f},{d['ci_high']:.3f}]"
        print(f"PHYS2 {args.name} {label:22s} {v:26s} AUROC {g(r['tip_auroc'])} | accuracy {acc:.3f} | mass pairs {g(r.get('outcome_flip_mass'))} | CoM pairs {g(r.get('outcome_flip_com'))}", flush=True)
json.dump(res, open(ROOT / args.out / f"v2_physics_stage2_{args.name}.json", "w"), indent=1)
