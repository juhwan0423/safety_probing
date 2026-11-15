"""Physics-informed tip-over prediction: identified payload -> rigid-body stability model -> risk.

  risk = lateral demand of the turn / static rollover threshold,   threshold = g d(m, y) / h(m)
where d (CoM to the tipping axis of the support triangle) and h (CoM height) follow from the payload mass m and its
lateral CoM offset y through the rigid-body mass model of truecom.py. The payload comes from
  true      : ground truth (upper bound)
  psq       : out-of-fold readout predictions from the observed video (psq_cv.py)
  nominal   : the training-set mean payload (no identification)
and the demand from
  realised  : peak chassis speed x yaw rate during the turn (uses the future; upper bound)
  planned   : a two-parameter power law of the speed at the start of the turn and the steering command, fitted on the
              training folds (available before the turn)
Reports tip-over AUROC and the accuracy on outcome-discordant matched pairs (same command, one payload change).
Run:  PYTHONPATH=. ../.venv/bin/python physics_pipeline.py [cv prediction name, default logo9_full_tc_obs65]
"""
import json, sys
import numpy as np
from common import Context, bs, physical_config
import truecom as tc, rollover_physics as rp

G = 9.81
POLY = np.array([(-0.910, 0.499), (-0.910, -0.494), (-0.718, -0.494), (0.580, 0.008), (-0.718, 0.499)])
STEER = {"wide": 1.0, "medium": 1.25, "tight": 1.5}


def threshold(mass: float, offset: float, out_y: float) -> tuple[float, float, float]:
    c = tc.com_body(mass, [0.0, offset, 0.0])
    d, _ = rp.ray_edge(c[:2], POLY, np.array([0.0, out_y]))
    return float(d), float(c[2]), float(G * d / c[2])


name = sys.argv[1] if len(sys.argv) > 1 else "logo9_full_tc_obs65"
c = Context("artifacts/targets/thesis_v1_obs65_truecom.pt")
M = {r["cid"]: r for r in json.load(open("out/mechanics_true_v1.json"))}
R = [M[i] for i in c.ids]
A = lambda k: np.array([r[k] for r in R], dtype=float)
z = np.load(f"out/cv_pred_{name}.npz", allow_pickle=True)
fold = z["fold"]; pred = z["static"].mean(0)                         # seed-mean out-of-fold payload prediction [N, 2]
tip = c.tnp["future_tip"].astype(int); tm = c.tnp["future_tip_mask"].astype(bool)
cfg, group, man = physical_config(c)
steer = np.array([STEER[man[i]["steering_label"]] for i in c.ids]); v0 = A("v0"); out_y = A("out_y")
check = np.array([threshold(r["mass"], r["offset"], r["out_y"])[2] for r in R])
print(f"static threshold from the mass model vs per-run log geometry: max abs difference {np.max(np.abs(check - A('a_crit'))):.4f} m/s^2")

# planned demand: log a = b0 + b1 log v0 + b2 log steer, fitted out-of-fold (same folds as the readout)
planned = np.zeros(len(R))
for k in np.unique(fold):
    tr = fold != k
    X = np.c_[np.ones(tr.sum()), np.log(v0[tr]), np.log(steer[tr])]
    beta, *_ = np.linalg.lstsq(X, np.log(A("a_kin")[tr]), rcond=None)
    te = fold == k
    planned[te] = np.exp(np.c_[np.ones(te.sum()), np.log(v0[te]), np.log(steer[te])] @ beta)
print("planned-demand model exponents (last fold): speed %.2f, steering %.2f | correlation with the realised demand %.3f" % (beta[1], beta[2], np.corrcoef(planned, A("a_kin"))[0, 1]))

mean_payload = np.zeros((len(R), 2))
for k in np.unique(fold):
    mean_payload[fold == k] = np.c_[A("mass"), A("offset")][fold != k].mean(0)
payloads = {"true": np.c_[A("mass"), A("offset")], "psq": pred, "nominal": mean_payload}
thr = {key: np.array([threshold(float(np.clip(p[0], 100, 900)), float(np.clip(p[1], -0.4, 0.4)), oy)[2] for p, oy in zip(P, out_y)]) for key, P in payloads.items()}
demand = {"realised": A("a_kin"), "planned": planned}

# outcome-discordant matched pairs
index = {cid: i for i, cid in enumerate(c.ids)}
runs = {(str(r["pair_id"]), str(r["run_label"])): str(r["condition_id"]) for r in c.run_rows}
pairs, seen = [], set()
for row in c.pair_rows:
    a, b = index[runs[(str(row["pair_id"]), "A")]], index[runs[(str(row["pair_id"]), "B")]]
    if not (tm[a] and tm[b]) or tip[a] == tip[b] or (min(a, b), max(a, b)) in seen: continue
    seen.add((min(a, b), max(a, b))); pairs.append(((a, b) if tip[a] else (b, a)) + (row["intervention_type"],))
H, L = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs]); kind = np.array([p[2] for p in pairs])


def flip(score):
    d = score[H] - score[L]
    f = lambda s: float(np.mean((d[s] > 0) + 0.5 * (d[s] == 0)))
    return f(np.ones(len(d), bool)), f(kind == "load_mass"), f(kind == "com_shift")


out = {}
print(f"\n{'payload':8s} {'demand':9s} | tip-over AUROC | discordant pairs: all (n={len(pairs)})  mass  CoM")
for dk, dv in demand.items():
    for pk, tv in thr.items():
        s = dv / tv
        auc = bs.auc(tip[tm], s[tm]); fa, fm, fc = flip(s)
        out[f"{pk}+{dk}"] = dict(auroc=float(auc), flip_all=fa, flip_mass=fm, flip_com=fc)
        print(f"{pk:8s} {dk:9s} |     {auc:.3f}      |        {fa:.3f}           {fm:.3f} {fc:.3f}")
    auc = bs.auc(tip[tm], dv[tm]); fa, fm, fc = flip(dv)
    out[f"none+{dk}"] = dict(auroc=float(auc), flip_all=fa, flip_mass=fm, flip_com=fc)
    print(f"{'(none)':8s} {dk:9s} |     {auc:.3f}      |        {fa:.3f}           {fm:.3f} {fc:.3f}   <- demand alone")
score = z["tip_score"].mean(0); fa, fm, fc = flip(score)
out["readout"] = dict(auroc=float(bs.auc(tip[tm], score[tm])), flip_all=fa, flip_mass=fm, flip_com=fc)
print(f"{'learned readout (tip score)':18s} |     {out['readout']['auroc']:.3f}      |        {fa:.3f}           {fm:.3f} {fc:.3f}")
err = np.abs(thr["psq"] - thr["true"]); print(f"\nthreshold error from the identified payload: median {np.median(err):.3f} m/s^2 (nominal payload: {np.median(np.abs(thr['nominal'] - thr['true'])):.3f})")
json.dump(dict(name=name, n_pairs=len(pairs), results=out), open(f"out/physics_pipeline_{name}.json", "w"), indent=1)
