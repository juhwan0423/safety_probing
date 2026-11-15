"""Outcome-discordant matched pairs: same planned command, one payload change, one member tips and the other recovers.

A predictor that only knows the command scores both members identically, so it is at chance (0.5) by construction.
Metric: fraction of such pairs in which the member that actually tipped receives the higher predicted tip-over score
(and, separately, the larger predicted peak roll). Uses out-of-fold predictions written by psq_cv.py.
Run:  PYTHONPATH=. ../.venv/bin/python pair_outcome_flip.py logo9_full config5_full ...
"""
import sys, json
import numpy as np
from common import Context, bs

c = Context("artifacts/targets/safety_query_full_comfix_med5.pt")
t = c.tnp
index = {cid: i for i, cid in enumerate(c.ids)}
runs = {(str(r["pair_id"]), str(r["run_label"])): str(r["condition_id"]) for r in c.run_rows}
man = {str(r["condition_id"]): r for r in c.canonical_rows}
tip, tipm = t["future_tip"].astype(bool), t["future_tip_mask"].astype(bool)
pairs = []                                   # (tipped index, recovered index, kind, route)
seen = set()
for row in c.pair_rows:
    a, b = index[runs[(str(row["pair_id"]), "A")]], index[runs[(str(row["pair_id"]), "B")]]
    if not (tipm[a] and tipm[b]) or tip[a] == tip[b] or (min(a, b), max(a, b)) in seen:
        continue
    seen.add((min(a, b), max(a, b)))
    hi, lo = (a, b) if tip[a] else (b, a)
    pairs.append((hi, lo, row["intervention_type"], str(man[c.ids[a]]["route_id"])))
kinds = np.array([p[2] for p in pairs]); route = np.array([p[3] for p in pairs]); _, cl = np.unique(route, return_inverse=True)
H, L = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
print(f"outcome-discordant pairs: {len(pairs)} (mass {np.sum(kinds == 'load_mass')}, lateral CoM {np.sum(kinds == 'com_shift')}) over {cl.max() + 1} routes")
rng = np.random.default_rng(bs.BOOTSTRAP_SEED)
out = {}
for name in sys.argv[1:]:
    z = np.load(f"out/cv_pred_{name}.npz", allow_pickle=True)
    score = z["tip_score"]; roll = np.abs(z["future_state"][:, :, :, 0]).max(2)
    res = {}
    for key, v in (("tip_score", score), ("peak_roll", roll)):
        d = v[:, H] - v[:, L]; correct = (d > 0) + 0.5 * (d == 0)                       # [S, P]; ties count one half
        for sel_name, sel in (("all", np.ones(len(pairs), bool)), ("mass", kinds == "load_mass"), ("com", kinds == "com_shift")):
            if sel.sum() == 0: continue
            res[f"{key}_{sel_name}"] = bs.hierarchical_mean(correct[:, sel], cl[sel], bootstrap=5000, rng=rng)
    out[name] = {k: {kk: v[kk] for kk in ("point", "ci_low", "ci_high", "n")} for k, v in res.items()}
    f = lambda k: f"{res[k]['point']:.3f}[{res[k]['ci_low']:.3f},{res[k]['ci_high']:.3f}]" if k in res else "n/a"
    print(f"FLIP {name:32s} tip-score ordering all={f('tip_score_all')} mass={f('tip_score_mass')} com={f('tip_score_com')} | peak-roll ordering all={f('peak_roll_all')}")
prev = json.load(open("out/pair_outcome_flip.json")) if __import__("os").path.exists("out/pair_outcome_flip.json") else {}
prev.update(out); prev["_n_pairs"] = {"all": len(pairs), "mass": int(np.sum(kinds == "load_mass")), "com": int(np.sum(kinds == "com_shift"))}
json.dump(prev, open("out/pair_outcome_flip.json", "w"), indent=1)
