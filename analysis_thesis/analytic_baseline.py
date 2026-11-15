"""Quasi-static rollover criterion driven by identified payload parameters.
score = h_sys * a_y / g  +  outward lateral shift of system CoM
  h_sys(m): system CoM height from rigid-body composition (fit on train logs)
  CoM shift: (m / M) * y_payload toward the outside of the turn (payload CoM offset; NOT in the logged system CoM)
  a_y: lateral-acceleration demand |v*omega| at the end of the observation window
AUROC is threshold-free, so the effective half-track is not needed. One sign convention is fixed on train."""
import json, numpy as np, torch
from common import Context, bs, ROOT
c = Context(); g = 9.81
d = {r["condition_id"]: r for r in json.load(open("out/dense_table.json"))}
rows = [d[i] for i in c.ids]
A = lambda k: np.array([r[k] for r in rows], dtype=float)
m, yl, M, cz = A("load_mass"), A("load_com_y"), A("sys_mass"), A("com_z0")
turn = np.array([1.0 if r["turn"] == "right" else -1.0 for r in rows])
Mv = float(np.median(M - m)); w, *_ = np.linalg.lstsq(np.c_[np.ones(len(c.train)), m[c.train]], (M * cz)[c.train], rcond=None)
h = lambda mm: (w[0] + w[1] * mm) / (Mv + mm)
tipm = c.tnp["future_tip_mask"].astype(bool); tip = c.tnp["future_tip"].astype(int)
def auroc(score, idx):
    k = idx[tipm[idx]]; return bs.auc(tip[k], score[k])
ay_gt = A("ay_obs_max"); v_gt = A("speed_obs_end")
def score(mm, yy, ay, sgn): return h(mm) * ay / g + sgn * turn * (mm / (Mv + mm)) * yy
sgn = max((+1, -1), key=lambda s: auroc(score(m, yl, ay_gt, s), c.train)); print("sign fixed on train:", sgn)
out = {}
def rep(name, s):
    out[name] = dict(train=auroc(s, c.train), test=auroc(s, c.test)); print(f"{name:55s} train {out[name]['train']:.3f}  test {out[name]['test']:.3f}")
rep("demand only (a_y, GT kinematics)", ay_gt)
rep("criterion, GT mass+CoM, GT obs kinematics", score(m, yl, ay_gt, sgn))
rep("criterion without CoM term (GT mass)", h(m) * ay_gt / g)
rep("criterion with nominal payload (no identification)", score(np.full_like(m, 480.0), np.zeros_like(yl), ay_gt, sgn))
# PSQ-identified parameters (video only)
P = bs.load_method_predictions(c.paths, c.ids)["vjepa_prefix"]
ck = torch.load(c.paths["vjepa_seed_17"], map_location="cpu", weights_only=False)
print("pred keys", list(ck["predictions_physical"].keys()))
ch = c.targets["channels"]["state"]; iv, iw = ch.index("linear_vel_body_x"), ch.index("imu_ang_vel_z")
obs = []
for s in bs.QUERY_SEEDS:
    k = torch.load(c.paths[f"vjepa_seed_{s}"], map_location="cpu", weights_only=False)["predictions_physical"]["observed_state"].numpy()
    obs.append(np.abs(k[:, :, iv] * k[:, :, iw]).max(1))
gt_obs = np.abs(c.tnp["observed_state"][:, :, iv] * c.tnp["observed_state"][:, :, iw]).max(1)
rep("criterion, GT mass+CoM, GT sampled obs kinematics", score(m, yl, gt_obs, sgn))
per_seed = [score(P["static"][i][:, 0], P["static"][i][:, 1], obs[i], sgn) for i in range(3)]
for i, s in enumerate(per_seed): rep(f"criterion, PSQ mass+CoM+kinematics (seed {bs.QUERY_SEEDS[i]})", s)
rep("criterion, PSQ mass+CoM, GT sampled obs kinematics", score(P["static"].mean(0)[:, 0], P["static"].mean(0)[:, 1], gt_obs, sgn))
out["psq_pipeline_test_mean"] = float(np.mean([auroc(s, c.test) for s in per_seed])); print("PSQ pipeline test mean", out["psq_pipeline_test_mean"])
rep("PSQ direct tip logit (seed mean)", P["tip_score"].mean(0))
json.dump(out, open("out/analytic_baseline.json", "w"), indent=1)
