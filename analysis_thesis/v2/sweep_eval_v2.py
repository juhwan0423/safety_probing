"""Boundary sweep of ForkSafe-CF v2: where does tip-over start, and can the video place a payload on that map?

The sweep (build_v2_sweep_config.py) is a regular grid that no model was trained on:
  mass map  payload mass 240..720 kg x drive command 16..24 (left turn, steering 1.25, lateral CoM offset 0)
  CoM map   lateral CoM offset -0.25..0.25 m x drive command 17.5..22.5 at 480 kg
Readouts are trained on the p5 training pairs (psq_v2.py on the merged set, sweep runs marked "unused").
This script reports, on the sweep runs:
  1. reproducibility of the replicate 480 kg runs and the critical drive command per payload (true boundary)
  2. readout metrics (mass, roll, tip-over AUROC) and comparisons between runs with the same command
  3. a two-stage predictor: [planned command, payload mass] -> future, trained on p5 with the true mass, applied
     with the true mass, with the mass read from the video (1.96 s = before the turn, and 2.71 s) and with no mass
  4. the critical drive command predicted for each payload from its video mass estimate against the true one
Run from the safety_probing root:
  PYTHONPATH=analysis_thesis/v2:src .venv/bin/python analysis_thesis/v2/sweep_eval_v2.py
"""
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from common_v2 import ContextV2, ROOT, bs

torch.set_num_threads(4)
ap = argparse.ArgumentParser(); ap.add_argument("--set", default="p5sweep"); ap.add_argument("--sweep-name", default="sweep"); ap.add_argument("--out", default="analysis_thesis/out"); ap.add_argument("--tag", default="")
args = ap.parse_args()
c = ContextV2(f"artifacts/v2/{args.set}/targets_truecom.pt", f"artifacts/v2/{args.set}/manifests"); t = c.tnp; N = len(c.ids); OUT = ROOT / args.out
man = [c.man[i] for i in c.ids]
sw = np.array([m.get("source_set") == args.sweep_name for m in man]); p5 = ~sw; SW = np.flatnonzero(sw)
mass = t["static"][:, 0].astype(np.float32); offset = np.array([m["load_com_offset_local_xyz_m"][1] for m in man], np.float32)
cmd = np.array([[m["drive_velocity_command"], m["steering_position_command"]] for m in man], np.float32); drive = np.abs(cmd[:, 0])
tipk = t["future_tip_mask"].astype(bool); tipv = t["future_tip"].astype(bool)
part = np.array([Path(m["dataset_root"]).name for m in man]); rep = np.array([m["route_id"].endswith("_rep") for m in man])
massmap = sw & (part == "forksafe_v2_sweep_01"); commap = sw & ((part == "forksafe_v2_sweep_02") | (massmap & (np.abs(mass - 480) < 1) & (drive >= 17.5) & (drive <= 22.5) & ~rep))
print(f"sweep runs {sw.sum()} (mass map {massmap.sum()}, CoM map incl. the 480 kg centre runs {commap.sum()}); outcomes known {tipk[sw].sum()}, tip-over {tipv[sw & tipk].sum()}", flush=True)
res = {"n_sweep": int(sw.sum())}

# 1. replicates and the true boundary --------------------------------------------------------------------------
R = []
for d in sorted(set(drive[massmap].tolist())):
    a = np.flatnonzero(massmap & (drive == d) & (np.abs(mass - 480) < 1) & ~rep); b = np.flatnonzero(massmap & (drive == d) & (np.abs(mass - 480) < 1) & rep)
    if len(a) == 1 and len(b) == 1:
        ra, rb = t["future_state"][a[0], :, 0], t["future_state"][b[0], :, 0]
        R.append(dict(drive=float(d), same_outcome=bool(tipv[a[0]] == tipv[b[0]]), max_future_roll_diff_deg=float(np.abs(ra - rb).max()), max_roll_a=float(man[a[0]]["max_roll_deg"]), max_roll_b=float(man[b[0]]["max_roll_deg"])))
res["replicates"] = R
if R: print(f"replicate 480 kg runs: {len(R)} drive levels, same outcome in {sum(r['same_outcome'] for r in R)}, future roll difference median {np.median([r['max_future_roll_diff_deg'] for r in R]):.3f} deg, max {max(r['max_future_roll_diff_deg'] for r in R):.2f} deg", flush=True)


def critical(levels, tips):
    """Critical drive command from outcomes on sorted drive levels: midpoint between the last recovery below the first tip-over and that tip-over."""
    levels, tips = np.asarray(levels), np.asarray(tips, bool); o = np.argsort(levels); levels, tips = levels[o], tips[o]
    if tips.all(): return float(levels[0] - 0.25), True
    if not tips.any(): return float(levels[-1] + 0.25), True
    first = int(np.flatnonzero(tips)[0]); mono = bool(tips[first:].all())
    return float(0.5 * (levels[first - 1] + levels[first])) if first > 0 else float(levels[0] - 0.25), mono


CRIT = {}
for m in sorted(set(np.round(mass[massmap]).tolist())):
    s = massmap & (np.abs(mass - m) < 1) & tipk & ~rep; CRIT[m], mono = critical(drive[s], tipv[s]); print(f"  mass {m:5.0f} kg: critical drive command {CRIT[m]:.2f} {'(not monotone)' if not mono else ''}", flush=True)
res["critical_drive_by_mass"] = {str(int(k)): v for k, v in CRIT.items()}
CRITO = {}
for y in sorted(set(np.round(offset[commap], 3).tolist())):
    s = commap & (np.abs(offset - y) < 1e-3) & tipk; CRITO[y], mono = critical(drive[s], tipv[s]); print(f"  offset {y:+.3f} m (480 kg): critical drive command {CRITO[y]:.2f} {'(not monotone)' if not mono else ''}", flush=True)
res["critical_drive_by_offset"] = {f"{k:+.3f}": v for k, v in CRITO.items()}


def same_command_accuracy(score, sel):
    """All pairs of runs in sel with the same drive level and different outcomes: is the tipped run scored higher? Bootstrap over drive levels."""
    per = defaultdict(list)
    for d in sorted(set(drive[sel].tolist())):
        idx = np.flatnonzero(sel & (drive == d) & tipk)
        for i in idx[tipv[idx]]:
            for j in idx[~tipv[idx]]:
                if np.abs(mass[i] - mass[j]) < 1 and np.abs(offset[i] - offset[j]) < 1e-3: continue        # replicates carry no payload difference
                per[d].append(float(score[i] > score[j]) + 0.5 * float(score[i] == score[j]))
    vals = [v for d in per for v in per[d]]
    if not vals: return None
    rng = np.random.default_rng(0); keys = list(per); boots = [np.mean([v for k in rng.choice(len(keys), len(keys)) for v in per[keys[k]]]) for _ in range(2000)]
    return dict(point=float(np.mean(vals)), ci_low=float(np.quantile(boots, .025)), ci_high=float(np.quantile(boots, .975)), n_pairs=len(vals), n_levels=len(keys))


# 2. video readouts on the sweep ------------------------------------------------------------------------------------
READ = {"video 2.71 s": args.set, "video 2.71 s + command": f"{args.set}_cmd", "video 1.96 s": f"{args.set}_obs47", "video 1.96 s + command": f"{args.set}_obs47_cmd", "command only": f"{args.set}_cmdonly"}
PRED, res["readouts"] = {}, {}
for label, name in READ.items():
    f = OUT / f"v2_pred_{name}.npz"
    if not f.exists(): print("missing", f.name); continue
    P = np.load(f, allow_pickle=True); assert [str(i) for i in P["ids"]] == c.ids; PRED[label] = P
    r = c.evaluate({k: P[k][:, SW] for k in ("static", "future_state", "future_summary", "tip_score")}, SW, bootstrap=2000)
    score = P["tip_score"].mean(0); am, ac = same_command_accuracy(score, massmap), same_command_accuracy(score, commap)
    res["readouts"][label] = dict(mass_mae_kg=r["mass_mae_kg"], mass_r2=r["mass_r2"], roll_mae_deg=r["roll_mae_deg"], tip_auroc=r.get("tip_auroc"), same_command_mass_map=am, same_command_com_map=ac)
    g = lambda d: "n/a" if d is None else f"{d['point']:.3f}[{d['ci_low']:.3f},{d['ci_high']:.3f}]"
    print(f"SWEEP {label:24s} mass MAE {g(r['mass_mae_kg'])} kg (R2 {r['mass_r2']:.2f}) | roll {g(r['roll_mae_deg'])} deg | AUROC {g(r.get('tip_auroc'))} | same command, mass map {g(am)} (n={am and am['n_pairs']}) | CoM map {g(ac)} (n={ac and ac['n_pairs']})", flush=True)

# 3. two-stage predictor trained on p5 --------------------------------------------------------------------------------
feat = lambda cm, ms: np.c_[cm, np.abs(cm[:, 1]), np.sign(cm[:, 1]), ms].astype(np.float32)
Y = np.concatenate([t["future_state"][:, :, 0], t["future_summary"][:, :1]], 1).astype(np.float32); M = np.concatenate([t["future_state_mask"][:, :, 0], t["future_summary_mask"][:, :1]], 1).astype(bool)
tip_t = torch.tensor(t["future_tip"]).float(); tipm_t = torch.tensor(tipk)
pairs_p5 = np.unique(c.cluster[p5]); val_pairs = np.random.default_rng(2026).choice(pairs_p5, size=int(round(0.15 * len(pairs_p5))), replace=False)
va_m = p5 & np.isin(c.cluster, val_pairs); tr_m = p5 & ~va_m; tr, va = torch.tensor(np.flatnonzero(tr_m)), torch.tensor(np.flatnonzero(va_m)); trn = tr.numpy()
ymu = np.array([Y[trn, j][M[trn, j]].mean() for j in range(Y.shape[1])], np.float32); ysd = np.array([Y[trn, j][M[trn, j]].std() + 1e-6 for j in range(Y.shape[1])], np.float32)
Yn, Mt = torch.tensor((Y - ymu) / ysd), torch.tensor(M); F = feat(cmd, mass); mu, sd = F[trn].mean(0), F[trn].std(0) + 1e-6; X = torch.tensor((F - mu) / sd)
loss_fn = lambda out, i: ((out[:, :-1] - Yn[i]) ** 2)[Mt[i]].mean() + nn.functional.binary_cross_entropy_with_logits(out[:, -1][tipm_t[i]], tip_t[i][tipm_t[i]])
MASS_IN = {"true mass": mass[SW], "no mass (train mean)": np.full(len(SW), mass[trn].mean(), np.float32)}
for label in ("video 1.96 s", "video 2.71 s"):
    if label in PRED: MASS_IN[f"mass from {label}"] = PRED[label]["static"][:, SW, 0].mean(0).astype(np.float32)
GM, GD = np.linspace(240, 720, 49, dtype=np.float32), np.linspace(16, 24, 81, dtype=np.float32); gm, gd = np.meshgrid(GM, GD, indexing="ij")
grid = feat(np.c_[-gd.ravel(), np.full(gd.size, 1.25, np.float32)], gm.ravel())
S2 = {k: {"roll": [], "zmp": [], "score": []} for k in MASS_IN}; pmap = []
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
            o = model(torch.tensor((feat(cmd[SW], ms) - mu) / sd)).numpy(); phys = o[:, :-1] * ysd + ymu
            S2[k]["roll"].append(phys[:, :10]); S2[k]["zmp"].append(phys[:, 10]); S2[k]["score"].append(o[:, -1])
        pmap.append(torch.sigmoid(model(torch.tensor((grid - mu) / sd))[:, -1]).numpy().reshape(gm.shape))
pmap = np.mean(pmap, 0); res["two_stage"] = {}


def crit_from_map(m):
    """Drive command at which the predicted tip-over probability crosses 0.5 for payload mass m (interpolated on the dense map)."""
    row = np.array([np.interp(m, GM, pmap[:, j]) for j in range(len(GD))]); above = np.flatnonzero(row >= 0.5)
    if len(above) == 0: return float(GD[-1])
    j = above[0]
    return float(GD[0]) if j == 0 else float(GD[j - 1] + (0.5 - row[j - 1]) / (row[j] - row[j - 1]) * (GD[j] - GD[j - 1]))


for k in MASS_IN:
    full = {"static": np.repeat(t["static"][None, SW], 3, 0).astype(np.float32), "future_state": np.stack(S2[k]["roll"])[..., None], "future_summary": np.stack(S2[k]["zmp"])[..., None], "tip_score": np.stack(S2[k]["score"])}
    r = c.evaluate(full, SW, bootstrap=2000); score = np.full(N, np.nan); score[SW] = full["tip_score"].mean(0)
    am = same_command_accuracy(score, massmap); pred_tip = score[SW] > 0; acc = float((pred_tip == tipv[SW])[tipk[SW]].mean())
    res["two_stage"][k] = dict(roll_mae_deg=r["roll_mae_deg"], tip_auroc=r.get("tip_auroc"), accuracy=acc, same_command_mass_map=am)
    g = lambda d: "n/a" if d is None else f"{d['point']:.3f}[{d['ci_low']:.3f},{d['ci_high']:.3f}]"
    print(f"TWOSTAGE {k:26s} roll {g(r['roll_mae_deg'])} deg | AUROC {g(r.get('tip_auroc'))} | accuracy {acc:.3f} | same command, mass map {g(am)}", flush=True)

# 4. critical drive command per payload ---------------------------------------------------------------------------------
adv, CP = {}, {}
true_c = np.array([CRIT[round(float(m))] for m in mass[massmap]])
for k, ms in MASS_IN.items():
    est = np.full(N, np.nan, np.float32); est[SW] = ms; pc = np.array([crit_from_map(m) for m in est[massmap]])
    CP[k] = pc
    adv[k] = dict(mae=float(np.abs(pc - true_c).mean()), bias=float((pc - true_c).mean()), within_half_level=float((np.abs(pc - true_c) <= 0.5).mean()))
    print(f"CRITICAL {k:26s} |predicted - true critical drive command|: mean {adv[k]['mae']:.2f}, bias {adv[k]['bias']:+.2f}, within 0.5: {adv[k]['within_half_level']:.2f}", flush=True)
res["critical_drive_prediction"] = adv; res["critical_drive_model_true_mass"] = {str(int(m)): crit_from_map(m) for m in CRIT}
json.dump(res, open(OUT / f"v2_sweep_eval{args.tag}.json", "w"), indent=1)
np.savez(OUT / f"v2_sweep_eval{args.tag}.npz", ids=np.array(c.ids)[SW], mass=mass[SW], offset=offset[SW], drive=drive[SW], tip=tipv[SW], known=tipk[SW], massmap=massmap[SW], commap=commap[SW], rep=rep[SW],
         GM=GM, GD=GD, pmap=pmap, crit_true=true_c, **{f"crit_pred|{k}": v for k, v in CP.items()}, **{f"mass_in|{k}": v for k, v in MASS_IN.items()}, **{f"score|{k}": np.stack(S2[k]["score"]).mean(0) for k in MASS_IN},
         **{f"readout_score|{k}": P["tip_score"][:, SW].mean(0) for k, P in PRED.items()}, **{f"readout_mass|{k}": P["static"][:, SW, 0].mean(0) for k, P in PRED.items()})
print("wrote", OUT / f"v2_sweep_eval{args.tag}.json")
