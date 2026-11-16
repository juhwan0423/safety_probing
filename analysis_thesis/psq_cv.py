"""Cross-validated observation PSQ with physically disjoint folds (seed replicas kept together).

  --protocol mos5    : 5-fold over the 45 (payload mass, lateral offset, speed) groups. This is the only clean v1 protocol:
                       the simulator is deterministic and the six routes of one speed share the whole pre-turn segment, so
                       under the two protocols below the payload of a test run can be looked up from a training run with
                       the same pre-turn video. Here every route that shares a pre-turn segment is in the same fold.
  --protocol logo9   : leave-one-maneuver-group-out (speed x steering, both turn directions)   [shares pre-turn segments]
  --protocol config5 : 5-fold over the 270 physical configurations                             [shares pre-turn segments]
  --variant full | spatial_mean | command (adds the planned speed/steer command as an extra memory token)
Validation for early stopping = a config-grouped 15% of the non-test configurations. Paper training loop reused unchanged.
"""
import sys, json, argparse, numpy as np, torch
from common import Context, ROOT, bs, physical_config, evaluate_any, fmt_any, to_gpu
sys.path.insert(0, str(ROOT / "src"))
from safety_probing import safety_query_probe as sq

ap = argparse.ArgumentParser()
ap.add_argument("--protocol", required=True, choices=("logo9", "config5", "mos5")); ap.add_argument("--variant", default="full")
ap.add_argument("--targets", default="artifacts/targets/safety_query_full_comfix_med5.pt"); ap.add_argument("--seeds", default="17,29,43")
ap.add_argument("--features", default="artifacts/features/vjepa_2_1_full_observation_spatial12.pt"); ap.add_argument("--tag", default=""); ap.add_argument("--with-command", action="store_true")
args = ap.parse_args()
name = f"{args.protocol}_{args.variant}{'_cmd' if args.with_command else ''}{args.tag}"

c = Context(args.targets)
config = json.load(open(ROOT / "configs/experiments/vjepa_safety_query.json"))
cache = torch.load(ROOT / args.features, map_location="cpu", weights_only=False)
assert [str(i) for i in cache["contract"]["condition_ids"]] == c.ids
features, targets = cache["features"], c.targets
if args.variant == "spatial_mean":
    features = features.float().mean((2, 3), keepdim=True).half()
cfg, group, man = physical_config(c)
if args.with_command or args.variant == "command_only":
    # planned manoeuvre (drive command, signed steering command): known before the turn starts, so it is a legitimate input
    SPEED = {"slow": -15.0, "medium": -20.0, "fast": -25.0}; STEER = {"wide": 1.0, "medium": 1.25, "tight": 1.5}
    cmd = np.array([[SPEED[man[i]["speed_label"]], (1.0 if man[i]["turn_direction"] == "left" else -1.0) * STEER[man[i]["steering_label"]]] for i in c.ids], np.float32)
    z = torch.tensor((cmd - cmd.mean(0)) / cmd.std(0))
    if args.variant == "command_only":
        features = z[:, None, None, None, :].expand(-1, 10, 1, 1, -1).contiguous().half()
    else:
        scale = float(features[:32].float().std())
        features = torch.cat([features, (z * scale)[:, None, None, None, :].expand(-1, *features.shape[1:4], -1).to(features.dtype)], dim=-1)
features = to_gpu(features)
cfg_ids = {k: j for j, k in enumerate(sorted(set(cfg)))}; cfg_idx = np.array([cfg_ids[k] for k in cfg])
route = np.array([k[2] for k in cfg])
rs = np.random.default_rng(20261116)
speed = np.array([man[i]["speed_label"] for i in c.ids])
mos = np.array([f"{k[0]:.0f}|{k[1]:+.2f}|{sp}" for k, sp in zip(cfg, speed)])          # shared pre-turn dynamics
if args.protocol == "logo9":
    fold_of = {g: j for j, g in enumerate(sorted(set(group)))}; fold = np.array([fold_of[g] for g in group]); clusters = group
elif args.protocol == "mos5":
    groups = sorted(set(mos)); perm = rs.permutation(len(groups)); fold_of = {g: int(j % 5) for g, j in zip(np.array(groups)[perm], range(len(groups)))}
    fold = np.array([fold_of[g] for g in mos]); clusters = mos
    cfg_idx = np.array([groups.index(g) for g in mos])                                   # validation carved out by whole groups too
else:
    perm = rs.permutation(len(cfg_ids)); fold_cfg = np.empty(len(cfg_ids), int); fold_cfg[perm] = np.arange(len(cfg_ids)) % 5
    fold = fold_cfg[cfg_idx]; clusters = route
n_fold = fold.max() + 1
seeds = [int(s) for s in args.seeds.split(",")]; tr = config["training"]
N = len(c.ids)
OOF = {"static": np.zeros((len(seeds), N, 2), np.float32), "future_state": np.zeros((len(seeds), N, 10, 1), np.float32),
       "future_summary": np.zeros((len(seeds), N, 4), np.float32), "tip_score": np.zeros((len(seeds), N), np.float32)}
base_err = {k: np.zeros(N) for k in ("mass", "com", "roll", "zmp")}
t = c.tnp
for k in range(n_fold):
    test = fold == k
    rest_cfg = np.unique(cfg_idx[~test]); val_cfg = set(np.random.default_rng(1000 + k).choice(rest_cfg, size=max(1, int(round(0.15 * len(rest_cfg)))), replace=False).tolist())
    split = np.where(test, "test", np.where(np.isin(cfg_idx, list(val_cfg)), "val", "train"))
    assert not (set(cfg_idx[split == "train"]) & set(cfg_idx[split == "test"])) and not (set(cfg_idx[split == "val"]) & set(cfg_idx[split != "val"]))
    records = [dict(r) for r in targets["records"]]
    for r, s in zip(records, split): r["split"] = str(s)
    train_idx = sq._indices(records, "train"); normalizers = sq.fit_normalizers(targets, train_idx)
    ti, te = np.flatnonzero(split == "train"), np.flatnonzero(test)
    for key, (arr, msk, chn) in {"mass": ("static", "static_mask", 0), "com": ("static", "static_mask", 1), "zmp": ("future_summary", "future_summary_mask", 0)}.items():
        mean = t[arr][ti, chn][t[msk][ti, chn]].mean(); base_err[key][te] = np.abs(t[arr][te, chn] - mean)
    mean = t["future_state"][ti, :, 0][t["future_state_mask"][ti, :, 0]].mean()
    base_err["roll"][te] = bs.condition_mae(np.full_like(t["future_state"][te, :, 0], mean), t["future_state"][te, :, 0], t["future_state_mask"][te, :, 0])
    for si, seed in enumerate(seeds):
        p, _, rep, rt = sq.train_one_seed(features, targets, records, config, normalizers, seed=seed, device="cuda", max_epochs=int(tr["max_epochs"]), patience=int(tr["patience"]))
        den = lambda kk, v: sq._denormalize(v, normalizers[kk]).numpy()
        OOF["static"][si, te] = den("static", p["static"])[te]; OOF["future_state"][si, te, :, 0] = den("future_state", p["future_state"])[te, :, 0]
        OOF["future_summary"][si, te] = den("future_summary", p["future_summary"])[te]; OOF["tip_score"][si, te] = p["future_tip_logits"].numpy()[te]
        print(f"[cv] {name} fold {k + 1}/{n_fold} seed {seed} train={len(ti)} val={(split == 'val').sum()} test={len(te)} best_epoch={rep['best_epoch']} {rt['duration_s']:.0f}s", flush=True)
np.savez(f"out/cv_pred_{name}.npz", fold=fold, clusters=clusters, **OOF)
res = evaluate_any(c, OOF, np.arange(N), clusters)
res["train_mean_baseline"] = {k: float(v.mean()) for k, v in base_err.items()}
per_fold = {}
for k in range(n_fold):
    idx = np.flatnonzero(fold == k); sub = {kk: v[:, idx] for kk, v in OOF.items()}
    r = evaluate_any(c, sub, idx, np.array([cfg_idx[i] for i in range(N)]), bootstrap=500)
    per_fold[str(sorted(set(group[idx]))[0] if args.protocol == "logo9" else k)] = {kk: (vv["point"] if isinstance(vv, dict) else vv) for kk, vv in r.items()}
res["per_fold"] = per_fold
json.dump(res, open(f"out/cv_{name}.json", "w"), indent=1)
print("CVRESULT", name, fmt_any(res), flush=True)
print("CVBASE", name, res["train_mean_baseline"], flush=True)
for kf, r in per_fold.items():
    print(f"CVFOLD {name} {kf:16s} mass={r['mass_mae_kg']:.1f} com={r['com_mae_m']:.3f} roll={r['roll_mae_deg']:.2f} tip={r.get('tip_auroc', float('nan')):.3f} com_dir={r.get('com_direction', float('nan')):.3f}", flush=True)
