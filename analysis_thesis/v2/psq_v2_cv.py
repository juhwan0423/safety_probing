"""Pair-grouped K-fold cross-validation of the observation PSQ on ForkSafe-CF v2.

Every run is predicted by a model that never saw its pair (both members of a pair share a fold), so all runs
contribute to the test metrics. Early stopping uses a pair-grouped 15% of the training folds.
  --variant full | spatial_mean | command_only ; --with-command appends the planned drive/steering command
Outputs: analysis_thesis/out/v2cv_{name}.json, v2cv_pred_{name}.npz ; prints V2CV / V2CVBASE lines.
"""
import argparse, json, numpy as np, torch
from common_v2 import ContextV2, ROOT, bs, fmt, to_gpu
from safety_probing import safety_query_probe as sq

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--features", required=True); ap.add_argument("--targets", required=True); ap.add_argument("--manifest-dir", required=True)
ap.add_argument("--variant", default="full"); ap.add_argument("--with-command", action="store_true"); ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--seeds", default="17,29,43"); ap.add_argument("--out", default="analysis_thesis/out")
args = ap.parse_args()
c = ContextV2(args.targets, args.manifest_dir)
cache = torch.load(ROOT / args.features, map_location="cpu", weights_only=False)
assert [str(i) for i in cache["contract"]["condition_ids"]] == c.ids
features, targets = cache["features"], c.targets
if args.variant == "spatial_mean": features = features.float().mean((2, 3), keepdim=True).half()
cmd = np.array([[c.man[i]["drive_velocity_command"], c.man[i]["steering_position_command"]] for i in c.ids], np.float32)
z = torch.tensor((cmd - cmd.mean(0)) / (cmd.std(0) + 1e-6))          # command standardisation uses no labels
if args.variant == "command_only":
    features = z[:, None, None, None, :].expand(-1, 10, 1, 1, -1).contiguous().half()
elif args.with_command:
    scale = float(features[:32].float().std())
    features = torch.cat([features, (z * scale)[:, None, None, None, :].expand(-1, *features.shape[1:4], -1).to(features.dtype)], dim=-1)
features = to_gpu(features)
n_pairs = int(c.cluster.max() + 1); N = len(c.ids)
perm = np.random.default_rng(20261116).permutation(n_pairs); fold_of_pair = np.empty(n_pairs, int); fold_of_pair[perm] = np.arange(n_pairs) % args.folds
fold = fold_of_pair[c.cluster]
config = json.load(open(ROOT / "configs/experiments/vjepa_safety_query.json")); tr = config["training"]
seeds = [int(s) for s in args.seeds.split(",")]
OOF = {"static": np.zeros((len(seeds), N, 2), np.float32), "future_state": np.zeros((len(seeds), N, 10, 1), np.float32),
       "future_summary": np.zeros((len(seeds), N, 4), np.float32), "tip_score": np.zeros((len(seeds), N), np.float32)}
t = c.tnp; base = {k: np.full(N, np.nan) for k in ("mass_mae_kg", "com_mae_m", "zmp_mae_m", "roll_mae_deg")}
for k in range(args.folds):
    test = fold == k
    rest = np.unique(c.cluster[~test]); val_pairs = np.random.default_rng(1000 + k).choice(rest, size=max(1, int(round(0.15 * len(rest)))), replace=False)
    split = np.where(test, "test", np.where(np.isin(c.cluster, val_pairs), "val", "train"))
    records = [dict(r) for r in targets["records"]]
    for r, s in zip(records, split): r["split"] = str(s)
    normalizers = sq.fit_normalizers(targets, sq._indices(records, "train"))
    ti, te = np.flatnonzero(split == "train"), np.flatnonzero(test)
    b = c.train_mean_baseline(ti, te)
    for key, (arr, msk, ch) in {"mass_mae_kg": ("static", "static_mask", 0), "com_mae_m": ("static", "static_mask", 1), "zmp_mae_m": ("future_summary", "future_summary_mask", 0)}.items():
        mean = t[arr][ti, ch][t[msk][ti, ch]].mean(); base[key][te] = np.where(t[msk][te, ch], np.abs(t[arr][te, ch] - mean), np.nan)
    mean = t["future_state"][ti, :, 0][t["future_state_mask"][ti, :, 0]].mean()
    base["roll_mae_deg"][te] = bs.condition_mae(np.full_like(t["future_state"][te, :, 0], mean), t["future_state"][te, :, 0], t["future_state_mask"][te, :, 0])
    for si, seed in enumerate(seeds):
        p, _, rep, rt = sq.train_one_seed(features, targets, records, config, normalizers, seed=seed, device="cuda", max_epochs=int(tr["max_epochs"]), patience=int(tr["patience"]))
        den = lambda kk, v: sq._denormalize(v, normalizers[kk]).numpy()
        OOF["static"][si, te] = den("static", p["static"])[te]; OOF["future_state"][si, te, :, 0] = den("future_state", p["future_state"])[te, :, 0]
        OOF["future_summary"][si, te] = den("future_summary", p["future_summary"])[te]; OOF["tip_score"][si, te] = p["future_tip_logits"].numpy()[te]
        print(f"[v2cv] {args.name} fold {k + 1}/{args.folds} seed {seed} train={len(ti)} val={(split == 'val').sum()} test={len(te)} best_epoch={rep['best_epoch']} {rt['duration_s']:.0f}s", flush=True)
np.savez(ROOT / args.out / f"v2cv_pred_{args.name}.npz", fold=fold, ids=np.array(c.ids), **OOF)
res = c.evaluate(OOF, np.arange(N))
res["train_mean_baseline"] = {k: float(np.nanmean(v)) for k, v in base.items()}; res["counts"] = {"runs": N, "pairs": n_pairs, "folds": args.folds}
json.dump(res, open(ROOT / args.out / f"v2cv_{args.name}.json", "w"), indent=1)
print("V2CV", args.name, fmt(res), flush=True); print("V2CVBASE", args.name, res["train_mean_baseline"], flush=True)
