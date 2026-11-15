"""Observation PSQ on ForkSafe-CF v2 (continuous payload/manoeuvre parameters). Paper training loop reused unchanged.
  --train-parts a b ...   restrict the training pairs to these dataset parts (learning curve); val/test stay fixed
  --variant full | spatial_mean | command_only   (command_only: the planned drive/steering command without any video)
"""
import argparse, json, sys, numpy as np, torch
from common_v2 import ContextV2, ROOT, bs, fmt, to_gpu
from safety_probing import safety_query_probe as sq

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--features", required=True); ap.add_argument("--targets", required=True); ap.add_argument("--manifest-dir", required=True)
ap.add_argument("--train-parts", nargs="*"); ap.add_argument("--variant", default="full"); ap.add_argument("--seeds", default="17,29,43")
ap.add_argument("--max-epochs", type=int); ap.add_argument("--with-command", action="store_true"); ap.add_argument("--debug-split", action="store_true"); ap.add_argument("--out", default="analysis_thesis/out")
args = ap.parse_args()
c = ContextV2(args.targets, args.manifest_dir)
cache = torch.load(ROOT / args.features, map_location="cpu", weights_only=False)
assert [str(i) for i in cache["contract"]["condition_ids"]] == c.ids
features, targets = cache["features"], c.targets
if args.variant == "spatial_mean": features = features.float().mean((2, 3), keepdim=True).half()
split = c.split.copy()
if args.debug_split:
    order = np.argsort(c.cluster, kind="stable"); n = len(order); split[order[: int(.6 * n)]] = "train"; split[order[int(.6 * n): int(.8 * n)]] = "val"; split[order[int(.8 * n):]] = "test"
if args.train_parts:
    split = np.where((split == "train") & ~np.isin(c.part, args.train_parts), "unused", split)
if args.with_command or args.variant == "command_only":
    # planned manoeuvre (drive and signed steering command) appended to every memory token: action-conditioned prediction
    cmd = np.array([[c.man[i]["drive_velocity_command"], c.man[i]["steering_position_command"]] for i in c.ids], np.float32)
    tri = np.flatnonzero(split == "train"); z = torch.tensor((cmd - cmd[tri].mean(0)) / (cmd[tri].std(0) + 1e-6))
    if args.variant == "command_only":       # no video at all: what the planned manoeuvre alone predicts
        features = z[:, None, None, None, :].expand(-1, 10, 1, 1, -1).contiguous().half()
    else:
        scale = float(features[:32].float().std())
        extra = (z * scale)[:, None, None, None, :].expand(-1, *features.shape[1:4], -1).to(features.dtype)
        features = torch.cat([features, extra], dim=-1)
features = to_gpu(features)
records = [dict(r) for r in targets["records"]]
for r, s in zip(records, split): r["split"] = str(s)
tr_idx, te_idx = np.flatnonzero(split == "train"), np.flatnonzero(split == "test")
print(args.name, {k: int((split == k).sum()) for k in ("train", "val", "test", "unused")}, flush=True)
config = json.load(open(ROOT / "configs/experiments/vjepa_safety_query.json")); tr = config["training"]
normalizers = sq.fit_normalizers(targets, sq._indices(records, "train"))
P = {"static": [], "future_state": [], "future_summary": [], "tip_score": []}
for seed in [int(s) for s in args.seeds.split(",")]:
    p, state, rep, rt = sq.train_one_seed(features, targets, records, config, normalizers, seed=seed, device="cuda",
                                          max_epochs=int(args.max_epochs or tr["max_epochs"]), patience=int(tr["patience"]))
    den = lambda k, v: sq._denormalize(v, normalizers[k]).numpy()
    P["static"].append(den("static", p["static"])); P["future_state"].append(den("future_state", p["future_state"])[:, :, :1])
    P["future_summary"].append(den("future_summary", p["future_summary"])); P["tip_score"].append(p["future_tip_logits"].numpy())
    torch.save({"state": state, "normalizers": normalizers, "report": rep}, ROOT / args.out / f"v2_ckpt_{args.name}_seed{seed}.pt")
    print(args.name, "seed", seed, rep, round(rt["duration_s"]), flush=True)
P = {k: np.stack(v) for k, v in P.items()}
np.savez(ROOT / args.out / f"v2_pred_{args.name}.npz", split=split, ids=np.array(c.ids), **P)
res = c.evaluate({k: v[:, te_idx] for k, v in P.items()}, te_idx)
res["train_mean_baseline"] = c.train_mean_baseline(tr_idx, te_idx); res["counts"] = {k: int((split == k).sum()) for k in ("train", "val", "test", "unused")}
json.dump(res, open(ROOT / args.out / f"v2_{args.name}.json", "w"), indent=1)
print("V2RESULT", args.name, fmt(res), flush=True); print("V2BASE", args.name, res["train_mean_baseline"], flush=True)
