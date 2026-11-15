"""Generated-future audit on ForkSafe-CF v2 (continuous parameters, no repeated pre-turn segments).

A readout is trained on frozen LTX-VAE latents of the REAL future (training pairs) and then applied, unchanged, to
  real    : real-future latents of the held-out test pairs        (what the readout can read when the future is real)
  rollout : LTX continuations of the same test observations       (what survives in a generated future)
Predictions over rollout seeds are averaged as in the paper (mean probability for the tip logit).
Reuses the paper's LTX Safety Query model and training loop; metrics from common_v2.ContextV2.

Inputs (see run_v2_ltx.sh): real-future cache, rollout cache, LTX target cache (true-CoM), manifest dir.
Outputs: analysis_thesis/out/v2ltx_{name}.json, v2ltx_pred_{name}.npz ; prints V2LTX lines.
"""
import argparse, json
import numpy as np, torch
from common_v2 import ContextV2, ROOT, fmt
from safety_probing import ltx_safety_query as lq
from safety_probing import safety_query_probe as sq

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--real-cache", required=True); ap.add_argument("--rollout-cache", required=True)
ap.add_argument("--targets", required=True); ap.add_argument("--manifest-dir", required=True); ap.add_argument("--seeds", default="17,29,43")
ap.add_argument("--experiment", default="configs/experiments/ltx_safety_query.json"); ap.add_argument("--out", default="analysis_thesis/out")
args = ap.parse_args()
c = ContextV2(args.targets, args.manifest_dir)
config = json.load(open(ROOT / args.experiment)); tr = config["training"]
real = torch.load(ROOT / args.real_cache, map_location="cpu", weights_only=False)
roll = torch.load(ROOT / args.rollout_cache, map_location="cpu", weights_only=False)
assert [str(i) for i in real["contract"]["condition_ids"]] == c.ids, "real-future cache and targets differ in condition order"
targets = c.targets
records = [dict(r) for r in targets["records"]]
for r, s in zip(records, c.split): r["split"] = str(s)
rollout_seeds, by_seed, target_indices, _ = lq._generated_layout(roll, records)
gen_idx = np.asarray(target_indices)
assert set(c.split[gen_idx]) == {"test"}, f"rollouts must cover test runs only, got {set(c.split[gen_idx])}"
te_idx = np.flatnonzero(c.split == "test")
print(args.name, "real", tuple(real["features"].shape), "rollouts", tuple(roll["features"].shape), "rollout seeds", rollout_seeds,
      "| test runs", len(te_idx), "with rollouts", len(gen_idx), flush=True)
normalizers = lq.fit_ltx_normalizers(targets, sq._indices(records, "train"))
den = lambda k, v: sq._denormalize(v, normalizers[k]).numpy()


def pack(p):
    return {"static": den("static", p["static"]), "future_state": den("future_state", p["future_state"])[:, :, :1],
            "future_summary": den("future_summary", p["future_summary"]), "tip_score": p["future_tip_logits"].numpy()}


R, GEN = [], []
for seed in [int(s) for s in args.seeds.split(",")]:
    model, real_pred, state, rep, rt = lq.train_one_seed(real["features"], targets, records, config, normalizers, seed=seed, device="cuda",
                                                         max_epochs=int(tr["max_epochs"]), patience=int(tr["patience"]))
    R.append(pack(real_pred))
    per_seed = lq._predict_rollouts(model, roll["features"], by_seed, batch_size=int(tr["batch_size"]), device="cuda")
    GEN.append(pack(lq.aggregate_rollout_predictions(per_seed)))
    print(args.name, "seed", seed, rep, round(rt["duration_s"]), flush=True)
real_all = {k: np.stack([r[k] for r in R]) for k in R[0]}                 # [S, N, ...]
gen = {k: np.stack([g[k] for g in GEN]) for k in GEN[0]}                  # [S, len(gen_idx), ...]
np.savez(ROOT / args.out / f"v2ltx_pred_{args.name}.npz", gen_idx=gen_idx, split=c.split, ids=np.array(c.ids),
         **{f"real_{k}": v for k, v in real_all.items()}, **{f"gen_{k}": v for k, v in gen.items()})
res = {"real_future": c.evaluate({k: v[:, gen_idx] for k, v in real_all.items()}, gen_idx), "generated_future": c.evaluate(gen, gen_idx),
       "train_mean_baseline": c.train_mean_baseline(np.flatnonzero(c.split == "train"), gen_idx), "rollout_seeds": [int(s) for s in rollout_seeds],
       "counts": {"train": int((c.split == "train").sum()), "val": int((c.split == "val").sum()), "test_with_rollouts": int(len(gen_idx))}}
json.dump(res, open(ROOT / args.out / f"v2ltx_{args.name}.json", "w"), indent=1)
print("V2LTX", args.name, "real future     ", fmt(res["real_future"]), flush=True)
print("V2LTX", args.name, "generated future", fmt(res["generated_future"]), flush=True)
print("V2LTXBASE", args.name, res["train_mean_baseline"], flush=True)
