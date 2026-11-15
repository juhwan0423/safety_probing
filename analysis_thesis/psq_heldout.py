"""Held-out generalisation of the observation PSQ, reusing the paper's training loop unchanged.

  --hold-key load_mass_kg --hold-value 480     train without one payload-mass level, test on it
  --hold-key com --hold-value 0.0              train without one lateral-CoM level
  --hold-key speed_label --hold-value medium   train without one speed regime
  --alt-split left_to_right                    train on left turns, test on right turns
Eval sets: 'strict' = held-out level within the main test components, 'all' = every held-out condition,
'indist' = main test conditions of the trained levels (reference).
"""
import sys, json, argparse, numpy as np, torch
from common import Context, ROOT, bs
sys.path.insert(0, str(ROOT / "src"))
from safety_probing import safety_query_probe as sq

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--hold-key"); ap.add_argument("--hold-value"); ap.add_argument("--alt-split")
ap.add_argument("--targets", default="artifacts/targets/safety_query_full_comfix_med5.pt"); ap.add_argument("--seeds", default="17,29,43")
args = ap.parse_args()

c = Context(args.targets)
config = json.load(open(ROOT / "configs/experiments/vjepa_safety_query.json"))
cache = torch.load(ROOT / "artifacts/features/vjepa_2_1_full_observation_spatial12.pt", map_location="cpu", weights_only=False)
assert [str(i) for i in cache["contract"]["condition_ids"]] == c.ids
features, targets = cache["features"], c.targets
man = {str(r["condition_id"]): r for r in c.canonical_rows}
main = np.array([man[i]["splits"]["main"] for i in c.ids])
attr = {"load_mass_kg": np.array([float(man[i]["load_mass_kg"]) for i in c.ids]),
        "com": np.array([float(man[i]["load_com_offset_local_xyz_m"][1]) for i in c.ids]),
        "speed_label": np.array([man[i]["speed_label"] for i in c.ids]),
        "steering_label": np.array([man[i]["steering_label"] for i in c.ids]),
        "turn_direction": np.array([man[i]["turn_direction"] for i in c.ids])}
records = [dict(r) for r in targets["records"]]
if args.alt_split:
    split = np.array([man[i]["splits"][args.alt_split] for i in c.ids])
    sets = {"test": split == "test"}
else:
    values = attr[args.hold_key]
    held = (values == (float(args.hold_value) if values.dtype.kind == "f" else args.hold_value))
    split = np.where(held, "heldout", main)
    sets = {"strict": held & (main == "test"), "all": held, "indist": (~held) & (main == "test")}
for r, s in zip(records, split): r["split"] = str(s)
print(args.name, {k: int(v.sum()) for k, v in sets.items()}, "train", int((split == "train").sum()), "val", int((split == "val").sum()), flush=True)

train_idx = sq._indices(records, "train")
normalizers = sq.fit_normalizers(targets, train_idx)
tr = config["training"]; seeds = [int(s) for s in args.seeds.split(",")]
P = {"static": [], "roll": [], "summary": [], "tip": []}
for seed in seeds:
    p, _, rep, rt = sq.train_one_seed(features, targets, records, config, normalizers, seed=seed, device="cuda",
                                      max_epochs=int(tr["max_epochs"]), patience=int(tr["patience"]))
    den = lambda k, v: sq._denormalize(v, normalizers[k]).numpy()
    P["static"].append(den("static", p["static"])); P["roll"].append(den("future_state", p["future_state"])[:, :, 0])
    P["summary"].append(den("future_summary", p["future_summary"])); P["tip"].append(p["future_tip_logits"].numpy())
    print(args.name, "seed", seed, rep, round(rt["duration_s"]), flush=True)
P = {k: np.stack(v) for k, v in P.items()}
np.savez(f"analysis_thesis/out/heldout_pred_{args.name}.npz", split=split, **P)

t = c.tnp; clusters = c.all_clusters
runs = {(str(r["pair_id"]), str(r["run_label"])): str(r["condition_id"]) for r in c.run_rows}
index = {cid: i for i, cid in enumerate(c.ids)}
rng = np.random.default_rng(bs.BOOTSTRAP_SEED)
out = {}
for name, S in sets.items():
    idx = np.flatnonzero(S)
    if idx.size == 0: continue
    r = {"n": int(idx.size)}
    def hm(v): return bs.hierarchical_mean(v, clusters[idx], bootstrap=2000, rng=rng)
    r["mass_mae_kg"] = hm(np.abs(P["static"][:, idx, 0] - t["static"][idx, 0]))
    r["mass_bias_kg"] = hm(P["static"][:, idx, 0] - t["static"][idx, 0])
    r["com_mae_m"] = hm(np.abs(P["static"][:, idx, 1] - t["static"][idx, 1]))
    r["com_bias_m"] = hm(P["static"][:, idx, 1] - t["static"][idx, 1])
    r["roll_mae_deg"] = hm(np.stack([bs.condition_mae(s[idx], t["future_state"][idx, :, 0], t["future_state_mask"][idx, :, 0]) for s in P["roll"]]))
    r["zmp_mae_m"] = hm(np.stack([bs.condition_mae(s[idx, 0], t["future_summary"][idx, 0], t["future_summary_mask"][idx, 0]) for s in P["summary"]]))
    m = t["future_tip_mask"][idx].astype(bool); lab = t["future_tip"][idx][m].astype(int)
    if len(np.unique(lab)) == 2:
        r["tip_auroc"] = {"point": float(np.mean([bs.auc(lab, s[idx][m]) for s in P["tip"]])), "pos": int(lab.sum()), "neg": int((1 - lab).sum())}
    pairs = {"load_mass": [], "com_shift": []}; pc = {"load_mass": [], "com_shift": []}
    for row in c.pair_rows:
        a, b = index.get(runs[(str(row["pair_id"]), "A")]), index.get(runs[(str(row["pair_id"]), "B")])
        if a is None or b is None or not (S[a] and S[b]): continue
        pairs[str(row["intervention_type"])].append((a, b)); pc[str(row["intervention_type"])].append(c.pair_to_component[str(row["pair_id"])])
    for kind, ch, key in (("load_mass", 0, "mass_direction"), ("com_shift", 1, "com_direction")):
        if pairs[kind]:
            corr = bs.pair_correctness(P["static"][:, :, ch], t["static"][:, ch], pairs[kind])
            r[key] = bs.hierarchical_mean(corr, np.asarray(pc[kind]), bootstrap=2000, rng=rng)
    r["pred_mass_mean"] = float(P["static"][:, idx, 0].mean()); r["pred_mass_sd"] = float(P["static"][:, idx, 0].std())
    r["pred_com_mean_abs"] = float(np.abs(P["static"][:, idx, 1]).mean())
    out[name] = r
    f = lambda k: (f"{r[k]['point']:.3f}[{r[k]['ci_low']:.3f},{r[k]['ci_high']:.3f}]" if k in r else "n/a")
    print(f"RESULT {args.name:22s} {name:7s} n={r['n']:3d} mass_mae={f('mass_mae_kg')} bias={f('mass_bias_kg')} com_mae={f('com_mae_m')} roll={f('roll_mae_deg')} "
          f"tip_auroc={r.get('tip_auroc', {}).get('point', float('nan')):.3f} mass_dir={f('mass_direction')} com_dir={f('com_direction')}", flush=True)
json.dump(out, open(f"analysis_thesis/out/heldout_{args.name}.json", "w"), indent=1)
