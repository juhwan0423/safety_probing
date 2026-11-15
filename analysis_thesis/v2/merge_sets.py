"""Concatenate v2 sets (manifests, true-CoM targets, V-JEPA feature caches) into one set directory.

Used to evaluate a readout trained on one set (e.g. p5) on another, fully held-out set (e.g. the boundary sweep): the
runs of every set named in --unused get the split label "unused", so the unchanged training scripts never train,
validate or test on them but still write their predictions.
Run from the safety_probing root:
  .venv/bin/python analysis_thesis/v2/merge_sets.py --sets p5 sweep --unused sweep --out p5sweep [--suffix _obs47]
"""
import argparse, json
from pathlib import Path
import torch

ROOT = Path("/home/jude/isaac_ws/safety_probing/artifacts/v2")
ap = argparse.ArgumentParser(); ap.add_argument("--sets", nargs="+", required=True); ap.add_argument("--unused", nargs="*", default=[]); ap.add_argument("--out", required=True)
ap.add_argument("--suffix", default="", help="observation-length suffix of the caches, e.g. _obs47")
args = ap.parse_args()
out = ROOT / args.out; (out / "manifests").mkdir(parents=True, exist_ok=True)
for fname in ("canonical_runs.jsonl", "pairs.jsonl"):
    with open(out / "manifests" / fname, "w") as fh:
        for s in args.sets:
            for line in open(ROOT / s / "manifests" / fname):
                r = json.loads(line)
                if s in args.unused: r["splits"] = {"main": "unused"}
                r["source_set"] = s; fh.write(json.dumps(r) + "\n")
T = [torch.load(ROOT / s / f"targets_truecom{args.suffix}.pt", map_location="cpu", weights_only=False) for s in args.sets]
merged = {k: v for k, v in T[0].items() if not torch.is_tensor(v) and k not in ("records", "contract")}
for k, v in T[0].items():
    if torch.is_tensor(v): merged[k] = torch.cat([t[k] for t in T], 0)
merged["records"] = [r for t in T for r in t["records"]]
merged["contract"] = dict(T[0]["contract"]); merged["contract"]["condition_ids"] = [i for t in T for i in t["contract"]["condition_ids"]]
for key in ("observed_times_s", "future_times_s"):
    if key in T[0]["contract"]: assert all(list(t["contract"][key]) == list(T[0]["contract"][key]) for t in T), f"{key} differs between sets"
torch.save(merged, out / f"targets_truecom{args.suffix}.pt")
F = [torch.load(ROOT / s / f"vjepa_spatial12{args.suffix}.pt", map_location="cpu", weights_only=False) for s in args.sets]
assert all(f["preprocessing"] == F[0]["preprocessing"] for f in F), "feature caches were made with different preprocessing"
feat = {k: v for k, v in F[0].items() if k not in ("features", "records", "contract")}
feat["features"] = torch.cat([f["features"] for f in F], 0); feat["records"] = [r for f in F for r in f["records"]]
feat["contract"] = dict(F[0]["contract"]); feat["contract"]["condition_ids"] = [i for f in F for i in f["contract"]["condition_ids"]]; feat["contract"]["run_count"] = len(feat["contract"]["condition_ids"])
assert [str(i) for i in feat["contract"]["condition_ids"]] == [str(i) for i in merged["contract"]["condition_ids"]], "feature and target order differ"
assert len(set(map(str, feat["contract"]["condition_ids"]))) == len(feat["contract"]["condition_ids"]), "duplicate condition ids across sets"
torch.save(feat, out / f"vjepa_spatial12{args.suffix}.pt")
print("merged", args.sets, "->", out, "| runs", len(merged["contract"]["condition_ids"]), "| features", tuple(feat["features"].shape), "| unused:", args.unused)
