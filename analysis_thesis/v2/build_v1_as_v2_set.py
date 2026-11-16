"""Expose the v1 grid (ForkSafe-CF v1, 810 runs) in the v2 set layout, for cross-dataset tests.

Writes artifacts/v2/v1grid/ with
  manifests/canonical_runs.jsonl   v1 rows plus the planned drive / steering command read from the run metadata
  manifests/pairs.jsonl            v1 pairs with the condition ids of their two runs
  targets_truecom[_obsF].pt, vjepa_spatial12[_obsF].pt   links to the existing v1 true-CoM targets and feature caches
Nothing is recomputed and no v1 file is modified. Use merge_sets.py --unused v1grid to evaluate v2-trained models on it.
Run from the safety_probing root:  .venv/bin/python analysis_thesis/v2/build_v1_as_v2_set.py
"""
import json, os
from pathlib import Path

ROOT = Path("/home/jude/isaac_ws/safety_probing"); SRC = ROOT / "artifacts/current/manifests"; OUT = ROOT / "artifacts/v2/v1grid"
(OUT / "manifests").mkdir(parents=True, exist_ok=True)
rows = [json.loads(l) for l in open(SRC / "canonical_runs.jsonl")]
for r in rows:
    route = json.load(open(r["metadata_path"]))["config_snapshot"]["route"]
    r["drive_velocity_command"], r["steering_position_command"] = float(route["drive_velocity_command"]), float(route["steering_position_command"])
    r["v1_split"] = r["splits"]["main"]
with open(OUT / "manifests/canonical_runs.jsonl", "w") as fh:
    for r in rows: fh.write(json.dumps(r) + "\n")
cid = {(str(r["pair_id"]), str(r["run_label"])): str(r["condition_id"]) for r in map(json.loads, open(SRC / "runs.jsonl"))}
known = {str(r["condition_id"]) for r in rows}; n = 0
with open(OUT / "manifests/pairs.jsonl", "w") as fh:
    for p in map(json.loads, open(SRC / "pairs.jsonl")):
        a, b = cid[(str(p["pair_id"]), "A")], cid[(str(p["pair_id"]), "B")]
        if a not in known or b not in known: continue
        fh.write(json.dumps({"pair_id": p["pair_id"], "intervention_type": p["intervention_type"], "intervention_value_A": p["intervention_value_A"], "intervention_value_B": p["intervention_value_B"],
                             "route_id": p["route_id"], "turn_direction": p["turn_direction"], "quality_tier": p["quality_tier"], "splits": p.get("splits"), "condition_id_A": a, "condition_id_B": b,
                             "run_A_safety_outcome": p["run_A_safety_outcome"], "run_B_safety_outcome": p["run_B_safety_outcome"]}) + "\n"); n += 1
LINKS = {"targets_truecom.pt": "artifacts/targets/thesis_v1_obs65_truecom.pt", "vjepa_spatial12.pt": "artifacts/features/vjepa_2_1_full_observation_spatial12.pt"}
for f in (24, 36, 47, 56):
    LINKS[f"targets_truecom_obs{f}.pt"] = f"artifacts/targets/thesis_v1_obs{f}_truecom.pt"; LINKS[f"vjepa_spatial12_obs{f}.pt"] = f"artifacts/features/thesis_v1_obs{f}_spatial12.pt"
for name, src in LINKS.items():
    dst = OUT / name
    if dst.is_symlink() or dst.exists(): dst.unlink()
    assert (ROOT / src).exists(), src; os.symlink(ROOT / src, dst)
print("wrote", OUT, "| runs", len(rows), "| pairs", n, "| links", len(LINKS))
