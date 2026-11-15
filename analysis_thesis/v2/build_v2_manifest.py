"""Manifest for ForkSafe-CF v2 (continuous payload/manoeuvre parameters), compatible with the existing
feature extractor and target builder. Labels use the same post-hoc rules as v1 (safety_probing.analysis/labeling).
Splits are assigned per pair by a hash of the pair id (70/15/15), so they stay fixed as more parts arrive."""
import argparse, hashlib, json, os, sys
from pathlib import Path
ROOT = Path("/home/jude/isaac_ws/safety_probing"); sys.path.insert(0, str(ROOT / "src"))
from safety_probing import analysis, labeling
from safety_probing.config import LabelingConfig

SIM_ROOT = Path("/home/jude/isaac_ws/ExoFork-CF/project_march/sim_project/dataset_root")
ap = argparse.ArgumentParser(); ap.add_argument("--parts", nargs="+", required=True); ap.add_argument("--out", default="artifacts/v2/manifests")
args = ap.parse_args()
lab_cfg = LabelingConfig.from_dict(json.load(open(ROOT / "configs/local_dataset.json"))["labeling"])
out_dir = ROOT / args.out; out_dir.mkdir(parents=True, exist_ok=True)

def split_of(pair_id: str) -> str:
    h = int(hashlib.sha256(pair_id.encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 70 else ("val" if h < 85 else "test")

runs, pairs, skipped = [], [], 0
for part in args.parts:
    root = SIM_ROOT / part
    for pair_dir in sorted((root / "pairs").iterdir()):
        pm_path = pair_dir / "metadata_pair.json"
        if not pm_path.is_file(): skipped += 1; continue
        pm = json.load(open(pm_path)); rows = {}
        ok = True
        for label in ("A", "B"):
            rd = pair_dir / f"run_{label}"; md_path = rd / "metadata_run.json"; dense = rd / "dense_log.csv"; rgb = rd / "rgb"
            if not (md_path.is_file() and dense.is_file() and rgb.is_dir()): ok = False; break
            cams = {d.name: str(d) for d in sorted(rgb.iterdir()) if d.is_dir()}
            if not cams and any(f.suffix == ".png" for f in rgb.iterdir()):     # single-camera runs: frames directly under rgb/
                cams = {"main_fixed_northwest": str(rgb)}
            if len(os.listdir(cams.get("main_fixed_northwest", rd))) != 120: ok = False; break
            md = json.load(open(md_path))
            ds = analysis.analyze_dense_log(dense, lab_cfg)
            if ds.get("row_count") != 360: ok = False; break
            lab = labeling.derive_safety_label(md, ds, lab_cfg)
            route = md["config_snapshot"]["route"]
            rows[label] = {
                "condition_id": analysis._condition_id(md), "run_id": md["run_id"], "pair_id": md["pair_id"], "run_label": label,
                "dataset_root": str(root), "run_dir": str(rd), "dense_log_path": str(dense), "metadata_path": str(md_path), "rgb_dirs": cams,
                "load_mass_kg": float(md["load_state"]["load_mass_kg"]), "load_com_offset_local_xyz_m": [float(v) for v in md["load_state"]["load_com_offset_local_xyz_m"]],
                "intervention_type": md["intervention_type"], "intervention_value": float(md["intervention_value"]),
                "route_id": md["route_id"], "turn_direction": md["turn_direction"], "speed_label": md.get("speed_label"), "steering_label": md.get("steering_label"),
                "drive_velocity_command": float(route["drive_velocity_command"]), "steering_position_command": float(route["steering_position_command"]),
                "seed": md.get("seed"), "tip_over": int(bool(md.get("tip_over"))), "max_roll_deg": md.get("max_roll_deg"), "max_pitch_deg": md.get("max_pitch_deg"),
                "dense_summary": ds, "label_evidence": lab, "failure_time_s": lab["event_time_s"],
                **{k: lab[k] for k in ("safety_outcome", "binary_tip_label", "future_tip_label", "future_prediction_eligible", "event_observed", "event_reason", "censor_time_s", "followup_time_s")},
                "splits": {"main": split_of(md["pair_id"])},
            }
        if not ok: skipped += 1; continue
        a, b = rows["A"], rows["B"]
        q = labeling.modernized_quality(outcome_a=a["safety_outcome"], outcome_b=b["safety_outcome"], same_action=bool(pm.get("same_action_trajectory", True)),
                                        max_roll_deg_delta=(b["max_roll_deg"] - a["max_roll_deg"]), max_pitch_deg_delta=(b["max_pitch_deg"] - a["max_pitch_deg"]))
        for r in (a, b): r["quality_tier"], r["quality_accepted"] = q["tier"], q["accepted"]
        runs += [a, b]
        pairs.append({"pair_id": pm["pair_id"], "intervention_type": pm["intervention_type"], "intervention_value_A": pm["intervention_value_A"], "intervention_value_B": pm["intervention_value_B"],
                      "route_id": pm["route_id"], "turn_direction": pm["turn_direction"], "dataset_root": str(root), "splits": a["splits"], "quality_tier": q["tier"],
                      "condition_id_A": a["condition_id"], "condition_id_B": b["condition_id"], "run_A_safety_outcome": a["safety_outcome"], "run_B_safety_outcome": b["safety_outcome"]})
ids = [r["condition_id"] for r in runs]; assert len(ids) == len(set(ids)), "duplicate condition ids"
runs.sort(key=lambda r: r["condition_id"])
for name, rows_ in (("canonical_runs.jsonl", runs), ("pairs.jsonl", pairs)):
    with open(out_dir / name, "w") as fh:
        for r in rows_: fh.write(json.dumps(r) + "\n")
from collections import Counter
print("pairs", len(pairs), "runs", len(runs), "skipped/incomplete pair dirs", skipped)
print("split (runs):", dict(Counter(r["splits"]["main"] for r in runs)), "| outcome:", dict(Counter(r["safety_outcome"] for r in runs)), "| tier:", dict(Counter(p["quality_tier"] for p in pairs)), "| kind:", dict(Counter(p["intervention_type"] for p in pairs)))
