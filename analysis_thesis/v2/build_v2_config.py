"""ForkSafe-CF v2: continuous payload and manoeuvre parameters (no grid, no replicas).

Every pair is an independent random draw:
  payload mass      m ~ U[240, 720] kg
  lateral CoM       y ~ U[-0.25, 0.25] m      (hidden mass property; visible load pose fixed)
  drive command     v ~ U[15, 25]             (wheel velocity command magnitude)
  steering pulse    s ~ U[1.0, 1.5], direction left/right
and one counterfactual intervention (run B) changing only the mass or only the lateral CoM.
Same vehicle, cameras, physics rate, logging and detector settings as the v1 shards.
Writes a new config under forklift_dataset/forksafe_v2/ and a new dataset root; nothing existing is modified.
"""
import argparse, copy, json
from pathlib import Path
import numpy as np

SIM = Path("/home/jude/isaac_ws/ExoFork-CF/project_march/sim_project")
ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--pairs", type=int, required=True); ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--mass", type=float, nargs=2, default=(240.0, 720.0)); ap.add_argument("--com", type=float, nargs=2, default=(-0.25, 0.25))
ap.add_argument("--speed", type=float, nargs=2, default=(15.0, 25.0)); ap.add_argument("--steer", type=float, nargs=2, default=(1.0, 1.5))
ap.add_argument("--min-dmass", type=float, default=60.0); ap.add_argument("--min-dcom", type=float, default=0.08)
ap.add_argument("--single-camera", action="store_true", help="keep only main_fixed_northwest (frames are then written directly under run_X/rgb/)")
args = ap.parse_args()

template = json.loads((SIM / "forklift_dataset/eccv_workshop_shards/eccv_workshop_grid_part_01.json").read_text())
rng = np.random.default_rng(args.seed)
cfg = {k: copy.deepcopy(template[k]) for k in ("seed", "headless", "physics_dt", "settle_steps", "assets", "logging", "detector")}
cfg["dataset_root"] = f"dataset_root/{args.name}"
if args.single_camera:
    cfg["logging"]["cameras"] = [c for c in cfg["logging"]["cameras"] if c["name"] == "main_fixed_northwest"]
base_scene, base_route = template["scenes"][0], template["routes"][0]
scenes, routes, pairs = [], [], []
for k in range(args.pairs):
    m, y = float(rng.uniform(*args.mass)), float(rng.uniform(*args.com))
    v, s = float(rng.uniform(*args.speed)), float(rng.uniform(*args.steer))
    direction = "left" if rng.random() < 0.5 else "right"
    sign = 1.0 if direction == "left" else -1.0
    sid, rid = f"{args.name}_scene_{k:04d}", f"{args.name}_route_{k:04d}"
    scene = copy.deepcopy(base_scene)
    scene.update(scenario_id=sid, description=f"v2 continuous payload: mass={m:.2f} kg, lateral CoM={y:+.4f} m", load_base_mass_kg=m,
                 load_base_com_offset_local_xyz_m=[0.0, y, 0.0], notes="ForkSafe-CF v2. Only mass or hidden load-local lateral CoM changes within a pair.")
    route = copy.deepcopy(base_route)
    route.update(route_id=rid, description=f"{direction} turn, drive command {-v:.3f}, 1-second steering pulse {sign * s:+.3f}, straight recovery.",
                 turn_direction=direction, mirror_group_id=rid, speed_label="continuous", steering_label="continuous",
                 drive_velocity_command=-v, steering_position_command=sign * s)
    for phase in route["phases"]:
        phase["joint_velocities"] = ["NaN", "NaN", -v]
        phase["joint_positions"] = [0.0, sign * s if phase["phase_name"] == "turn" else 0.0, "NaN"]
    if k % 2 == 0:
        while True:
            m2 = float(rng.uniform(*args.mass))
            if abs(m2 - m) >= args.min_dmass: break
        kind, base, mod, tag = "load_mass", m, m2, "A_mass"
    else:
        while True:
            y2 = float(rng.uniform(*args.com))
            if abs(y2 - y) >= args.min_dcom: break
        kind, base, mod, tag = "com_shift", y, y2, "B_lcom"
    scenes.append(scene); routes.append(route)
    pairs.append({"intervention_type": kind, "base_value": base, "modified_value": mod, "route_id": rid, "scenario_id": sid,
                  "pair_id": f"{tag}_{args.name}_{k:04d}", "seed": int(template["seed"]),
                  "notes": "v2 counterfactual: manoeuvre commands, visible load pose and the other payload property fixed."})
cfg.update(scenes=scenes, routes=routes, pair_requests=pairs)
cfg["forksafe_v2"] = {"name": args.name, "sampling_seed": args.seed, "pairs": args.pairs, "mass_range_kg": list(args.mass), "com_range_m": list(args.com),
                      "drive_command_range": list(args.speed), "steering_range": list(args.steer), "min_delta_mass_kg": args.min_dmass, "min_delta_com_m": args.min_dcom}
if args.single_camera:
    cfg["forksafe_v2"]["cameras"] = "main_fixed_northwest only (frames written directly under run_X/rgb/)"
out = SIM / "forklift_dataset/forksafe_v2" / f"{args.name}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(cfg, indent=2) + "\n")
print("wrote", out, "| pairs", len(pairs), "| runs", 2 * len(pairs), "| dataset root", cfg["dataset_root"])
