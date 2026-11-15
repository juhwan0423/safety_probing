"""ForkSafe-CF v2 boundary sweep: a regular grid that maps where tip-over starts.

Fixed left turn with steering command 1.25; the drive command is swept in steps of 0.5 inside 17.5-22.5 and 1.0 outside.
  mass map : payload mass 240..720 kg in steps of 60 (lateral CoM offset 0) at every drive level
             (pairs 240/480, 300/540, 360/600, 420/660 and 720/480; the second 480 kg run uses a route with a
              different id but identical commands, so it is a replicate that checks reproducibility)
  CoM map  : payload mass 480 kg with lateral CoM offset -0.25/+0.25 and -0.125/+0.125 m at the drive levels 17.5-22.5
Same vehicle, camera, physics rate, logging and detector settings as the band parts (single camera).
Writes configs under forklift_dataset/forksafe_v2/ and new dataset roots; nothing existing is modified.
Run:  python analysis_thesis/v2/build_v2_sweep_config.py
"""
import copy, json
from pathlib import Path

SIM = Path("/home/jude/isaac_ws/ExoFork-CF/project_march/sim_project")
template = json.loads((SIM / "forklift_dataset/forksafe_v2/forksafe_v2_band_01.json").read_text())
STEER, DIRECTION = 1.25, "left"
DRIVE = [16.0, 17.0] + [17.5 + 0.5 * k for k in range(11)] + [23.0, 24.0]
BAND = [d for d in DRIVE if 17.5 <= d <= 22.5]
MASS_PAIRS = [(240.0, 480.0), (300.0, 540.0), (360.0, 600.0), (420.0, 660.0)]
COM_PAIRS = [(-0.25, 0.25), (-0.125, 0.125)]


def scene(name, sid, m, y):
    s = copy.deepcopy(template["scenes"][0])
    s.update(scenario_id=sid, description=f"v2 sweep payload: mass={m:.1f} kg, lateral CoM={y:+.4f} m", load_base_mass_kg=m, load_base_com_offset_local_xyz_m=[0.0, y, 0.0],
             notes="ForkSafe-CF v2 boundary sweep. Only mass or hidden load-local lateral CoM changes within a pair.")
    return s


def route(rid, v):
    r = copy.deepcopy(template["routes"][0])
    r.update(route_id=rid, description=f"{DIRECTION} turn, drive command {-v:.3f}, 1-second steering pulse {STEER:+.3f}, straight recovery.", turn_direction=DIRECTION,
             mirror_group_id=rid, speed_label="sweep", steering_label="sweep", drive_velocity_command=-v, steering_position_command=STEER)
    for phase in r["phases"]:
        phase["joint_velocities"] = ["NaN", "NaN", -v]; phase["joint_positions"] = [0.0, STEER if phase["phase_name"] == "turn" else 0.0, "NaN"]
    return r


def build(name, mass_map, com_map):
    cfg = {k: copy.deepcopy(template[k]) for k in ("seed", "headless", "physics_dt", "settle_steps", "assets", "logging", "detector")}
    cfg["dataset_root"] = f"dataset_root/{name}"
    scenes, routes, pairs = {}, {}, []
    def sc(m, y):
        sid = f"{name}_scene_m{m:.0f}_y{y:+.3f}"; scenes.setdefault(sid, scene(name, sid, m, y)); return sid
    def rt(v, rep=False):
        rid = f"{name}_route_d{v:04.1f}{'_rep' if rep else ''}"; routes.setdefault(rid, route(rid, v)); return rid
    def pair(kind, base, mod, sid, rid, tag):
        pairs.append({"intervention_type": kind, "base_value": base, "modified_value": mod, "route_id": rid, "scenario_id": sid, "pair_id": f"{tag}_{name}_{len(pairs):04d}",
                      "seed": int(template["seed"]), "notes": "v2 boundary sweep: manoeuvre commands, visible load pose and the other payload property fixed."})
    if mass_map:
        for v in DRIVE:
            for a, b in MASS_PAIRS: pair("load_mass", a, b, sc(a, 0.0), rt(v), "A_mass")
            pair("load_mass", 720.0, 480.0, sc(720.0, 0.0), rt(v, rep=True), "A_mass")          # the 480 kg run is a replicate
    if com_map:
        for v in BAND:
            for a, b in COM_PAIRS: pair("com_shift", a, b, sc(480.0, a), rt(v), "B_lcom")
    cfg.update(scenes=list(scenes.values()), routes=list(routes.values()), pair_requests=pairs)
    cfg["forksafe_v2"] = {"name": name, "design": "boundary sweep", "steering_command": STEER, "turn_direction": DIRECTION, "drive_levels": DRIVE if mass_map else BAND,
                          "mass_pairs_kg": MASS_PAIRS + [(720.0, 480.0)] if mass_map else [], "com_pairs_m": COM_PAIRS if com_map else [], "cameras": "main_fixed_northwest only"}
    out = SIM / "forklift_dataset/forksafe_v2" / f"{name}.json"; out.write_text(json.dumps(cfg, indent=2) + "\n")
    print("wrote", out, "| pairs", len(pairs), "| runs", 2 * len(pairs), "| scenes", len(scenes), "| routes", len(routes))


build("forksafe_v2_sweep_01", mass_map=True, com_map=False)
build("forksafe_v2_sweep_02", mass_map=False, com_map=True)
