"""One-off Isaac Sim probe: ground-truth mass properties of the forklift + payload, and how the dataset logger
pairs link masses with link poses. Writes nothing into the dataset; no rendering.

Run from the simulator project directory (so that `forklift_dataset` imports):
  cd /home/jude/isaac_ws/ExoFork-CF/project_march/sim_project
  ../../../env_isaaclab/bin/python /home/jude/isaac_ws/safety_probing/analysis_thesis/sim_probe/probe_mass_properties.py \
      --config /home/jude/isaac_ws/safety_probing/analysis_thesis/sim_probe/probe_config.json --headless --out <json>
"""
import json, sys

args = sys.argv[1:]
cfg_path = args[args.index("--config") + 1]
out_path = args[args.index("--out") + 1]

from forklift_dataset import generator as gen          # boots SimulationApp (headless when --headless is in argv)
from forklift_dataset.config import load_experiment_config
from forklift_dataset.scene import attach_payload_cube, build_load_state, build_scene
from forklift_dataset import state_reader as sr
import numpy as np
import omni.usd
from pxr import Gf, UsdPhysics

config = load_experiment_config(cfg_path)
results = []
try:
    for scene in config.scenes:
        handles = build_scene(config, scene)
        load_state = build_load_state(scene, "load_mass", scene.load_base_mass_kg)
        attach_payload_cube(config, scene, handles, load_state)
        handles.world.reset(); handles.forklift.initialize(); handles.world.play()
        for _ in range(config.settle_steps):
            handles.world.step(render=False)
        stage = omni.usd.get_context().get_stage()
        av = handles.forklift._articulation_view
        masses = np.asarray(av.get_body_masses(clone=True), dtype=float).reshape(-1)
        com_result = av.get_body_coms(clone=True)
        coms = np.asarray(com_result[0] if isinstance(com_result, tuple) else com_result, dtype=float)
        coms = coms.reshape(-1, coms.shape[-1])
        com_orient = None
        if isinstance(com_result, tuple) and len(com_result) > 1:
            com_orient = np.asarray(com_result[1], dtype=float).reshape(-1, np.asarray(com_result[1]).shape[-1]).tolist()
        body_names_attr = list(getattr(av, "body_names", []) or [])
        pv = getattr(av, "_physics_view", None)
        link_paths, link_names = None, None
        if pv is not None:
            try: link_paths = [str(p) for p in (pv.link_paths[0] if len(pv.link_paths) and isinstance(pv.link_paths[0], (list, tuple)) else pv.link_paths)]
            except Exception as exc: link_paths = f"error:{type(exc).__name__}:{exc}"
            try: link_names = [str(n) for n in pv.shared_metatype.link_names]
            except Exception as exc: link_names = f"error:{type(exc).__name__}:{exc}"
        _, traversal = sr._rigid_body_paths_by_key(handles.forklift.prim_path)

        def world_of(path, local):
            prim = stage.GetPrimAtPath(path)
            m = omni.usd.get_world_transform_matrix(prim)
            w = m.Transform(Gf.Vec3d(float(local[0]), float(local[1]), float(local[2])))
            return [float(w[0]), float(w[1]), float(w[2])]

        root_path = sr._resolve_forklift_pose_prim_path(handles, config.assets)
        root_origin = world_of(root_path, (0, 0, 0)); root_x = world_of(root_path, (1, 0, 0)); root_y = world_of(root_path, (0, 1, 0))
        # what the logger does (masses by index, poses by USD traversal order)
        logged = sr._read_system_com(handles, previous_state=None, dt=None)
        snapshot = sr.read_runtime_state(handles, config.assets, gravity_mps2=config.physics_context.gravity_mps2)
        # correct pairing when PhysX exposes the link paths
        truth = None
        if isinstance(link_paths, list) and len(link_paths) == len(masses):
            pts = np.array([world_of(p, coms[i][:3]) for i, p in enumerate(link_paths)])
            truth = {"position": (pts * masses[:, None]).sum(0) / masses.sum(), "per_link": [
                {"path": p, "mass": float(masses[i]), "local_com": coms[i][:3].tolist(), "world_com": pts[i].tolist(), "world_origin": world_of(p, (0, 0, 0))} for i, p in enumerate(link_paths)]}
            truth["position"] = truth["position"].tolist()
        load_path = handles.load_prim_path
        load_info = None
        if load_path:
            lp = stage.GetPrimAtPath(load_path); mapi = UsdPhysics.MassAPI(lp)
            load_info = {"path": load_path, "mass_attr": mapi.GetMassAttr().Get(), "com_attr": [float(v) for v in (mapi.GetCenterOfMassAttr().Get() or (0, 0, 0))],
                         "world_origin": world_of(load_path, (0, 0, 0)), "world_com": world_of(load_path, mapi.GetCenterOfMassAttr().Get() or (0, 0, 0)),
                         "fixed_to_articulation": bool(sr._load_is_fixed_to_articulation(handles))}
        results.append({
            "scenario": scene.scenario_id, "load_mass_kg": float(scene.load_base_mass_kg), "load_com_offset_local": [float(v) for v in scene.load_base_com_offset_local_xyz_m],
            "masses": masses.tolist(), "local_coms": coms.tolist(), "com_orientations": com_orient, "body_names_attr": body_names_attr, "link_paths": link_paths, "link_names": link_names,
            "traversal_paths": traversal, "traversal_world_origins": [world_of(p, (0, 0, 0)) for p in traversal],
            "root_path": root_path, "root_origin": root_origin, "root_x_axis_point": root_x, "root_y_axis_point": root_y,
            "logger_system_com": {"position": logged.position, "total_mass": logged.total_mass_kg, "source": logged.source, "components": logged.component_count},
            "forklift_pose": {"position": list(snapshot.forklift_pose.position) if hasattr(snapshot.forklift_pose, "position") else None},
            "truth_system_com": truth, "load": load_info,
        })
        print("PROBE", scene.scenario_id, "masses", np.round(masses, 2).tolist(), "sum", round(float(masses.sum()), 2), flush=True)
        print("PROBE link_paths", link_paths, flush=True)
        print("PROBE traversal ", traversal, flush=True)
        print("PROBE logger CoM", logged.position, "| truth", None if truth is None else truth["position"], "| root origin", root_origin, flush=True)
        handles.world.stop()
        json.dump(results, open(out_path, "w"), indent=1, default=str)
finally:
    gen.simulation_app.close()
