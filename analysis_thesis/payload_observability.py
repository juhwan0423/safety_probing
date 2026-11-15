"""How large is the kinematic trace that a payload change leaves in the observed motion? (ForkSafe-CF v1)

For every manoeuvre (route) the grid contains the same run with the other payload levels, and the simulator is
deterministic. We therefore compare, over time,
    mass       : 720 kg vs 240 kg   (same lateral offset, same route)
    lateral CoM: +0.25 m vs -0.25 m (same mass, same route; configured offsets, 0.225 m physically)
and report the difference in chassis position and heading. One pixel of the 384-px model input covers about 3 cm.
Writes out/payload_observability_v1.json and figures/fig_payload_observability (PDF/PNG when matplotlib is available).
Run:  PYTHONPATH=. ../external/LTX-2/.venv/bin/python payload_observability.py
"""
import csv, json
import numpy as np
import comfix as cf

T = np.arange(0.1, 2.75, 0.05)
man = cf.load_manifest()
runs = {}
for path, r in man.items():
    key = (float(r["load_mass_kg"]), float(r["load_com_offset_local_xyz_m"][1]), r["route_id"])
    if key in runs:
        continue                                  # replicas are identical
    with open(path) as fh:
        d = sorted(csv.DictReader(fh), key=lambda x: float(x["sim_time"]))
    f = lambda k: np.array([float(x[k]) if x[k] != "" else np.nan for x in d])
    t = f("sim_time"); k = np.abs(t[None] - T[:, None]).argmin(1)
    runs[key] = dict(x=f("forklift_pos_x")[k], y=f("forklift_pos_y")[k], yaw=np.unwrap(np.radians(f("yaw_deg")))[k], roll=f("roll_deg")[k])
masses = sorted({k[0] for k in runs}); offsets = sorted({k[1] for k in runs}); routes = sorted({k[2] for k in runs})


def diff(a, b):
    return np.hypot(a["x"] - b["x"], a["y"] - b["y"]), np.abs(np.degrees(a["yaw"] - b["yaw"]))


pos = {"mass": [], "com": []}; yaw = {"mass": [], "com": []}
for route in routes:
    for off in offsets:
        if (masses[-1], off, route) in runs and (masses[0], off, route) in runs:
            p, y = diff(runs[(masses[-1], off, route)], runs[(masses[0], off, route)]); pos["mass"].append(p); yaw["mass"].append(y)
    for m in masses:
        if (m, offsets[-1], route) in runs and (m, offsets[0], route) in runs:
            p, y = diff(runs[(m, offsets[-1], route)], runs[(m, offsets[0], route)]); pos["com"].append(p); yaw["com"].append(y)
S = {k: {"pos": np.array(pos[k]), "yaw": np.array(yaw[k])} for k in pos}
out = {"t": T.tolist()}
for k in S:
    for q in ("pos", "yaw"):
        out[f"{k}_{q}_median"] = np.median(S[k][q], 0).tolist(); out[f"{k}_{q}_q25"] = np.quantile(S[k][q], .25, 0).tolist(); out[f"{k}_{q}_q75"] = np.quantile(S[k][q], .75, 0).tolist()
    out[f"{k}_n"] = int(len(S[k]["pos"]))
json.dump(out, open("out/payload_observability_v1.json", "w"))
at = lambda arr, tt: float(np.median(arr, 0)[int(np.argmin(np.abs(T - tt)))])
print(f"comparisons: mass {out['mass_n']} (720 vs 240 kg), lateral CoM {out['com_n']} (+0.25 vs -0.25 m)")
for tt in (1.0, 1.95, 2.3, 2.7):
    print(f"t = {tt:4.2f} s | mass: position {at(S['mass']['pos'], tt) * 100:6.1f} cm, heading {at(S['mass']['yaw'], tt):5.2f} deg | lateral CoM: position {at(S['com']['pos'], tt) * 100:5.2f} cm, heading {at(S['com']['yaw'], tt):5.2f} deg")

try:
    import thesis_style as ts
    from thesis_style import plt
    ts.apply()
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for a, q, lab in ((ax[0], "pos", "difference in chassis position (cm)"), (ax[1], "yaw", "difference in heading (deg)")):
        sc = 100.0 if q == "pos" else 1.0
        for k, col, name in (("mass", ts.PRED, "payload mass, 720 vs 240 kg"), ("com", ts.GEN, "lateral CoM offset, $+$0.25 vs $-$0.25 m")):
            med = np.array(out[f"{k}_{q}_median"]) * sc; lo = np.array(out[f"{k}_{q}_q25"]) * sc; hi = np.array(out[f"{k}_{q}_q75"]) * sc
            a.fill_between(T, np.maximum(lo, 1e-3), hi, color=col, alpha=0.15, lw=0); a.plot(T, med, color=col, label=name)
        a.set_yscale("log"); a.axvline(2.0, color=ts.BASE, lw=0.8, ls=":"); a.set_xlabel("time (s)"); a.set_ylabel(lab)
    ax[0].axhline(3.1, color=ts.INK, lw=0.8, ls="--"); ax[0].text(0.15, 3.6, "1 pixel at the model input", fontsize=8, va="bottom")
    ax[0].text(2.03, ax[0].get_ylim()[0] * 1.6, "steering\nstarts", fontsize=7.5, color=ts.BASE, va="bottom"); ts.panel(ax[0], "a"); ts.panel(ax[1], "b")
    handles, labels = ax[0].get_legend_handles_labels()
    fig.tight_layout(w_pad=1.6, rect=(0, 0, 1, 0.9)); fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    ts.save(fig, "fig_payload_observability")
except ModuleNotFoundError as exc:
    print("figure skipped:", exc)
