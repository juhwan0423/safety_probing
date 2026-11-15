"""Seed replicas: how similar are they (video features), and how much do they inflate test metrics?"""
import json, collections, numpy as np, torch
from common import Context, ROOT, bs
c = Context()                                   # original paper targets
man = {str(r["condition_id"]): r for r in c.canonical_rows}
cfg = [(float(man[i]["load_mass_kg"]), float(man[i]["load_com_offset_local_xyz_m"][1]), man[i]["route_id"]) for i in c.ids]
by = collections.defaultdict(list)
for n, k in enumerate(cfg): by[k].append(n)
split = c.split
twin_train = np.array([any(split[j] == "train" for j in by[cfg[n]] if j != n) for n in range(len(cfg))])
feat = torch.load(ROOT / "artifacts/features/vjepa_2_1_full_observation.pt", map_location="cpu", weights_only=False)["features"].float().reshape(len(cfg), -1).numpy()
fn = feat / np.linalg.norm(feat, axis=1, keepdims=True)
# replica similarity vs nearest non-replica similarity
rep_sim, other_sim, l2_rep, l2_other = [], [], [], []
S = fn @ fn.T
D = None
for n in range(len(cfg)):
    reps = [j for j in by[cfg[n]] if j != n]
    mask = np.ones(len(cfg), bool); mask[by[cfg[n]]] = False
    rep_sim.append(S[n, reps].max()); other_sim.append(S[n, mask].max())
    l2_rep.append(np.linalg.norm(feat[n] - feat[reps], axis=1).min()); l2_other.append(np.linalg.norm(feat[n] - feat[mask], axis=1).min())
rep_sim, other_sim, l2_rep, l2_other = map(np.array, (rep_sim, other_sim, l2_rep, l2_other))
print("pooled V-JEPA feature cosine: nearest replica median %.6f (min %.6f) | nearest non-replica median %.6f" % (np.median(rep_sim), rep_sim.min(), np.median(other_sim)))
print("L2 distance: nearest replica median %.4f | nearest non-replica median %.4f | replica is the nearest neighbour for %.1f%% of conditions" % (np.median(l2_rep), np.median(l2_other), 100 * np.mean(l2_rep < l2_other)))
print("exactly identical replica features (L2==0):", int((l2_rep == 0).sum()), "of", len(cfg))
# error by split / twin status using the paper checkpoints (predictions exist for all 810 conditions)
P = bs.load_method_predictions(c.paths, c.ids)
t = c.tnp
def errs(method):
    st = P[method]["static"]
    if method == "ltx_generated": return None
    mass = np.abs(st[:, :, 0] - t["static"][:, 0]).mean(0); com = np.abs(st[:, :, 1] - t["static"][:, 1]).mean(0)
    roll = np.mean([bs.condition_mae(s[:, :, 0], t["future_state"][:, :, 0], t["future_state_mask"][:, :, 0]) for s in P[method]["future_state"]], 0)
    return mass, com, roll
groups = {"train": split == "train", "val": split == "val", "test, twin in train (n=%d)" % int(((split == "test") & twin_train).sum()): (split == "test") & twin_train,
          "test, twin only in val (n=%d)" % int(((split == "test") & ~twin_train).sum()): (split == "test") & ~twin_train}
res = {}
for method in ("vjepa_prefix", "ltx_oracle"):
    mass, com, roll = errs(method); print(f"\n{method}: MAE  mass(kg)  CoM(m)  future roll(deg)")
    for name, g in groups.items():
        print(f"   {name:32s} {mass[g].mean():7.2f}  {com[g].mean():.4f}  {roll[g].mean():6.2f}")
        res[f"{method}/{name}"] = [float(mass[g].mean()), float(com[g].mean()), float(roll[g].mean())]
json.dump(dict(rep_cos_median=float(np.median(rep_sim)), other_cos_median=float(np.median(other_sim)), replica_nearest_frac=float(np.mean(l2_rep < l2_other)), errors=res), open("out/replica_leak.json", "w"), indent=1)
