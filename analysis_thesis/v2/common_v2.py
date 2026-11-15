"""Evaluation helpers for ForkSafe-CF v2 (continuous parameters; pairs are the independent bootstrap unit)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path("/home/jude/isaac_ws/safety_probing")
sys.path.insert(0, "/home/jude/CWM_paper/Safety_Probing_paper/scripts"); sys.path.insert(0, str(ROOT / "src"))
import build_statistics as bs  # noqa: E402


class ContextV2:
    def __init__(self, targets_path: str, manifest_dir: str):
        self.targets = torch.load(ROOT / targets_path, map_location="cpu", weights_only=False)
        self.ids = [str(i) for i in self.targets["contract"]["condition_ids"]]
        self.tnp = {k: v.numpy() for k, v in self.targets.items() if hasattr(v, "shape")}
        rows = [json.loads(l) for l in open(ROOT / manifest_dir / "canonical_runs.jsonl")]
        self.man = {str(r["condition_id"]): r for r in rows}
        self.pair_rows = [json.loads(l) for l in open(ROOT / manifest_dir / "pairs.jsonl")]
        self.split = np.array([self.man[i]["splits"]["main"] for i in self.ids])
        self.part = np.array([Path(self.man[i]["dataset_root"]).name for i in self.ids])
        pair_ids = sorted({self.man[i]["pair_id"] for i in self.ids}); pid = {p: j for j, p in enumerate(pair_ids)}
        self.cluster = np.array([pid[self.man[i]["pair_id"]] for i in self.ids])
        self.index = {cid: n for n, cid in enumerate(self.ids)}

    def evaluate(self, pred: dict, idx: np.ndarray, bootstrap: int = 5000) -> dict:
        """pred arrays [S, len(idx), ...] aligned with idx."""
        rng = np.random.default_rng(bs.BOOTSTRAP_SEED)
        t = {k: v[idx] for k, v in self.tnp.items()}
        _, cl = np.unique(self.cluster[idx], return_inverse=True)
        out = {"n": int(len(idx)), "n_pairs": int(cl.max() + 1)}
        for k, v in bs.metric_errors(pred, t).items():
            out[k] = bs.hierarchical_mean(v, cl, bootstrap=bootstrap, rng=rng)
        m = t["future_tip_mask"].astype(bool); labels = t["future_tip"][m].astype(np.int64)
        if len(np.unique(labels)) == 2:
            out["tip_auroc"] = bs.hierarchical_auc(labels, pred["tip_score"][:, m], cl[m], bootstrap=min(bootstrap, 2000), rng=rng)
            out["tip_rate"] = float(labels.mean())
        # continuous-regression diagnostics (seed-mean prediction)
        for name, ch in (("mass", 0), ("com", 1)):
            p, y = pred["static"][:, :, ch].mean(0), t["static"][:, ch]
            slope, icpt = np.polyfit(y, p, 1)
            out[f"{name}_r2"] = float(1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2)); out[f"{name}_slope"] = float(slope)
            out[f"{name}_pearson"] = float(np.corrcoef(p, y)[0, 1])
        pos = {int(g): j for j, g in enumerate(idx)}
        kinds = {"mass_direction": [], "com_direction": [], "gold_zmp_ordering": []}; pcl = {k: [] for k in kinds}
        for row in self.pair_rows:
            a, b = self.index.get(row["condition_id_A"]), self.index.get(row["condition_id_B"])
            if a is None or b is None or a not in pos or b not in pos: continue
            a, b = pos[a], pos[b]
            key = "mass_direction" if row["intervention_type"] == "load_mass" else "com_direction"
            kinds[key].append((a, b)); pcl[key].append(cl[a])
            if row["quality_tier"] == "gold": kinds["gold_zmp_ordering"].append((a, b)); pcl["gold_zmp_ordering"].append(cl[a])
        for key, (vals, tgt) in {"mass_direction": (pred["static"][:, :, 0], t["static"][:, 0]), "com_direction": (pred["static"][:, :, 1], t["static"][:, 1]),
                                 "gold_zmp_ordering": (pred["future_summary"][:, :, 0], t["future_summary"][:, 0])}.items():
            if kinds[key]:
                out[key] = bs.hierarchical_mean(bs.pair_correctness(vals, tgt, kinds[key]), np.asarray(pcl[key]), bootstrap=bootstrap, rng=rng)
        # outcome-discordant pairs (same command, one payload change, one member tips): does the tipped member get the higher tip score?
        tipv, tipk = t["future_tip"].astype(bool), t["future_tip_mask"].astype(bool); flip, fcl, fkind = [], [], []
        for key in ("mass_direction", "com_direction"):
            for (a, b), g in zip(kinds[key], pcl[key]):
                if tipk[a] and tipk[b] and tipv[a] != tipv[b]:
                    flip.append((a, b) if tipv[a] else (b, a)); fcl.append(g); fkind.append(key)
        out["n_outcome_flip_pairs"] = len(flip)
        if flip:
            hi, lo = np.array([f[0] for f in flip]), np.array([f[1] for f in flip]); d = pred["tip_score"][:, hi] - pred["tip_score"][:, lo]
            out["outcome_flip"] = bs.hierarchical_mean((d > 0) + 0.5 * (d == 0), np.asarray(fcl), bootstrap=bootstrap, rng=rng)
        return out

    def train_mean_baseline(self, train_idx: np.ndarray, idx: np.ndarray) -> dict:
        t = self.tnp; out = {}
        for key, (arr, msk, ch) in {"mass_mae_kg": ("static", "static_mask", 0), "com_mae_m": ("static", "static_mask", 1), "zmp_mae_m": ("future_summary", "future_summary_mask", 0)}.items():
            mean = t[arr][train_idx, ch][t[msk][train_idx, ch]].mean(); out[key] = float(np.abs(t[arr][idx, ch] - mean)[t[msk][idx, ch]].mean())
        mean = t["future_state"][train_idx, :, 0][t["future_state_mask"][train_idx, :, 0]].mean()
        out["roll_mae_deg"] = float(bs.condition_mae(np.full_like(t["future_state"][idx, :, 0], mean), t["future_state"][idx, :, 0], t["future_state_mask"][idx, :, 0]).mean())
        return out


def fmt(res: dict) -> str:
    keys = ["mass_mae_kg", "com_mae_m", "roll_mae_deg", "zmp_mae_m", "tip_auroc", "mass_direction", "com_direction", "gold_zmp_ordering", "outcome_flip"]
    s = f"n={res['n']} pairs={res['n_pairs']} | " + " | ".join(f"{k}={res[k]['point']:.3f}[{res[k]['ci_low']:.3f},{res[k]['ci_high']:.3f}]" for k in keys if k in res)
    return s + f" | R2 mass={res['mass_r2']:.3f} (slope {res['mass_slope']:.2f}) com={res['com_r2']:.3f} (slope {res['com_slope']:.2f})"

def to_gpu(features, reserve_gb: float = 4.0):
    """Keep the feature cache on the GPU when it fits (training then no longer indexes a CPU tensor per batch)."""
    import torch
    if not torch.cuda.is_available():
        return features
    free, _ = torch.cuda.mem_get_info()
    need = features.numel() * features.element_size()
    if free > need + reserve_gb * 2 ** 30:
        return features.to("cuda")
    print(f"[features] kept on CPU (free GPU memory {free / 2 ** 30:.1f} GB)", flush=True)
    return features
