"""Shared evaluation context for thesis-extension analyses.

Reuses the paper's build_statistics.py definitions (split, components, pairs,
hierarchical bootstrap) so new numbers are directly comparable with Table 1.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/home/jude/isaac_ws/safety_probing")
PAPER = Path("/home/jude/CWM_paper/Safety_Probing_paper")
sys.path.insert(0, str(PAPER / "scripts"))
import build_statistics as bs  # noqa: E402


class Context:
    def __init__(self, target_cache: str = "artifacts/targets/safety_query_full.pt"):
        paths = bs.source_paths(ROOT, "full")
        stats = bs.read_json(paths["dataset_statistics"])
        source_config = Path(stats["provenance"]["source_config"]).resolve()
        self.canonical_rows = bs.read_jsonl(paths["canonical_runs"])
        self.run_rows = bs.read_jsonl(paths["runs"])
        self.pair_rows = bs.read_jsonl(paths["pairs"])
        self.targets = torch.load(ROOT / target_cache, map_location="cpu", weights_only=False)
        self.ids = [str(i) for i in self.targets["contract"]["condition_ids"]]
        rec = self.targets["records"]
        self.split = np.array([r["split"] for r in rec])
        self.train = np.flatnonzero(self.split == "train")
        self.val = np.flatnonzero(self.split == "val")
        self.test = np.flatnonzero(self.split == "test")
        self.test_ids = [self.ids[i] for i in self.test]
        self.test_clusters, self.pair_to_component, _ = bs.build_component_labels(
            ROOT, source_config, paths["split_manifest"], self.run_rows, self.test_ids)
        self.all_clusters, _, _ = bs.build_component_labels(
            ROOT, source_config, paths["split_manifest"], self.run_rows, self.ids)
        self.tnp = {k: v.numpy() for k, v in self.targets.items() if hasattr(v, "shape")}
        self.tsub = {k: v[self.test] for k, v in self.tnp.items()}
        self.pairs = bs.build_pair_indices(self.pair_rows, self.run_rows, self.test_ids, self.pair_to_component)
        self.paths = paths

    def evaluate(self, pred: dict[str, np.ndarray], bootstrap: int = 10000) -> dict:
        """pred: static [S,Ntest,2], future_state [S,Ntest,10,>=1] (roll ch0),
        future_summary [S,Ntest,>=1] (min ZMP ch0), tip_score [S,Ntest]."""
        rng = np.random.default_rng(bs.BOOTSTRAP_SEED)
        out = {}
        errs = bs.metric_errors(pred, self.tsub)
        for k, v in errs.items():
            out[k] = bs.hierarchical_mean(v, self.test_clusters, bootstrap=bootstrap, rng=rng)
        m = self.tsub["future_tip_mask"].astype(bool)
        labels = self.tsub["future_tip"][m].astype(np.int64)
        out["tip_auroc"] = bs.hierarchical_auc(labels, pred["tip_score"][:, m], self.test_clusters[m], bootstrap=bootstrap, rng=rng)
        ts = self.tsub["static"]
        for name, vals, tgt, src in (
            ("mass_direction", pred["static"][:, :, 0], ts[:, 0], "load_mass"),
            ("com_direction", pred["static"][:, :, 1], ts[:, 1], "com_shift"),
            ("gold_zmp_ordering", pred["future_summary"][:, :, 0], self.tsub["future_summary"][:, 0], "gold_zmp"),
        ):
            pc = bs.pair_correctness(vals, tgt, self.pairs[src]["pairs"])
            out[name] = bs.hierarchical_mean(pc, np.asarray(self.pairs[src]["clusters"], dtype=np.int64), bootstrap=bootstrap, rng=rng)
        return out


def fmt(res: dict) -> str:
    order = ["mass_mae_kg", "com_mae_m", "roll_mae_deg", "zmp_mae_m", "tip_auroc", "mass_direction", "com_direction", "gold_zmp_ordering"]
    return " | ".join(f"{k}={res[k]['point']:.3f}[{res[k]['ci_low']:.3f},{res[k]['ci_high']:.3f}]" for k in order)


def physical_config(context: "Context"):
    """(mass, lateral CoM, route) per condition: the 810 conditions are 270 physical configurations x 3 seed replicas."""
    man = {str(r["condition_id"]): r for r in context.canonical_rows}
    cfg = [(float(man[i]["load_mass_kg"]), float(man[i]["load_com_offset_local_xyz_m"][1]), str(man[i]["route_id"])) for i in context.ids]
    group = [f'{man[i]["speed_label"]}_{man[i]["steering_label"]}' for i in context.ids]      # maneuver group (both turn directions)
    return cfg, np.array(group), man


def evaluate_any(context: "Context", pred: dict, idx: np.ndarray, clusters: np.ndarray, bootstrap: int = 5000) -> dict:
    """Paper metrics on an arbitrary condition subset. pred arrays are [S, len(idx), ...] aligned with idx;
    clusters gives the bootstrap unit for every one of the 810 conditions."""
    rng = np.random.default_rng(bs.BOOTSTRAP_SEED)
    t = {k: v[idx] for k, v in context.tnp.items()}
    cl = np.asarray(clusters)[idx]
    _, cl_int = np.unique(cl, return_inverse=True)
    out = {"n": int(len(idx)), "n_clusters": int(cl_int.max() + 1)}
    for k, v in bs.metric_errors(pred, t).items():
        out[k] = bs.hierarchical_mean(v, cl_int, bootstrap=bootstrap, rng=rng)
    m = t["future_tip_mask"].astype(bool)
    labels = t["future_tip"][m].astype(np.int64)
    if len(np.unique(labels)) == 2:
        try:
            out["tip_auroc"] = bs.hierarchical_auc(labels, pred["tip_score"][:, m], cl_int[m], bootstrap=min(bootstrap, 2000), rng=rng)
        except RuntimeError:
            out["tip_auroc"] = {"point": float(np.mean([bs.auc(labels, s) for s in pred["tip_score"][:, m]])), "ci_low": float("nan"), "ci_high": float("nan")}
    pos = {int(g): j for j, g in enumerate(idx)}
    index = {cid: i for i, cid in enumerate(context.ids)}
    runs = {(str(r["pair_id"]), str(r["run_label"])): str(r["condition_id"]) for r in context.run_rows}
    kinds = {"mass_direction": [], "com_direction": [], "gold_zmp_ordering": []}
    pcl = {k: [] for k in kinds}
    for row in context.pair_rows:
        a, b = pos.get(index[runs[(str(row["pair_id"]), "A")]]), pos.get(index[runs[(str(row["pair_id"]), "B")]])
        if a is None or b is None:
            continue
        key = "mass_direction" if row["intervention_type"] == "load_mass" else "com_direction"
        kinds[key].append((a, b)); pcl[key].append(cl_int[a])
        if row["quality_tier"] == "gold":
            kinds["gold_zmp_ordering"].append((a, b)); pcl["gold_zmp_ordering"].append(cl_int[a])
    for key, (vals, tgt) in {"mass_direction": (pred["static"][:, :, 0], t["static"][:, 0]), "com_direction": (pred["static"][:, :, 1], t["static"][:, 1]),
                             "gold_zmp_ordering": (pred["future_summary"][:, :, 0], t["future_summary"][:, 0])}.items():
        if kinds[key]:
            out[key] = bs.hierarchical_mean(bs.pair_correctness(vals, tgt, kinds[key]), np.asarray(pcl[key]), bootstrap=bootstrap, rng=rng)
    return out


def fmt_any(res: dict) -> str:
    keys = ["mass_mae_kg", "com_mae_m", "roll_mae_deg", "zmp_mae_m", "tip_auroc", "mass_direction", "com_direction", "gold_zmp_ordering"]
    return f"n={res['n']} | " + " | ".join(f"{k}={res[k]['point']:.3f}[{res[k]['ci_low']:.3f},{res[k]['ci_high']:.3f}]" for k in keys if k in res)

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
