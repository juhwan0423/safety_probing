"""Paper-split table (Table 1 of the workshop paper) on the true-CoM targets, next to the earlier label variants.
Outputs out/truecom_table1.json. Run after run_truecom_paper_split.sh:  PYTHONPATH=. ../.venv/bin/python truecom_eval.py"""
import json
from common import Context, ROOT, bs, fmt

VARIANTS = (("paper", "artifacts/targets/safety_query_full.pt", "outputs/safety_query/full", "outputs/ltx_safety_query/full_prompt_v1"),
            ("truecom", "artifacts/targets/thesis_v1_obs65_truecom.pt", "outputs/safety_query/full_truecom", "outputs/ltx_safety_query/full_prompt_v1_truecom"))
out = {}
for variant, targets, vj, lt in VARIANTS:
    vj, lt = ROOT / vj, ROOT / lt
    if not all((vj / f"seed_{s:04d}.pt").is_file() and (lt / f"seed_{s:04d}.pt").is_file() for s in bs.QUERY_SEEDS):
        print(variant, "not ready"); continue
    c = Context(targets)
    paths = dict(c.paths)
    for s in bs.QUERY_SEEDS:
        paths[f"vjepa_seed_{s}"] = vj / f"seed_{s:04d}.pt"; paths[f"ltx_seed_{s}"] = lt / f"seed_{s:04d}.pt"
    methods = bs.load_method_predictions(paths, c.ids)
    out[variant] = {}
    for name in ("vjepa_prefix", "ltx_oracle", "ltx_generated"):
        r = c.evaluate(bs.align_predictions(methods[name], name, c.ids, c.test_ids))
        out[variant][name] = r
        print(f"{variant:8s} {name:14s} {fmt(r)}", flush=True)
    tb = bs.train_mean_baselines(c.targets, c.train, c.test); out[variant]["train_mean"] = tb
    print(f"{variant:8s} train-mean baseline MAE: {tb}")
json.dump(out, open("out/truecom_table1.json", "w"), indent=1)
