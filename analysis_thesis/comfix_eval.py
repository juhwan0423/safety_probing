"""Table 1 under the original, bug-fixed (comfix) and bug-fixed + de-spiked (comfix_med5) labels."""
import json, sys
from pathlib import Path
from common import Context, ROOT, bs, fmt
out = {}
for variant, suffix in (("original", ""), ("comfix", "_comfix"), ("comfix_med5", "_comfix_med5")):
    vj = ROOT / f"outputs/safety_query/full{suffix}"; lt = ROOT / f"outputs/ltx_safety_query/full_prompt_v1{suffix}"
    if not all((vj / f"seed_{s:04d}.pt").is_file() and (lt / f"seed_{s:04d}.pt").is_file() for s in bs.QUERY_SEEDS):
        print(variant, "not ready"); continue
    c = Context(f"artifacts/targets/safety_query_full{suffix}.pt")
    paths = dict(c.paths)
    for s in bs.QUERY_SEEDS:
        paths[f"vjepa_seed_{s}"] = vj / f"seed_{s:04d}.pt"; paths[f"ltx_seed_{s}"] = lt / f"seed_{s:04d}.pt"
    methods = bs.load_method_predictions(paths, c.ids)
    out[variant] = {}
    for name in ("vjepa_prefix", "ltx_oracle", "ltx_generated"):
        r = c.evaluate(bs.align_predictions(methods[name], name, c.ids, c.test_ids))
        out[variant][name] = r
        print(f"{variant:12s} {name:14s} {fmt(r)}", flush=True)
    tb = bs.train_mean_baselines(c.targets, c.train, c.test); out[variant]["train_mean"] = tb
    print(f"{variant:12s} train-mean baseline MAE: {tb}")
json.dump(out, open("out/comfix_table1.json", "w"), indent=1)
