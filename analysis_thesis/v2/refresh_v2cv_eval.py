"""Re-evaluate saved pair-grouped CV predictions with the current ContextV2.evaluate and add any new keys to the
result JSON (e.g. accuracy on outcome-discordant pairs by the kind of payload change). Existing values are checked
to be unchanged (same bootstrap seed and draw count), so this only adds information.
Run from the safety_probing root:
  PYTHONPATH=analysis_thesis/v2:src .venv/bin/python analysis_thesis/v2/refresh_v2cv_eval.py --set p5 --names p5 p5_cmd p5_cmdonly
"""
import argparse, json
import numpy as np
from common_v2 import ContextV2, ROOT

ap = argparse.ArgumentParser(); ap.add_argument("--set", required=True); ap.add_argument("--names", nargs="+", required=True); ap.add_argument("--out", default="analysis_thesis/out")
args = ap.parse_args()
c = ContextV2(f"artifacts/v2/{args.set}/targets_truecom.pt", f"artifacts/v2/{args.set}/manifests")
for name in args.names:
    jf, pf = ROOT / args.out / f"v2cv_{name}.json", ROOT / args.out / f"v2cv_pred_{name}.npz"
    if not (jf.exists() and pf.exists()): print("skip", name); continue
    old = json.load(open(jf)); P = np.load(pf, allow_pickle=True); assert [str(i) for i in P["ids"]] == c.ids
    new = c.evaluate({k: P[k] for k in ("static", "future_state", "future_summary", "tip_score")}, np.arange(len(c.ids)))
    for k in ("roll_mae_deg", "tip_auroc", "mass_mae_kg", "outcome_flip"):
        if k in old: assert abs(old[k]["point"] - new[k]["point"]) < 1e-9 and abs(old[k]["ci_low"] - new[k]["ci_low"]) < 1e-9, f"{name}: {k} changed"
    added = [k for k in new if k not in old]; old.update({k: new[k] for k in added}); json.dump(old, open(jf, "w"), indent=1)
    g = lambda k: f"{old[k]['point']:.3f}[{old[k]['ci_low']:.3f},{old[k]['ci_high']:.3f}]" if k in old else "n/a"
    print(f"{name:16s} added {added} | pairs with different outcomes: mass {g('outcome_flip_mass')} (n={old.get('n_outcome_flip_mass')}), CoM {g('outcome_flip_com')} (n={old.get('n_outcome_flip_com')})")
