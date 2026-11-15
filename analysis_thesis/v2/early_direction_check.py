"""Why does the video-only roll error stay at the baseline until the turn is visible? (ForkSafe-CF v2, fixed split)

Before the steering pulse (t < 2.0 s) the video cannot show which way the vehicle will turn, so the SIGN of the future
roll is unknown although its size may already be predictable. For each observation length this prints, on the test
runs: the signed roll MAE (the usual metric), the MAE between |predicted roll| and |true roll| (size only), and how
often the predicted roll has the sign of the true roll at the final future step.
Run from the safety_probing root:  PYTHONPATH=analysis_thesis/v2:src .venv/bin/python analysis_thesis/v2/early_direction_check.py p5
"""
import json, sys
import numpy as np
from common_v2 import ContextV2, ROOT

name = sys.argv[1] if len(sys.argv) > 1 else "p5"
c = ContextV2(f"artifacts/v2/{name}/targets_truecom.pt", f"artifacts/v2/{name}/manifests"); t = c.tnp
steer = np.array([c.man[i]["steering_position_command"] for i in c.ids]); print("turn directions (sign of the steering command):", {int(s): int((np.sign(steer) == s).sum()) for s in (-1, 1)})
out = {}
for F in (24, 36, 47, 56, 65):
    for suf in ("", "_cmd"):
        P = np.load(ROOT / f"analysis_thesis/out/v2_pred_{name}_obs{F}{suf}.npz", allow_pickle=True)
        te = np.flatnonzero(P["split"] == "test"); pred = P["future_state"][:, :, :, 0]                      # [seeds, n_test or N, 10]
        pred = pred[:, te] if pred.shape[1] == len(c.ids) else pred
        y, m = t["future_state"][te, :, 0], t["future_state_mask"][te, :, 0].astype(bool)
        signed = np.mean([np.abs(p - y)[m].mean() for p in pred]); size = np.mean([np.abs(np.abs(p) - np.abs(y))[m].mean() for p in pred])
        last = np.array([np.flatnonzero(mm)[-1] if mm.any() else -1 for mm in m]); ok = last >= 0; big = ok & (np.abs(y[np.arange(len(te)), last]) > 5.0)
        sign_ok = np.mean([(np.sign(p[np.arange(len(te)), last]) == np.sign(y[np.arange(len(te)), last]))[big].mean() for p in pred])
        base_size = np.abs(np.abs(y) - np.abs(t["future_state"][P["split"] == "train", :, 0])[t["future_state_mask"][P["split"] == "train", :, 0].astype(bool)].mean())[m].mean()
        out[f"obs{F}{suf}"] = dict(signed_mae_deg=float(signed), size_mae_deg=float(size), sign_agreement=float(sign_ok), n_sign=int(big.sum()), size_baseline_deg=float(base_size))
        print(f"obs {F / 24:.2f} s {'video+cmd' if suf else 'video    '}: signed roll MAE {signed:5.2f} deg | size-only MAE {size:5.2f} deg (train-mean size baseline {base_size:.2f}) | sign of final roll correct {sign_ok:.2f} (n={big.sum()})")
json.dump(out, open(ROOT / f"analysis_thesis/out/v2_early_direction_{name}.json", "w"), indent=1)
