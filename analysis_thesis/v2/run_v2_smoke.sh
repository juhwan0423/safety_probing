#!/usr/bin/env bash
# Smoke test on three v2 parts: true-CoM target builder with a v2 manifest, pair-grouped CV, command-only variant.
set -uo pipefail
cd /home/jude/isaac_ws/safety_probing
D=artifacts/v2/p3; M=$D/manifests/canonical_runs.jsonl; export PYTHONUNBUFFERED=1
COMFIX_MANIFEST=$M PYTHONPATH=src .venv/bin/python analysis_thesis/truecom_build_targets.py --experiment configs/experiments/vjepa_safety_query.json --manifest $M --shard full --output $D/targets_truecom.pt 2>&1 | tail -1
A="--features $D/vjepa_spatial12.pt --targets $D/targets_truecom.pt --manifest-dir $D/manifests"
PYTHONPATH=analysis_thesis/v2:src .venv/bin/python analysis_thesis/v2/psq_v2_cv.py --name p3_smoke $A --folds 2 --seeds 17 2>&1 | grep -E "V2CV|Traceback|Error"
PYTHONPATH=analysis_thesis/v2:src .venv/bin/python analysis_thesis/v2/psq_v2.py --name p3_smoke_cmdonly $A --variant command_only --seeds 17 2>&1 | grep -E "V2RESULT|Traceback|Error"
PYTHONPATH=analysis_thesis/v2:src .venv/bin/python analysis_thesis/v2/psq_v2.py --name p3_smoke_cmd $A --with-command --seeds 17 2>&1 | grep -E "V2RESULT|Traceback|Error"
COMFIX_MANIFEST=$M PYTHONPATH=analysis_thesis ../safety_probing/.venv/bin/python analysis_thesis/mechanics_true.py --manifest $M --out analysis_thesis/out/mechanics_true_v2_p3.json 2>&1 | grep -E "runs|ZMP - CoP|stability arm|fraction|AUROC" | cut -c1-260
echo "[smoke] done"
