#!/usr/bin/env bash
# Boundary sweep: manifest, features (2.71 s and 1.96 s observation), true-CoM targets, readouts trained on p5 and
# applied to the sweep, mechanics, and the sweep evaluation. Starts when the sweep generation has finished.
set -uo pipefail
cd /home/jude/isaac_ws/safety_probing
PY=.venv/bin/python; export PYTHONUNBUFFERED=1
GEN=run_logs/forksafe_v2_generation.log
until grep -q "SWEEP COMPLETE" $GEN 2>/dev/null; do sleep 60; done
D=artifacts/v2/sweep; M=$D/manifests/canonical_runs.jsonl; mkdir -p $D
echo "[sweep] $(date +%T) manifest"
$PY analysis_thesis/v2/build_v2_manifest.py --parts forksafe_v2_sweep_01 forksafe_v2_sweep_02 --out $D/manifests || exit 1
echo "[sweep] $(date +%T) features and targets"
PYTHONPATH=src $PY -m safety_probing.vjepa_spatial_features --manifest $M --shard full --output $D/vjepa_spatial12.pt --batch-size 4 2>&1 | tail -1
COMFIX_MANIFEST=$M PYTHONPATH=src $PY analysis_thesis/truecom_build_targets.py --experiment configs/experiments/vjepa_safety_query.json --manifest $M --shard full --output $D/targets_truecom.pt 2>&1 | tail -1
EXP=configs/experiments/thesis_vjepa_obs47.json
PYTHONPATH=src $PY -m safety_probing.vjepa_spatial_features --experiment $EXP --manifest $M --shard full --output $D/vjepa_spatial12_obs47.pt --batch-size 4 2>&1 | tail -1
COMFIX_MANIFEST=$M COMFIX_MED5=1 PYTHONPATH=src $PY analysis_thesis/truecom_build_targets.py --experiment $EXP --manifest $M --shard full --output $D/targets_truecom_obs47.pt 2>&1 | tail -1
echo "[sweep] $(date +%T) merge with p5"
$PY analysis_thesis/v2/merge_sets.py --sets p5 sweep --unused sweep --out p5sweep || exit 1
$PY analysis_thesis/v2/merge_sets.py --sets p5 sweep --unused sweep --out p5sweep --suffix _obs47 || exit 1
S=artifacts/v2/p5sweep
run() { PYTHONPATH=analysis_thesis/v2:src $PY analysis_thesis/v2/psq_v2.py "$@" --manifest-dir $S/manifests 2>&1 | grep "V2RESULT\|Traceback\|Error"; }
echo "[sweep] $(date +%T) readouts trained on the p5 training pairs"
run --name p5sweep --features $S/vjepa_spatial12.pt --targets $S/targets_truecom.pt
run --name p5sweep_cmd --features $S/vjepa_spatial12.pt --targets $S/targets_truecom.pt --with-command
run --name p5sweep_cmdonly --features $S/vjepa_spatial12.pt --targets $S/targets_truecom.pt --variant command_only
run --name p5sweep_obs47 --features $S/vjepa_spatial12_obs47.pt --targets $S/targets_truecom_obs47.pt
run --name p5sweep_obs47_cmd --features $S/vjepa_spatial12_obs47.pt --targets $S/targets_truecom_obs47.pt --with-command
echo "[sweep] $(date +%T) mechanics and evaluation"
COMFIX_MANIFEST=$M PYTHONPATH=analysis_thesis $PY analysis_thesis/mechanics_true.py --manifest $M --out analysis_thesis/out/mechanics_true_v2_sweep.json 2>&1 | grep -E "runs|fraction|AUROC" | cut -c1-200
PYTHONPATH=analysis_thesis/v2:src $PY analysis_thesis/v2/sweep_eval_v2.py 2>&1 | grep -v "^fold"
echo "[sweep] $(date +%T) ALL DONE"
