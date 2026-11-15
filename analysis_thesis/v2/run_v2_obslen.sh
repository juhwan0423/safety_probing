#!/usr/bin/env bash
# How early can the future be predicted? Observation truncated to 1.0 / 1.5 / 1.96 / 2.33 / 2.71 s, with and without the planned command.
# Usage: run_v2_obslen.sh <N parts>   (uses artifacts/v2/p<N>)
set -uo pipefail
cd /home/jude/isaac_ws/safety_probing
PY=.venv/bin/python; export PYTHONUNBUFFERED=1
# TARGET_BUILDER=truecom uses targets from the true rigid-body CoM (default: comfix, kept for the earlier p3 run)
N=$1; D=artifacts/v2/p$N; M=$D/manifests/canonical_runs.jsonl; TB=${TARGET_BUILDER:-comfix}
if [ $TB = truecom ]; then SUF=truecom; else SUF=med5; fi
for F in 24 36 47 56 65; do
  if [ $F = 65 ]; then FEAT=$D/vjepa_spatial12.pt; TGT=$D/targets_$SUF.pt; else
    EXP=configs/experiments/thesis_vjepa_obs$F.json; FEAT=$D/vjepa_spatial12_obs$F.pt; TGT=$D/targets_${SUF}_obs$F.pt
    [ -f $FEAT ] || PYTHONPATH=src $PY -m safety_probing.vjepa_spatial_features --experiment $EXP --manifest $M --shard full --output $FEAT --batch-size 4 2>&1 | tail -1
    [ -f $TGT ] || COMFIX_MANIFEST=$M COMFIX_MED5=1 PYTHONPATH=src $PY analysis_thesis/${TB}_build_targets.py --experiment $EXP --manifest $M --shard full --output $TGT 2>&1 | tail -1
  fi
  PYTHONPATH=analysis_thesis/v2:src $PY analysis_thesis/v2/psq_v2.py --name p${N}_obs$F --features $FEAT --targets $TGT --manifest-dir $D/manifests 2>&1 | grep "V2RESULT\|V2BASE"
  PYTHONPATH=analysis_thesis/v2:src $PY analysis_thesis/v2/psq_v2.py --name p${N}_obs${F}_cmd --features $FEAT --targets $TGT --manifest-dir $D/manifests --with-command 2>&1 | grep "V2RESULT\|V2BASE"
done
echo "[obslen] ALL DONE"
