#!/usr/bin/env bash
# Analysis of the boundary-band parts (drive command 18-22) and of all ten v2 parts together.
# Pair-grouped 5-fold CV with true-CoM targets: video only / video + planned command / planned command only.
set -uo pipefail
cd /home/jude/isaac_ws/safety_probing
PY=.venv/bin/python; export PYTHONUNBUFFERED=1
GEN=run_logs/forksafe_v2_generation.log
until grep -q "BAND COMPLETE" $GEN 2>/dev/null; do sleep 60; done
V2="PYTHONPATH=analysis_thesis/v2:src $PY analysis_thesis/v2"
BAND=$(for i in 1 2 3 4 5; do printf "forksafe_v2_band_%02d " $i; done)
MAIN=$(for i in 1 2 3 4 5; do printf "forksafe_v2_part_%02d " $i; done)
prep() {   # prep <dir> <parts...>
  local D=$1; shift; mkdir -p $D; local M=$D/manifests/canonical_runs.jsonl
  $PY analysis_thesis/v2/build_v2_manifest.py --parts "$@" --out $D/manifests || return 1
  PYTHONPATH=src $PY -m safety_probing.vjepa_spatial_features --manifest $M --shard full --output $D/vjepa_spatial12.pt --batch-size 4 2>&1 | tail -1
  COMFIX_MANIFEST=$M PYTHONPATH=src $PY analysis_thesis/truecom_build_targets.py --experiment configs/experiments/vjepa_safety_query.json --manifest $M --shard full --output $D/targets_truecom.pt 2>&1 | tail -1
}
cv() { eval "$V2/psq_v2_cv.py $*" 2>&1 | grep "V2CV\|Traceback\|Error"; }
for SET in band5 all10; do
  D=artifacts/v2/$SET
  echo "[v2-band] $(date +%T) $SET"
  if [ $SET = band5 ]; then prep $D $BAND; else prep $D $MAIN $BAND; fi
  A="--features $D/vjepa_spatial12.pt --targets $D/targets_truecom.pt --manifest-dir $D/manifests"
  cv --name ${SET} $A
  cv --name ${SET}_cmd $A --with-command
  cv --name ${SET}_cmdonly $A --variant command_only
  eval "$V2/kinematics_v2.py --name $SET --targets $D/targets_truecom.pt --manifest-dir $D/manifests" 2>&1 | grep "V2KIN\|Traceback\|Error"
  COMFIX_MANIFEST=$D/manifests/canonical_runs.jsonl PYTHONPATH=analysis_thesis $PY analysis_thesis/mechanics_true.py --manifest $D/manifests/canonical_runs.jsonl --out analysis_thesis/out/mechanics_true_v2_$SET.json 2>&1 | grep -E "runs|fraction|AUROC" | cut -c1-200
done
echo "[v2-band] $(date +%T) ALL DONE"
