#!/usr/bin/env bash
# v1 experiments on the true-CoM targets (truecom.py). Jobs: "<protocol>:<frames>:<with_command 0|1|only>".
# Skips finished jobs, so several workers can run disjoint lists. 3 seeds, physically disjoint folds.
set -uo pipefail
cd /home/jude/isaac_ws/safety_probing/analysis_thesis
for JOB in "$@"; do
  IFS=: read -r PROTO F CMD <<< "$JOB"
  TGT=artifacts/targets/thesis_v1_obs${F}_truecom.pt
  if [ "$F" = 65 ]; then FEAT=artifacts/features/vjepa_2_1_full_observation_spatial12.pt; else FEAT=artifacts/features/thesis_v1_obs${F}_spatial12.pt; fi
  case "$CMD" in
    1)    NAME=${PROTO}_full_cmd_tc_obs$F;      EXTRA="--with-command" ;;
    only) NAME=${PROTO}_command_only_tc_obs$F;  EXTRA="--variant command_only" ;;
    *)    NAME=${PROTO}_full_tc_obs$F;          EXTRA="" ;;
  esac
  if [ -f out/cv_$NAME.json ]; then echo "[tc] skip $NAME"; continue; fi
  until [ -f ../$TGT ]; do sleep 20; done; sleep 5
  echo "[tc] $(date +%T) start $NAME"
  PYTHONUNBUFFERED=1 ../.venv/bin/python psq_cv.py --protocol $PROTO --features $FEAT --targets $TGT --tag _tc_obs$F $EXTRA 2>&1 | grep "CVRESULT\|CVBASE\|CVFOLD\|Traceback\|Error"
done
echo "[tc] $(date +%T) worker done: $*"
