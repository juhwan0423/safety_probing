#!/usr/bin/env bash
# Early-prediction sweep worker: run the listed "<frames>:<with_command 0|1>" jobs, skipping those already finished.
# Several workers can run side by side with disjoint job lists (5-fold CV over physical configurations, 3 seeds each).
set -uo pipefail
cd /home/jude/isaac_ws/safety_probing/analysis_thesis
for JOB in "$@"; do
  F=${JOB%%:*}; CMD=${JOB##*:}
  if [ "$F" = 65 ]; then FEAT=artifacts/features/vjepa_2_1_full_observation_spatial12.pt; TGT=artifacts/targets/safety_query_full_comfix_med5.pt
  else FEAT=artifacts/features/thesis_v1_obs${F}_spatial12.pt; TGT=artifacts/targets/thesis_v1_obs${F}_med5.pt; fi
  if [ "$CMD" = 1 ]; then NAME=config5_full_cmd_obs$F; EXTRA=--with-command; else NAME=config5_full_obs$F; EXTRA=; fi
  if [ -f out/cv_$NAME.json ]; then echo "[early] skip $NAME"; continue; fi
  echo "[early] $(date +%T) start $NAME"
  PYTHONUNBUFFERED=1 ../.venv/bin/python psq_cv.py --protocol config5 --features $FEAT --targets $TGT --tag _obs$F $EXTRA 2>&1 | grep "CVRESULT\|CVBASE\|Traceback\|Error"
done
echo "[early] $(date +%T) worker done: $*"
