#!/usr/bin/env bash
# Early prediction on v1: how long before the event can tip-over be predicted, with and without the planned command?
# Protocol: 5-fold CV over physical configurations (seed replicas kept together), 3 seeds.
set -uo pipefail
cd /home/jude/isaac_ws/safety_probing
PY=.venv/bin/python; export PYTHONUNBUFFERED=1
for F in 47 24 36 56; do
  EXP=configs/experiments/thesis_vjepa_obs$F.json
  [ -f artifacts/features/thesis_v1_obs${F}_spatial12.pt ] || PYTHONPATH=src $PY -m safety_probing.vjepa_spatial_features --experiment $EXP --shard full --output artifacts/features/thesis_v1_obs${F}_spatial12.pt --batch-size 4 2>&1 | tail -1
  [ -f artifacts/targets/thesis_v1_obs${F}_med5.pt ] || COMFIX_MED5=1 PYTHONPATH=src $PY analysis_thesis/comfix_build_targets.py --experiment $EXP --shard full --output artifacts/targets/thesis_v1_obs${F}_med5.pt 2>&1 | tail -1
done
cd analysis_thesis
run() { ../.venv/bin/python psq_cv.py --protocol config5 "$@" 2>&1 | grep "CVRESULT\|CVBASE\|Traceback\|Error"; }
run --variant command_only --tag _obs0
for F in 47 65 24 36 56; do
  if [ $F = 65 ]; then FEAT=artifacts/features/vjepa_2_1_full_observation_spatial12.pt; TGT=artifacts/targets/safety_query_full_comfix_med5.pt
  else FEAT=artifacts/features/thesis_v1_obs${F}_spatial12.pt; TGT=artifacts/targets/thesis_v1_obs${F}_med5.pt; fi
  run --features $FEAT --targets $TGT --tag _obs$F --with-command
  run --features $FEAT --targets $TGT --tag _obs$F
done
echo "[early] ALL DONE"
