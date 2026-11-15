#!/usr/bin/env bash
# Early prediction with all runs as test data: pair-grouped 5-fold CV for each observation length (1.0 / 1.5 / 1.96 / 2.33 s),
# video only and video + planned command. The 2.71 s runs already exist as v2cv_p<N>{,_cmd}.json.
# Usage: run_v2_obslen_cv.sh <N parts>   (needs the truncated feature caches written by run_v2_obslen.sh)
set -uo pipefail
cd /home/jude/isaac_ws/safety_probing
PY=.venv/bin/python; export PYTHONUNBUFFERED=1
N=$1; D=artifacts/v2/p$N
for F in 47 36 56 24; do
  FEAT=$D/vjepa_spatial12_obs$F.pt; TGT=$D/targets_truecom_obs$F.pt
  PYTHONPATH=analysis_thesis/v2:src $PY analysis_thesis/v2/psq_v2_cv.py --name p${N}_obs$F --features $FEAT --targets $TGT --manifest-dir $D/manifests 2>&1 | grep "V2CV"
  PYTHONPATH=analysis_thesis/v2:src $PY analysis_thesis/v2/psq_v2_cv.py --name p${N}_obs${F}_cmd --features $FEAT --targets $TGT --manifest-dir $D/manifests --with-command 2>&1 | grep "V2CV"
done
echo "[obslen-cv] ALL DONE"
