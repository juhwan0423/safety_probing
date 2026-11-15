#!/usr/bin/env bash
# Final ForkSafe-CF v2 analysis, started once every dataset part has been generated:
#   full readout (video only / video + planned command / command only), no-video kinematics ceiling,
#   learning curve over training parts, observation-length sweep, hypothesis testing and what-if prediction.
# Usage: run_v2_final.sh <N parts> [wait-pattern]
set -uo pipefail
cd /home/jude/isaac_ws/safety_probing
PY=.venv/bin/python; export PYTHONUNBUFFERED=1
N=${1:-5}; WAIT=${2:-"ALL DONE"}
GEN=run_logs/forksafe_v2_generation.log
until grep -q "$WAIT" $GEN 2>/dev/null; do sleep 60; done
PART() { for i in $(seq 1 $1); do printf "forksafe_v2_part_%02d " $i; done; }
D=artifacts/v2/p$N; mkdir -p $D; M=$D/manifests/canonical_runs.jsonl
V2="PYTHONPATH=analysis_thesis/v2:src $PY analysis_thesis/v2"
echo "[v2-final] $(date +%T) parts=$N"
$PY analysis_thesis/v2/build_v2_manifest.py --parts $(PART $N) --out $D/manifests || { echo "manifest failed"; exit 1; }
PYTHONPATH=src $PY -m safety_probing.vjepa_spatial_features --manifest $M --shard full --output $D/vjepa_spatial12.pt --batch-size 4 2>&1 | tail -1
# targets from the true rigid-body CoM (truecom.py); the logger's CoM/ZMP columns are not used
COMFIX_MANIFEST=$M PYTHONPATH=src $PY analysis_thesis/truecom_build_targets.py --experiment configs/experiments/vjepa_safety_query.json \
    --manifest $M --shard full --output $D/targets_truecom.pt 2>&1 | tail -1
T=$D/targets_truecom.pt
A="--features $D/vjepa_spatial12.pt --targets $T --manifest-dir $D/manifests"
run() { eval "$V2/psq_v2.py $*" 2>&1 | grep "V2RESULT\|V2BASE\|Traceback\|Error"; }
echo "[v2-final] $(date +%T) main readout"
run --name p$N $A
run --name p${N}_cmd $A --with-command
run --name p${N}_cmdonly $A --variant command_only
run --name p${N}_spatialmean $A --variant spatial_mean
eval "$V2/kinematics_v2.py --name p$N --targets $T --manifest-dir $D/manifests" 2>&1 | grep "V2KIN\|Traceback\|Error"
echo "[v2-final] $(date +%T) pair-grouped 5-fold CV (all runs out-of-fold)"
cv() { eval "$V2/psq_v2_cv.py $*" 2>&1 | grep "V2CV\|Traceback\|Error"; }
cv --name p${N} $A
cv --name p${N}_cmd $A --with-command
cv --name p${N}_cmdonly $A --variant command_only
echo "[v2-final] $(date +%T) learning curve"
for K in $(seq 1 $((N - 1))); do
  run --name p${N}_n$K $A --train-parts $(PART $K)
  run --name p${N}_n${K}_cmd $A --train-parts $(PART $K) --with-command
done
echo "[v2-final] $(date +%T) hypothesis testing"
for OBS in kin latent; do
  eval "$V2/hypothesis_v2.py --name p${N}_$OBS --obs $OBS --targets $T --manifest-dir $D/manifests --features $D/vjepa_spatial12.pt" 2>&1 | grep "HYPV2\|latent code\|Error\|Traceback"
done
echo "[v2-final] $(date +%T) observation-length sweep"
TARGET_BUILDER=truecom bash analysis_thesis/v2/run_v2_obslen.sh $N 2>&1 | grep "V2RESULT\|V2BASE\|Traceback\|Error\|ALL DONE"
echo "[v2-final] $(date +%T) ALL DONE"
