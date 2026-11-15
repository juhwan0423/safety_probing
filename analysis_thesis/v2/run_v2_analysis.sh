#!/usr/bin/env bash
# As dataset parts finish, rebuild the v2 manifest/features/targets and train the video readout + the no-video ceiling.
set -uo pipefail
cd /home/jude/isaac_ws/safety_probing
PY=.venv/bin/python; export PYTHONUNBUFFERED=1
GEN=run_logs/forksafe_v2_generation.log
for N in 1 2 3 5; do
  until grep -q "done forksafe_v2_part_0$N" $GEN 2>/dev/null; do sleep 60; done
  PARTS=$(for i in $(seq 1 $N); do echo -n "forksafe_v2_part_0$i "; done)
  D=artifacts/v2/p$N; mkdir -p $D
  echo "[v2-analysis] $(date +%T) parts=$N"
  $PY analysis_thesis/v2/build_v2_manifest.py --parts $PARTS --out $D/manifests || { echo "manifest failed"; continue; }
  PYTHONPATH=src $PY -m safety_probing.vjepa_spatial_features --manifest $D/manifests/canonical_runs.jsonl --shard full --output $D/vjepa_spatial12.pt --batch-size 4 2>&1 | tail -1
  COMFIX_MANIFEST=$D/manifests/canonical_runs.jsonl COMFIX_MED5=1 PYTHONPATH=src $PY analysis_thesis/comfix_build_targets.py --experiment configs/experiments/vjepa_safety_query.json \
      --manifest $D/manifests/canonical_runs.jsonl --shard full --output $D/targets_med5.pt 2>&1 | tail -1
  PYTHONPATH=analysis_thesis/v2:src $PY analysis_thesis/v2/psq_v2.py --name p$N --features $D/vjepa_spatial12.pt --targets $D/targets_med5.pt --manifest-dir $D/manifests 2>&1 | grep "V2RESULT\|V2BASE\|^p$N {" 
  PYTHONPATH=analysis_thesis/v2:src $PY analysis_thesis/v2/kinematics_v2.py --name p$N --targets $D/targets_med5.pt --manifest-dir $D/manifests 2>&1 | grep "V2KIN"
done
echo "[v2-analysis] ALL DONE"
