#!/usr/bin/env bash
# Generated-future audit on a v2 set. Needs the GPU to itself (LTX-2.3 takes about 25 GB): run only when no
# simulator or training job is active.
# Usage: run_v2_ltx.sh <set dir under artifacts/v2, e.g. p5> [rollout seeds, default "17 29"]
set -uo pipefail
cd /home/jude/isaac_ws/safety_probing
SET=${1:-p5}; SEEDS=${2:-"17 29"}
D=artifacts/v2/$SET; M=$D/manifests/canonical_runs.jsonl
LPY=external/LTX-2/.venv/bin/python; PY=.venv/bin/python; export PYTHONUNBUFFERED=1
ROLL=outputs/ltx_v2data/${SET}_test
echo "[v2-ltx] $(date +%T) real-future latents ($SET)"
[ -f $D/ltx_real_future.pt ] || PYTHONPATH=src $LPY -m safety_probing.ltx_real_latents --manifest $M --shard full --output $D/ltx_real_future.pt --work-root $D/ltx_real_work 2>&1 | tail -2
echo "[v2-ltx] $(date +%T) rollouts for the test split, seeds: $SEEDS"
SEEDARGS=$(for s in $SEEDS; do printf -- "--seed %s " $s; done)
PYTHONPATH=src $LPY -m safety_probing.ltx_batch --experiment configs/experiments/ltx_2_3_continuation_prompt_v1.json --manifest $M --shard full --split test $SEEDARGS --output-root $ROLL 2>&1 | tail -3
echo "[v2-ltx] $(date +%T) rollout cache"
PYTHONPATH=src $LPY -m safety_probing.ltx_rollout_cache --real-cache $D/ltx_real_future.pt --rollout-root $ROLL --output $D/ltx_rollout_future.pt 2>&1 | tail -2
echo "[v2-ltx] $(date +%T) targets and readout"
[ -f $D/ltx_targets_truecom.pt ] || COMFIX_MANIFEST=$M PYTHONPATH=src $PY analysis_thesis/truecom_build_targets.py --experiment configs/experiments/ltx_safety_query.json --manifest $M --shard full --output $D/ltx_targets_truecom.pt 2>&1 | tail -1
PYTHONPATH=analysis_thesis/v2:src $PY analysis_thesis/v2/ltx_v2.py --name $SET --real-cache $D/ltx_real_future.pt --rollout-cache $D/ltx_rollout_future.pt --targets $D/ltx_targets_truecom.pt --manifest-dir $D/manifests 2>&1 | grep "V2LTX\|Traceback\|Error\|rollouts"
echo "[v2-ltx] $(date +%T) done"
