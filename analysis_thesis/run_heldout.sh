#!/usr/bin/env bash
set -uo pipefail
cd /home/jude/isaac_ws/safety_probing
PY=.venv/bin/python
run() { $PY analysis_thesis/psq_heldout.py "$@" || echo "FAILED $*"; }
(cd analysis_thesis && true)
export PYTHONPATH=analysis_thesis:src PYTHONUNBUFFERED=1
run --name mass480 --hold-key load_mass_kg --hold-value 480
run --name mass360 --hold-key load_mass_kg --hold-value 360
run --name mass600 --hold-key load_mass_kg --hold-value 600
run --name com0 --hold-key com --hold-value 0.0
run --name speed_medium --hold-key speed_label --hold-value medium
run --name left_to_right --alt-split left_to_right
run --name right_to_left --alt-split right_to_left
run --name speed_fast --hold-key speed_label --hold-value fast
echo "[heldout] ALL DONE"
