#!/usr/bin/env bash
set -uo pipefail
cd /home/jude/isaac_ws/safety_probing/analysis_thesis
until grep -q "ALL DONE" ../run_logs/heldout_pipeline.log 2>/dev/null; do sleep 30; done
export PYTHONUNBUFFERED=1
../.venv/bin/python psq_cv.py --protocol config5 || echo "FAILED config5"
../.venv/bin/python psq_cv.py --protocol logo9 || echo "FAILED logo9"
echo "[cv] ALL DONE"
