"""Build a Safety Query target cache whose system CoM, ZMP and outward margins come from the true rigid-body CoM
(truecom.py) instead of the logger's mis-paired CoM.
Usage (from the safety_probing root): python analysis_thesis/truecom_build_targets.py <builder args>
Environment: COMFIX_MANIFEST=<canonical_runs.jsonl> for datasets other than v1 (run -> payload lookup)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "src"))
from safety_probing import safety_query_targets as sqt
import truecom
sqt._read_dense_rows = lambda path: truecom.corrected_rows(path, med5=True)
print("[truecom] dense rows with rigid-body CoM / ZMP / margins (5-sample median on ZMP)", flush=True)
sqt.main()
