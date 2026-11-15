"""Build a Safety Query target cache from corrected dense rows (new file; originals untouched).
Usage (from the safety_probing root): python analysis_thesis/comfix_build_targets.py <builder args>
COMFIX_MED5=1 additionally de-spikes the finite-difference ZMP with a 5-sample running median."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "src"))
from safety_probing import safety_query_targets as sqt
import comfix
MED5 = os.environ.get("COMFIX_MED5") == "1"
sqt._read_dense_rows = lambda path: comfix.corrected_rows(path, med5=MED5)
print(f"[comfix] corrected dense rows, med5={MED5}", flush=True)
sqt.main()
