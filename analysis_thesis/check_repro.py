import json, numpy as np
from common import Context, bs, fmt
c = Context()
methods = bs.load_method_predictions(c.paths, c.ids)
ref = {}
for name in ("vjepa_prefix", "ltx_oracle", "ltx_generated"):
    al = bs.align_predictions(methods[name], name, c.ids, c.test_ids)
    r = c.evaluate(al, bootstrap=2000)
    ref[name] = r
    print(name, fmt(r))
json.dump(ref, open("out/repro_table1.json", "w"), indent=1)
