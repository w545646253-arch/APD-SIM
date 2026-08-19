from __future__ import annotations
import csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from _common import run, ROOT

def execute(cfg, output):
    source=(ROOT/Path(cfg["inputs"][0]["path"])).resolve()
    rows=list(csv.DictReader(source.open("r",encoding="utf-8-sig",newline="")))
    required={"sample_id","structure_class","method","psnr","ssim"}
    if len(rows)!=120 or not required.issubset(rows[0]): raise RuntimeError("expected complete 30-FOV x 4-method metrics table")
    groups=defaultdict(list)
    for row in rows:
        for metric in ("psnr","ssim"):
            groups[(row["structure_class"],row["method"],metric)].append(float(row[metric]))
    out=[]
    for (structure,method,metric),values in sorted(groups.items()):
        a=np.asarray(values,dtype=np.float64)
        out.append({"structure_class":structure,"method":method,"metric":metric,"n":a.size,"mean":a.mean(),"sd":a.std(ddof=1)})
    path=output/"class_specific_summary.csv"
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=out[0]); w.writeheader(); w.writerows(out)
    return {"rows":len(out),"output":str(path)}

if __name__ == '__main__': raise SystemExit(run('class_specific_statistics','configs/reproduction/class_specific_statistics.json',execute))
