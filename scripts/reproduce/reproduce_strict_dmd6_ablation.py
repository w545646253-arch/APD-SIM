from pathlib import Path
from _common import run
def execute(cfg, output):
    from unisim.revision_r1 import physmap6_experiment as m
    m.OUTPUT_BASE=output.parent
    return m.run(preflight_only=bool(cfg.get("parameters",{}).get("preflight_only",False)))

if __name__ == '__main__': raise SystemExit(run('strict_dmd6_ablation', 'configs/reproduction/strict_dmd6_ablation.json', execute))
