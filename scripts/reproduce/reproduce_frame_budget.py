from pathlib import Path
from _common import run
def execute(cfg, output):
    from unisim.revision_r1 import frame_budget_r1c2 as m
    m.OUTPUT_ROOT=output.parent; m.allocate_run_dir=lambda: output
    return {"exit_code":m.main(Path(__file__).resolve())}

if __name__ == '__main__': raise SystemExit(run('frame_budget', 'configs/reproduction/frame_budget.json', execute))
