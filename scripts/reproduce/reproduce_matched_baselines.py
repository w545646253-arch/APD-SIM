from pathlib import Path
from _common import run
def execute(cfg, output):
    import tools.run_revision_matched_dmd6 as m
    m.OUTPUT=output
    return {"exit_code":m.main()}

if __name__ == '__main__': raise SystemExit(run('matched_baselines', 'configs/reproduction/matched_baselines.json', execute))
