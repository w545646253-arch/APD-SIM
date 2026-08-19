from pathlib import Path
from _common import run
def execute(cfg, output):
    import tools.run_revision_seed_sensitivity as m
    m.OUTPUT=output
    return {"exit_code":m.main()}

if __name__ == '__main__': raise SystemExit(run('seed_sensitivity', 'configs/reproduction/seed_sensitivity.json', execute))
