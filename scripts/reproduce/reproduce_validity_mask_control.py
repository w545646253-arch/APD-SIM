from pathlib import Path
from _common import run
def execute(cfg, output):
    from unisim.revision_r1 import validity_mask_control as m
    m.OUTPUT_ROOT=output.parent; m.allocate_run_directory=lambda: output
    return {"exit_code":m.main(Path(__file__).resolve())}

if __name__ == '__main__': raise SystemExit(run('validity_mask_control', 'configs/reproduction/validity_mask_control.json', execute))
