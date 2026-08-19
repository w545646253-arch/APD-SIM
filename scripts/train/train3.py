"""Right-click entrypoint for formal APD-SIM DMD 3-frame 2-D training."""

from __future__ import annotations

import os
from pathlib import Path

from unisim.formal_training_2d import run_formal_training


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_ID = "DMD_3F_1O3P"
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "apd_dmd_r2"
    / "train3_formal_restart_simple_r1.json"
)


def main() -> None:
    env_preflight = os.environ.get("APD_DMD_PREFLIGHT_ONLY", "").strip() == "1"
    run_formal_training(
        protocol_id=PROTOCOL_ID,
        config_path=CONFIG_PATH,
        preflight_only=env_preflight,
    )


if __name__ == "__main__":
    main()
