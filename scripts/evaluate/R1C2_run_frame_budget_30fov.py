"""PyCharm right-click entry for the formal Reviewer 1 Comment 2 experiment.

No command-line arguments are required.  Set APD_R1C2_PREFLIGHT_ONLY=1 in the
PyCharm run environment to execute only the CPU/read-only preflight.
"""

from pathlib import Path

from unisim.revision_r1.frame_budget_r1c2 import main as _run


def main() -> int:
    return _run(Path(__file__).resolve())


if __name__ == "__main__":
    raise SystemExit(main())
