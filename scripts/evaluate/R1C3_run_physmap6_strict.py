"""PyCharm right-click entry for Reviewer #1 strict PhysMap-6 ablation.

Default execution performs strict preflight followed by the finite smoke and
formal pipeline.  Set ``APD_R1C3_PREFLIGHT_ONLY=1`` to stop after read-only
identity checks and receipts.  No command-line arguments are required.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unisim.revision_r1.physmap6_experiment import run


def main() -> None:
    preflight_only = os.environ.get("APD_R1C3_PREFLIGHT_ONLY", "0") == "1"
    result = run(preflight_only=preflight_only)
    if result["status"] not in {"R1C3_PREFLIGHT_PASS", "R1C3_PHYSMAP6_STRICT_READY"}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
