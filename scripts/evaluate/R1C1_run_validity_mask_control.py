"""PyCharm right-click entry point for Reviewer 1 validity-mask control."""

from pathlib import Path

from unisim.revision_r1.validity_mask_control import main


if __name__ == "__main__":
    raise SystemExit(main(Path(__file__).resolve()))
