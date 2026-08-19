"""Compatibility lifecycle helpers for formal APD-DMD R2 training.

The production entrypoints use :mod:`unisim.formal_training_2d` directly.
This module retains the atomic process lock and lazily delegates its historical
``formal_preflight``/``run_formal_training`` API to the strict two-dimensional
engine.  It therefore cannot return the retired configuration/data-recovery
blockers from the earlier audit phase.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Union


FORMAL_OUTPUT_NAME = "APD_DMD_GEOMETRY_TRAINING_R2"
BLOCK_STATUS = "DEPRECATED_GATE_DELEGATES_TO_FORMAL_2D"


class FormalTrainingBlocked(RuntimeError):
    """Raised by a fail-closed lifecycle condition such as an active lock."""

    def __init__(self, status: str, message: str):
        super().__init__(f"{status}: {message}")
        self.status = status
        self.detail = message


def sha256_file(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
            process_query_limited_information, False, int(pid)
        )
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
            return True
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@dataclass
class TrainingLock:
    """Atomic, non-destructive singleton lock for one formal GPU run."""

    formal_output_root: Path
    script_name: str
    protocol_id: str
    gpu: str
    config_hash: str
    acquired: bool = False

    @property
    def path(self) -> Path:
        return self.formal_output_root / "training.lock"

    def acquire(self) -> None:
        if self.acquired:
            return
        self.formal_output_root.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                current = json.loads(self.path.read_text(encoding="utf-8"))
                current_pid = int(current["pid"])
            except Exception as exc:
                raise FormalTrainingBlocked(
                    "TRAINING_GPU_LOCK_ACTIVE",
                    f"Unreadable lock must be inspected manually: {self.path}: {exc}",
                ) from exc
            if _pid_is_running(current_pid):
                raise FormalTrainingBlocked(
                    "TRAINING_GPU_LOCK_ACTIVE",
                    f"PID {current_pid} owns {self.path}; the existing process was not terminated",
                )
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            stale = self.formal_output_root / f"training.lock.stale.{stamp}.{current_pid}.json"
            os.replace(self.path, stale)

        payload = {
            "pid": os.getpid(),
            "script_name": self.script_name,
            "protocol_id": self.protocol_id,
            "gpu": self.gpu,
            "start_time_utc": datetime.now(timezone.utc).isoformat(),
            "config_hash": self.config_hash,
        }
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        fd = os.open(self.path, flags)
        try:
            os.write(fd, (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"))
        finally:
            os.close(fd)
        self.acquired = True

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if int(payload.get("pid", -1)) == os.getpid():
                self.path.unlink()
        finally:
            self.acquired = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()


def formal_preflight(
    protocol_id: str,
    config_path: Union[str, Path],
    *,
    project_root: Optional[Union[str, Path]] = None,
) -> Dict[str, object]:
    """Delegate the legacy API name to the active strict-2D preflight."""
    del project_root
    from .formal_training_2d import formal_preflight_2d

    return formal_preflight_2d(protocol_id, config_path)


def print_preflight(result: Dict[str, object]) -> None:
    from .formal_training_2d import print_preflight_2d

    print_preflight_2d(result)


def run_formal_training(
    protocol_id: str,
    config_path: Union[str, Path],
    *,
    project_root: Optional[Union[str, Path]] = None,
    preflight_only: bool = False,
) -> Dict[str, object]:
    """Delegate the legacy API name to the active strict-2D engine."""
    del project_root
    from .formal_training_2d import run_formal_training as run_formal_training_2d

    return run_formal_training_2d(
        protocol_id=protocol_id,
        config_path=config_path,
        preflight_only=preflight_only,
    )


__all__ = [
    "BLOCK_STATUS",
    "FORMAL_OUTPUT_NAME",
    "FormalTrainingBlocked",
    "TrainingLock",
    "formal_preflight",
    "print_preflight",
    "run_formal_training",
    "sha256_file",
]
