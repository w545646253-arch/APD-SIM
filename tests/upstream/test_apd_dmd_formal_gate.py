from __future__ import annotations

import json
from pathlib import Path

import pytest

import unisim.formal_training_gate as gate
from unisim.formal_training_2d import formal_preflight_2d


def test_formal_preflight_reads_identity_receipt_but_no_sealed_tiff(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    opened = []
    original_open = Path.open
    original_read_text = Path.read_text

    def guarded_open(self, *args, **kwargs):
        opened.append(str(self.resolve()))
        assert self.suffix.lower() not in (".tif", ".tiff")
        return original_open(self, *args, **kwargs)

    def guarded_read_text(self, *args, **kwargs):
        opened.append(str(self.resolve()))
        assert self.suffix.lower() not in (".tif", ".tiff")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    result = formal_preflight_2d(
        "DMD_6F_2O3P",
        root / "configs" / "apd_dmd_r2" / "train6_formal.json",
    )
    assert result["status"] == "FORMAL_2D_PREFLIGHT_PASS"
    assert result["protocol_id"] == "DMD_6F_2O3P"
    assert result["test_files_not_accessible_to_training_runtime"] is True
    assert result["sealed_test_runtime_tiff_accesses"] == 0
    assert any("sealed_test_manifest.json" in path for path in opened)
    assert all(Path(path).suffix.lower() not in (".tif", ".tiff") for path in opened)


def test_live_training_lock_blocks_second_owner_without_terminating(tmp_path):
    lock1 = gate.TrainingLock(tmp_path, "a.py", "DMD_9F_3O3P", "cuda:0", "a" * 64)
    lock2 = gate.TrainingLock(tmp_path, "b.py", "DMD_6F_2O3P", "cuda:0", "b" * 64)
    lock1.acquire()
    try:
        with pytest.raises(gate.FormalTrainingBlocked) as error:
            lock2.acquire()
        assert error.value.status == "TRAINING_GPU_LOCK_ACTIVE"
        payload = json.loads((tmp_path / "training.lock").read_text(encoding="utf-8"))
        assert payload["script_name"] == "a.py"
    finally:
        lock1.release()


def test_stale_lock_is_archived(tmp_path, monkeypatch):
    stale = tmp_path / "training.lock"
    stale.write_text(
        json.dumps(
            {
                "pid": 99999999,
                "script_name": "old.py",
                "protocol_id": "DMD_9F_3O3P",
                "gpu": "cuda:0",
                "start_time_utc": "2000-01-01T00:00:00Z",
                "config_hash": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(gate, "_pid_is_running", lambda pid: False)
    lock = gate.TrainingLock(tmp_path, "new.py", "DMD_9F_3O3P", "cuda:0", "a" * 64)
    lock.acquire()
    try:
        archived = list(tmp_path.glob("training.lock.stale.*.json"))
        assert len(archived) == 1
    finally:
        lock.release()


@pytest.mark.parametrize(
    "script_name,protocol_id",
    [
        ("train9.py", "DMD_9F_3O3P"),
        ("train6.py", "DMD_6F_2O3P"),
        ("train3.py", "DMD_3F_1O3P"),
    ],
)
def test_right_click_scripts_have_fixed_protocol_binding(script_name, protocol_id):
    root = Path(__file__).resolve().parents[1]
    text = (root / script_name).read_text(encoding="utf-8")
    assert "Path(__file__).resolve()" in text
    assert f'PROTOCOL_ID = "{protocol_id}"' in text
    assert "run_formal_training" in text
    assert "argparse" not in text
