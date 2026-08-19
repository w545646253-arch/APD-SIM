from pathlib import Path
import json, re
import numpy as np

ROOT=Path(__file__).resolve().parents[2]

def test_protocols_load_and_masks():
    from unisim.protocols import protocol_registry
    expected={"DMD_3F_1O3P":3,"DMD_6F_2O3P":6,"DMD_9F_3O3P":9}
    for pid,n in expected.items():
        p=protocol_registry.require(pid)
        assert p.frame_count==n and sum(p.validity_mask)==n and len(p.validity_mask)==15

def test_synthetic_asset_identity():
    z=np.load(ROOT/"examples/synthetic/smoke_input.npz",allow_pickle=False)
    assert z["gt"].shape==(64,64) and np.isfinite(z["gt"]).all()

def test_json_parses_and_no_local_absolute_paths():
    bad=re.compile(r"(?i)(?:(?<![A-Za-z])[A-Z]:[\\/]|/"+"home/|C:"+r"\\Users\\)")
    for p in ROOT.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".py",".json",".md",".txt",".toml",".yml",".yaml",".cff",".csv"}:
            text=p.read_text(encoding="utf-8-sig")
            assert not bad.search(text), p
            if p.suffix.lower()==".json": json.loads(text)

def test_no_large_git_files():
    assert all(p.stat().st_size <= 100*1024*1024 for p in ROOT.rglob("*") if p.is_file())
