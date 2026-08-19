from __future__ import annotations
import argparse, hashlib, json, os, shutil, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def _sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024), b""): h.update(b)
    return h.hexdigest()

def run(task: str, default_config: str, execute):
    p=argparse.ArgumentParser(description=f"Fail-closed APD-SIM reproduction: {task}")
    p.add_argument("--config", type=Path, default=ROOT/default_config)
    p.add_argument("--output", type=Path, default=ROOT/"reproduction_outputs"/task)
    p.add_argument("--receipt", type=Path, default=ROOT/"reproduction_receipts"/f"{task}.json")
    p.add_argument("--execute", action="store_true", help="Run after validation; default only validates")
    p.add_argument("--overwrite", action="store_true")
    args=p.parse_args(); started=datetime.now(timezone.utc).isoformat()
    receipt={"schema_version":1,"task":task,"started_utc":started,"training_executed":False,"test_set_tuning":False}
    try:
        cfg=json.loads(args.config.resolve().read_text(encoding="utf-8"))
        if cfg.get("task") != task or cfg.get("test_set_tuning") is not False: raise RuntimeError("task/no-tuning contract mismatch")
        resolved=[]
        for item in cfg.get("inputs",[]):
            raw=Path(item["path"]); path=(ROOT/raw).resolve() if not raw.is_absolute() else raw.resolve()
            row={"role":item["role"],"path":str(path),"required":bool(item.get("required",True)),"exists":path.is_file() or path.is_dir(),"expected_sha256":item.get("sha256")}
            if row["required"] and not row["exists"]: raise FileNotFoundError(f"missing required input: {item['role']} -> {path}")
            if path.is_file() and item.get("sha256"):
                row["actual_sha256"]=_sha(path)
                if row["actual_sha256"].lower()!=str(item["sha256"]).lower(): raise RuntimeError(f"hash mismatch: {item['role']}")
            if path.is_file() and item.get("size_bytes") is not None and path.stat().st_size!=int(item["size_bytes"]): raise RuntimeError(f"size mismatch: {item['role']}")
            resolved.append(row)
        receipt["resolved_configuration"]={"config":str(args.config.resolve()),"output":str(args.output.resolve()),"inputs":resolved,"parameters":cfg.get("parameters",{})}
        print(json.dumps(receipt["resolved_configuration"], indent=2, sort_keys=True))
        if args.output.exists() and any(args.output.iterdir()) and not args.overwrite: raise FileExistsError(f"output exists; pass --overwrite: {args.output}")
        if args.overwrite and args.output.exists(): shutil.rmtree(args.output)
        if args.execute:
            args.output.mkdir(parents=True, exist_ok=False)
            result=execute(cfg, args.output)
            receipt.update({"status":"COMPLETED","execution_result":result})
        else: receipt["status"]="CONFIGURATION_VALIDATED_NO_EXECUTION"
        code=0
    except Exception as exc:
        receipt.update({"status":"BLOCKED","error_type":type(exc).__name__,"error":str(exc)}); code=2
    receipt["completed_utc"]=datetime.now(timezone.utc).isoformat(); args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True)); return code
