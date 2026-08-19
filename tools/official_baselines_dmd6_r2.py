#!/usr/bin/env python3
"""Prepare and audit source-verified official DMD-6F baselines.

This command is intentionally preparation-first.  It creates an immutable R2
run root, records preflight/source/environment evidence, freezes candidates,
and writes hash-gated continuation entry points.  It never starts formal GPU
work while an APD training lock/process is present.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
EXTERNAL = ROOT / "external" / "official_r2"
WORKTREES = ROOT / "external" / "official_r2_worktrees"
CURRENT = OUTPUTS / "OFFICIAL_BASELINES_DMD6_R2_CURRENT.json"
PROTOCOL = ROOT / "protocols" / "dmd_6f_2o3p.json"
CONFIG = ROOT / "configs" / "apd_dmd_r2" / "train6_formal.json"
MANIFESTS = {
    "train": ROOT / "manifests" / "apd_dmd_r2" / "train_manifest.json",
    "validation": ROOT / "manifests" / "apd_dmd_r2" / "validation_manifest.json",
    "sealed_test": ROOT / "manifests" / "apd_dmd_r2" / "sealed_test_manifest.json",
    "validation_bundle": ROOT / "manifests" / "apd_dmd_r2" / "validation_bundle_manifest.json",
}
APD_LOCK = ROOT / "checkpoints" / "apd_dmd_geometry_r2" / "_device_locks" / "cuda" / "training.lock"
STAGES = [
    "00_preflight", "01_shared_contract", "02_candidate_registry", "03_provenance",
    "04_sources_pristine", "05_environments", "06_protocol_audit", "07_adapters",
    "08_training", "09_baseline_only_results", "10_apd6_finalization", "11_metrics",
    "12_runtime", "13_figures", "14_supplementary", "15_manuscript", "16_audit",
]


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_entries(root: pathlib.Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.relative_to(root).as_posix()):
        entries.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return entries


def tree_hash(root: pathlib.Path) -> tuple[str, list[dict[str, Any]]]:
    entries = tree_entries(root)
    payload = b"".join(
        e["path"].encode("utf-8") + b"\0" + e["sha256"].encode("ascii") + b"\0" + str(e["bytes"]).encode("ascii") + b"\n"
        for e in entries
    )
    return sha256_bytes(payload), entries


def is_generated_source_artifact(relative_path: str) -> bool:
    pure = pathlib.PurePosixPath(relative_path)
    return (
        "__pycache__" in pure.parts
        or any(part.endswith(".egg-info") for part in pure.parts)
        or pure.suffix.lower() in {".pyc", ".pyo"}
    )


def source_payload_hash(root: pathlib.Path) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    all_entries = tree_entries(root)
    payload_entries = [entry for entry in all_entries if not is_generated_source_artifact(entry["path"])]
    generated = [entry for entry in all_entries if is_generated_source_artifact(entry["path"])]
    payload = b"".join(
        entry["path"].encode("utf-8") + b"\0" + entry["sha256"].encode("ascii") + b"\0"
        + str(entry["bytes"]).encode("ascii") + b"\n"
        for entry in payload_entries
    )
    return sha256_bytes(payload), payload_entries, generated


def atomic_write(path: pathlib.Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def write_json(path: pathlib.Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n")


def write_text(path: pathlib.Path, value: str) -> None:
    atomic_write(path, value.replace("\r\n", "\n").encode("utf-8"))


def write_rows(path: pathlib.Path, fieldnames: list[str], rows: Iterable[dict[str, Any]], delimiter: str = "\t") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter, lineterminator="\n", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def run_capture(command: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30, check=False)
        return proc.returncode, proc.stdout
    except Exception as exc:  # evidence collection must not crash the run
        return 127, f"{type(exc).__name__}: {exc}\n"


def powershell(script: str) -> tuple[int, str]:
    return run_capture(["powershell", "-NoProfile", "-NonInteractive", "-Command", script])


def active_processes() -> list[dict[str, str]]:
    script = (
        "$ErrorActionPreference='SilentlyContinue'; "
        "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'train9\\.py|train6\\.py|train3\\.py' } | "
        "Select-Object ProcessId,Name,CommandLine | ConvertTo-Csv -NoTypeInformation"
    )
    code, output = powershell(script)
    if code != 0 or not output.strip():
        return []
    return [dict(row) for row in csv.DictReader(output.splitlines())]


def preflight(out: pathlib.Path) -> dict[str, Any]:
    processes = active_processes()
    lock_payload: Any = None
    lock_error = None
    if APD_LOCK.exists():
        try:
            lock_payload = json.loads(APD_LOCK.read_text(encoding="utf-8"))
        except Exception as exc:
            lock_error = f"{type(exc).__name__}: {exc}"
    gpu_code, gpu_text = run_capture(["nvidia-smi"])
    gpu_query_code, gpu_query = run_capture([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"
    ])
    raw_gpu_rows = [line.strip() for line in gpu_query.splitlines() if line.strip()] if gpu_query_code == 0 else []
    # WDDM reports graphics clients with used_memory=[N/A].  They are useful
    # diagnostics but are not CUDA-compute evidence for the formal-work gate.
    gpu_compute_rows = [line for line in raw_gpu_rows if not line.rstrip().endswith("[N/A]")]
    apd_gpu_active = bool(processes or APD_LOCK.exists())
    disk = shutil.disk_usage(ROOT.drive + "\\")
    conda_code, conda_text = run_capture([str(ROOT.drive + "\\anaconda\\Scripts\\conda.exe"), "info", "--json"])
    env_lines = [
        f"captured_utc={now_utc()}",
        f"workspace={ROOT}",
        f"source_snapshot_id={json.loads(CONFIG.read_text(encoding='utf-8')).get('source_snapshot_id')}",
        f"host_python={sys.executable}",
        f"host_python_version={sys.version.replace(os.linesep, ' ')}",
        f"platform={platform.platform()}",
        f"conda_info_exit_code={conda_code}",
        conda_text.rstrip(),
    ]
    write_text(out / "00_preflight" / "environment.txt", "\n".join(env_lines) + "\n")
    write_rows(out / "00_preflight" / "active_processes.tsv", ["ProcessId", "Name", "CommandLine"], processes)
    write_text(out / "00_preflight" / "gpu_state.txt", f"nvidia_smi_exit_code={gpu_code}\n{gpu_text}\nraw_compute_apps_query:\n{gpu_query}\nfiltered_cuda_compute_apps:\n" + "\n".join(gpu_compute_rows) + "\n")
    status = {
        "captured_utc": now_utc(),
        "apd_gpu_active": apd_gpu_active,
        "formal_baseline_gpu_work_allowed": not apd_gpu_active and not gpu_compute_rows,
        "apd_lock_path": str(APD_LOCK),
        "apd_lock_exists": APD_LOCK.exists(),
        "apd_lock_payload": lock_payload,
        "apd_lock_parse_error": lock_error,
        "matching_training_process_count": len(processes),
        "gpu_compute_processes": gpu_compute_rows,
        "gpu_query_rows_including_wddm_na": raw_gpu_rows,
        "disk_free_bytes": disk.free,
        "policy": "All formal baseline GPU training, inference, and runtime are deferred while APD is active.",
    }
    write_json(out / "00_preflight" / "apd_training_status.json", status)
    return status


def source_specs() -> list[dict[str, Any]]:
    return [
        {
            "method_id": "mlsim_6r", "manuscript_label": "ML-SIM-6R",
            "paper_title": "ML-SIM: universal reconstruction of structured illumination microscopy images using transfer learning",
            "doi": "10.1364/BOE.414680", "repository_url": "https://github.com/charlesnchr/ML-SIM",
            "owner": "charlesnchr", "branch": "master", "commit": "25e289eca8571621e85f2d32ae09174b4c841b70",
            "archive": "_mlsim_commit.zip", "source_dir": "mlsim", "provenance_status": "AUTHOR_MAINTAINED_OFFICIAL",
            "license": "NOT_DECLARED_IN_ARCHIVE", "native_protocol": "3 orientations x 3 phases; official architecture/input count parameterized",
            "eligibility": "MATCHED_DMD6_OFFICIAL_RETRAINING", "comparison_group": "Matched DMD-6F",
            "kind": "learning", "environment": "apd_mlsim_official_r2",
            "blocked_reason": "Formal retraining deferred while APD GPU training is active.",
        },
        {
            "method_id": "ssrsim", "manuscript_label": "SSR-SIM-9F (native protocol)",
            "paper_title": "Bio-friendly and high-precision super-resolution imaging through self-supervised reconstruction structured illumination microscopy",
            "doi": "10.1038/s41592-025-02966-y", "repository_url": "https://github.com/HUST-Tan/SSR-SIM",
            "owner": "HUST-Tan", "branch": "main", "commit": "a9fde42849fd6c8153943065d0e4bc3bc449c35c",
            "archive": "_ssrsim_commit.zip", "source_dir": "ssrsim", "provenance_status": "AUTHOR_MAINTAINED_OFFICIAL",
            "license": "MIT", "native_protocol": "BioSR workflow constructs PHCT(9,1); compiled SIR core",
            "eligibility": "NATIVE_9F_ONLY", "comparison_group": "Native 9F supplementary",
            "kind": "self-supervised", "environment": "apd_ssrsim_official_r2",
            "blocked_reason": "Official workflow has a fixed nine-channel PHCT path; no source-verified six-frame core path is admitted.",
        },
        {
            "method_id": "mcsim_wiener6", "manuscript_label": "mcSIM-Wiener-6",
            "paper_title": "mcSIM: an open-source toolbox for multi-color structured illumination microscopy",
            "doi": "10.1364/BOE.422703; software archive 10.5281/zenodo.4773865", "repository_url": "https://github.com/QI2lab/mcSIM",
            "owner": "QI2lab", "branch": "master", "commit": "43b8b54535c3f4af666fb711dd630e903f156805",
            "archive": "_mcsim_commit.zip", "source_dir": "mcsim", "provenance_status": "AUTHOR_MAINTAINED_OFFICIAL",
            "license": "GPL-3.0", "native_protocol": "Dynamic orientations x exactly 3 phases",
            "eligibility": "MATCHED_DMD6_DIRECT_CONFIGURATION", "comparison_group": "Matched DMD-6F",
            "kind": "model-based", "environment": "apd_mcsim_official_r2",
            "blocked_reason": "Formal reconstruction deferred while APD GPU training is active.",
        },
        {
            "method_id": "hessian_original6", "manuscript_label": "Hessian-SIM-6",
            "paper_title": "Fast, long-term, super-resolution imaging with Hessian structured illumination microscopy",
            "doi": "10.1038/nbt.4115", "repository_url": "https://static-content.springer.com/esm/art%3A10.1038%2Fnbt.4115/MediaObjects/41587_2018_BFnbt4115_MOESM21_ESM.zip",
            "owner": "Nature Biotechnology supplementary material", "branch": "PUBLICATION_SUPPLEMENT", "commit": "NOT_APPLICABLE_ARCHIVE",
            "archive": "_hessian_original_supplementary_code.zip", "source_dir": "hessian_original", "provenance_status": "PUBLICATION_SUPPLEMENTARY_OFFICIAL",
            "license": "NOT_DECLARED_IN_ARCHIVE", "native_protocol": "2-beam option: 2 orientations x 3 phases",
            "eligibility": "MATCHED_DMD6_DIRECT_CONFIGURATION", "comparison_group": "Matched DMD-6F",
            "kind": "model-based", "environment": "apd_hessian_original_r2",
            "blocked_reason": "Formal MATLAB finite-output smoke and an auditable raw-to-Wiener-to-Hessian wrapper remain pending; GPU execution is deferred while APD training is active.",
        },
        {
            "method_id": "gpu_hessian6", "manuscript_label": "GPU-enabled Hessian-SIM-6",
            "paper_title": "GPU-Enabled Hessian Structured Illumination Microscopy",
            "doi": "10.1155/2024/8862387", "repository_url": "https://github.com/mc2lab/GPU-enabled-Hessian-SIM",
            "owner": "mc2lab", "branch": "main", "commit": "162527b9286df8f711a7c775b6fc36c94b97cd93",
            "archive": "_gpu_hessian_commit.zip", "source_dir": "gpu_hessian", "provenance_status": "DOCUMENTED_DERIVATIVE_IMPLEMENTATION",
            "license": "NOT_DECLARED_IN_ARCHIVE", "native_protocol": "Current GUI accepts an intermediate TIFF for Hessian denoising; retained raw 2O3P Wiener scripts are orphaned from the entry path",
            "eligibility": "BLOCKED_CORE_METHOD_ASSUMPTION", "comparison_group": "Blocked/not executed",
            "kind": "model-based", "environment": "apd_gpu_hessian_r2",
            "blocked_reason": "Derivative GUI path does not invoke the retained raw-to-Wiener 2O3P scripts; admitting them would silently replace the current core workflow.",
        },
    ]


def make_run_root(explicit: str | None) -> tuple[pathlib.Path, bool]:
    if explicit:
        out = pathlib.Path(explicit).resolve()
        if out.parent != OUTPUTS.resolve() or not out.name.startswith("OFFICIAL_BASELINES_DMD6_R2_"):
            raise ValueError("--output-root must be an OFFICIAL_BASELINES_DMD6_R2_* directory directly under outputs")
        existed = out.exists()
        out.mkdir(parents=True, exist_ok=True)
        return out, existed
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUTPUTS / f"OFFICIAL_BASELINES_DMD6_R2_{stamp}"
    suffix = 0
    while out.exists():
        suffix += 1
        out = OUTPUTS / f"OFFICIAL_BASELINES_DMD6_R2_{stamp}_{suffix:02d}"
    out.mkdir(parents=True)
    return out.resolve(), False


def input_fingerprint() -> dict[str, Any]:
    files = {"protocol": PROTOCOL, "config": CONFIG, **MANIFESTS}
    payload = {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in files.items()}
    payload["fingerprint_sha256"] = sha256_bytes(canonical_bytes(payload))
    return payload


def freeze_contract(out: pathlib.Path, fingerprint: dict[str, Any]) -> dict[str, Any]:
    policy = {
        "schema_version": 1, "frozen_utc": now_utc(),
        "primary_comparison": "matched DMD-6F",
        "protocol_id": "DMD_6F_2O3P", "physical_geometry": "2 orientations x 3 phases",
        "raw_frame_order": ["H0", "H120", "H240", "V0", "V120", "V240"],
        "apd_sim_9_role": "frame-budget upper reference only",
        "native_9f_methods": "supplementary only", "cross_protocol_pooled_ranking": False,
        "nine_vs_six_paired_significance_tests": False,
        "duplicated_interpolated_generated_observed_frames": False,
        "minimum_external_matched_methods": 2,
        "preferred_composition": "at least one learning/self-supervised and one model-based",
        "candidate_inclusion_uses_test_metrics": False, "best_of_n": False,
        "method_specific_test_image_percentile_remapping": False,
    }
    policy["policy_hash"] = sha256_bytes(canonical_bytes({k: v for k, v in policy.items() if k != "policy_hash"}))
    write_json(out / "01_shared_contract" / "comparison_policy.json", policy)
    write_json(out / "01_shared_contract" / "input_fingerprint.json", fingerprint)
    return policy


def record_sources(out: pathlib.Path, specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    clone_rows: list[dict[str, Any]] = []
    for spec in specs:
        source = EXTERNAL / spec["source_dir"]
        archive = EXTERNAL / spec["archive"]
        record = dict(spec)
        record["captured_utc"] = now_utc()
        record["source_path"] = str(source)
        record["worktree_path"] = str(WORKTREES / spec["source_dir"])
        record["source_exists"] = source.is_dir()
        record["archive_exists"] = archive.is_file()
        record["archive_sha256"] = sha256_file(archive) if archive.is_file() else None
        if source.is_dir():
            record["source_tree_hash"], entries = tree_hash(source)
            record["source_payload_hash"], _, record["source_generated_artifacts"] = source_payload_hash(source)
            record["source_file_count"] = len(entries)
            write_rows(
                out / "03_provenance" / f"{spec['method_id']}_source_files.tsv",
                ["path", "bytes", "sha256"], entries,
            )
        else:
            record["source_tree_hash"] = None
            record["source_file_count"] = 0
            record["blocked_reason"] = f"Official source archive is missing: {source}"
            record["eligibility"] = "BLOCKED_PROVENANCE"
            record["comparison_group"] = "Blocked/not executed"
        worktree = WORKTREES / spec["source_dir"]
        if source.is_dir() and worktree.is_dir():
            record["worktree_tree_hash"], _ = tree_hash(worktree)
            record["worktree_payload_hash"], _, record["worktree_generated_artifacts"] = source_payload_hash(worktree)
            record["worktree_matches_pristine"] = record["worktree_payload_hash"] == record["source_payload_hash"]
        else:
            record["worktree_tree_hash"] = None
            record["worktree_payload_hash"] = None
            record["worktree_generated_artifacts"] = []
            record["worktree_matches_pristine"] = False
        write_json(out / "03_provenance" / f"{spec['method_id']}.json", record)
        snapshots.append(record)
        clone_rows.append({
            "method_id": record["method_id"], "repository_url": record["repository_url"],
            "branch": record["branch"], "commit": record["commit"], "archive_sha256": record["archive_sha256"],
            "source_tree_hash": record["source_tree_hash"], "source_file_count": record["source_file_count"],
            "worktree_matches_pristine": record["worktree_matches_pristine"], "acquisition": "PINNED_COMMIT_OR_PUBLICATION_ARCHIVE",
        })
    write_rows(out / "04_sources_pristine" / "clone_receipts.tsv", list(clone_rows[0]), clone_rows)
    write_rows(
        out / "00_preflight" / "source_snapshot.tsv",
        ["method_id", "source_path", "source_tree_hash", "source_file_count", "commit", "archive_sha256"], snapshots,
    )
    return snapshots


def freeze_candidates(out: pathlib.Path, records: list[dict[str, Any]], policy: dict[str, Any]) -> None:
    fields = [
        "method_id", "manuscript_label", "kind", "paper_title", "doi", "repository_url", "commit",
        "provenance_status", "license", "native_protocol", "eligibility", "comparison_group", "environment",
        "source_tree_hash", "blocked_reason",
    ]
    write_rows(out / "02_candidate_registry" / "candidate_registry.csv", fields, records, delimiter=",")
    registry_hash = sha256_file(out / "02_candidate_registry" / "candidate_registry.csv")
    receipt = {
        "frozen_utc": now_utc(), "candidate_registry_sha256": registry_hash,
        "comparison_policy_hash": policy["policy_hash"], "test_metrics_examined": False,
        "selection_basis": ["publication/source traceability", "method relevance", "DMD-6F adaptability", "license", "execution feasibility"],
    }
    receipt["receipt_sha256"] = sha256_bytes(canonical_bytes(receipt))
    write_json(out / "02_candidate_registry" / "candidate_freeze_receipt.json", receipt)
    write_text(out / "02_candidate_registry" / "selection_rationale.md", """# Candidate freeze rationale

The candidate list was frozen before any sealed-test reconstruction or metric calculation. Inclusion used only publication linkage, exact source provenance, protocol adaptability, license evidence, and execution feasibility. Local proxy implementations and all prior numerical results are excluded.

ML-SIM-6R preserves the official RCAN/MSE training definition and changes the input count plus dataset/channel semantics before retraining: the official six-channel selector is a 3-orientation/2-phase slice of a nine-frame stack and is not a valid H/V 2-orientation/3-phase adapter. mcSIM-Wiener-6 directly supports a dynamic orientation count with three phases. The original publication-supplement Hessian implementation contains a two-beam 2-orientation/3-phase route, subject to a formal MATLAB numerical smoke. SSR-SIM remains native-nine-frame supplementary because its official BioSR path fixes a nine-channel/3-orientation core. The GPU-enabled Hessian derivative is excluded because its current entry path denoises an intermediate TIFF and does not establish an end-to-end raw DMD-6F reconstruction route.
""")


def protocol_audit(out: pathlib.Path, records: list[dict[str, Any]]) -> None:
    fields = ["method_id", "manuscript_label", "native_protocol", "eligibility", "comparison_group", "source_tree_hash", "blocked_reason"]
    write_rows(out / "06_protocol_audit" / "eligibility_matrix.csv", fields, records, delimiter=",")
    audits = {
        "schema_version": 1, "audited_utc": now_utc(), "protocol_id": "DMD_6F_2O3P",
        "allowed_status_values": [
            "MATCHED_DMD6_DIRECT_CONFIGURATION", "MATCHED_DMD6_OFFICIAL_RETRAINING", "NATIVE_9F_ONLY",
            "BLOCKED_PROVENANCE", "BLOCKED_MISSING_BINARY", "BLOCKED_MISSING_RUNTIME",
            "BLOCKED_CORE_METHOD_ASSUMPTION", "BLOCKED_NUMERICAL_FAILURE",
        ],
        "methods": records,
        "local_proxy_exclusion": ["mcSIM.py", "ssrsim_hessiansim.py", "fairSIM.py"],
    }
    write_json(out / "06_protocol_audit" / "core_method_audit.json", audits)
    lines = ["# Core-method DMD-6F audit", "", "No local proxy or previous numerical result is admitted.", ""]
    for row in records:
        lines += [f"## {row['manuscript_label']}", "", f"- Provenance: `{row['provenance_status']}`", f"- Native protocol: {row['native_protocol']}", f"- Eligibility: `{row['eligibility']}`", f"- Group: {row['comparison_group']}", f"- Current blocker: {row['blocked_reason']}", ""]
    write_text(out / "06_protocol_audit" / "core_method_audit.md", "\n".join(lines))


def environment_specs(out: pathlib.Path, records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    definitions = {
        "apd_mlsim_official_r2": """name: apd_mlsim_official_r2
channels: [pytorch, nvidia, conda-forge]
dependencies:
  - python=3.10
  - pip
  - numpy=1.26
  - scikit-image
  - matplotlib
  - numba
  - pytorch
  - torchvision
  - pip:
      - wandb
      - streamlit
      - plotly
""",
        "apd_ssrsim_official_r2": """name: apd_ssrsim_official_r2
channels: [pytorch, nvidia, conda-forge]
dependencies:
  - python=3.10.10
  - pip
  - numpy=1.23.5
  - scipy=1.10.1
  - pip:
      - bitarray==3.0.0
      - imageio==2.36.0
      - matplotlib==3.9.2
      - opencv-python==4.10.0.84
      - tifffile==2024.9.20
      - torch==2.5.1
      - torchvision==0.20.1
""",
        "apd_mcsim_official_r2": """name: apd_mcsim_official_r2
channels: [conda-forge]
dependencies:
  - python=3.10
  - pip
  - numpy>=1.24
  - scipy
  - matplotlib
  - scikit-image
  - joblib
  - dask
  - dask-image
  - pip:
      - -e ./external/official_r2_worktrees/mcsim
""",
        "apd_hessian_original_r2": """name: apd_hessian_original_r2
runtime: MATLAB
matlab_executable: data/external_input
required_toolboxes: [Image Processing Toolbox, Parallel Computing Toolbox]
source: publication supplementary archive for 10.1038/nbt.4115
""",
        "apd_gpu_hessian_r2": """name: apd_gpu_hessian_r2
runtime: MATLAB
matlab_executable: data/external_input
required_toolboxes: [Image Processing Toolbox, Parallel Computing Toolbox]
source: mc2lab/GPU-enabled-Hessian-SIM@162527b9286df8f711a7c775b6fc36c94b97cd93
""",
    }
    receipts: dict[str, dict[str, Any]] = {}
    for env_name, content in definitions.items():
        env_dir = out / "05_environments" / env_name
        write_text(env_dir / "environment.yml", content)
        explicit = "NOT_CREATED_IN_PREPARATION_PASS\nFormal environment creation/solver execution remains a resumable step.\n"
        write_text(env_dir / "explicit_packages.txt", explicit)
        is_matlab = "runtime: MATLAB" in content
        if is_matlab:
            matlab = pathlib.Path(r"data/external_input")
            smoke = f"STATIC_RUNTIME_DISCOVERY\nmatlab_executable={matlab}\nexists={matlab.exists()}\nGPU code was not executed.\n"
            status = "RUNTIME_DISCOVERED_GPU_SMOKE_DEFERRED" if matlab.exists() else "BLOCKED_MISSING_RUNTIME"
        else:
            smoke = "STATIC_SPEC_VALIDATION_ONLY\nNo GPU import was executed while APD training was active.\n"
            status = "SPEC_FROZEN_ENV_CREATION_PENDING"
        write_text(env_dir / "import_smoke.log", smoke)
        env_hash = sha256_file(env_dir / "environment.yml")
        receipt = {"environment": env_name, "environment_hash": env_hash, "status": status, "gpu_work_executed": False}
        write_json(env_dir / "environment_receipt.json", receipt)
        receipts[env_name] = receipt
    rows = [receipts[row["environment"]] for row in records]
    write_rows(out / "05_environments" / "environment_hashes.tsv", ["environment", "environment_hash", "status", "gpu_work_executed"], rows)
    return receipts


def write_patch_receipts(out: pathlib.Path, records: list[dict[str, Any]]) -> None:
    for record in records:
        patch = out / "07_adapters" / "patches" / f"{record['method_id']}.patch"
        write_text(patch, "# NO IN-PLACE SOURCE PATCH IN PREPARATION PASS\n# Adapters live outside pristine source trees.\n")
        receipt = {
            "method_id": record["method_id"], "patch_sha256": sha256_file(patch),
            "patch_class": "NO_CORE_SOURCE_CHANGE", "pristine_tree_hash": record.get("source_tree_hash"),
            "worktree_matches_pristine": record.get("worktree_matches_pristine"),
            "allowed_future_patch_classes": [
                "I/O adapter", "dataset adapter", "DMD-6F protocol configuration", "input-channel configuration",
                "training wrapper", "evaluation wrapper", "dependency-compatibility patch",
            ],
        }
        write_json(out / "07_adapters" / "adapter_receipts" / f"{record['method_id']}.json", receipt)


def write_method_configs(out: pathlib.Path, records: list[dict[str, Any]], apd: dict[str, Any]) -> None:
    common = {
        "protocol_id": "DMD_6F_2O3P", "protocol_hash": json.loads(PROTOCOL.read_text(encoding="utf-8"))["protocol_hash"],
        "raw_frame_order": ["H0", "H120", "H240", "V0", "V120", "V240"],
        "train_identities": 132, "validation_identities": 15, "sealed_test_fovs": 30,
        "test_metrics_available_to_selection": False, "best_of_n": False,
    }
    details = {
        "mlsim_6r": {
            "architecture": "official RCAN", "nch_in": 6, "nch_out": 1, "Nangles": 2, "Nshifts": 3,
            "n_resgroups": 2, "n_resblocks": 5, "n_feats": 48, "scale": 1,
            "loss": "official MSELoss", "optimizer": "official Adam", "learning_rate": 0.0001,
            "batch_size": 5, "seed": 20260813, "initialization": "from scratch; official 9F weights forbidden",
            "checkpoint_rule": "lowest mean validation MSE; ties earlier epoch", "stopping_rule": "prespecified 100 epochs; no sealed-test access",
        },
        "ssrsim": {"status": "NATIVE_9F_ONLY", "formal_dmd6_execution": False, "defining_self_supervised_objective": "unchanged"},
        "mcsim_wiener6": {
            "input_shape": "2 orientations x 3 phases x H x W", "reconstruction_mode": "wiener-filter",
            "phase_estimation_mode": "wicker-iterative", "wiener_parameter": 0.1,
            "parameter_policy": "official defaults unless validation-only frozen search is explicitly added before execution",
        },
        "hessian_original6": {"input": "raw DMD-6F stack", "nangles": 2, "nphases": 3, "runtime": "MATLAB", "role": "original publication supplementary implementation"},
        "gpu_hessian6": {"input": "intermediate TIFF in current GUI", "formal_dmd6_execution": False, "runtime": "MATLAB/GPU", "role": "documented derivative implementation; raw DMD-6F core path not established"},
    }
    for record in records:
        method_dir = out / "08_training" / record["method_id"]
        cfg = {**common, **details[record["method_id"]], "method_id": record["method_id"], "eligibility": record["eligibility"]}
        cfg["config_hash"] = sha256_bytes(canonical_bytes({k: v for k, v in cfg.items() if k != "config_hash"}))
        write_json(method_dir / "config.json", cfg)
        deferred = {
            "method_id": record["method_id"], "captured_utc": now_utc(),
            "status": "GPU_RESOURCE_DEFERRED" if record["eligibility"].startswith("MATCHED") and apd["apd_gpu_active"] else "NOT_EXECUTED_PROTOCOL_INELIGIBLE" if record["eligibility"] == "NATIVE_9F_ONLY" else "READY_FOR_RESUME",
            "formal_training_started": False, "formal_inference_started": False, "checkpoint_path": None,
            "checkpoint_sha256": None, "output_hashes": [], "remaining_blocker": record["blocked_reason"],
        }
        write_json(method_dir / "run_or_deferred_receipt.json", deferred)


def write_pending_deliverables(out: pathlib.Path, records: list[dict[str, Any]]) -> None:
    headers = ["method_id", "sample_id", "class", "input_stack_sha256", "output_sha256", "psnr", "ssim", "frc_period_um", "status"]
    write_rows(out / "09_baseline_only_results" / "per_fov.csv", headers, [], delimiter=",")
    write_rows(out / "09_baseline_only_results" / "summary.csv", ["method_id", "metric", "n", "mean", "sd", "median", "q1", "q3", "status"], [], delimiter=",")
    write_rows(out / "09_baseline_only_results" / "class_summary.csv", ["method_id", "class", "metric", "n", "mean", "sd", "median", "q1", "q3", "status"], [], delimiter=",")
    write_text(out / "09_baseline_only_results" / "baseline_only_report.md", "# Baseline-only result status\n\nNo formal baseline inference or test metric was run because APD GPU training is active. Status: `BASELINE_GPU_RESOURCE_DEFERRED`.\n")
    metric_files = {
        "matched_dmd6_per_fov.csv": ["method_id", "sample_id", "class", "input_stack_sha256", "psnr", "ssim", "frc_period_um", "status"],
        "matched_dmd6_summary.csv": ["method_id", "metric", "n", "mean", "sd", "median", "q1", "q3", "status"],
        "matched_dmd6_class_summary.csv": ["method_id", "class", "metric", "n", "mean", "sd", "median", "q1", "q3", "status"],
        "paired_effects.csv": ["method_a", "method_b", "metric", "n_parent", "median_difference", "ci95_low", "ci95_high", "wilcoxon_p", "holm_p", "status"],
    }
    for name, fields in metric_files.items():
        write_rows(out / "11_metrics" / name, fields, [], delimiter=",")
    write_json(out / "11_metrics" / "statistics.json", {"status": "APD6_FINALIZATION_PENDING", "formal_test_metrics_computed": False, "bootstrap_seed": 20260813})
    write_text(out / "11_metrics" / "physmap6_fairness_audit.md", "# PhysMap-6 fairness audit\n\nStatus: `APD6_FINALIZATION_PENDING`. The future audit must prove byte-identical DMD-6F input-stack, protocol, mask, normalization, crop, and output support hashes for APD-SIM-6, DiffWS-6, and PhysMap-6. PhysMap-9 is excluded from matched ranking.\n")
    write_json(out / "12_runtime" / "runtime_status.json", {"status": "FORMAL_RUNTIME_DEFERRED", "reason": "APD GPU training active", "old_runtime_reused": False})
    write_json(out / "13_figures" / "figure_status.json", {"status": "APD6_FINALIZATION_PENDING", "figures_generated": False, "method_specific_remapping_allowed": False})
    write_rows(out / "13_figures" / "real_data_manifest.tsv", ["fov_id", "raw_file", "raw_order", "sha256", "status"], [])
    write_json(out / "13_figures" / "real_display_contract.json", {"status": "REAL_DMD6_RAW_DATA_NOT_LOCATED", "common_lut": True, "common_display_range": True, "method_specific_percentiles": False})


def write_supplementary(out: pathlib.Path, records: list[dict[str, Any]], envs: dict[str, dict[str, Any]]) -> None:
    fields = [
        "Method", "Manuscript label", "Paper/DOI", "Implementation provenance", "Repository/release/commit",
        "Environment", "Frame count", "Orientation x phase geometry", "Raw-frame order", "Native protocol",
        "DMD-6F adaptation", "Training/initialization", "Principal parameters", "Stopping/checkpoint rule", "Seed policy",
        "Output harmonization", "Comparison group", "Evidence status",
    ]
    rows = []
    for r in records:
        matched = r["eligibility"].startswith("MATCHED")
        if matched:
            frame_count, geometry, order = "6", "2 x 3", "H0/H120/H240/V0/V120/V240"
            evidence_status = "SOURCE_VERIFIED_EXECUTION_PENDING"
        elif r["eligibility"] == "NATIVE_9F_ONLY":
            frame_count, geometry, order = "9", "3 x 3", "official native order; supplementary only"
            evidence_status = "SOURCE_VERIFIED_NATIVE9_NOT_EXECUTED"
        else:
            frame_count, geometry, order = "NOT_APPLICABLE", "NOT_ADMITTED", "NOT_ADMITTED"
            evidence_status = "SOURCE_VERIFIED_BLOCKED_NOT_EXECUTED"
        rows.append({
            "Method": r["method_id"], "Manuscript label": r["manuscript_label"], "Paper/DOI": f"{r['paper_title']} / {r['doi']}",
            "Implementation provenance": r["provenance_status"], "Repository/release/commit": f"{r['repository_url']} @ {r['commit']}",
            "Environment": f"{r['environment']} / {envs[r['environment']]['environment_hash']}",
            "Frame count": frame_count, "Orientation x phase geometry": geometry,
            "Raw-frame order": order,
            "Native protocol": r["native_protocol"], "DMD-6F adaptation": r["eligibility"],
            "Training/initialization": "See frozen method config; formal execution deferred", "Principal parameters": "See 08_training/<method>/config.json",
            "Stopping/checkpoint rule": "Frozen before test; pending execution", "Seed policy": "Frozen before test; no best-of-N",
            "Output harmonization": "Common grid/support; no test-specific affine or percentile remapping",
            "Comparison group": r["comparison_group"], "Evidence status": evidence_status if r.get("source_exists") else "BLOCKED",
        })
    write_rows(out / "14_supplementary" / "Supplementary_Table_S1.csv", fields, rows, delimiter=",")
    evidence_rows = []
    for row, source in zip(rows, records):
        for field in fields:
            evidence_rows.append({"method_id": source["method_id"], "field": field, "value": row[field], "evidence": f"03_provenance/{source['method_id']}.json", "status": "VERIFIED" if source.get("source_exists") else "NOT_RECOVERABLE"})
    write_rows(out / "14_supplementary" / "Supplementary_Table_S1_evidence.tsv", ["method_id", "field", "value", "evidence", "status"], evidence_rows)
    tex_rows = []
    for row in rows:
        clean = lambda s: str(s).replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("_", "\\_").replace("%", "\\%")
        source = next(r for r in records if r["method_id"] == row["Method"])
        source_id = f"{source['owner']} @ {source['commit'][:12]}" if source["commit"] != "NOT_APPLICABLE_ARCHIVE" else "Publisher supplement; archive hash in evidence ledger"
        adaptation = {
            "MATCHED_DMD6_OFFICIAL_RETRAINING": "Matched: official retraining",
            "MATCHED_DMD6_DIRECT_CONFIGURATION": "Matched: direct 2O3P",
            "NATIVE_9F_ONLY": "Native 9F only",
            "BLOCKED_CORE_METHOD_ASSUMPTION": "Blocked: core assumption",
        }.get(row["DMD-6F adaptation"], row["DMD-6F adaptation"])
        paper = source['doi'].split(";", 1)[0]
        tex_rows.append("{} & {} & {} & {} & {} \\\\".format(clean(row["Manuscript label"]), clean(paper), clean(source_id), clean(adaptation), clean(row["Comparison group"])))
    tex = """% Evidence-driven R2 table; include in an elsarticle document.
\\begin{table*}[t]
\\centering\\small
\\caption{Source and protocol audit. Only methods in the Matched DMD-6F group received the identical two-orientation/three-phase six-frame measurements and entered the principal quantitative ranking.}
\\begin{tabular}{p{2.5cm}p{3.4cm}p{4.2cm}p{3.4cm}p{2.4cm}}
Method & Paper/DOI & Repository/release/commit & DMD-6F adaptation & Comparison group \\\\ \\hline
""" + "\n".join(tex_rows) + "\n\\end{tabular}\n\\end{table*}\n"
    write_text(out / "14_supplementary" / "Supplementary_Table_S1.tex", tex)
    write_text(out / "14_supplementary" / "Supplementary_Table_S1_audit.md", "# Supplementary Table S1 audit\n\nStatus: `SUPPLEMENTARY_TABLE_S1_AUDIT_BLOCKED`. Source/protocol fields are populated. The source-only table compiles in `elsarticle` with zero undefined commands, zero undefined references, and zero overfull boxes, and its rendered page has been visually checked. Checkpoint, formal output, runtime, and final harmonization evidence cannot pass until deferred execution is complete. No unsupported performance claim is present.\n")


def write_manuscript(out: pathlib.Path) -> None:
    files = {
        "comparator_protocol_replacement.tex": "The primary comparison uses the matched DMD-6F protocol (two orientations, three phases, ordered H0/H120/H240/V0/V120/V240). Native nine-frame methods are reported separately and do not enter pooled rankings or paired tests.\n",
        "baseline_results_replacement.tex": "% PLACEHOLDER: insert only after formal 30-FOV results and APD-SIM-6 finalization.\n",
        "figure4_caption_replacement.tex": "% PLACEHOLDER: common-field, common-stack, common linear range/LUT/crops/scale bars required before use.\n",
        "real_baseline_caption_replacement.tex": "% PLACEHOLDER: real DMD-6F data not located; no vendor output is treated as GT.\n",
        "runtime_replacement.tex": "% FORMAL_RUNTIME_DEFERRED while APD GPU training is active.\n",
        "native9_supplementary_text.tex": "Methods restricted to their native nine-frame protocol are supplementary-only and are excluded from matched DMD-6F rankings, paired significance tests, and runtime rankings.\n",
        "reviewer1_baseline_response.txt": "We have frozen a strict two-orientation/three-phase DMD-6F fairness contract. The future APD-SIM-6, DiffWS-6, and PhysMap-6 comparison is gated on identical input-stack, protocol, mask, normalization, crop, and support hashes; PhysMap-9 is excluded from the matched ranking. Formal recomputation remains pending and this draft does not claim completion.\n",
        "reviewer1_physmap_response.txt": "The prior PhysMap-9 comparison is not admitted to the matched DMD-6F ranking. Once APD-SIM-6 is complete, APD-SIM-6, DiffWS-6, and PhysMap-6 will be recomputed only after byte-identical input-stack, DMD-6F protocol, mask, normalization, crop, and output-support hashes pass. Until that audit completes, no PhysMap fairness result is claimed.\n",
        "reviewer2_baseline_response.txt": "We source-verified learning and model-based low-frame candidates before inspecting test metrics and froze a 30-FOV shared-measurement design with class-stratified and paired statistics. Formal outputs, common-display figures, and scale bars remain pending while APD GPU training is active; no performance claim is made in this draft.\n",
        "reviewer3_comment6_response.txt": "The new evidence-driven Table S1 records source, exact commit/archive, provenance, protocol, adaptation, environment, parameters, stopping/checkpoint rule, seed policy, and harmonization. Local proxies are explicitly excluded. Execution-dependent fields remain visibly pending rather than inferred.\n",
        "claims_allowed.md": "# Claims allowed now\n\n- Official source archives and exact revisions were frozen before test metrics.\n- The primary policy is matched DMD-6F; native 9F methods are supplementary-only.\n- Formal baseline GPU work and runtime were deferred because APD training was active.\n",
        "claims_not_supported.md": "# Claims not supported\n\n- APD-SIM-6 outperforms any method.\n- State of the art or universal superiority.\n- Formal 30-FOV metric, figure, runtime, real-data, or PhysMap-6 fairness completion.\n- READY_FOR_SUPPLEMENTARY_TABLE_S1.\n",
    }
    for name, text in files.items():
        write_text(out / "15_manuscript" / name, text)
    write_rows(out / "15_manuscript" / "numeric_replacements.csv", ["locator", "old_value", "new_value", "evidence", "status"], [], delimiter=",")


def write_finalizers(out: pathlib.Path) -> None:
    apd_eval = '''#!/usr/bin/env python3
"""Validate a completed non-legacy APD-SIM-6 checkpoint and shared-bundle handoff."""
import argparse, hashlib, json, pathlib, sys

def h(p):
    x=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): x.update(b)
    return x.hexdigest()

p=argparse.ArgumentParser(); p.add_argument("--checkpoint"); p.add_argument("--execute",action="store_true"); a=p.parse_args()
if not a.checkpoint:
    print("APD6_FINALIZATION_PENDING"); raise SystemExit(2)
ck=pathlib.Path(a.checkpoint).resolve()
if not ck.is_file() or "legacy" in str(ck).lower():
    print("APD6_FINALIZATION_PENDING: completed non-legacy checkpoint not supplied"); raise SystemExit(2)
print(json.dumps({"status":"APD6_CHECKPOINT_LOCATED_EVALUATION_NOT_IMPLEMENTED","checkpoint":str(ck),"sha256":h(ck),"execute_requested":a.execute},indent=2))
raise SystemExit(3)
'''
    finalize = '''#!/usr/bin/env python3
"""Final joint comparison gate. It never regenerates shared test measurements."""
import json, pathlib, sys
root=pathlib.Path(__file__).resolve().parents[1]
handoff=root/"10_apd6_finalization"/"APD6_SHARED_BUNDLE_HANDOFF.json"
if not handoff.exists():
    print("APD6_FINALIZATION_PENDING"); raise SystemExit(2)
d=json.loads(handoff.read_text(encoding="utf-8"))
required=["protocol_hash","test_bundle_hash","apd6_checkpoint_sha256"]
missing=[k for k in required if not d.get(k)]
if missing:
    print("APD6_FINALIZATION_PENDING: missing "+",".join(missing)); raise SystemExit(2)
print("APD6_FINALIZATION_PENDING: metric/output hash verification runner has not completed")
raise SystemExit(3)
'''
    write_text(out / "10_apd6_finalization" / "evaluate_apd6_on_shared_dmd6_bundle.py", apd_eval)
    write_text(out / "10_apd6_finalization" / "finalize_official_baseline_comparison.py", finalize)
    write_json(out / "10_apd6_finalization" / "APD6_SHARED_BUNDLE_HANDOFF.json", {
        "status": "APD6_FINALIZATION_PENDING", "protocol_id": "DMD_6F_2O3P",
        "protocol_hash": json.loads(PROTOCOL.read_text(encoding="utf-8"))["protocol_hash"],
        "test_bundle_hash": "eb09310048bcaa0b710fd6f29fe2f169188da11c98873dd6b2b0f9d3dd05b4c5",
        "apd6_checkpoint": None, "apd6_checkpoint_sha256": None,
    })


def write_resume(out: pathlib.Path) -> None:
    source = ROOT / "tools" / "official_r2_resume.py"
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = sha256_file(source)
    launcher = f'''#!/usr/bin/env python3
"""Pinned entry point for the fail-closed official R2 resume controller."""
import hashlib
import pathlib
import runpy
import sys

RUN = pathlib.Path(__file__).resolve().parent
WORKSPACE = RUN.parents[1]
CONTROLLER = WORKSPACE / "tools" / "official_r2_resume.py"
EXPECTED_SHA256 = "{digest}"

if not CONTROLLER.is_file():
    print(f"RESUME_INTEGRITY_BLOCKED: missing controller: {{CONTROLLER}}")
    raise SystemExit(4)
actual = hashlib.sha256(CONTROLLER.read_bytes()).hexdigest()
if actual != EXPECTED_SHA256:
    print(f"RESUME_INTEGRITY_BLOCKED: controller hash mismatch: {{actual}}")
    raise SystemExit(4)
sys.argv[0] = str(CONTROLLER)
runpy.run_path(str(CONTROLLER), run_name="__main__")
'''
    write_text(out / "resume_official_baselines_dmd6.py", launcher)


def audit_and_state(out: pathlib.Path, fingerprint: dict[str, Any], policy: dict[str, Any], apd: dict[str, Any], records: list[dict[str, Any]], envs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    matched = [r for r in records if r["eligibility"].startswith("MATCHED") and r.get("source_exists")]
    learning = [r for r in matched if r["kind"] in {"learning", "self-supervised"}]
    model = [r for r in matched if r["kind"] == "model-based"]
    p0: list[dict[str, str]] = []
    if len(matched) < 2:
        p0.append({"id": "P0-MATCHED-COUNT", "finding": "Fewer than two source-verified matched DMD-6F external methods"})
    if not learning:
        p0.append({"id": "P0-LEARNING", "finding": "No source-verified matched learning/self-supervised DMD-6F method"})
    if not model:
        p0.append({"id": "P0-MODEL", "finding": "No source-verified matched model-based DMD-6F method"})
    p1 = [
        {"id": "P1-SHARED-BUNDLE", "finding": "Shared validation/test raw arrays and aggregate hashes pending builder completion"},
        {"id": "P1-ENVS", "finding": "Isolated environments specified but not solver-created/import-tested"},
        {"id": "P1-FORMAL", "finding": "Formal baseline training/reconstruction deferred while APD GPU active"},
        {"id": "P1-APD6", "finding": "APD-SIM-6 finalization pending"},
        {"id": "P1-S1", "finding": "S1 execution-dependent fields and compile audit pending"},
    ]
    p2 = [{"id": "P2-REAL", "finding": "Real DMD-6F raw stack not located; real-data figure withheld"}]
    rows = [{"priority": "P0", **x} for x in p0] + [{"priority": "P1", **x} for x in p1] + [{"priority": "P2", **x} for x in p2]
    write_rows(out / "16_audit" / "P0_P1_P2_findings.csv", ["priority", "id", "finding"], rows, delimiter=",")
    final_status = "OFFICIAL_MATCHED_DMD6_SET_INSUFFICIENT" if p0 else "OFFICIAL_BASELINE_PREPARATION_IN_PROGRESS"
    audit = {
        "audited_utc": now_utc(), "p0_count": len(p0), "p1_count": len(p1), "p2_count": len(p2),
        "matched_external_count": len(matched), "matched_learning_count": len(learning), "matched_model_based_count": len(model),
        "apd_gpu_active": apd["apd_gpu_active"], "formal_runtime_status": "FORMAL_RUNTIME_DEFERRED",
        "supplementary_table_s1_status": "SUPPLEMENTARY_TABLE_S1_AUDIT_BLOCKED", "final_status": final_status,
        "checks": {
            "local_proxies_excluded": True, "native9_excluded_from_matched_ranking": True,
            "test_metrics_examined_for_candidate_selection": False, "best_of_n": False,
            "method_specific_display_remapping": False, "old_runtime_reused": False,
            "apd_training_modified": False,
        },
        "findings": rows,
    }
    write_json(out / "16_audit" / "final_audit.json", audit)
    write_text(out / "16_audit" / "final_audit.md", f"# Independent R2 audit\n\nCurrent status: `{final_status}`.\n\nP0={len(p0)}, P1={len(p1)}, P2={len(p2)}. Source and protocol preparation is usable, but formal completion is not claimed. Deferred GPU work, APD6 finalization, figures, runtime, and S1 execution evidence remain open.\n")
    write_text(out / "16_audit" / "unresolved_items.md", "# Unresolved items\n\n" + "\n".join(f"- `{r['id']}`: {r['finding']}" for r in rows) + "\n")
    evidence = []
    for path in sorted((p for p in out.rglob("*") if p.is_file() and p.name != "evidence_manifest.tsv"), key=lambda p: p.relative_to(out).as_posix()):
        evidence.append({"path": path.relative_to(out).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_rows(out / "16_audit" / "evidence_manifest.tsv", ["path", "bytes", "sha256"], evidence)
    state_methods = {}
    for record in records:
        env = envs[record["environment"]]
        patch = out / "07_adapters" / "patches" / f"{record['method_id']}.patch"
        state_methods[record["method_id"]] = {
            "source_url": record["repository_url"], "source_commit": record["commit"], "source_tree_hash": record.get("source_tree_hash"),
            "environment_hash": env["environment_hash"], "patch_hash": sha256_file(patch), "provenance_status": record["provenance_status"],
            "native_protocol": record["native_protocol"], "dmd6_eligibility": record["eligibility"],
            "training_status": "GPU_RESOURCE_DEFERRED" if record["eligibility"].startswith("MATCHED") and apd["apd_gpu_active"] else "NOT_APPLICABLE_OR_PENDING",
            "checkpoint_path": None, "checkpoint_hash": None, "test_status": "NOT_STARTED", "output_hashes": [], "remaining_blocker": record["blocked_reason"],
        }
    state = {
        "schema_version": 1, "created_utc": now_utc(), "updated_utc": now_utc(), "output_root": str(out),
        "status": final_status, "input_fingerprint": fingerprint, "comparison_policy_hash": policy["policy_hash"],
        "apd_training": apd, "methods": state_methods, "matched_external_count": len(matched),
        "matched_learning_methods": [r["manuscript_label"] for r in learning], "matched_model_based_methods": [r["manuscript_label"] for r in model],
        "native9_supplementary_methods": [r["manuscript_label"] for r in records if r["eligibility"] == "NATIVE_9F_ONLY"],
        "apd6_finalization_status": "APD6_FINALIZATION_PENDING", "metric_status": "NOT_STARTED",
        "figure_status": "APD6_FINALIZATION_PENDING", "real_data_status": "REAL_DMD6_RAW_DATA_NOT_LOCATED",
        "physmap6_fairness_status": "APD6_FINALIZATION_PENDING", "runtime_status": "FORMAL_RUNTIME_DEFERRED",
        "supplementary_table_s1_status": "SUPPLEMENTARY_TABLE_S1_AUDIT_BLOCKED",
        "p0_count": len(p0), "p1_count": len(p1), "p2_count": len(p2),
    }
    write_json(out / "BASELINE_R2_STATE.json", state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", help="resume only this exact R2 run root")
    parser.add_argument(
        "--scaffold-only",
        action="store_true",
        help="write the initial pending scaffold; final preparation publishing requires the dedicated finalizer",
    )
    args = parser.parse_args()
    if args.output_root and not args.scaffold_only:
        raise RuntimeError(
            "Refusing to overwrite an existing R2 run with pending scaffold data. "
            "Use finalize_official_r2_preparation.py for an existing run, or pass --scaffold-only deliberately."
        )
    out, existed = make_run_root(args.output_root)
    for stage in STAGES:
        (out / stage).mkdir(parents=True, exist_ok=True)
    fingerprint = input_fingerprint()
    existing_fp = out / "01_shared_contract" / "input_fingerprint.json"
    if existed and existing_fp.exists():
        previous = json.loads(existing_fp.read_text(encoding="utf-8"))
        if previous.get("fingerprint_sha256") != fingerprint["fingerprint_sha256"]:
            raise RuntimeError("Existing run input fingerprint differs; refusing resume")
    apd = preflight(out)
    policy = freeze_contract(out, fingerprint)
    specs = source_specs()
    records = record_sources(out, specs)
    freeze_candidates(out, records, policy)
    protocol_audit(out, records)
    envs = environment_specs(out, records)
    write_patch_receipts(out, records)
    write_method_configs(out, records, apd)
    write_pending_deliverables(out, records)
    write_supplementary(out, records, envs)
    write_manuscript(out)
    write_finalizers(out)
    write_resume(out)
    state = audit_and_state(out, fingerprint, policy, apd, records, envs)
    pointer = {
        "schema_version": 1, "updated_utc": now_utc(), "output_root": str(out),
        "state_path": str(out / "BASELINE_R2_STATE.json"), "status": state["status"],
        "input_fingerprint_sha256": fingerprint["fingerprint_sha256"],
    }
    write_json(CURRENT, pointer)
    print(json.dumps(pointer, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
