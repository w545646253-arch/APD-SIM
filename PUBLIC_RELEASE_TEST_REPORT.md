# Public release test report

- Status: **PASS**
- Test date (UTC): 2026-08-19
- Training executed: no
- Full GPU inference executed: no
- Dependency installation or environment mutation: no

## CPU-safe gates

| Gate | Result |
|---|---|
| Compile every Python file | PASS |
| Public release tests | 4 passed |
| Protocol runtime, BioSR manifest loader, and official metric tests | 39 passed |
| Formal 2-D core, checkpoint contract, and validity-mask statistics tests | 43 passed |
| Import from a clean working directory | PASS |
| Sanitized training config and manifest hashes | PASS (3/3 configs) |
| JSON parsing and README path references | PASS |
| Secret, local absolute path, and private-IP scan | PASS (0 public findings) |
| Ordinary Git file size limit | PASS (no file above 100 MiB) |
| Missing-asset fail-closed smoke test | PASS (`BLOCKED`, exit code 2, no execution) |

The 86 passing tests are unique across the three successful pytest invocations. The fail-closed smoke deliberately omitted checkpoint assets and confirmed that the frame-budget wrapper stopped before inference and wrote a machine-readable receipt.

## Not executed

Checkpoint-dependent single-GT tests, the sealed 30-FOV workflows, external baseline programs, CUDA inference, and all training were not executed. Their required binaries/data are intentionally absent from the Git tree. Configuration validation and identity checks were used instead. Asset-dependent upstream tests are retained as reference tests but are outside the default `pytest` gate.

Validated environment: Python 3.11.0, NumPy 1.26.4, SciPy 1.16.0, PyTorch 2.8.0.dev20250615+cu128, tifffile 2025.3.30, Pillow 10.4.0, matplotlib 3.8.4, pandas 2.2.3, openpyxl 3.1.5, pytest 9.1.1.
