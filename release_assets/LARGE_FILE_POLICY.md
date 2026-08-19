# Large-file policy

The complete workspace contains 1709 files larger than 10 MiB. Classification counts: {'OPTIONAL_RESULT': 465, 'RESTRICTED_DATA': 7, 'EXCLUDE': 1234, 'CHECKPOINT_REQUIRED': 3}. Per-file classifications are in `public_release_audit/REPOSITORY_INVENTORY.csv` (`size_bytes` plus disposition). No file larger than 100 MiB is in the Git tree.

- Required final checkpoints are release assets and are never committed normally.
- Restricted data remain with their licensors/custodians.
- Generated results, failed runs, caches, duplicates, and third-party source archives are excluded.
- Use Git LFS only if the authors deliberately decide to version checkpoint assets; ordinary release hosting with the exact SHA-256 manifest is preferred.
