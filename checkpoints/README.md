# Checkpoint placement

Checkpoint binaries are excluded from Git. Place downloaded assets at the exact relative paths in `release_assets/CHECKPOINT_MANIFEST.csv`, verify byte size and SHA-256, and do not rename them. The code loads the EMA branch and rejects incompatible protocol metadata. See `release_assets/LARGE_FILE_POLICY.md`.
