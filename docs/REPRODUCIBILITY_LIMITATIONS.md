# Reproducibility limitations

- The first-party licence is unresolved.
- BioSR, sealed 30-FOV data, DMD controller binaries/bitmaps, and checkpoint binaries are not redistributed.
- The historical DMD-9F initializer used for the final DMD-3F training run is absent and had no recorded SHA-256, so exact DMD-3F retraining is blocked.
- Full evaluation and inference require a compatible GPU and were not rerun for packaging.
- Protocol provenance paths were redacted while scientific protocol hashes were preserved; public file hashes bind the redacted JSON.
- GT-referenced FRC is a numerical comparison, not an independent experimental optical-resolution measurement.
- No public DOI, repository URL, tag, or source commit was recoverable.
