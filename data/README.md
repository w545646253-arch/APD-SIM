# Data placement

No image data are distributed. Place authorized files as follows:

- `BioSR_GT/`: TIFF names and SHA-256 values must match the sanitized train/validation manifests.
- `sealed_test_gt/`: sealed GT files, only for final evaluation; never for training or selection.
- `reproduction/dmd6/`: the frozen six-frame bundle and `test30_dmd6_manifest.tsv`.
- `reproduction/frame_budget/test30/`: the protocol-specific 30-FOV inputs.
- `third_party/`: locally generated third-party baseline checkpoints/outputs acquired under upstream terms.

The release does not assert that BioSR or any sealed data may be redistributed. Verify the data licence independently.
