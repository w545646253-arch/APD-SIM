# APD-SIM public reproducibility release

This release accompanies **Low-Frame DMD Structured Illumination Microscopy with Validity-Masked Measurement-Consistent Diffusion Reconstruction**. It contains the authoritative first-party numerical code and small identity metadata; it contains no BioSR/sealed-test images, checkpoints, generated paper figures, private controller binaries, or third-party baseline source.

## Protocols

All models use a fixed 15-slot conditioning tensor. A validity value of 1 means that a slot contains an acquired measurement and participates in the measurement-consistency objective; 0 means absent/padded and must not contribute.

| Label | Protocol ID | Raw-frame order | 15-slot validity mask |
|---|---|---|---|
| DMD-3F | `DMD_3F_1O3P` | X0, X120, X240 | 111000000000000 |
| DMD-6F | `DMD_6F_2O3P` | H0, H120, H240, V0, V120, V240 | 111111000000000 |
| DMD-9F | `DMD_9F_3O3P` | X0, X120, X240, Y0, Y120, Y240, Z0, Z120, Z240 | 111111111000000 |

The protocol scientific hashes are preserved. Only provenance path strings pointing to local controller/audit files were redacted; both original and public file hashes are recorded in `public_release_audit/SOURCE_SELECTION_RECEIPT.json`.

## Environment

Python 3.10 or newer is required. From the repository root:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m pytest tests/public
```

GPU execution requires a CUDA-compatible PyTorch installation selected for the host; do not blindly replace the CPU-safe test environment.

## Data placement

Obtain BioSR from its official distributor under its own terms and place the 147 manifest-bound TIFFs directly in `data/BioSR_GT/`. Place authorized sealed/reviewer bundles only under `data/reproduction/` as described in `data/README.md`. Test data are never used for training or checkpoint selection.

## Checkpoints

Checkpoints are release assets, not ordinary Git files. Download the three final files and place them at the paths in `release_assets/CHECKPOINT_MANIFEST.csv`, then verify `release_assets/SHA256SUMS.txt`. DMD-3F historical training initialization is not present in the audited workspace and has no recoverable SHA-256; this blocks exact-from-scratch DMD-3F retraining but does not block inference with the final DMD-3F checkpoint.

## Training

Training is not a smoke test and was not run while creating this release. After placing authorized BioSR data and required initialization assets:

```bash
python scripts/train/train3.py
python scripts/train/train6.py
python scripts/train/train9.py
```

Set `APD_DMD_PREFLIGHT_ONLY=1` to validate a training configuration without optimization. The configs preserve the reported 100,000 scheduled-iteration policy. The DMD-3F command fails closed until its historical DMD-9F initialization checkpoint and completion receipt are supplied.

## Single-GT inference

```bash
python scripts/infer/run_single_gt_apd369_frc_spectra.py --gt examples/your_gt.tif --output-root reproduction_outputs/single_gt --seed 20260812
```

The repaired R4 DMD-9F identity is mandatory. The script independently simulates each protocol; it does not form 3F/6F inputs by retrospective subsampling of a common 9F stack.

## Reproduction and evaluation

Every wrapper validates identities, prints resolved inputs, writes JSON receipts, refuses existing non-empty output directories, and requires `--overwrite` to replace generated output. Omit `--execute` for validation only.

```bash
python scripts/reproduce/reproduce_frame_budget.py --execute
python scripts/reproduce/reproduce_validity_mask_control.py --execute
python scripts/reproduce/reproduce_strict_dmd6_ablation.py --execute
python scripts/reproduce/reproduce_matched_baselines.py --execute
python scripts/reproduce/reproduce_class_specific_statistics.py --execute
python scripts/reproduce/reproduce_seed_sensitivity.py --execute
```

The strict DMD-6F comparison is WF-6 / DiffWS-6 / PhysMap-6 / APD-SIM-6 on identical six-frame hashes. External baselines are acquired from upstream repositories at the identities in `THIRD_PARTY_NOTICES.md`; their source is deliberately absent here.

Expected generated roots are `reproduction_outputs/<task>/` and `reproduction_receipts/<task>.json`. Figure/table postprocessing consumes those receipts and sample-level CSV files; it must not tune on the sealed cohort.

## Interpretation boundary

PSNR, SSIM, GT-referenced FRC cutoff/AUC, and synthetic forward-model ablations are numerical, GT-referenced evaluations. They are not independent hardware resolution measurements and do not establish sub-100-nm optical resolution. Controller-defined nominal geometry is not a substitute for a historical acquisition receipt.

## Citation and limitations

Use `CITATION.cff` and replace the pending author metadata before publication. No DOI, repository URL, tag, or source commit was recoverable and none is invented. BioSR redistribution, sealed test redistribution, controller binaries, checkpoint hosting, and the project licence require separate authorization. See `LICENSE_PENDING.md`, `docs/REPRODUCIBILITY_LIMITATIONS.md`, and the audit reports.
