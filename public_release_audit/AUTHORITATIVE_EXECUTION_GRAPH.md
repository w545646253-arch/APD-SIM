# Authoritative execution graph

## Training

`scripts/train/train{3,6,9}.py` → `unisim.formal_training_2d.run_formal_training` → BioSR manifest loader (`unisim.datasets`) → protocol registry/runtime → 2-D forward model → APD-conditioned U-Net/diffusion → checkpoint contract. DMD-3F additionally requires the historical verified DMD-9F initializer, which is currently unavailable and therefore fails closed.

## Inference and evaluation

- Single arbitrary GT: `scripts/infer/run_single_gt_apd369_frc_spectra.py` → `unisim.apd369_single_gt_repaired` → R4 repair provenance → protocol-specific forward calls → registered EMA/DDIM Stage 1 → masked Stage 2 → GT-referenced FRC and display assets.
- Formal 30-FOV APD-3/6/9: `scripts/evaluate/evaluate_apd369_repair_r4.py` → `evaluate_apd369_protocols_final.py` plus R4 DMD-9 tiled Stage-1 support → `tools.apd369_final_contract` / `revision_dmd6_common`.
- Frame budget: reproduction wrapper → `unisim.revision_r1.frame_budget_r1c2`.
- Validity mask: reproduction wrapper → `unisim.revision_r1.validity_mask_control`; correct and mask-blind branches share raw data/noise/model and differ only in the logical mask.
- Strict DMD-6F: reproduction wrapper → `physmap6_experiment` → `physmap6_pipeline` / `physmap6_core` / `physmap6_reporting`; compares WF-6, DiffWS-6, PhysMap-6, APD-SIM-6 on identical six-frame hashes.
- Matched baselines: wrapper → `tools.run_revision_matched_dmd6` → first-party adapters; third-party source/checkpoints remain external.
- Class statistics: wrapper consumes the authoritative per-FOV matched CSV and computes ddof=1 grouped summaries without test tuning.
- Seed sensitivity: wrapper → `tools.run_revision_seed_sensitivity`; fixed raw/checkpoint/Stage-2 and prespecified five-seed policy.
- FRC: `tools.official_r2_common_metrics` and `tools.revision_dmd6_common.gt_frc`; 5% edge crop, mean centering, Tukey alpha 0.20, 100 radial annuli, first downward 1/7 crossing, no claim of independent optical resolution.
- Tables/figures: `tools.finalize_revision_experiments` consumes completed sample-level outputs; manuscript-specific internal build products are excluded.

## Superseded or excluded paths

Legacy 3-D code, pre-repair DMD-9/R3 exporters, failed runs, proxy baselines, internal one-off manuscript patchers, and duplicate orchestration snapshots are not in the public execution closure.
