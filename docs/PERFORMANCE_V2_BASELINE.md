# Performance V2 baseline

The immutable machine-readable baseline is stored in
`artifacts/performance_v2/baseline_snapshot.json`; the corresponding source,
configuration, test, manifest, and checkpoint hashes are stored in
`artifacts/performance_v2/baseline_files.sha256`.

The original snapshot is never rewritten.  A corrected schema-v3 immutable companion at
`artifacts/performance_v2/baseline_snapshot_binding_v3_20260716.json` atomically
binds the original snapshot, checkpoint, effective config, baseline-owned manifest copies, later
three-domain benchmark, ROI audit, per-organ/per-marker/surrogate-group metrics,
CPU inference throughput, and the CPU smoke value `peak_vram_bytes=0`.  It deliberately records
GPU inference peak as unavailable rather than describing the CPU zero as a GPU measurement or
reconstructing a value that was not measured during the legacy run.  Re-running
`scripts/freeze_performance_v2_baseline.ps1` against the same output fails
instead of overwriting either baseline artifact.

The corrected companion has SHA256
`4c7905b0c36f9814bb7abc7cbc2602854b7f55bdc6e3a6c57319c3dcb37cd78f`. Its provenance
verifies the original snapshot SHA256
`a06c6da373e417c14a8d71344f819470e6c49affcc6a19b1a06b326872226480`, original hash-list
SHA256 `4477b56b7bbafc08a51d3c06a1c7433a4e90354da06c3c6ce80addeeb3e3f0bd`, and later
benchmark SHA256 `2f84862e78981e112d0061f9d7c0693dde093213b04e1167a0bfff01397e0016`.
Schema v3 is a companion, not a replacement or migration, so later validator results
cannot silently rewrite capture-time evidence.

The earlier schema-v2 companion is retained as failed audit evidence. It was
created while another process was rebuilding the active manifests and therefore
contains three transient hashes. It is explicitly superseded and must not be
used. Schema v3 freezes four byte-for-byte copies under
`artifacts/performance_v2/baseline_manifests/`; every copy is verified against
the original schema-v1 snapshot before v3 can be created.  A later read-only audit
also confirmed that all four bytes and hashes match the corresponding entries in
`baseline_files.sha256`; the hash-list itself remains bound as immutable evidence.

## Baseline capture state

At the immutable capture time the repository was not a Git repository and had
no completed competition-model training run.  The frozen checkpoint is a
two-epoch CD68 engineering smoke run using a 660,907-parameter Residual U-Net
trained on 16 images.  It is retained for compatibility and provenance, not
presented as a competition result.  Later V2 smoke checkpoints do not alter this
captured baseline.

The best available combination inside that checkpoint is the raw model with
D4 TTA.  On all 258 current validation rows it obtains float, uint8, and final
JPEG round-trip SSIM values of 0.406936, 0.406534, and 0.406125 respectively;
the corresponding PSNR values are 11.091808, 11.091856, and 11.092188 dB.  The
short-run EMA is substantially worse because it was updated for only sixteen
optimizer steps.  These values remain immutable inside
`baseline_snapshot.json`.

## Implemented benchmark rerun

After the V2 three-domain validator was implemented, `benchmark-baseline`
re-evaluated the same checkpoint and all 258 validation rows.  The persisted
result is `artifacts/performance_v2/baseline_benchmark.json`:

| weights / TTA | float SSIM / PSNR | uint8 SSIM / PSNR | JPG SSIM / PSNR |
|---|---:|---:|---:|
| raw / none | 0.394293 / 11.084787 | 0.393903 / 11.084804 | 0.393385 / 11.085175 |
| raw / D4 | 0.407567 / 11.092067 | 0.407169 / 11.092082 | 0.406767 / 11.092427 |
| EMA / none | 0.222996 / 9.678874 | 0.222812 / 9.678828 | 0.222246 / 9.678773 |
| EMA / D4 | 0.304139 / 9.808947 | 0.303858 / 9.808947 | 0.303197 / 9.809074 |

The frozen snapshot and the later benchmark were produced by different
evaluation revisions, so the snapshot is not rewritten and their values are not
mixed.  V2 experiment reports cite the later benchmark artifact; provenance
checks cite the immutable snapshot.  Both remain surrogate-split engineering
references rather than competition metrics.

## Validation limitation

All local image stems are anonymous numbers.  The current 32-image surrogate
groups do not represent verified ROIs, and image-border evidence shows spatially
continuous train/validation pairs.  These metrics are engineering references;
they cannot prove ROI-grouped improvement.  Context remains disabled until the
official `ROI_row_col` data or an authoritative coordinate mapping is present.

## Fair A0 status

The later sample screen trained the old base-32 `MultiMarkerRestorer` for 20
epochs on all 1,088 current train rows and evaluated all 258 validation rows.
A0 raw/no-TTA JPG SSIM/PSNR was 0.779860/24.470988.  A1 regressed slightly;
A2 showed a sample-only positive trend; A3 was blocked before training because
the filename grid is not authoritative.  These runs use the same surrogate
split that failed the ROI-neighborhood audit, so none is a promotable
ROI-grouped performance result and the retained default remains the rollback
configuration.

The strict-screen snapshot points at several paths that were intentionally extended
later: `p0.yaml` gained A4–A8 definitions, the ROI audit was regenerated, and the
experiment registry received new rows.  The immutable companion
`artifacts/performance_v2/strict_p0_screen_verification_20260716.json` therefore
does not pretend those mutable paths still contain their capture-time bytes.  It
re-verifies the original 10-file A0/A1/A2 run aggregates exactly, freezes the
capture-time registry prefix, confirms the train/validation manifests and screen
report, and marks the old P0 YAML and ROI-audit bytes as unavailable.  Its SHA256 is
`cf585d54b949082ac416556f1b36b80b131ce3b14ca014f6e226faa92d842df3`.
