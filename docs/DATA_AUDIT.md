# Data audit

- Data root: `D:\大学学习\竞赛\全球人工智能精英赛\project\dataset\dataset_sample`
- Audited training images: 1346
- ROI leakage verifiable from filenames: False
- Official test status: missing

## Marker storage and value statistics

| Marker | Count | Modes | Logical channels | Mean | Std | Essentially grayscale RGB |
|---|---:|---|---:|---:|---:|---|
| DAPI | 1346 | RGB | 1 | 0.14819 | 0.14609 | True |
| HLA-DR | 1346 | RGB | 1 | 0.13924 | 0.14698 | True |
| CD45RO | 1346 | RGB | 1 | 0.13864 | 0.13351 | True |
| Vimentin | 1346 | RGB | 1 | 0.14102 | 0.14677 | True |
| CD68 | 1346 | RGB | 1 | 0.16097 | 0.12051 | True |

## Pairing and leakage

The manifest was paired by normalized stem, not directory traversal order. When real ROI identifiers were absent, consecutive numeric stems were grouped into explicit surrogate blocks; this reduces adjacent-patch leakage but cannot prove true ROI separation.

## Alignment

- HLA-DR: approximately_aligned; median offset (0, 0), shift augmentation=False.
- CD45RO: approximately_aligned; median offset (0, 0), shift augmentation=False.
- Vimentin: approximately_aligned; median offset (0, 0), shift augmentation=False.
- CD68: approximately_aligned; median offset (0, 0), shift augmentation=False.

No automatic registration was applied. Alignment diagnostics are reporting and augmentation guidance only; labels remain byte-for-byte untouched.

## Test isolation

Official test inputs are excluded from all statistics, normalization, model selection, and visualizations. The smoke manifest is held-out validation input and is explicitly non-official.
