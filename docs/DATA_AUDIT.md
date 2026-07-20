# Data audit

- Data root: `D:\\大学学习\\竞赛\\全球人工智能精英赛\\project\\dataset\\official`
- Audited training images: 6296
- ROI leakage verifiable from filenames: True
- Official test status: available

## Marker storage and value statistics

| Marker | Count | Modes | Logical channels | Mean | Std | Essentially grayscale RGB |
|---|---:|---|---:|---:|---:|---|
| DAPI | 6296 | RGB | 1 | 0.13741 | 0.13840 | True |
| HLA-DR | 6296 | RGB | 1 | 0.12971 | 0.13103 | True |
| CD45RO | 6296 | RGB | 1 | 0.13346 | 0.11767 | True |
| Vimentin | 6296 | RGB | 1 | 0.12799 | 0.13720 | True |
| CD68 | 6296 | RGB | 1 | 0.15200 | 0.11153 | True |

## Pairing and leakage

The manifest was paired by normalized stem, not directory traversal order. All
6296 training stems provided authoritative `ROI_row_col` coordinates. The local
split holds out five complete ROIs, with no ROI, canonical-key, file-hash, or
adjacent-neighborhood overlap between training and validation.

## Alignment

- HLA-DR: approximately_aligned; median offset (0, 0), shift augmentation=False.
- CD45RO: approximately_aligned; median offset (0, 0), shift augmentation=False.
- Vimentin: approximately_aligned; median offset (0, 0), shift augmentation=False.
- CD68: approximately_aligned; median offset (0, 0), shift augmentation=False.

No automatic registration was applied. Alignment diagnostics are reporting and
augmentation guidance only; labels remain byte-for-byte untouched.

## Test isolation

Official test inputs are excluded from all statistics, normalization, model
selection, and visualizations. The smoke manifest is held-out validation input
and is explicitly non-official.
