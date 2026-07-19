# Research-to-code map for CAMP-VS v2

This project uses clean-room implementations built from published architectural
ideas.  No third-party research repository is vendored and no external model
weight is downloaded.

| Research direction | Project implementation | Status and boundary |
|---|---|---|
| ProtoMTG | Shared and marker-specific cosine prototypes, extended to multiple scales and optional organ banks | Reimplemented; prototype residuals are zero-initialized and monitored for usage collapse |
| PyramidPix2Pix | Fixed Gaussian/Laplacian pyramids and a base/detail reconstruction head | Adopted without adversarial training |
| NAFNet | Local high-resolution restoration blocks | Existing tested NAF blocks are reused |
| Restormer | Efficient low-resolution channel/spatial mixing | Reimplemented in pure PyTorch at 1/8 and 1/16 only |
| MambaIRv2 | Optional low-resolution state-space experiment | P2 only; not a core dependency and never silently substituted |
| HookNet | Separate local and wider-context branches | Used as a design principle for center/context fusion |
| UNIStainNet/PGVMS | Marker conditioning and spatial modulation | Marker/organ embeddings and identity-initialized FiLM; no external H&E foundation model |
| FAMO | Optional reconstruction-task balancing | P1 flag; equal weighting remains the reference |
| Registration literature | Audit-gated shift-tolerant training loss | Disabled when alignment audit shows paired data are already aligned |
| Model Soups | Architecture-compatible weight averaging | Validation-JPG-gated and separate from prediction ensemble |
| EnsIR/ensemble methods | Validation/OOF-only nonnegative prediction weights | No per-test-image fitting or target leakage |

The implementation records whether an idea is enabled, disabled, blocked by
data, or rejected by ablation.  License information for any future code-derived
component must be recorded before it enters the repository.

