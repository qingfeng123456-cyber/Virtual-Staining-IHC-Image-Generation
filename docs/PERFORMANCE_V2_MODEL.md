# CAMP-VS v2 model

CAMP-VS v2 is an incremental restoration model.  Its center branch processes a
full 256×256 DAPI patch with the existing NAF blocks and exposes 1/2, 1/4, 1/8,
and 1/16 features.  A small shared context encoder converts a verified 3×3 DAPI
neighborhood into nine tokens.  Missing neighbors copy the center pixels but
remain masked, so they cannot affect attention or pooling.

Context enters the center branch through zero-initialized FiLM.  The initial
context-enabled model is therefore numerically equivalent to local-only at the
fusion residuals.  Bottleneck cross-attention is a separate flag and belongs to
A4, not A3.  Global restoration mixing is restricted to low-resolution stages.

Marker and organ embeddings condition lightweight residual adapters.  Unknown
organ is a real embedding index, while a single-organ configuration can disable
organ conditioning.  Hierarchical shared/marker/organ prototypes are optional
and expose attention and usage diagnostics.

Prototype diagnostics are separate from model selection and attention-image export
is disabled by default. When enabled, a fixed seed and canonical-key SHA256 select a
bounded validation subset; every RGB PNG and the deterministic manifest are hashed.
The real A2 contract run used four validation samples and produced 64 maps (eight
shared plus eight CD68-task maps per sample). Its manifest SHA256 is
`366507a3d2ac3613df9c55c7fd44cc724fd4e0c35fc97680fae9dc0145cbbb5d`.
Test and official-test metadata are rejected before any image is written.

The decoder returns low-frequency base logits, bounded detail logits, final
predictions, and deep supervision.  A bounded calibrator predicts logit gain and
bias and is initialized to exact identity.  Every branch can be disabled without
changing legacy model or checkpoint behavior.

Prediction ensemble and model soup are external evaluation tools, not hidden CAMP-VS
branches. Ensemble fitting requires externally anchored manifest/ROI-audit/fold CSV
evidence plus matching sidecars; model soup requires full JPG per-image validation
evidence. Neither tool may infer safety from a CLI source string.
Soup requires matching architecture, state schema, initialization lineage, and
weight source by default. Unsafe lineage override is visibly marked and cannot be a
promotable default. Neither tool currently has a formal A8 performance result.
