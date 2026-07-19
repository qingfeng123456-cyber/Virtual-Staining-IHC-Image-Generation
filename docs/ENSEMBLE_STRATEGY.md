# Ensemble and model-soup strategy

Uniform prediction averaging is the first reference.  Learned weights are
nonnegative, sum to one, and are fitted only from validation or out-of-fold
predictions.  Global weights precede marker/organ-specific weights, and no test
image receives individually optimized weights.

Every learned-weight prediction and target array requires a JSON sidecar, but a CLI
`source` string and the sidecar are only assertions. The loader reads and hashes the
actual full manifest, the audited train/validation manifest pair, the ROI audit, and
the OOF fold assignment before it trusts an array. Sidecars must exactly match those
external anchors plus file/content hashes, role, shape/dtype, ordered sample keys and
organs, JPG domain, complete coverage, fold/group assignments, fold count, and unique
artifact ID. Strict validation requires full filename coordinates, verified direction
and boundary continuity, and no train/validation ROI or adjacent-patch leakage.
`test` and `official_test` are rejected even through the unsafe engineering path.

Grouped OOF requires every manifest sample exactly once, every fold present, and no
ROI group crossing folds. With `cross_validate_weights=true`, each fold's weights
are fitted on the other folds and scored only on held-out groups. Reports retain
sample/group counts, nonnegative unit-sum weights, learned/uniform scores, optimizer,
evaluation count, and fallback reason. Validation-only arrays cannot request this
grouped OOF cross-validation.

Weight-space soup is limited to identical architectures, state-dict keys,
shapes, initialization lineage, and compatible prototype/calibrator state.
Greedy soup starts with the best validation-JPG member and keeps another member
only when complete validation does not regress.  Soup is revalidated from
scratch and remains distinct from prediction ensemble.

Strict soup is default (`allow_unsafe_model_soup_lineage=false`). Missing legacy
provenance, mixed raw/EMA/SWA sources, duplicate members, architecture tampering,
different lineages, different non-floating buffers, truncated/non-JPG validation, or
an unverified ROI audit are rejected. Supplied ranking scores are ignored for model
selection: every individual member and greedy combination is re-evaluated on the full
JPG protocol, then the final soup is validated again. The checkpoint binds exact
manifest, ROI audit, and per-image evidence hashes. Explicit unsafe lineage or
validation flags force marked filenames/reports/checkpoints; such output is rejected
by a later strict soup.

Reports compare raw, EMA, optional SWA, soup, and prediction ensemble.  Range-wise
selection is a P2 experiment and remains disabled unless cross-validated gains
are stable.

These contracts have engineering-test coverage, but no official grouped OOF arrays,
formal A8 ensemble run, or formal soup performance result exists locally. Neither
learned weights nor soup enters the retained configuration.
