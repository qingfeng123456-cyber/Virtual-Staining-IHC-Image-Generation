# Project Rules

- Run the relevant automated tests before claiming that a change is complete.
- Do not leave TODO markers, empty `pass` statements, pseudocode, or unimplemented functions.
- Use `pathlib.Path` for filesystem paths and support Windows paths containing spaces or non-ASCII characters.
- Use only official competition data. Never download external training data or pretrained weights for the default pipeline.
- Test images must not participate in training, normalization statistics, self-supervision, pseudo-labeling, or model selection.
- After changing data discovery, manifests, datasets, or transforms, run the manifest, dataset, and smoke-pipeline tests.
- After changing a model or loss, run shape, backward, finite-loss, and smoke-train tests.
- After changing inference or submission code, run inference, submission, and ZIP-structure tests.
- Windows defaults to `num_workers=0`; increasing it must be an explicit configuration choice.
- Preserve original image dimensions, channel semantics, value range, filenames, and counts unless an audited conversion is explicitly recorded.
- Final reports must list commands actually executed and their real outcomes.
- Never claim a long training run or metric was completed unless it actually finished.
- If GPU execution is unavailable, explicitly state that only CPU smoke validation was completed.
- Official-test and smoke-submission artifacts must remain clearly separated.

## Performance V2 Rules

- Performance V2 is an incremental upgrade; preserve the stable legacy baseline and its immutable hash snapshot.
- Every new architecture, loss, sampler, optimizer, context, calibration, TTA, and ensemble feature must have an explicit feature flag and rollback path.
- Neighborhoods must never cross organ, split, or ROI boundaries, and context may use DAPI inputs only.
- A numeric stem or image-content edge match is audit evidence, not an authoritative ROI coordinate.
- Require a verified filename coordinate grid or authoritative mapping before enabling context in a promotable experiment.
- Select performance models using ROI-grouped final JPEG round-trip metrics; always report float and uint8 metrics as well.
- A module that has not passed ROI-grouped ablation remains disabled in the final default configuration.
- Never use test data to fit normalization, pretraining, model selection, calibrators, ensemble weights, or any other learned state.
- GAN and diffusion branches are not part of the default performance path.
- After neighborhood changes, run ROI parsing, neighborhood, context-mask, and context-transform tests.
- After CAMP model changes, run local/context shape, forward/backward, AMP, and smoke tests.
- After scheduled-loss changes, run finite-value, interpolation-continuity, and resume tests.
- After ensemble or soup changes, verify architecture compatibility, nonnegative weights, unit weight sum, and validation/OOF-only fitting.
- Record raw and EMA metrics independently; do not assume EMA is better.
- Do not claim a full train, A3 improvement, official submission, or leaderboard result that was not actually completed.
- Final reports must list commands run, real results, failed experiments, rollback decisions, and the exact retained configuration.
