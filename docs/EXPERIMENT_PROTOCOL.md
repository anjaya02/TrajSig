# Precommitted Phase-0 protocol

## Primary endpoint

Mean balanced accuracy of six-way condition classification across five leave-one-seed-identity-out folds, using internal-only telemetry and fixed multinomial logistic regression (`C=1`). Macro F1, per-fold results, predictions, confusion matrices, and per-condition errors are retained.

## Representation grid

- Endpoint.
- Seven-statistic per-channel summary.
- Eight-point downsampled and flattened trajectory.
- Signature and log-signature, depths 1–4.
- Every signature with and without explicit normalized time.
- Primary global trajectories in normal, reversed, and three deterministic shuffled orders.
- The stage-update-ratio sensitivity construction in normal order.
- Internal-only and internal-plus-ordinary-metrics versions are separate.

No depth is selected after looking at test performance. Signature level zero is omitted because it is constant.

## Secondary endpoints

- Seed identity under leave-one-condition-out evaluation.
- Training-fold standardized within-condition versus between-condition distances.
- Feature dimension, dense float64 bytes, raw compressed telemetry bytes, and extraction time.

## Temporal semantics

Time augmentation occurs before the temporal permutation. Reversal and shuffle therefore move the original time-value pairs together. This asks whether the representation relies on the actual ordered history rather than assigning fresh monotonic positions after scrambling.

## Decision rules

KILL if signatures fail on unseen seeds, mainly identify seeds, tie simple baselines without a meaningful compression benefit, lack temporal-order effects, or impose uncompensated cost.

UNCERTAIN if useful signal exists but baseline ties, seed variance, condition-specific failure, low sample size, or a concrete confound prevents a reliable conclusion. Exactly one highest-information follow-up may then be selected.

GO only for repeatable held-out-seed signal with a signature-specific advantage, meaningful temporal/time behavior, or a fair and substantial information/compression advantage.

Smoke outputs are categorically ineligible for these rules.

## Known limitations fixed in advance

- Five seed identities yield imprecise uncertainty estimates.
- Telemetry is once per epoch; within-epoch geometry is unobserved.
- Gradient moments describe the final minibatch, not an epoch mean.
- Gini uses at most 4,096 deterministic, evenly spaced elements per tensor.
- C5 necessarily changes example exposure, although optimizer steps and trajectory lengths are fixed.
- One dataset and architecture cannot establish generality.

