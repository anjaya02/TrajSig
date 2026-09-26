# New Phase-0 protocol: #5 Training Diagnostics

This is a new experiment on the already-completed 30-run telemetry archive. It does not revise the #7 signature verdict. No training or signature/log-signature extraction is permitted.

## Fixed tasks and data

- Evaluate prefixes ending after epochs 3, 5, 10, and 15; no feature may use a later epoch.
- Benign: C0/C1/C2. Primary harmful: C3/C4. C5 is scored and reported separately, never fitted or calibrated as a known class.
- Five outer leave-one-seed-identity-out folds. All five or six conditions of the held-out seed stay together.
- Internal channels are the existing eight-channel global aggregate: weight RMS, gradient RMS, update RMS, update/weight ratio, gradient mean/std, near-zero fraction, and gradient Gini. Signed `log1p(abs(x))` is applied independently to these channels before summarizing. Ordinary baseline uses train loss, train accuracy, and test accuracy; learning rate is excluded. No condition ID enters the feature vector.
- Multivariate prefixes use the mean and final observed value of each channel. Single-feature controls use only the prefix mean of gradient RMS or update/weight ratio.

## Five fixed scores

1. `ordinary_logistic`: regularized logistic regression on six ordinary loss/accuracy summary features.
2. `gradient_threshold`: absolute standardized deviation of prefix-mean gradient RMS from pooled benign training runs.
3. `update_ratio_threshold`: the same one-scalar rule for prefix-mean update/weight ratio.
4. `telemetry_logistic`: the same logistic model as (1) on 16 internal-telemetry summary features. This is the primary supervised candidate.
5. `telemetry_envelope`: a benign-only, three-prototype envelope on the same 16 telemetry features. Prototypes are the training-seed means for C0, C1, and C2. Distance is the minimum root-mean-square standardized distance; scales use pooled within-prototype variability with a 25%-of-global-scale floor. Condition labels are used only to fit the benign prototypes, not supplied at prediction time.

Logistic regression uses a training-only standard scaler, `C=0.1`, `class_weight=balanced`, and no hyperparameter search. All scores are oriented so larger means more warning.

## Seed calibration and readout

Inside each outer fold, fit each score four more times, leaving out one of the four training seed identities each time. Score the three benign runs of that inner-held-out seed, giving 12 cross-fitted benign calibration scores. The operational threshold is the *higher* empirical 90th percentile of these 12 scores; issue a warning only for a score strictly above it. Then fit the score on all four outer-training seeds and score all six conditions of the untouched outer-held-out seed. This is an empirical seed-block calibration, **not** a claim of finite-sample conformal coverage: only four seed blocks and 12 benign calibration runs are available.

Report pooled held-out false alarms among 15 benign runs, separate C3/C4 detection among five runs each, and C5 warning frequency among five runs. Also report fold-level variation, pooled and fold-wise AUROC/AUPRC on C0–C4, cross-fitted calibration exceedances, and first warning among the four checkpoints (right-censored after epoch 15). The 5% operating point is not meaningfully resolvable: one of 15 benign held-out runs is already 6.7%. The primary operational comparison is therefore nominal 10% calibration. As a *descriptive ranking comparison only*, report harmful sensitivity at an oracle matched budget of at most one false alarm among the 15 held-out benign runs; that threshold uses test labels and must never be presented as deployable calibration.

## Decision rule fixed before scoring

GO requires, at or before epoch 10, a pre-specified multivariate telemetry score with at most one false alarm among the 15 held-out benign runs, at least three of five C3 and three of five C4 runs warned, and at least two more of the ten harmful runs warned than **each** ordinary or single-feature comparator at the same one-of-15 false-alarm budget. The effect must not hinge on a single held-out seed. KILL if the best simple comparator essentially matches telemetry (within one harmful run at matched false-alarm budget), benign optimizer/schedule changes generate repeated false alarms, or useful detection appears only at epoch 15. UNCERTAIN is reserved for one concrete unresolved issue with one specified follow-up. The four checkpoints and five scores above are the full search grid; no post-hoc variants will be added to rescue a result.
