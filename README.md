# Optimization-Trajectory Signatures: Falsifiable Phase-0

This repository tests whether path signatures/log-signatures of neural-network optimization telemetry preserve training-history information that generalizes across random seeds. It is an early kill test, not a demonstration and not a security or proof-of-training system.

Current project state: implementation, automated tests, and a CPU smoke experiment are intended to be run locally; the scientific CIFAR-10 experiment requires the generated Colab workflow. No KILL/UNCERTAIN/GO verdict is valid until all 30 full runs and Stage-5 analysis are complete.

## Experimental design

The full design uses CIFAR-10, a CIFAR-adapted ResNet-18 trained from scratch for 30 epochs, six controlled conditions, and five seed identities (30 runs). Thirty epochs expose an early/middle/late schedule, including both C2 step decays at epochs 10 and 20, while keeping 900 total T4 training epochs within Phase-0 scale. Every run uses 390 full 128-example optimizer steps per epoch; the C5 subset loader cycles so dataset size cannot identify the condition through batch size, update count, or trajectory length.

Conditions:

- C0: SGD, cosine schedule.
- C1: AdamW, cosine schedule.
- C2: SGD, step schedule at epochs 10 and 20.
- C3/C4: deterministic, nested 10%/20% symmetric label corruption.
- C5: deterministically remove half of examples from CIFAR classes 0 and 1. This changes class composition while preserving all other classes, architecture, transforms, epochs, and update counts. Removed indices are recorded.

The raw telemetry tensor has shape `epochs × selected_parameter_tensors × 8`. Selected tensors are trainable weights with at least two dimensions. Statistics are weight norm, last-minibatch gradient norm, epoch displacement norm, displacement/weight ratio, gradient mean, gradient SD, near-zero fraction, and a bounded deterministic gradient-Gini estimate. Full gradients are never retained.

The primary path is an interpretable 8-channel parameter-count-aware global aggregation. `stage_update_ratios` adds stage-level update/weight ratios as an aggregation sensitivity check. Ordinary loss, accuracy, test accuracy, and learning rate are excluded from the primary analysis and added only in a labelled control.

## Leakage controls

- Run files use opaque deterministic hashes; condition and seed are not in filenames.
- Condition/seed metadata and affected data indices live outside telemetry arrays.
- Leave-one-seed-identity-out is mandatory for condition prediction.
- Seed identity is evaluated with leave-one-condition-out folds.
- Path normalization is fit on training runs only, before nonlinear signature extraction.
- Classifier feature scaling is independently fit on training representations only.
- Split-specific representation caches include training run IDs and normalizer values in their cache key.
- All runs have compatible feature definitions and equal trajectory length.

## Local setup and tests

Python 3.10–3.13 is supported. A clean environment is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
pytest
```

`esig` is wrapped only by `src/trajsig/signatures/adapter.py`; the dependency can be replaced without changing evaluation code.

## Smoke experiment (not scientific evidence)

```powershell
python -m trajsig pipeline --config configs/smoke.yaml --output results/smoke
```

This trains a tiny CNN on deterministic synthetic images for three epochs, three conditions, and two seed identities. It validates training, resume/cache behavior, telemetry, signatures, held-out evaluation, audits, figures, and reporting. Its scores must not be used for a scientific verdict.

## Full T4 experiment

This machine has no suitable local GPU. Use [notebooks/phase0_colab.ipynb](notebooks/phase0_colab.ipynb) with `phase0_source.zip`:

1. Upload `phase0_colab.ipynb` to Google Colab and open it.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Run all cells. Authorize the Google Drive mount. The setup cell prints the downloaded source ZIP's SHA-256 hash and rejects a stale bundle.
4. The notebook downloads the current `phase0_source.zip` from this repository. If GitHub is unavailable, upload the ZIP when prompted.
5. Leave the output directory at `MyDrive/trajsig_phase0_results`. A disconnected session can be resumed by running all cells again; complete runs are skipped and an interrupted run resumes at its last saved epoch checkpoint. Work after the last saved epoch must repeat.
6. Wait for all 30 runs and the audit to complete.
7. Download `phase0_training_artifacts.zip` from the displayed Drive path.
8. Return exactly that archive to this repository/conversation. Stage 5 will regenerate fold-specific representations and perform the scientific analysis here.

The notebook refuses to train without CUDA and prints the actual device. It prints startup status, a heartbeat every 60 seconds, the current run and a batch count every 50 training steps, then an overall epoch count and rough ETA after each saved epoch. `MyDrive/trajsig_phase0_results/progress.json` records stages such as `checking_saved_runs`, `setting_up_run`, and `training`, as well as the last saved epoch. A checkpoint protects an in-progress run at every epoch boundary. Closing Colab or losing its runtime can interrupt execution; reopen it and run all cells to resume. Colab runtime availability is controlled by Google and cannot be guaranteed by the notebook.

## Manual commands

```bash
python -m trajsig train --config configs/full.yaml --output /path/to/persistent/results
python -m trajsig audit --config configs/full.yaml --output /path/to/persistent/results
python -m trajsig evaluate --config configs/full.yaml --output /path/to/persistent/results
python -m trajsig report --config configs/full.yaml --output /path/to/persistent/results
python -m trajsig package --config configs/full.yaml --output /path/to/persistent/results
```

`train` is resumable and skips complete hashed run IDs. Raw telemetry, manifests, data-index records, split-specific derived representations, evaluation tables, figures, and reports are stored in separate subdirectories.

## Output schema

```text
results/<experiment>/
  experiment_config.json
  progress.json
  audit.json
  telemetry/<opaque-run-id>.npz
  manifests/<opaque-run-id>.json
  data_specs/<opaque-run-id>.{json,npz}
  checkpoints/<opaque-run-id>.pt
  representations/<split-cache-key>.npz
  evaluation/{per_fold,aggregate,predictions,per_class,distance,compression}_results.csv
  figures/*.png
  reports/*.md
```

The representation cache is deliberately fold-specific: signatures of standardized paths cannot be safely computed once using normalization estimated from the entire dataset.

## Interpretation discipline

All configured depths are reported. A small accuracy advantage with overlapping uncertainty is a tie. A signature result is only interesting if it robustly beats simple competitors, reveals real temporal-order/time behavior, or offers a fair and material compression tradeoff. Strong seed prediction, shuffled-order parity, ordinary-metric dominance, or baseline parity are evidence against the hypothesis.

Until full compute is returned: **FULL PHASE-0 COMPUTE PENDING.**
