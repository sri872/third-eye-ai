# Phase 13 Report — Scene/Model Evaluation

Date: 2026-11-25  
Workspace: `/Users/srishanthbiradar/Projects/third-eye-v2`  
Python: `3.12.3`  
No retraining, threshold tuning on held-out data, dependency installation, or model overwrite was performed.

## 1. Reproducible evaluation artifact

A new read-only evaluation script was added:

- `scripts/evaluate_accident_model.py`

It loads only existing artifacts:

- `models/accident_classifier_metrics.json`
- `models/accident_classifier.joblib`
- `data/embeddings.npz`
- `data/frames/metadata.csv`
- `data/frames/**/*.jpg`

It writes a documented audit artifact:

- `models/accident_evaluation_report.json`

Exact command and output:

```text
.venv/bin/python scripts/evaluate_accident_model.py
exit=0
```

Saved output: `/private/var/folders/bn/2bj93s1s2zz_lshjg99fp0200000gn/T/phase13_evaluation_output.txt`.

The script asserts that dataset/video IDs/labels match the embedding arrays and that video-level train/validation/test IDs are disjoint before producing metrics.

## 2. Deployed model and dataset audit

- Model: frozen torchvision ResNet-18 embeddings + scikit-learn `LogisticRegression(C=0.01)`.
- Dataset: 400 videos, 1,200 frames, 600 positive and 600 negative frames.
- Splits: 280/60/60 videos and 840/180/180 frames.
- Train/val/test video ID intersections: empty for all pairs.
- All sampled frames are `224x224`.
- Labels are derived from source directory/class metadata only. No annotator identity, adjudication record, or accident-confirmation ground truth exists in the project.
- The geometric rule evaluator remains out of runtime use; the trained classifier remains authoritative.

## 3. Published metrics exactly reproduced

At the published default threshold `0.50`, on the 180 held-out test frames:

| Metric | Value |
|---|---:|
| Accuracy | 0.6778 |
| Precision | 0.6600 |
| Recall | 0.7333 |
| F1 | 0.6947 |
| ROC-AUC | 0.7452 |
| Brier score | 0.2066 |
| 10-bin ECE | 0.0855 |
| FPR | 0.3778 |
| FNR | 0.2667 |

Confusion matrix `[[TN, FP], [FN, TP]]`:

```text
[[56, 34],
 [24, 66]]
```

This exactly matches `models/accident_classifier_metrics.json`; the artifact records `matches_published_metrics: true`.

## 4. Threshold sweep (reporting only, not selected for deployment)

| Threshold | Accuracy | Precision | Recall | F1 | FPR | FNR |
|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 0.5667 | 0.5380 | 0.9444 | 0.6855 | 0.8111 | 0.0556 |
| 0.40 | 0.6278 | 0.5839 | 0.8889 | 0.7048 | 0.6333 | 0.1111 |
| 0.50 | 0.6778 | 0.6600 | 0.7333 | 0.6947 | 0.3778 | 0.2667 |
| 0.60 | 0.7056 | 0.7534 | 0.6111 | 0.6748 | 0.2000 | 0.3889 |
| 0.70 | 0.6444 | 0.7600 | 0.4222 | 0.5429 | 0.1333 | 0.5778 |
| 0.80 | 0.6167 | 0.8621 | 0.2778 | 0.4202 | 0.0444 | 0.7222 |

These values are descriptive. No threshold was chosen from this held-out sweep. The deployed UI bands remain unchanged:

- `>= 0.60` → `LIKELY ACCIDENT`
- `0.40–0.60` → `UNCERTAIN`
- `< 0.40` → `NO ACCIDENT SIGNAL`

On held-out test frames, those bands distribute as:

| Band | Counts |
|---|---|
| `>=0.60` | 55 true positives, 18 false positives |
| `0.40–0.60` | 25 positives, 39 negatives |
| `<0.40` | 10 false negatives, 33 true negatives |

The published metrics use threshold `0.50`, not the deployment band cutoffs.

## 5. Calibration feasibility

Validation-only Platt scaling was fitted on decision scores from the 180 validation frames and applied to held-out test scores. Test labels were never used for fitting.

Validation raw metrics:

- Accuracy `0.6611`, precision `0.6355`, recall `0.7556`, F1 `0.6904`, ROC-AUC `0.7030`.
- Brier `0.2247`, ECE `0.1047`.

Held-out test after validation-only Platt transform:

- Accuracy `0.7167`, precision `0.7349`, recall `0.6778`, F1 `0.7052`, ROC-AUC `0.7452`.
- Brier `0.2095`, ECE `0.1015`.

Decision: **not deployed**. Reasons:

1. Validation was already used to select `C=0.01`.
2. The validation set is small (180 frames).
3. Runtime stills/videos are out-of-domain relative to the dataset framing.
4. ECE after Platt remains `0.1015`, so it is not demonstrably reliable.

Runtime scores remain **uncalibrated ranking scores**, not accident probabilities.

## 6. Leakage and duplicate audit

Using the explicit 8-bit dHash implementation in the evaluation script:

- Exact byte-duplicate groups: `0`.
- Frame pairs with Hamming distance `<= 5`: `308`.
- Cross-split near-duplicate pairs: `124`.

Even though video IDs are disjoint, cross-split near-duplicates indicate a real visual-leakage risk that can make held-out metrics optimistic. The counts are from this reproducible implementation; no claim is made that all perceptual leakage has been removed.

## 7. Test evidence

Targeted Phase 13 suite:

```text
.venv/bin/pytest -q tests/test_model_evaluation.py \
  tests/test_accident_analysis.py tests/test_fusion.py
26 passed, 5 warnings in 13.74s
```

Saved output: `/private/var/folders/bn/2bj93s1s2zz_lshjg99fp0200000gn/T/phase13_targeted.txt`.

Full suite after Phase 13:

```text
.venv/bin/pytest --cache-clear -q
193 passed, 6 warnings in 74.17s (0:01:14)
```

Saved output: `/private/var/folders/bn/2bj93s1s2zz_lshjg99fp0200000gn/T/phase13_full_suite.txt`.

## 8. Honest limitations

- No labelled real accident video ground truth exists. Video-level accuracy remains unverifiable.
- The dataset labels come from source folders, not documented human adjudication.
- ROC-AUC `0.7452` indicates useful but imperfect ranking, not operational reliability.
- At default `0.50`, FPR is `0.3778`: many negatives are incorrectly flagged.
- No calibration artifact was deployed.
- No geometric rule may override the trained model.
- Git diff/status remain unavailable because the workspace has no `.git` directory.

## 9. Phase 13 outcome

Phase 13 is complete. The published held-out metrics were reproduced exactly, precision/recall/F1/ROC-AUC/confusion/FPR/FNR were documented, threshold bands were analyzed without tuning, near-duplicate leakage risk was quantified, and validation-only calibration feasibility was investigated but deliberately not deployed.
