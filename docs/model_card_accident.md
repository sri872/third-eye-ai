# Model Card: ResNet-18 + Logistic Regression Accident Classifier

**Model Name**: Third-Eye ResNet-18 Logistic Regression Accident Classifier  
**Task**: Single-frame binary classification of vehicular collision presence / imminent accident signal.  
**Version**: 1.0.0 (Phase 3b)  
**Date**: 2026-11-24  

---

## 1. Model Details

- **Architecture**:
  - **Feature Extractor**: Frozen torchvision ResNet-18 (`torchvision.models.resnet18` with ImageNet-1K pretrained weights `ResNet18_Weights.DEFAULT`). Final classification layer replaced with `torch.nn.Identity()`, producing 512-dimensional continuous feature embeddings.
  - **Classifier**: Scikit-Learn `LogisticRegression(C=0.01, max_iter=1000, random_state=42)`.
- **Inference Hardware**: Apple Silicon MPS / CPU fallback for feature extraction; Scikit-Learn on CPU.
- **Artifact Location**: `models/accident_classifier.joblib` (Classifier) and cached PyTorch checkpoints.

---

## 2. Training & Evaluation Data

- **Source Dataset**: Nexar Dashcam Crash Prediction Dataset (`~/Datasets/ThirdEye/Nexar/test-public/`).
- **Data Extracted**:
  - Sampled from 200 positive (accident/imminent collision) videos and 200 negative (normal driving) videos.
  - Exactly 3 frames sampled per video at uniform relative timeline positions: **60%, 75%, and 90%** of total video duration.
  - Total frames: **1,200** (600 positive, 600 negative).
  - Image dimensions: 224x224 RGB JPEG format.
- **Split Strategy (Strict Zero Leakage)**:
  - Splitting was performed **strictly by video ID** (never by frame), ensuring all 3 frames of a video exist in exactly one split.
  - Stratified 70/15/15 split with fixed seed `42`:
    - **Train Set**: 280 videos (840 frames) — 420 positive, 420 negative.
    - **Validation Set**: 60 videos (180 frames) — 90 positive, 90 negative.
    - **Test Set**: 60 videos (180 frames) — 90 positive, 90 negative.
  - Video ID overlap: **0 videos shared across splits** (proven by set intersection tests).

---

## 3. Evaluation Metrics (Held-Out Test Set)

Evaluated exactly **once** on the held-out test split (180 frames from 60 unseen videos):

| Metric | Result | Baseline (Majority Class) |
|---|:---:|:---:|
| **Majority-Class Baseline** | 50.00% | 50.00% |
| **Accuracy** | **67.78%** | 50.00% |
| **Precision** | **66.00%** | — |
| **Recall** | **73.33%** | — |
| **F1 Score** | **0.6947** | 0.5000 |
| **ROC-AUC** | **0.7452** | 0.5000 |

### Confusion Matrix (Test Set):
```text
                  Predicted Negative    Predicted Positive
Actual Negative           56 (TN)               34 (FP)
Actual Positive           24 (FN)               66 (TP)
```

---

## 4. Decision Thresholds & Output Calibration

Because probabilities represent uncalibrated logistic sigmoid outputs on dashcam embeddings:
- **`LIKELY ACCIDENT`**: $P(\text{accident}) \ge 0.60$
- **`UNCERTAIN`**: $0.40 \le P(\text{accident}) < 0.60$
- **`NO ACCIDENT SIGNAL`**: $P(\text{accident}) < 0.40$

The rule-based geometric overlap features are retained strictly as **explanation text** and are never presented as the verdict.

---

## 5. Limitations & Caveats

1. **Dataset Bias & Generalization**:
   - The model was trained on forward-facing dashcam footage from the Nexar dataset. It may not generalize reliably to high-angle CCTV or overhead surveillance angles.
2. **Imminent vs Post-Impact**:
   - Nexar positive clips terminate within 500ms–1500ms of collision. Frames capture imminent impact trajectories; post-impact wreckage without dynamic approach motion may have lower sensitivity.
3. **Uncalibrated Probabilities**:
   - Outputs must be interpreted as ranking scores rather than absolute true-world accident probabilities.
