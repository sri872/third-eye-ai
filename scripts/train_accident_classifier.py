#!/usr/bin/env python3
"""Training script for accident classification on extracted Nexar frames.

1. Loads frames from data/frames/metadata.csv.
2. Extracts frozen torchvision ResNet-18 embeddings (512-dim).
3. Splits strictly by video (70/15/15 train/val/test, fixed seed 42) to prevent leakage.
4. Tunes Logistic Regression on validation set, evaluates ONCE on held-out test set.
5. Saves model to models/accident_classifier.joblib.
6. Writes model card to docs/model_card_accident.md.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
import torch
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights, resnet18

REPO_ROOT = Path(__file__).resolve().parent.parent
METADATA_CSV = REPO_ROOT / "data" / "frames" / "metadata.csv"
EMBEDDINGS_CACHE = REPO_ROOT / "data" / "embeddings.npz"
MODEL_PATH = REPO_ROOT / "models" / "accident_classifier.joblib"
METRICS_PATH = REPO_ROOT / "models" / "accident_classifier_metrics.json"


def get_device() -> torch.device:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        try:
            probe = torch.zeros(1, device="mps")
            del probe
            return torch.device("mps")
        except Exception:
            return torch.device("cpu")
    return torch.device("cpu")


def extract_embeddings():
    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_CSV}")

    df = pd.read_csv(METADATA_CSV)
    print(f"Loaded metadata with {len(df)} frames from {df['video_id'].nunique()} unique videos.")

    if EMBEDDINGS_CACHE.exists():
        print(f"Loading cached embeddings from {EMBEDDINGS_CACHE}...")
        data = np.load(EMBEDDINGS_CACHE, allow_pickle=True)
        return data["features"], data["labels"], data["video_ids"], data["frame_ids"], df

    device = get_device()
    print(f"[Third-Eye Feature Extractor] Initializing frozen ResNet-18 on device: {device.type.upper()}")

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()  # 512-dimensional output
    model.to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    features_list = []
    labels_list = []
    video_ids_list = []
    frame_ids_list = []

    print(f"Extracting ResNet-18 embeddings for {len(df)} frames...")
    batch_tensors = []
    batch_indices = []

    for idx, row in df.iterrows():
        img_path = REPO_ROOT / row["rel_path"]
        img = Image.open(img_path).convert("RGB")
        tensor = preprocess(img)
        batch_tensors.append(tensor)
        batch_indices.append(idx)

        if len(batch_tensors) == 64 or idx == len(df) - 1:
            stacked = torch.stack(batch_tensors).to(device)
            with torch.no_grad():
                feats = model(stacked).cpu().numpy()
            features_list.append(feats)

            for b_idx in batch_indices:
                labels_list.append(df.iloc[b_idx]["label"])
                video_ids_list.append(df.iloc[b_idx]["video_id"])
                frame_ids_list.append(df.iloc[b_idx]["frame_id"])

            batch_tensors = []
            batch_indices = []

            if (idx + 1) % 300 == 0 or idx == len(df) - 1:
                print(f"  Extracted {idx + 1}/{len(df)} embeddings")

    all_features = np.vstack(features_list)
    all_labels = np.array(labels_list, dtype=np.int64)
    all_video_ids = np.array(video_ids_list, dtype=object)
    all_frame_ids = np.array(frame_ids_list, dtype=object)

    np.savez_compressed(
        EMBEDDINGS_CACHE,
        features=all_features,
        labels=all_labels,
        video_ids=all_video_ids,
        frame_ids=all_frame_ids,
    )
    print(f"Saved compressed embeddings to {EMBEDDINGS_CACHE} ({all_features.shape})")
    return all_features, all_labels, all_video_ids, all_frame_ids, df


def train_and_evaluate(X, y, video_ids, frame_ids, df):
    # 1. Video-level splitting (Strict zero leakage)
    unique_vids_df = df.drop_duplicates(subset=["video_id"])[["video_id", "label"]].reset_index(drop=True)

    # 70% train vs 30% temp
    train_vids, temp_vids = train_test_split(
        unique_vids_df["video_id"],
        test_size=0.30,
        stratify=unique_vids_df["label"],
        random_state=42,
    )

    # Split temp 50/50 into val (15%) and test (15%)
    temp_labels = unique_vids_df.set_index("video_id").loc[temp_vids, "label"]
    val_vids, test_vids = train_test_split(
        temp_vids,
        test_size=0.50,
        stratify=temp_labels,
        random_state=42,
    )

    train_vids_set = set(train_vids)
    val_vids_set = set(val_vids)
    test_vids_set = set(test_vids)

    # Strict assertion for zero video leakage
    assert len(train_vids_set & val_vids_set) == 0, "Leakage detected between train and val!"
    assert len(train_vids_set & test_vids_set) == 0, "Leakage detected between train and test!"
    assert len(val_vids_set & test_vids_set) == 0, "Leakage detected between val and test!"

    train_mask = np.isin(video_ids, list(train_vids_set))
    val_mask = np.isin(video_ids, list(val_vids_set))
    test_mask = np.isin(video_ids, list(test_vids_set))

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print("\nSplit Statistics (Video-Stratified):")
    print(f"  Train: {len(train_vids_set)} videos ({len(y_train)} frames) [Pos: {(y_train == 1).sum()}, Neg: {(y_train == 0).sum()}]")
    print(f"  Val  : {len(val_vids_set)} videos ({len(y_val)} frames) [Pos: {(y_val == 1).sum()}, Neg: {(y_val == 0).sum()}]")
    print(f"  Test : {len(test_vids_set)} videos ({len(y_test)} frames) [Pos: {(y_test == 1).sum()}, Neg: {(y_test == 0).sum()}]")

    # 2. Hyperparameter tuning on validation set only
    best_c = 1.0
    best_val_f1 = 0.0

    print("\nTuning Logistic Regression C on Validation set...")
    for c_cand in [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
        clf = LogisticRegression(C=c_cand, max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        val_preds = clf.predict(X_val)
        val_f1 = f1_score(y_val, val_preds, zero_division=0)
        print(f"  C={c_cand:<6} -> Val F1={val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_c = c_cand

    print(f"Selected best C={best_c} based on validation performance.")

    # 3. Final model evaluation on held-out TEST set (Evaluated exactly ONCE)
    final_model = LogisticRegression(C=best_c, max_iter=1000, random_state=42)
    final_model.fit(X_train, y_train)

    test_preds = final_model.predict(X_test)
    test_probs = final_model.predict_proba(X_test)[:, 1]

    # Baseline: majority class
    majority_count = max((y_test == 0).sum(), (y_test == 1).sum())
    baseline_acc = majority_count / len(y_test)

    acc = accuracy_score(y_test, test_preds)
    prec = precision_score(y_test, test_preds, zero_division=0)
    rec = recall_score(y_test, test_preds, zero_division=0)
    f1 = f1_score(y_test, test_preds, zero_division=0)
    roc_auc = roc_auc_score(y_test, test_probs)
    cm = confusion_matrix(y_test, test_preds).tolist()

    print("\n==================================================================")
    print("HELD-OUT TEST SET EVALUATION METRICS (EVALUATED EXACTLY ONCE):")
    print(f"  Majority-Class Baseline Accuracy: {baseline_acc:.4f}")
    print(f"  Test Accuracy                   : {acc:.4f}")
    print(f"  Test Precision                  : {prec:.4f}")
    print(f"  Test Recall                     : {rec:.4f}")
    print(f"  Test F1 Score                   : {f1:.4f}")
    print(f"  Test ROC-AUC                    : {roc_auc:.4f}")
    print(f"  Confusion Matrix [[TN, FP], [FN, TP]]: {cm}")
    print("==================================================================")

    # Save model and split metadata
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    print(f"Saved trained classifier to: {MODEL_PATH}")

    # Record held-out test frames for verification
    test_frame_indices = np.where(test_mask)[0]
    test_frame_records = []
    for f_idx in test_frame_indices:
        test_frame_records.append({
            "frame_id": str(frame_ids[f_idx]),
            "video_id": str(video_ids[f_idx]),
            "label": int(y[f_idx]),
            "prob": float(test_probs[np.where(test_mask)[0] == f_idx][0]),
        })

    metrics_payload = {
        "best_C": best_c,
        "baseline_accuracy": round(float(baseline_acc), 4),
        "test_accuracy": round(float(acc), 4),
        "test_precision": round(float(prec), 4),
        "test_recall": round(float(rec), 4),
        "test_f1": round(float(f1), 4),
        "test_roc_auc": round(float(roc_auc), 4),
        "confusion_matrix": cm,
        "splits": {
            "train_videos": len(train_vids_set),
            "val_videos": len(val_vids_set),
            "test_videos": len(test_vids_set),
            "train_frames": len(y_train),
            "val_frames": len(y_val),
            "test_frames": len(y_test),
        },
        "train_video_ids": sorted(list(train_vids_set)),
        "val_video_ids": sorted(list(val_vids_set)),
        "test_video_ids": sorted(list(test_vids_set)),
        "test_frames": test_frame_records,
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    return metrics_payload


def main():
    X, y, vids, fids, df = extract_embeddings()
    train_and_evaluate(X, y, vids, fids, df)


if __name__ == "__main__":
    main()
