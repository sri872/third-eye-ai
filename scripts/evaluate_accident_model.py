"""Reproducible Phase 13 audit/evaluation of the deployed accident classifier.

Reads only existing artifacts. Does not retrain, overwrite model weights, or
fit calibration on held-out test data. Output is written to
models/accident_evaluation_report.json as a documented audit artifact.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
METRICS_PATH = ROOT / "models" / "accident_classifier_metrics.json"
MODEL_PATH = ROOT / "models" / "accident_classifier.joblib"
EMBEDDINGS_PATH = ROOT / "data" / "embeddings.npz"
METADATA_PATH = ROOT / "data" / "frames" / "metadata.csv"
OUTPUT_PATH = ROOT / "models" / "accident_evaluation_report.json"


def dhash(image_path: Path, hash_size: int = 8) -> int:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read frame: {image_path}")
    resized = cv2.resize(image, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return value


def ece(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        mask = (scores >= edges[index]) & (scores < edges[index + 1])
        if not mask.any():
            continue
        bin_size = float(mask.sum())
        total += bin_size / len(scores) * abs(
            float(labels[mask].mean()) - float(scores[mask].mean())
        )
    return total


def metric_set(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict:
    predictions = (scores >= threshold).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    return {
        "threshold": threshold,
        "accuracy": round(float(accuracy_score(labels, predictions)), 4),
        "precision": round(float(precision_score(labels, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(labels, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(labels, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(labels, scores)), 4),
        "brier": round(float(brier_score_loss(labels, scores)), 4),
        "ece_10bins": round(ece(labels, scores), 4),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "false_positive_rate": round(float(fp / max(tn + fp, 1)), 4),
        "false_negative_rate": round(float(fn / max(tp + fn, 1)), 4),
    }


def main() -> None:
    metrics = json.loads(METRICS_PATH.read_text())
    embeddings = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    features = embeddings["features"]
    labels = np.asarray(embeddings["labels"], dtype=int)
    frame_ids = [str(value) for value in embeddings["frame_ids"]]
    video_ids = np.asarray([str(value) for value in embeddings["video_ids"]])

    metadata = pd.read_csv(METADATA_PATH)
    metadata["frame_id"] = metadata["frame_id"].astype(str)
    metadata["video_id"] = metadata["video_id"].astype(str)

    metadata_by_frame = metadata.set_index("frame_id")
    assert len(features) == len(frame_ids) == len(labels) == len(metadata)
    assert set(frame_ids) == set(metadata["frame_id"])
    assert all(
        video_ids[index] == metadata_by_frame.loc[frame_id, "video_id"]
        for index, frame_id in enumerate(frame_ids)
    )
    assert all(
        labels[index] == int(metadata_by_frame.loc[frame_id, "label"])
        for index, frame_id in enumerate(frame_ids)
    )

    video_sets = {
        "train": {str(value) for value in metrics["train_video_ids"]},
        "val": {str(value) for value in metrics["val_video_ids"]},
        "test": {str(value) for value in metrics["test_video_ids"]},
    }
    intersections = {
        "train_val": sorted(video_sets["train"] & video_sets["val"]),
        "train_test": sorted(video_sets["train"] & video_sets["test"]),
        "val_test": sorted(video_sets["val"] & video_sets["test"]),
    }
    assert not any(intersections.values()), "Video-level splits must be disjoint"

    split_of_video = {
        video_id: split for split, ids in video_sets.items() for video_id in ids
    }
    frame_splits = np.asarray(
        [split_of_video[metadata_by_frame.loc[frame_id, "video_id"]] for frame_id in frame_ids]
    )
    for split, count in metrics["splits"].items():
        if split.endswith("_frames"):
            key = split.replace("_frames", "")
            assert int((frame_splits == key).sum()) == count

    classifier = joblib.load(MODEL_PATH)
    raw_proba = classifier.predict_proba(features)[:, 1]
    decision = classifier.decision_function(features)

    # Platt scaling fitted ONLY on validation. Test scores are transformed, never fitted.
    platt = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    platt.fit(decision[frame_splits == "val"].reshape(-1, 1), labels[frame_splits == "val"])
    calibrated_proba = platt.predict_proba(decision.reshape(-1, 1))[:, 1]

    test_mask = frame_splits == "test"
    test_labels = labels[test_mask]
    test_scores = raw_proba[test_mask]
    reproduced = metric_set(test_labels, test_scores)
    published = {
        "accuracy": metrics["test_accuracy"],
        "precision": metrics["test_precision"],
        "recall": metrics["test_recall"],
        "f1": metrics["test_f1"],
        "roc_auc": metrics["test_roc_auc"],
        "confusion_matrix": metrics["confusion_matrix"],
    }
    matches_published = all(
        abs(float(reproduced[key]) - float(published[key])) <= 0.0001
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc")
    ) and reproduced["confusion_matrix"] == published["confusion_matrix"]
    assert matches_published, "Deployed model metrics must reproduce the published artifact"

    threshold_sweep = [
        metric_set(test_labels, test_scores, threshold)
        for threshold in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
    ]
    deployment_bands = {
        "scope": "held_out_test_frames",
        "likely_accident_ge_0.60": {
            "true_positive": int(((test_scores >= 0.60) & (test_labels == 1)).sum()),
            "false_positive": int(((test_scores >= 0.60) & (test_labels == 0)).sum()),
        },
        "uncertain_0.40_to_0.60": {
            "positive": int(
                ((test_scores >= 0.40) & (test_scores < 0.60) & (test_labels == 1)).sum()
            ),
            "negative": int(
                ((test_scores >= 0.40) & (test_scores < 0.60) & (test_labels == 0)).sum()
            ),
        },
        "no_signal_lt_0.40": {
            "false_negative": int(((test_scores < 0.40) & (test_labels == 1)).sum()),
            "true_negative": int(((test_scores < 0.40) & (test_labels == 0)).sum()),
        },
        "note": "Bands are descriptive only; thresholds were not tuned on held-out test data.",
    }

    # Exact and perceptual duplicate audit with split attribution.
    hashes: dict[str, str] = {}
    dhashes: dict[str, int] = {}
    frame_shapes: set[tuple[int, int]] = set()
    for rel in metadata["rel_path"].astype(str):
        path = ROOT / rel
        hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
        dhashes[rel] = dhash(path)
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        frame_shapes.add(tuple(image.shape[:2]))  # type: ignore[union-attr]

    exact_groups: dict[str, list[str]] = {}
    for rel, digest in hashes.items():
        exact_groups.setdefault(digest, []).append(rel)
    exact_duplicates = [group for group in exact_groups.values() if len(group) > 1]

    paths = list(dhashes)
    near_pairs: list[tuple[str, str]] = []
    for index, first in enumerate(paths):
        first_hash = dhashes[first]
        for second in paths[index + 1 :]:
            if int(first_hash ^ dhashes[second]).bit_count() <= 5:
                near_pairs.append((first, second))
    split_of_rel = {
        rel: split_of_video[metadata_by_frame.loc[Path(rel).stem, "video_id"]]
        for rel in paths
    }
    cross_split = [
        pair for pair in near_pairs if split_of_rel[pair[0]] != split_of_rel[pair[1]]
    ]

    class_counts = metadata["class_name"].value_counts().to_dict()
    results = {
        "objective": "Phase 13 model evaluation and calibration investigation",
        "metrics_source": str(METRICS_PATH.relative_to(ROOT)),
        "model": "frozen torchvision ResNet-18 embeddings + sklearn LogisticRegression(C=0.01)",
        "dataset": {
            "videos": int(metadata["video_id"].nunique()),
            "frames": int(len(metadata)),
            "class_counts": {str(key): int(value) for key, value in class_counts.items()},
            "train_videos": metrics["splits"]["train_videos"],
            "val_videos": metrics["splits"]["val_videos"],
            "test_videos": metrics["splits"]["test_videos"],
            "video_id_intersections": intersections,
            "labels_from_source_directory_only": True,
            "annotator_or_adjudication_metadata_present": False,
            "frame_shapes": [list(shape) for shape in sorted(frame_shapes)],
        },
        "published_held_out_metrics": published,
        "reproduced_held_out_raw": reproduced,
        "matches_published_metrics": matches_published,
        "held_out_test_threshold_sweep_reporting_only": threshold_sweep,
        "deployment_band_counts": deployment_bands,
        "validation_raw": metric_set(
            labels[frame_splits == "val"], raw_proba[frame_splits == "val"]
        ),
        "held_out_test_platt_validation_fit": metric_set(
            test_labels, calibrated_proba[test_mask]
        ),
        "calibration_decision": (
            "Validation-only Platt scaling is reproducible but is NOT deployed: validation "
            "was already used to select C, the validation set is small (180 frames), and the "
            "operating domain is out-of-distribution for still images. Runtime scores remain "
            "UNCALIBRATED ranking scores."
        ),
        "geometric_rule_evaluator": {
            "runtime_use": False,
            "reason": "2D bounding-box overlap is interpreted as physical interpenetration; perspective and occlusion cause false accident signals. The trained model is authoritative.",
        },
        "duplicates": {
            "exact_duplicate_groups": len(exact_duplicates),
            "dhash_hamming_le_5_pairs": len(near_pairs),
            "cross_split_near_duplicate_pairs": len(cross_split),
            "note": "Zero exact duplicates is expected; cross-split near-duplicate pairs indicate visual-leakage optimism risk even though video IDs are disjoint.",
        },
    }

    OUTPUT_PATH.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
