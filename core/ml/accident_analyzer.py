"""Accident analyzer module for Project Third-Eye (Phase 3b).

Combines:
1. Primary classification signal: Trained Logistic Regression on frozen ResNet-18 embeddings
   from the real Nexar collision dataset (ROC-AUC 0.7452 >= 0.70 threshold).
   Verdicts:
     - prob >= 0.60: "LIKELY ACCIDENT"
     - 0.40 <= prob < 0.60: "UNCERTAIN"
     - prob < 0.40: "NO ACCIDENT SIGNAL"
2. Object detection (Faster R-CNN ResNet-50 FPN) for vehicle counting and person detection.
3. Proximity calculation: counts people whose bounding boxes touch/intersect a vehicle box
   (persons_near_vehicles_count).
4. Geometric features (IoU, aspect ratios) used strictly as EXPLANATION text; never dictating the verdict.
5. Produces AccidentEvent schema with data_source="REAL", method="TRAINED_MODEL".
"""

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import uuid

import cv2
import joblib
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights, resnet18

from core.cv.detector import detect
from core.ml.accident_features import extract_accident_features
from core.ml.evidence_assessment import build_accident_assessment
from core.schemas import AccidentEvent, Detection
from db.database import save_accident_event

logger = logging.getLogger("third_eye.ml.accident_analyzer")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = REPO_ROOT / "models" / "accident_classifier.joblib"

# Global lazy-loaded cache
_EXTRACTOR_CACHE = {}


def get_device() -> torch.device:
    """Detect available compute device with MPS and CPU fallback."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        try:
            probe = torch.zeros(1, device="mps")
            del probe
            return torch.device("mps")
        except Exception:
            return torch.device("cpu")
    return torch.device("cpu")


def load_model_and_extractor():
    """Lazy load the frozen ResNet-18 extractor and Scikit-Learn classifier once."""
    if "extractor" not in _EXTRACTOR_CACHE:
        device = get_device()
        print(f"[Third-Eye Accident Analyzer] Loading frozen ResNet-18 on device: {device.type.upper()}")
        weights = ResNet18_Weights.DEFAULT
        extractor = resnet18(weights=weights)
        extractor.fc = torch.nn.Identity()
        extractor.to(device)
        extractor.eval()

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Trained accident classifier not found at {MODEL_PATH}")
        classifier = joblib.load(MODEL_PATH)

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        _EXTRACTOR_CACHE["extractor"] = extractor
        _EXTRACTOR_CACHE["classifier"] = classifier
        _EXTRACTOR_CACHE["preprocess"] = preprocess
        _EXTRACTOR_CACHE["device"] = device

    return (
        _EXTRACTOR_CACHE["extractor"],
        _EXTRACTOR_CACHE["classifier"],
        _EXTRACTOR_CACHE["preprocess"],
        _EXTRACTOR_CACHE["device"],
    )


def compute_persons_near_vehicles(detections: List[Detection]) -> int:
    """Count unique detected persons whose bounding boxes intersect any vehicle bounding box.

    Args:
        detections: List of Detection schema objects.

    Returns:
        Integer count of persons touching or overlapping a vehicle.
    """
    vehicle_labels = {"car", "truck", "bus", "motorcycle", "bicycle"}
    person_boxes = [d.bbox for d in detections if d.label == "person"]
    vehicle_boxes = [d.bbox for d in detections if d.label in vehicle_labels]

    if not person_boxes or not vehicle_boxes:
        return 0

    count = 0
    for p_box in person_boxes:
        px1, py1, px2, py2 = p_box
        near = False
        for v_box in vehicle_boxes:
            vx1, vy1, vx2, vy2 = v_box
            # Standard 2D bounding box intersection check
            if max(px1, vx1) < min(px2, vx2) and max(py1, vy1) < min(py2, vy2):
                near = True
                break
        if near:
            count += 1
    return count


def classify_frame_probability(image_pil: Image.Image) -> float:
    """Compute the uncalibrated accident probability from ResNet-18 + trained classifier."""
    extractor, classifier, preprocess, device = load_model_and_extractor()

    img_tensor = preprocess(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        features = extractor(img_tensor).cpu().numpy()

    # Predict probability for class 1 (accident)
    prob = float(classifier.predict_proba(features)[0, 1])
    return prob


def determine_verdict(prob: float) -> Tuple[str, str, str]:
    """Map model probability to 3 calibrated verdicts and corresponding severity/status.

    Thresholds:
      - prob >= 0.60: 'LIKELY ACCIDENT' (severity: 'high_probability', status: 'reported')
      - 0.40 <= prob < 0.60: 'UNCERTAIN' (severity: 'uncertain', status: 'reported')
      - prob < 0.40: 'NO ACCIDENT SIGNAL' (severity: 'none', status: 'normal_flow')
    """
    if prob >= 0.60:
        return "LIKELY ACCIDENT", "high_probability", "reported"
    elif prob >= 0.40:
        return "UNCERTAIN", "uncertain", "reported"
    else:
        return "NO ACCIDENT SIGNAL", "none", "normal_flow"


def analyze(
    image_path: Union[str, Path],
    location_name: str = "Monitored Roadway",
    latitude: float = 12.9716,
    longitude: float = 77.5946,
    conf_threshold: float = 0.5,
    save_to_db: bool = True,
    media_type: str = "image",
    temporal: Optional[dict] = None,
    human_verified_impact: Optional[dict] = None,
) -> AccidentEvent:
    """Analyze an image using the trained accident classifier and object detector.

    Args:
        image_path: Path to the scene photo.
        location_name: Identifier or road name for event location.
        latitude: WGS84 latitude coordinate.
        longitude: WGS84 longitude coordinate.
        conf_threshold: Minimum detection confidence threshold for Faster R-CNN.
        save_to_db: If True, saves event record to SQLite database.

    Returns:
        AccidentEvent schema instance with data_source="REAL" and method="TRAINED_MODEL".
    """
    path_obj = Path(image_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    # Load PIL image for feature extractor
    pil_img = Image.open(path_obj).convert("RGB")
    width, height = pil_img.size

    # Step 1: Model inference for accident probability (PRIMARY SIGNAL)
    prob = classify_frame_probability(pil_img)
    verdict, severity, status = determine_verdict(prob)

    # Step 2: Object detection for vehicle counts and person-vehicle proximity
    detections = detect(path_obj, conf_threshold=conf_threshold)
    vehicle_labels = {"car", "truck", "bus", "motorcycle", "bicycle"}
    vehicle_count = sum(1 for d in detections if d.label in vehicle_labels)
    persons_near_count = compute_persons_near_vehicles(detections)

    # Step 3: Compute geometric features for forensic explanation ONLY (never dictates verdict)
    features = extract_accident_features(detections, image_width=width, image_height=height)

    # Format forensic explanation text
    explanation_parts = [
        f"Classifier score: {prob:.4f} -> Verdict: {verdict}",
        f"Vehicles detected: {vehicle_count}",
        f"Persons near vehicles: {persons_near_count}",
        f"Geometric overlap context: max vehicle IoU={features['max_vehicle_iou']:.2f}, heavily overlapping pairs={features['heavily_overlapping_pairs']}",
    ]
    reasons_str = "; ".join(explanation_parts)

    # The classifier verdict stays a separate diagnostic field. The shared
    # four-tier assessment is the authoritative evidence result for all tabs.
    assessment = build_accident_assessment(
        model_verdict=verdict,
        model_score=prob,
        media_type=media_type,
        features=features,
        temporal=temporal,
        human_verified_impact=human_verified_impact,
    )
    reasons_str += (
        f"; Four-tier evidence assessment: {assessment.assessment_category}; "
        f"human review required: yes"
    )

    stem = path_obj.stem
    event_id = f"acc-{stem}-{uuid.uuid4().hex[:6]}"
    now_utc = datetime.now(timezone.utc)

    # Create AccidentEvent schema instance
    event = AccidentEvent(
        event_id=event_id,
        timestamp=now_utc,
        location_name=location_name,
        latitude=latitude,
        longitude=longitude,
        severity=severity,
        vehicles_involved=vehicle_count,
        casualties_estimated=persons_near_count,
        status=status,
        confidence=round(prob, 4),
        method="TRAINED_MODEL",
        score=round(prob, 4),
        verdict=verdict,
        persons_near_vehicles_count=persons_near_count,
        reasons=reasons_str,
        evidence_assessment=assessment,
        data_source="REAL",
    )

    # Step 4: Persist to SQLite
    if save_to_db:
        save_accident_event(event)

    return event
