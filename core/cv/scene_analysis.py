"""Scene interpretation from real detector output (no invented observations).

The module provides two independently testable pieces:

1. Class-aware duplicate suppression that removes near-identical boxes while
   preserving genuine adjacent/occluding objects of the same class.
2. A scene summary that cleanly separates *detected objects* from *derived
   observations* and lists only the limitations that actually apply.

Both use ``Detection`` records already produced by the Faster R-CNN detector.
Nothing here runs a model, reads a file, or infers weather, road surface, GPS,
speed, injury or occupant counts.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from core.schemas import Detection

VEHICLE_LABELS = frozenset({"car", "truck", "bus", "motorcycle", "bicycle"})

DEFAULT_IOU_THRESHOLD = 0.85
DEFAULT_CONTAINMENT_THRESHOLD = 0.95
# A small box wholly inside a much larger box is normally a different, occluded
# object. Containment alone must never remove it: require near-equal box areas
# before treating containment as a duplicate.
DEFAULT_AREA_RATIO_THRESHOLD = 0.90


@dataclass(frozen=True)
class SuppressionResult:
    """Raw and postprocessed detections with an auditable suppression trail."""

    raw: Tuple[Detection, ...]
    detections: Tuple[Detection, ...]
    suppressed: Tuple[Detection, ...]
    decisions: Tuple[Dict[str, Any], ...]

    @property
    def raw_count(self) -> int:
        return len(self.raw)

    @property
    def kept_count(self) -> int:
        return len(self.detections)

    @property
    def suppressed_count(self) -> int:
        return len(self.suppressed)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "raw_detection_count": self.raw_count,
            "postprocessed_detection_count": self.kept_count,
            "suppressed_count": self.suppressed_count,
            "iou_threshold": DEFAULT_IOU_THRESHOLD,
            "containment_threshold": DEFAULT_CONTAINMENT_THRESHOLD,
            "min_area_ratio_for_containment": DEFAULT_AREA_RATIO_THRESHOLD,
            "suppressed_detection_ids": [item.detection_id for item in self.suppressed],
            "decisions": [dict(item) for item in self.decisions],
        }


def _iou_and_containment(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0, 0.0
    area_a = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    iou = inter / (area_a + area_b - inter)
    containment = inter / min(area_a, area_b)
    return round(float(iou), 4), round(float(containment), 4)


def _confidence(detection: Detection) -> float:
    return float(detection.confidence) if detection.confidence is not None else -1.0


def suppress_duplicate_detections(
    detections: Iterable[Detection],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    containment_threshold: float = DEFAULT_CONTAINMENT_THRESHOLD,
) -> SuppressionResult:
    """Suppress near-duplicate boxes, preferring the highest-confidence copy.

    Suppression is class-aware: a ``person`` box inside a ``car`` box is never
    removed. Same-class boxes are suppressed only when they are near-identical
    (high IoU) or one is almost entirely contained in the other with a similar
    area. Moderate overlap from perspective, occlusion or adjacent vehicles is
    preserved.
    """
    raw = tuple(detections)
    ordered = sorted(raw, key=lambda item: (-_confidence(item), item.detection_id))
    kept: List[Detection] = []
    suppressed: List[Detection] = []
    decisions: List[Dict[str, Any]] = []

    for candidate in ordered:
        duplicate_of: Optional[Detection] = None
        reason = ""
        best_iou = 0.0
        best_containment = 0.0
        for retained in kept:
            if candidate.label != retained.label:
                continue
            iou, containment = _iou_and_containment(candidate.bbox, retained.bbox)
            best_iou = max(best_iou, iou)
            best_containment = max(best_containment, containment)
            if iou >= iou_threshold:
                duplicate_of = retained
                reason = f"same-class IoU {iou:.2f} >= {iou_threshold:.2f}"
                break
            area_candidate = max(
                1.0,
                (candidate.bbox[2] - candidate.bbox[0])
                * (candidate.bbox[3] - candidate.bbox[1]),
            )
            area_retained = max(
                1.0,
                (retained.bbox[2] - retained.bbox[0])
                * (retained.bbox[3] - retained.bbox[1]),
            )
            area_ratio = min(area_candidate, area_retained) / max(
                area_candidate, area_retained
            )
            if (
                containment >= containment_threshold
                and area_ratio >= DEFAULT_AREA_RATIO_THRESHOLD
            ):
                duplicate_of = retained
                reason = (
                    f"same-class containment {containment:.2f} >= {containment_threshold:.2f} "
                    f"with near-equal areas (ratio {area_ratio:.2f})"
                )
                break
        if duplicate_of is None:
            kept.append(candidate)
            continue
        suppressed.append(candidate)
        decisions.append(
            {
                "suppressed_id": candidate.detection_id,
                "kept_id": duplicate_of.detection_id,
                "label": candidate.label,
                "max_same_class_iou": best_iou,
                "max_same_class_containment": best_containment,
                "reason": reason,
            }
        )

    # Restore a readable (top-to-bottom, left-to-right) order for display while
    # keeping the highest-confidence copy of every duplicate.
    kept.sort(key=lambda item: (item.bbox[1], item.bbox[0], item.detection_id))
    return SuppressionResult(raw, tuple(kept), tuple(suppressed), tuple(decisions))


def build_scene_summary(
    detections: Iterable[Detection],
    image_width: int,
    image_height: int,
    media_type: str = "image",
    suppression: Optional[SuppressionResult | Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Summarise real detections without inventing scene facts.

    ``observations`` are derived from detector output (counts and distribution).
    ``limitations`` are explicit NOT DETERMINED statements about facts that a
    single image or a sampled video cannot establish.
    """
    items = tuple(detections)
    vehicles = [item for item in items if item.label in VEHICLE_LABELS]
    persons = [item for item in items if item.label == "person"]
    others = [item for item in items if item.label not in VEHICLE_LABELS and item.label != "person"]
    distribution = Counter(item.label for item in items)
    confidences = [float(item.confidence) for item in items if item.confidence is not None]

    observed = [
        f"{len(vehicles)} vehicle bounding box(es) detected"
        if vehicles
        else "no vehicle bounding box detected",
        f"{len(persons)} person bounding box(es) detected"
        if persons
        else "no person bounding box detected",
    ]
    if others:
        observed.append(
            "other allowed COCO classes detected: "
            + ", ".join(f"{label} x{count}" for label, count in sorted(distribution.items()) if label not in VEHICLE_LABELS and label != "person")
        )
    else:
        observed.append("no other allowed object class detected")

    confidence_summary: Dict[str, Any] = {
        "count_with_confidence": len(confidences),
        "min": round(min(confidences), 4) if confidences else None,
        "max": round(max(confidences), 4) if confidences else None,
        "mean": round(sum(confidences) / len(confidences), 4) if confidences else None,
        "low_confidence_count": sum(1 for value in confidences if value < 0.5),
    }

    limits = [
        "Weather and road-surface condition: NOT DETERMINED from the supplied pixels.",
        "GPS location: NOT DETERMINED from pixels; any location is a separate simulated selection.",
        "Traffic speed and direction of travel: NOT DETERMINED from a single frame.",
        "Accident occurrence, impact and vehicle damage: NOT DETERMINED from boxes alone.",
        "Injuries and occupants inside vehicles: NOT DETERMINED; only visible exterior people are counted.",
    ]
    if media_type == "video":
        limits.append(
            "Video evidence is limited to evenly sampled frames; no full-clip or tracking-based conclusion is available."
        )
    elif media_type != "image":
        limits.append("Media type: NOT DETERMINED.")

    if isinstance(suppression, Mapping):
        suppression_data = dict(suppression)
        raw_count = int(suppression_data.get("raw_detection_count", len(items)))
        kept_count = int(
            suppression_data.get("postprocessed_detection_count", len(items))
        )
    elif suppression is not None:
        suppression_data = suppression.as_dict()
        raw_count = suppression.raw_count
        kept_count = suppression.kept_count
    else:
        suppression_data = None
        raw_count = len(items)
        kept_count = len(items)

    return {
        "media_type": media_type,
        "image_dimensions_px": [int(image_width), int(image_height)],
        "bounding_box_convention": "[xmin, ymin, xmax, ymax] in displayed analysis-image pixels",
        "vehicle_count": len(vehicles),
        "person_count": len(persons),
        "other_object_count": len(others),
        "object_count": len(items),
        "label_distribution": dict(sorted(distribution.items())),
        "detected_objects": [
            {
                "detection_id": item.detection_id,
                "label": item.label,
                "bbox": list(item.bbox),
                "confidence": item.confidence,
                "data_source": item.data_source,
            }
            for item in items
        ],
        "observations": observed,
        "confidence": confidence_summary,
        "limitations": limits,
        "raw_detection_count": raw_count,
        "postprocessed_detection_count": kept_count,
        "duplicate_suppression": suppression_data,
    }


@dataclass(frozen=True)
class SceneAnalysisResult:
    """Conservative scene context derived only from pixels and real detections.

    Luminance, contrast and bottom-region averages are measured directly from
    the supplied pixels. Vehicle/person counts come strictly from ``Detection``
    records. Weather, GPS, speed, accident, damage, injury and road-safety facts
    are never inferred from a frame.
    """

    scene_type: str
    environment_classification: str
    lighting_condition: str
    mean_luminance: float
    contrast_condition: str
    contrast_std: float
    road_surface_luminance: Optional[float]
    road_surface_assessment: str
    vehicles_detected_count: int
    persons_visible_count: int
    visibility_assessment: str
    scene_description: str
    weather_condition: str = "not determined (requires external atmospheric sensor)"
    gps_coordinates: str = "not determined (no GPS sensor in media pixels)"


def analyze_scene_environment(
    image: Image.Image | np.ndarray,
    detections: Optional[Sequence[Detection]] = None,
    media_type: str = "image",
) -> SceneAnalysisResult:
    """Measure real image properties without fabricating scene observations.

    PIL images are converted to RGB before luminance extraction. NumPy arrays
    follow the OpenCV convention used by the detector (3-channel BGR); 2-D
    arrays are treated as already-grayscale. ``media_type`` is accepted for a
    stable call signature; it does not change any pixel measurement.
    """
    if isinstance(image, Image.Image):
        rgb = np.asarray(image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    elif isinstance(image, np.ndarray):
        if image.size == 0:
            raise ValueError("Empty image provided for scene analysis")
        if image.ndim == 2:
            gray = image
        elif image.ndim == 3 and image.shape[2] == 1:
            gray = image[:, :, 0]
        elif image.ndim == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 3 and image.shape[2] >= 4:
            gray = cv2.cvtColor(image[:, :, :4], cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError("Invalid image shape for scene analysis")
    else:
        raise ValueError("Invalid image input for scene analysis")

    if gray.size == 0 or gray.ndim != 2 or min(gray.shape) <= 0:
        raise ValueError("Empty image provided for scene analysis")

    # Luminance and contrast come only from the actual gray-pixel values.
    mean_lum = float(np.mean(gray))
    std_contrast = float(np.std(gray))

    if mean_lum < 50.0:
        lighting = "Night / Very low light"
    elif mean_lum < 95.0:
        lighting = "Dim / Overcast / Low daylight"
    elif mean_lum < 180.0:
        lighting = "Normal daylight"
    else:
        lighting = "Bright daylight / High glare"

    if std_contrast < 30.0:
        contrast = "Low contrast / Diffuse"
    elif std_contrast < 65.0:
        contrast = "Normal contrast"
    else:
        contrast = "High contrast"

    # Bottom 30% is a measured foreground luminance region, not a road-surface
    # sample. Pixel brightness cannot establish material, surface condition, or
    # even whether the foreground is roadway, so the assessment stays unknown.
    height = gray.shape[0]
    road_zone = gray[int(height * 0.70) :, :]
    road_mean: Optional[float] = float(np.mean(road_zone)) if road_zone.size > 0 else None
    road_surface = (
        "NOT DETERMINED (foreground brightness only; material unverified)"
        if road_mean is not None
        else "not determined"
    )

    det_list = list(detections or [])
    v_count = sum(1 for item in det_list if item.label in VEHICLE_LABELS)
    p_count = sum(1 for item in det_list if item.label == "person")

    if v_count == 0 and p_count == 0:
        visibility = "No vehicles or pedestrians localized in foreground"
    elif v_count > 0 and p_count == 0:
        visibility = f"{v_count} vehicle(s) visible; zero exterior pedestrians detected"
    elif v_count == 0 and p_count > 0:
        visibility = f"{p_count} person(s) visible; zero vehicles detected"
    else:
        visibility = f"{v_count} vehicle(s) and {p_count} person(s) localized in scene"

    # These labels describe detector evidence only. "Roadway" is not inferred
    # from brightness or from the absence of boxes.
    if v_count >= 4:
        scene_type = "Multiple vehicle boxes detected"
        environment_class = "Vehicle-box-dominant scene (environment not determined)"
    elif v_count >= 1:
        scene_type = "Vehicle boxes detected"
        environment_class = "Detected-vehicle scene (environment not determined)"
    elif p_count >= 1:
        scene_type = "Person boxes detected; scene type not determined"
        environment_class = "Detected-person scene (environment not determined)"
    else:
        scene_type = "Scene type not determined from pixels"
        environment_class = "No allowed object boxes detected"

    description = (
        f"{scene_type} observed under {lighting.lower()} with {contrast.lower()} "
        f"({visibility.lower()})."
    )

    return SceneAnalysisResult(
        scene_type=scene_type,
        environment_classification=environment_class,
        lighting_condition=lighting,
        mean_luminance=round(mean_lum, 2),
        contrast_condition=contrast,
        contrast_std=round(std_contrast, 2),
        road_surface_luminance=round(road_mean, 2) if road_mean is not None else None,
        road_surface_assessment=road_surface,
        vehicles_detected_count=v_count,
        persons_visible_count=p_count,
        visibility_assessment=visibility,
        scene_description=description,
        weather_condition="not determined (requires external atmospheric sensor)",
        gps_coordinates="not determined (no GPS sensor in media pixels)",
    )
