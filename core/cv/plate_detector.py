"""Number-plate region localisation with a verified OpenCV Zoo model.

Primary detector: vendored OpenCV Zoo LPD-YuNet ONNX, loaded through ``cv2.dnn``
on CPU with an exact SHA-256 check. Fallback detector: the deterministic OpenCV
heuristic in this module, used only when the ONNX cannot be loaded.

The output is a *candidate plate region*, never a registration number. The
detection score is a model/heuristic ranking value, not an OCR confidence and not
a calibrated accident probability. Candidates are passed to EasyOCR by the
caller; OCR failures stay unverified.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import ClassVar, Optional, Sequence, Tuple

import cv2
import numpy as np

from core.cv.plate_detector_yunet import METHOD as YUNET_METHOD, get_yunet_detector

BBox = Tuple[int, int, int, int]
METHOD = "HEURISTIC_OPENCV_V1"
OCR_FALLBACK_METHOD = "EASYOCR_WHOLE_CROP_FALLBACK"

_MIN_ASPECT = 1.7
_MAX_ASPECT = 6.5
_IDEAL_ASPECT = 3.0
_MAX_CANDIDATES = 4
_BRIGHT_PIXEL_MIN = 235
_PLATE_BORDER_PAD = 2
_OCR_FALLBACK_PAD = 8

# Explicit geometry-only association states. The no-candidate status matches the
# string already used by the UI and VehicleReport schemas.
ASSOCIATED = "ASSOCIATED"
AMBIGUOUS = "AMBIGUOUS"
UNASSOCIATED = "UNASSOCIATED"
NO_PLATE_REGION = "NO PLATE REGION DETECTED"
NO_CANDIDATES = NO_PLATE_REGION
_ASSOCIATION_MARGIN = 0.15
_MIN_CONTAINMENT = 0.5


@dataclass(frozen=True)
class PlateRegion:
    """One plausible plate region from a vehicle crop.

    ``bbox_local`` is in the vehicle-crop pixel coordinate system used for OCR.
    ``bbox_original`` is clamped to the supplied vehicle bounding box so the
    evidence can be related back to the original uploaded image/video frame.
    ``heuristic_score`` keeps its name for backward compatibility; for YuNet it
    stores the real model score, otherwise the heuristic ranking number. For an
    OCR-proposed fallback region it is 0.0 and ``ocr_confidence`` carries the
    real whole-crop EasyOCR confidence. Neither is a calibrated probability.
    """

    bbox_local: BBox
    bbox_original: BBox
    heuristic_score: float
    character_candidates: int
    aspect_ratio: float
    edge_density: float
    method: str = METHOD
    ocr_confidence: Optional[float] = None


def _iou(a: BBox, b: BBox) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    intersection = (x1 - x0) * (y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1)


def _clamp_bbox(bbox: Sequence[float]) -> Optional[BBox]:
    try:
        values = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        return None
    xmin, ymin, xmax, ymax = values
    if xmin >= xmax or ymin >= ymax:
        return None
    return int(round(xmin)), int(round(ymin)), int(round(xmax)), int(round(ymax))


def _character_boxes(gray: np.ndarray) -> list[BBox]:
    """Deduplicate MSER regions that look like compact characters."""
    try:
        mser = cv2.MSER_create(delta=5, min_area=8, max_area=1200)
        regions, _ = mser.detectRegions(gray)
    except cv2.error:
        return []

    boxes: list[BBox] = []
    seen: set[tuple[int, int, int, int]] = set()
    for region in regions:
        x, y, w, h = cv2.boundingRect(region)
        if w < 2 or h < 5 or w > 40 or h > 45:
            continue
        if h / max(w, 1) > 5.0 or w * h < 12:
            continue
        key = (x // 2, y // 2, w // 2, h // 2)
        if key in seen:
            continue
        seen.add(key)
        boxes.append((x, y, x + w, y + h))
    return boxes


def _bright_interior_bbox(gray_roi: np.ndarray) -> Optional[BBox]:
    """Find the plate's bright interior inside a contour ROI.

    The morphological close used to find candidate contours deliberately joins
    nearby edges. For a small plate, character strokes drawn above the plate can
    inflate the contour far beyond the actual plate. Measuring the bright plate
    face keeps minimum-size filtering tied to the plate rather than to that
    text blob. Returns ``None`` when no trustworthy bright face is present, so
    the caller can fall back to the contour box for darker real-world plates.
    """
    mask = (gray_roi >= _BRIGHT_PIXEL_MIN).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return None
    x, y, w, h, area = max((tuple(item) for item in stats[1:]), key=lambda item: item[4])
    if w < 8 or h < 3 or area < 20:
        return None
    aspect = w / max(h, 1)
    if not (_MIN_ASPECT <= aspect <= _MAX_ASPECT):
        return None
    return int(x), int(y), int(x + w), int(y + h)


def _line_score(gray: np.ndarray) -> tuple[float, int]:
    """Return a 0..1 heuristic score for a horizontal line of characters."""
    boxes = _character_boxes(gray)
    if len(boxes) < 3:
        return 0.0, 0

    heights = np.asarray([b[3] - b[1] for b in boxes], dtype=float)
    centers = np.asarray([(b[1] + b[3]) / 2.0 for b in boxes], dtype=float)
    median_height = float(np.median(heights))
    if median_height < 5.0:
        return 0.0, 0

    inliers = np.abs(centers - float(np.median(centers))) <= max(
        3.0, 0.6 * median_height
    )
    character_count = int(inliers.sum())
    if character_count < 3:
        return 0.0, character_count
    return min(character_count, 10) / 10.0, character_count


def _map_to_original(local: BBox, original: Optional[BBox]) -> BBox:
    if original is None:
        return local
    return (
        min(max(original[0] + local[0], original[0]), original[2]),
        min(max(original[1] + local[1], original[1]), original[3]),
        min(max(original[0] + local[2], original[0]), original[2]),
        min(max(original[1] + local[3], original[1]), original[3]),
    )


@dataclass(frozen=True)
class PlateAssociation:
    """Geometry-only decision about which plate region belongs to a vehicle.

    ``region`` is set only for an unambiguous ASSOCIATED result; AMBIGUOUS and
    UNASSOCIATED never auto-select a candidate. ``candidates`` keeps the plausible
    regions for audit, and ``reason`` explains the decision without inventing
    confidence or reading plate text.
    """

    status: str
    region: Optional[PlateRegion] = None
    reason: str = ""
    candidates: Tuple[PlateRegion, ...] = ()

    ASSOCIATED: ClassVar[str] = ASSOCIATED
    AMBIGUOUS: ClassVar[str] = AMBIGUOUS
    UNASSOCIATED: ClassVar[str] = UNASSOCIATED
    NO_PLATE_REGION: ClassVar[str] = NO_PLATE_REGION

    @property
    def bbox(self) -> Optional[BBox]:
        """Original-image bbox of the selected region, if one was selected."""
        return self.region.bbox_original if self.region is not None else None


def _overlap_area(a: BBox, b: BBox) -> int:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0
    return (x1 - x0) * (y1 - y0)


def _association_score(candidate: BBox, vehicle: BBox) -> tuple[float, float]:
    """Score containment and centredness in image coordinates only."""
    candidate_area = max((candidate[2] - candidate[0]) * (candidate[3] - candidate[1]), 1)
    containment = min(_overlap_area(candidate, vehicle) / candidate_area, 1.0)
    cx = (candidate[0] + candidate[2]) / 2.0
    cy = (candidate[1] + candidate[3]) / 2.0
    vx = (vehicle[0] + vehicle[2]) / 2.0
    vy = (vehicle[1] + vehicle[3]) / 2.0
    half_w = max((vehicle[2] - vehicle[0]) / 2.0, 1.0)
    half_h = max((vehicle[3] - vehicle[1]) / 2.0, 1.0)
    distance = min(math.hypot((cx - vx) / half_w, (cy - vy) / half_h), 1.0)
    centeredness = 1.0 - distance
    return containment * (0.5 + 0.5 * centeredness), containment


def associate_plate_regions(
    regions: Sequence[PlateRegion] | None,
    vehicle_bbox: Sequence[float] | None,
    image_width: int,
    image_height: int,
) -> PlateAssociation:
    """Associate candidate plate regions with one vehicle using geometry only.

    A region is plausible when its centre lies inside the vehicle box and at
    least half of it is contained by that box. Containment and centredness rank
    plausible regions; if two of them are near-ties, the result is AMBIGUOUS and
    no region is selected. Text, OCR confidence and detector scores are never
    consulted, so a plate is never inferred from text alone.
    """
    candidates = tuple(regions or ())
    if not candidates:
        return PlateAssociation(
            status=NO_PLATE_REGION,
            reason="No candidate plate regions supplied.",
        )

    vehicle = _clamp_bbox(vehicle_bbox) if vehicle_bbox is not None else None
    if vehicle is None:
        return PlateAssociation(
            status=UNASSOCIATED,
            reason="Invalid vehicle bounding box.",
            candidates=candidates,
        )
    try:
        width, height = int(image_width), int(image_height)
    except (TypeError, ValueError):
        return PlateAssociation(
            status=UNASSOCIATED,
            reason="Invalid image dimensions.",
            candidates=candidates,
        )
    if width <= 0 or height <= 0:
        return PlateAssociation(
            status=UNASSOCIATED,
            reason="Invalid image dimensions.",
            candidates=candidates,
        )

    bounds = (0, 0, width, height)
    vehicle = (
        max(vehicle[0], 0),
        max(vehicle[1], 0),
        min(vehicle[2], width),
        min(vehicle[3], height),
    )
    if vehicle[0] >= vehicle[2] or vehicle[1] >= vehicle[3]:
        return PlateAssociation(
            status=UNASSOCIATED,
            reason="Vehicle bounding box is outside the image.",
            candidates=candidates,
        )

    plausible: list[tuple[float, BBox, PlateRegion]] = []
    for region in candidates:
        local_box = _clamp_bbox(region.bbox_original) or _clamp_bbox(region.bbox_local)
        if local_box is None:
            continue
        box = (
            max(local_box[0], 0),
            max(local_box[1], 0),
            min(local_box[2], width),
            min(local_box[3], height),
        )
        if box[0] >= box[2] or box[1] >= box[3]:
            continue
        center_x = (box[0] + box[2]) / 2.0
        center_y = (box[1] + box[3]) / 2.0
        if not (
            vehicle[0] <= center_x <= vehicle[2]
            and vehicle[1] <= center_y <= vehicle[3]
        ):
            continue
        score, containment = _association_score(box, vehicle)
        if containment < _MIN_CONTAINMENT:
            continue
        plausible.append((score, box, region))

    if not plausible:
        return PlateAssociation(
            status=UNASSOCIATED,
            reason="No plate region centre and majority area lie inside the vehicle.",
            candidates=candidates,
        )

    plausible.sort(key=lambda item: (-item[0], item[1]))
    best_score, _, best_region = plausible[0]
    plausible_regions = tuple(item[2] for item in plausible)

    if len(plausible) > 1:
        second_score = plausible[1][0]
        if best_score - second_score < _ASSOCIATION_MARGIN:
            return PlateAssociation(
                status=AMBIGUOUS,
                region=None,
                reason=(
                    "Multiple plate regions overlap the vehicle with no clear "
                    "geometry winner; none was auto-selected."
                ),
                candidates=plausible_regions,
            )

    return PlateAssociation(
        status=ASSOCIATED,
        region=best_region,
        reason="Single best geometry match inside the vehicle bounding box.",
        candidates=plausible_regions,
    )


def detect_plate_regions(
    crop: np.ndarray,
    vehicle_bbox: Sequence[float] | None = None,
    *,
    min_width: int = 20,
    min_height: int = 7,
    max_candidates: int = _MAX_CANDIDATES,
    use_model: bool = True,
) -> tuple[PlateRegion, ...]:
    """Locate plausible number-plate regions inside one vehicle crop.

    Args:
        crop: BGR vehicle crop.
        vehicle_bbox: Optional original-image ``[xmin, ymin, xmax, ymax]`` used
            to translate local crop coordinates back to the source image.
        min_width / min_height: Minimum accepted heuristic candidate dimensions,
            measured on the bright plate face when one is available so joined
            character strokes cannot inflate a small plate past the limit.
        max_candidates: Maximum number of non-overlapping candidates returned.
        use_model: Prefer the verified YuNet ONNX when available. If loading or
            inference fails, the deterministic heuristic is used instead.

    Returns:
        Candidates ordered by detection score. Empty means "no plate region
        detected"; it does not mean an accident or absence of a plate in the
        physical world.
    """
    if crop is None or not isinstance(crop, np.ndarray) or crop.size == 0:
        return ()
    if crop.ndim != 3 or crop.shape[2] < 3:
        return ()
    if min_width <= 0 or min_height <= 0 or max_candidates <= 0:
        return ()

    original = _clamp_bbox(vehicle_bbox) if vehicle_bbox is not None else None
    if vehicle_bbox is not None and original is None:
        # Invalid detection boxes are rejected rather than silently guessed.
        return ()

    if use_model:
        detector = get_yunet_detector()
        if detector is not None:
            try:
                model_boxes = detector.detect(crop)[:max_candidates]
                return tuple(
                    PlateRegion(
                        bbox_local=local,
                        bbox_original=_map_to_original(local, original),
                        heuristic_score=round(score, 4),
                        character_candidates=0,
                        aspect_ratio=round(
                            (local[2] - local[0]) / max(local[3] - local[1], 1), 2
                        ),
                        edge_density=0.0,
                        method=YUNET_METHOD,
                    )
                    for local, score in model_boxes
                )
            except Exception:
                # A failed model inference falls back to the transparent heuristic.
                detector = None

    height, width = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    edges = cv2.Canny(clahe, 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    scored: list[tuple[BBox, float, int, float, float]] = []

    for contour in contours:
        x, y, box_w, box_h = cv2.boundingRect(contour)
        if box_w < min_width or box_h < min_height or box_w * box_h < 180:
            continue

        contour_roi = gray[y : y + box_h, x : x + box_w]
        interior = _bright_interior_bbox(contour_roi)
        if interior is not None:
            # The bright plate face is a tighter, less easily inflated measure
            # than the joined edge contour. Keep a small border around it for OCR.
            measured_w = interior[2] - interior[0]
            measured_h = interior[3] - interior[1]
            local = (
                max(0, x + interior[0] - _PLATE_BORDER_PAD),
                max(0, y + interior[1] - _PLATE_BORDER_PAD),
                min(width, x + interior[2] + _PLATE_BORDER_PAD),
                min(height, y + interior[3] + _PLATE_BORDER_PAD),
            )
            aspect = measured_w / max(measured_h, 1)
        else:
            measured_w, measured_h = box_w, box_h
            aspect = box_w / max(box_h, 1)
            local = (
                max(0, x - 2),
                max(0, y - 2),
                min(width, x + box_w + 2),
                min(height, y + box_h + 2),
            )

        if local[2] <= local[0] or local[3] <= local[1]:
            continue
        if measured_w < min_width or measured_h < min_height or measured_w * measured_h < 180:
            continue
        if not (_MIN_ASPECT <= aspect <= _MAX_ASPECT):
            continue

        roi = clahe[local[1] : local[3], local[0] : local[2]]
        line_score, character_count = _line_score(roi)
        if line_score <= 0.0:
            continue

        edge_density = float(np.mean(edges[local[1] : local[3], local[0] : local[2]] > 0))
        contrast = float(np.percentile(roi, 90) - np.percentile(roi, 10))
        aspect_score = 1.0 - min(abs(aspect - _IDEAL_ASPECT) / 2.5, 1.0)
        score = (
            0.45 * line_score
            + 0.25 * aspect_score
            + 0.20 * min(edge_density / 0.30, 1.0)
            + 0.10 * min(contrast / 120.0, 1.0)
        )
        scored.append((local, score, character_count, aspect, edge_density))

    scored.sort(key=lambda item: item[1], reverse=True)
    picked: list[tuple[BBox, float, int, float, float]] = []
    for candidate in scored:
        if all(_iou(candidate[0], other[0]) < 0.40 for other in picked):
            picked.append(candidate)
        if len(picked) >= max_candidates:
            break

    regions: list[PlateRegion] = []
    for local, score, character_count, aspect, edge_density in picked:
        regions.append(
            PlateRegion(
                bbox_local=local,
                bbox_original=_map_to_original(local, original),
                heuristic_score=round(score, 4),
                character_candidates=character_count,
                aspect_ratio=round(aspect, 2),
                edge_density=round(edge_density, 4),
            )
        )
    return tuple(regions)


def _plate_like_ocr_text(text: str) -> bool:
    """Accept only alphanumeric reads that look like a registration number."""
    cleaned = re.sub(r"[^A-Z0-9]", "", str(text).upper())
    # Indian registration marks are typically 8-10 alphanumeric characters with
    # at least three digits. Short reads such as "can09g" are not accepted.
    return len(cleaned) >= 8 and sum(char.isdigit() for char in cleaned) >= 3


def detect_plate_regions_with_ocr_fallback(
    crop: np.ndarray,
    vehicle_bbox: Sequence[float] | None = None,
    *,
    max_candidates: int = _MAX_CANDIDATES,
) -> tuple[PlateRegion, ...]:
    """Locate a plate, using real whole-crop OCR only if both detectors miss it.

    YuNet is trained on Chinese plates and the deterministic heuristic can also
    fail on a real Indian plate. When both return nothing, this runs EasyOCR on
    the entire vehicle crop and turns plate-like text boxes into candidate
    regions (with an 8 px padding). The caller then crops and re-runs OCR on the
    proposed region, preserving raw evidence and the 0.7 verification threshold.
    An empty tuple still means "no candidate region"; no text is synthesised.
    """
    if crop is None or not isinstance(crop, np.ndarray) or crop.size == 0:
        return ()

    regions = detect_plate_regions(crop, vehicle_bbox, max_candidates=max_candidates)
    if regions:
        return regions

    original = _clamp_bbox(vehicle_bbox) if vehicle_bbox is not None else None
    if vehicle_bbox is not None and original is None:
        return ()

    try:
        from core.ocr.plate_reader import get_ocr_reader

        reader, _ = get_ocr_reader()
        raw_results = reader.readtext(crop)
    except Exception:
        return ()

    height, width = crop.shape[:2]
    proposals: list[tuple[BBox, float, float]] = []
    for polygon, text, probability in raw_results:
        if not _plate_like_ocr_text(text):
            continue
        try:
            xs = [int(round(point[0])) for point in polygon]
            ys = [int(round(point[1])) for point in polygon]
        except (TypeError, ValueError, IndexError):
            continue
        local = (
            max(0, min(xs) - _OCR_FALLBACK_PAD),
            max(0, min(ys) - _OCR_FALLBACK_PAD),
            min(width, max(xs) + _OCR_FALLBACK_PAD),
            min(height, max(ys) + _OCR_FALLBACK_PAD),
        )
        if local[2] <= local[0] or local[3] <= local[1]:
            continue
        aspect = (local[2] - local[0]) / max(local[3] - local[1], 1)
        if not (1.5 <= aspect <= 7.0):
            continue
        confidence = float(probability)
        proposals.append((local, confidence, aspect))

    proposals.sort(key=lambda item: item[1], reverse=True)
    picked: list[tuple[BBox, float, float]] = []
    for proposal in proposals:
        if all(_iou(proposal[0], other[0]) < 0.4 for other in picked):
            picked.append(proposal)
        if len(picked) >= max_candidates:
            break

    return tuple(
        PlateRegion(
            bbox_local=local,
            bbox_original=_map_to_original(local, original),
            heuristic_score=0.0,
            character_candidates=0,
            aspect_ratio=round(aspect, 2),
            edge_density=0.0,
            method=OCR_FALLBACK_METHOD,
            ocr_confidence=round(confidence, 4),
        )
        for local, confidence, aspect in picked
    )
