"""Vehicle report generation pipeline for Project Third-Eye.

Coordinates:
1. Faster R-CNN detection for vehicle bounding boxes (filtering out persons).
2. Safe cropping and vehicle attribute extraction (dominant color, type).
3. EasyOCR license plate recognition with normalization and real confidence.
4. Persistence into SQLite via db/database.py.
5. Saving annotated vehicle crops to data/outputs/crops/.
"""

from dataclasses import replace
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union
import uuid

import cv2
import numpy as np

from core.cv.detector import detect, detect_with_orientation
from core.cv.plate_detector import detect_plate_regions
from core.cv.vehicle_attributes import crop_vehicle, dominant_color, get_vehicle_type
from core.ocr.plate_reader import preprocess_plate_region, read_plate_evidence
from core.schemas import VehicleReport
from db.database import save_vehicle_reports

logger = logging.getLogger("third_eye.cv.vehicle_report")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "crops"


def save_annotated_crop(
    crop: np.ndarray,
    report: VehicleReport,
    output_path: Path,
) -> Path:
    """Draw forensic attribute overlay on vehicle crop and save to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated = crop.copy()

    # Create top info banner
    v_type = (report.vehicle_type or "vehicle").upper()
    color_str = f"{report.color} ({report.color_share:.2f})" if report.color else "unknown"
    plate_str = f"PLATE: {report.license_plate}" if report.license_plate else "PLATE: NOT READABLE"
    if report.plate_confidence is not None:
        plate_str += f" ({report.plate_confidence:.2f})"

    line1 = f"{v_type} | {color_str}"
    line2 = plate_str

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1

    # Draw small semi-transparent banner at the top
    banner_height = 42
    h, w = annotated.shape[:2]
    if h > banner_height and w > 120:
        cv2.rectangle(annotated, (0, 0), (w, banner_height), (0, 0, 0), -1)
        cv2.putText(annotated, line1, (6, 16), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(annotated, line2, (6, 34), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    cv2.imwrite(str(output_path), annotated)
    return output_path


def build_vehicle_reports(
    image_path: Union[str, Path],
    conf_threshold: float = 0.5,
    save_to_db: bool = True,
    output_dir: Optional[Path] = None,
) -> Tuple[List[VehicleReport], List[Path]]:
    """Run detection and generate forensic VehicleReport records for each detected vehicle.

    Args:
        image_path: Path to input image.
        conf_threshold: Minimum detection confidence threshold (default 0.5).
        save_to_db: Whether to automatically persist generated reports into SQLite.
        output_dir: Target folder for saving annotated vehicle crops.

    Returns:
        (reports, crop_paths): List of VehicleReport schemas and paths to saved crops.
    """
    path_obj = Path(image_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    # Read image with OpenCV
    img = cv2.imread(str(path_obj))
    if img is None:
        raise ValueError(f"Failed to load image with OpenCV: {path_obj}")

    # Step 1: Run Faster R-CNN detection with orientation check
    detections, pil_img, info = detect_with_orientation(path_obj, conf_threshold=conf_threshold)
    if info.get("rotation_applied"):
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    target_dir = output_dir or DEFAULT_OUTPUT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    reports: List[VehicleReport] = []
    saved_crops: List[Path] = []
    stem = path_obj.stem
    now_utc = datetime.now(timezone.utc)

    # Step 2: Filter for vehicle classes only (ignore persons)
    vehicle_idx = 0
    for det in detections:
        v_type = get_vehicle_type(det.label)
        if v_type is None:
            continue

        vehicle_idx += 1
        vehicle_id = f"veh-{stem}-{vehicle_idx:02d}-{uuid.uuid4().hex[:4]}"

        # Step 3: Safe crop with bounds clamping
        try:
            crop = crop_vehicle(img, det)
        except Exception as e:
            logger.warning("Failed to crop vehicle %s: %s", vehicle_id, e)
            continue

        # Step 4: Extract dominant color via k-means
        color_name, color_share = dominant_color(crop)

        # Step 5: Localise a plate region, then run EasyOCR only on that region.
        crop_x0 = max(0, min(int(round(det.bbox[0])), img.shape[1] - 1))
        crop_y0 = max(0, min(int(round(det.bbox[1])), img.shape[0] - 1))
        vehicle_bbox = (
            crop_x0,
            crop_y0,
            crop_x0 + crop.shape[1],
            crop_y0 + crop.shape[0],
        )
        regions = detect_plate_regions(crop, vehicle_bbox)
        plate_text: Optional[str] = None
        ocr_conf: Optional[float] = None
        detection_status = "NO PLATE REGION DETECTED"
        plate_bbox: Optional[Tuple[int, int, int, int]] = None
        plate_detector: Optional[str] = None
        if regions:
            region = regions[0]
            lx0, ly0, lx1, ly1 = region.bbox_local
            plate_crop = crop[ly0:ly1, lx0:lx1]
            detection_status = "PLATE REGION DETECTED"
            plate_bbox = region.bbox_original
            plate_detector = region.method
            try:
                ev_proc = read_plate_evidence(preprocess_plate_region(plate_crop))
                ev_raw = read_plate_evidence(plate_crop)
                evidence = (
                    ev_raw
                    if (ev_raw.plate and (not ev_proc.plate or (ev_raw.plate_confidence or 0) > (ev_proc.plate_confidence or 0)))
                    else ev_proc
                )
                plate_text = evidence.plate
                ocr_conf = evidence.plate_confidence
            except Exception as exc:
                logger.warning("Plate OCR failed for %s: %s", vehicle_id, exc)

        # Step 6: Construct VehicleReport schema with REAL data_source
        report = VehicleReport(
            vehicle_id=vehicle_id,
            license_plate=plate_text,
            plate_confidence=ocr_conf,
            plate_detection_status=detection_status,
            plate_bbox=plate_bbox,
            plate_detector=plate_detector,
            vehicle_type=v_type,
            color=color_name,
            color_share=color_share,
            damage_severity=None,
            confidence=det.confidence,
            timestamp=det.timestamp or now_utc,
            data_source="REAL",
        )
        reports.append(report)

        # Step 7: Save annotated crop
        crop_path = target_dir / f"{stem}_{vehicle_id}_{v_type}.jpg"
        saved = save_annotated_crop(crop, report, crop_path)
        saved_crops.append(saved)

    # Step 8: Persist to SQLite
    if save_to_db and reports:
        save_vehicle_reports(reports)

    return reports, saved_crops
