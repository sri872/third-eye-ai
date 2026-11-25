"""Real-inference verification of the four mandatory demonstration assets.

Run from the repository root with .venv/bin/python scripts/verify_demo_assets.py.
No result is hardcoded: detector, OCR and classifier outputs come from the models.
"""

from pathlib import Path
import json
import sys

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.cv.detector import detect, detect_with_orientation
from core.cv.plate_detector import detect_plate_regions_with_ocr_fallback
from core.cv.plate_detector_yunet import get_yunet_detector
from core.cv.vehicle_attributes import crop_vehicle
from core.ocr.plate_reader import preprocess_plate_region, read_plate_evidence
from core.ml.accident_analyzer import analyze, classify_frame_probability, determine_verdict
from app.video_utils import probe_video, sample_video_frames
from app.video_analysis import analyze_frame_scores, build_video_assessment

print("=" * 70)
print("DEMONSTRATION ASSETS REAL INFERENCE VERIFICATION")
print("=" * 70)
findings: dict = {}

# 1. Image A
print("\n--- Image A: single_car_plate.jpg ---")
img_a_path = Path("data/samples/demo/single_car_plate.jpg")
raw_img_a = Image.open(img_a_path)
print(f"Raw dimensions: {raw_img_a.size[0]}x{raw_img_a.size[1]}")
dets_a, final_a, info_a = detect_with_orientation(img_a_path)
print(f"Orientation correction: rotation_applied={info_a['rotation_applied']}, degrees={info_a['rotation_degrees']}")
print(f"Post-orientation size: {final_a.size[0]}x{final_a.size[1]}")
print(f"Detections count: {len(dets_a)}")
for d in dets_a:
    print(f"  Detection: label={d.label}, conf={d.confidence:.4f}, bbox={[round(v, 2) for v in d.bbox]}")
findings["single_car_plate.jpg"] = {
    "source_size_px": info_a["source_size_px"],
    "analysed_size_px": info_a["analyze_size_px"],
    "orientation": info_a,
    "detections": [
        {"label": d.label, "confidence": d.confidence, "bbox": d.bbox} for d in dets_a
    ],
}

# Plate detection on Image A
yunet = get_yunet_detector()
# Use the exact analysed orientation returned by detect_with_orientation, so
# plate boxes are expressed in the same pixel coordinate system as the boxes.
cv_final_a = cv2.cvtColor(np.array(final_a), cv2.COLOR_RGB2BGR)

plate_findings = []
for d in dets_a:
    # Use the same crop and source-coordinate bbox as the Streamlit app so the
    # plate regions and OCR evidence are directly comparable with Tab 2.
    v_crop = crop_vehicle(cv_final_a, d)
    x1 = max(0, min(int(round(d.bbox[0])), cv_final_a.shape[1] - 1))
    y1 = max(0, min(int(round(d.bbox[1])), cv_final_a.shape[0] - 1))
    vehicle_bbox = (x1, y1, x1 + v_crop.shape[1], y1 + v_crop.shape[0])
    regions = detect_plate_regions_with_ocr_fallback(v_crop, vehicle_bbox)
    print(f"  Plate candidate regions in vehicle crop: {len(regions)}")
    p_cands = yunet.detect(v_crop) if yunet else ()
    print(f"  YuNet Candidates in vehicle crop: {len(p_cands)}")
    if p_cands:
        pbox_local, pconf = p_cands[0]
        pbox_global = (x1 + pbox_local[0], y1 + pbox_local[1], x1 + pbox_local[2], y1 + pbox_local[3])
        print(f"    YuNet Box: {pbox_global} (score: {pconf:.4f})")
    for region in regions:
        px1, py1, px2, py2 = region.bbox_local
        plate_crop = v_crop[max(0, py1):min(v_crop.shape[0], py2), max(0, px1):min(v_crop.shape[1], px2)]
        # Mirror the Streamlit pipeline exactly: try the preprocessed crop and
        # the raw crop, then retain the stronger plate candidate. This keeps the
        # machine-readable evidence identical to what the professor sees in Tab 2.
        processed = preprocess_plate_region(plate_crop)
        ev_proc = read_plate_evidence(processed)
        ev_raw = read_plate_evidence(plate_crop)
        ocr_ev = (
            ev_raw
            if (
                ev_raw.plate
                and (
                    not ev_proc.plate
                    or (ev_raw.plate_confidence or 0) > (ev_proc.plate_confidence or 0)
                )
            )
            else ev_proc
        )
        print(f"    Region {region.method}: bbox={region.bbox_original}")
        print(f"    EasyOCR: raw_text='{ocr_ev.raw_text}', raw_conf={ocr_ev.raw_confidence}, plate='{ocr_ev.plate}', conf={ocr_ev.plate_confidence}, status='{ocr_ev.ocr_status}'")
        plate_findings.append({
            "method": region.method,
            "bbox": region.bbox_original,
            "raw_text": ocr_ev.raw_text,
            "raw_confidence": ocr_ev.raw_confidence,
            "plate": ocr_ev.plate,
            "plate_confidence": ocr_ev.plate_confidence,
            "ocr_status": ocr_ev.ocr_status,
        })
findings["single_car_plate.jpg"]["plate_ocr"] = plate_findings

# 2. Image B
print("\n--- Image B: two_car_collision.jpg ---")
img_b_path = Path("data/samples/demo/two_car_collision.jpg")
dets_b, final_b, info_b = detect_with_orientation(img_b_path)
print(f"Dimensions: {final_b.size[0]}x{final_b.size[1]}")
cars_b = [d for d in dets_b if d.label == "car"]
persons_b = [d for d in dets_b if d.label == "person"]
print(f"Detections total: {len(dets_b)} (Cars: {len(cars_b)}, Persons: {len(persons_b)})")
for i, d in enumerate(cars_b[:2]):
    print(f"  Car {i+1}: conf={d.confidence:.4f}, bbox={[round(v, 2) for v in d.bbox]}")
score_b = classify_frame_probability(final_b)
verdict_b, status_b, _ = determine_verdict(score_b)
event_b = analyze(img_b_path, save_to_db=False)
print(f"ResNet-18 Accident Classifier Score (analysed orientation): {score_b:.4f}, Verdict: {verdict_b}")
print(f"core.ml.analyze() score: {event_b.score:.4f}, Verdict: {event_b.verdict}")
findings["two_car_collision.jpg"] = {
    "size_px": final_b.size,
    "orientation": info_b,
    "detections": [
        {"label": d.label, "confidence": d.confidence, "bbox": d.bbox} for d in dets_b
    ],
    "classifier_score": round(score_b, 4),
    "verdict": verdict_b,
    "scene_status": status_b,
}

# 3. Image C
print("\n--- Image C: heavy_traffic.jpg ---")
img_c_path = Path("data/samples/demo/heavy_traffic.jpg")
dets_c, final_c, info_c = detect_with_orientation(img_c_path)
print(f"Dimensions: {final_c.size[0]}x{final_c.size[1]}")
print(f"Detections count: {len(dets_c)}")
counts_by_class = {}
for d in dets_c:
    counts_by_class[d.label] = counts_by_class.get(d.label, 0) + 1
print(f"Breakdown by class: {counts_by_class}")
score_c = classify_frame_probability(final_c)
verdict_c, status_c, _ = determine_verdict(score_c)
event_c = analyze(img_c_path, save_to_db=False)
print(f"ResNet-18 Accident Classifier Score (analysed orientation): {score_c:.4f}, Verdict: {verdict_c}")
print(f"core.ml.analyze() score: {event_c.score:.4f}, Verdict: {event_c.verdict}")
findings["heavy_traffic.jpg"] = {
    "size_px": final_c.size,
    "orientation": info_c,
    "counts_by_class": counts_by_class,
    "detections": [
        {"label": d.label, "confidence": d.confidence, "bbox": d.bbox} for d in dets_c
    ],
    "classifier_score": round(score_c, 4),
    "verdict": verdict_c,
    "scene_status": status_c,
}

# 4. Video D
print("\n--- Video D: road vdo.mp4 ---")
vdo_path = Path("data/samples/demo/road vdo.mp4")
meta = probe_video(vdo_path)
print(f"Video metadata: {meta.frame_count} frames, {meta.fps} fps, {meta.duration_s:.1f}s, {meta.width}x{meta.height}")
sampled = sample_video_frames(vdo_path, sample_count=12)
frame_rows = []
for pos, frame in enumerate(sampled.frames, start=1):
    sc = classify_frame_probability(frame.image)
    verd, sev, stat = determine_verdict(sc)
    real_detections = detect(frame.image)
    frame_rows.append({
        "Frame": pos,
        "Source frame": frame.index,
        "Timestamp (s)": round(frame.timestamp_s, 2),
        "Score": round(sc, 4),
        "Verdict": verd,
        "Severity": sev,
        "Vehicles": sum(1 for item in real_detections if item.label in {
            "car", "truck", "bus", "motorcycle", "bicycle"
        }),
        "Persons": sum(1 for item in real_detections if item.label == "person"),
    })
temporal = analyze_frame_scores(frame_rows)
print(f"Temporal signal: {temporal['temporal_signal']}")
print(f"Classifier score range: {temporal['min_score']} - {temporal['max_score']} (mean: {temporal['mean_score']})")
print(f"Peak score at Source frame {temporal['peak_source_frame']}: {temporal['max_score']}")
for r in frame_rows:
    print(f"  Frame {r['Frame']:2d} (t={r['Timestamp (s)']:5.2f}s, frame #{r['Source frame']:3d}): score={r['Score']:.4f} -> {r['Verdict']} (vehicles={r['Vehicles']}, persons={r['Persons']})")
peak_verdict = frame_rows[int(temporal["peak_position"]) - 1]["Verdict"]
assessment = build_video_assessment(meta, frame_rows, temporal, peak_verdict)
print(f"Assessment: {assessment['assessment_text']}")
findings["road vdo.mp4"] = {
    "metadata": {
        "frame_count": meta.frame_count,
        "fps": meta.fps,
        "duration_s": meta.duration_s,
        "width": meta.width,
        "height": meta.height,
    },
    "sample_count": len(frame_rows),
    "frame_rows": frame_rows,
    "temporal": temporal,
    "assessment": assessment["assessment_text"],
}
output = REPO_ROOT / "data" / "outputs" / "demo_verification.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(findings, indent=2, default=str), encoding="utf-8")
print(f"Wrote real inference findings to {output.relative_to(REPO_ROOT)}")
print("=" * 70)
