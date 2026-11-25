#!/usr/bin/env python3
"""Automated batch incident reporting runner for Project Third-Eye (Phase 4).

Executes a zero-typing multimodal workflow:
1. Loads all sample photos from data/samples/
2. Extracts full vision evidence:
   - Object detections (Phase 1 Faster R-CNN)
   - Vehicle reports with OCR and color (Phase 2)
   - Accident event classification (Phase 3b ResNet-18)
   - Occupant and pedestrian reports (Phase 4)
3. Pairs each image with every sample emergency call in data/sample_calls/
   (clearly marked as SIMULATED pairing).
4. Prints human-readable fused incident reports and highlights cross-modal disagreement flags.
"""

from pathlib import Path
import sys
import cv2

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.cv.detector import detect
from core.cv.vehicle_report import build_vehicle_reports
from core.ml.accident_analyzer import analyze
from core.nlp.caller_extraction import extract_caller_info
from core.nlp.fusion import fuse_incident_evidence
from core.nlp.occupant_report import build_occupant_reports

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> int:
    samples_dir = REPO_ROOT / "data" / "samples"
    calls_dir = REPO_ROOT / "data" / "sample_calls"

    sample_images = sorted([
        f for f in samples_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTS
    ])
    sample_calls = sorted([
        f for f in calls_dir.glob("call_*.txt")
    ])

    print("==================================================================")
    print("Project Third-Eye | Multimodal Incident Reporting & Fusion (Phase 4)")
    print("Vision Stack     : Faster R-CNN + ResNet-18 + EasyOCR (REAL imagery)")
    print("NLP Stack        : spaCy en_core_web_sm + Rule Matchers (SIMULATED calls)")
    print("Fusion Logic     : Provenance tagging (VISION / CALLER_SIMULATED / MISSING)")
    print("Disagreement Rule: Show both signals; never silently override")
    print("UI Principle     : Zero typing, automated batch execution")
    print("==================================================================")

    if not sample_images:
        print(f"\n[ERROR] No sample images found in {samples_dir}")
        return 1
    if not sample_calls:
        print(f"\n[ERROR] No sample calls found in {calls_dir}")
        return 1

    print(f"\nFound {len(sample_images)} image(s) and {len(sample_calls)} emergency call transcript(s).")
    print(f"Total simulated multimodal pairings to evaluate: {len(sample_images) * len(sample_calls)}\n")

    pairing_count = 0
    total_disagreements = 0

    for img_idx, img_path in enumerate(sample_images, 1):
        print("******************************************************************")
        print(f"ANALYZING SCENE IMAGE [{img_idx}/{len(sample_images)}]: {img_path.name}")
        print("******************************************************************")

        # 1. Vision Pipeline
        img_bgr = cv2.imread(str(img_path))
        detections = detect(img_path, conf_threshold=0.5)
        vehicle_reports, _ = build_vehicle_reports(img_path, save_to_db=False)
        accident_event = analyze(img_path, location_name=f"Sample: {img_path.name}", save_to_db=False)
        occupant_reports, occ_summary = build_occupant_reports(detections, vehicle_reports)

        print(f"Vision Summary: Verdict=[{accident_event.verdict}], Vehicles={len(vehicle_reports)}, "
              f"Visible People={occ_summary['persons_visible_count']} "
              f"(Near Vehicles={occ_summary['persons_near_vehicles_count']})\n")

        # 2. Pair with each emergency call
        for call_idx, call_path in enumerate(sample_calls, 1):
            pairing_count += 1
            call_text = call_path.read_text(encoding="utf-8")
            caller_ext = extract_caller_info(call_text, call_id=call_path.stem)

            # Fuse evidence
            fused_report = fuse_incident_evidence(
                accident_event=accident_event,
                vehicle_reports=vehicle_reports,
                occupant_reports=occupant_reports,
                caller_extraction=caller_ext,
                image_name=img_path.name,
                save_to_db=True,
            )

            disagree_flag = "[DISAGREEMENT DETECTED]" if fused_report.disagreements else "[ALIGNED]"
            if fused_report.disagreements:
                total_disagreements += 1

            print(f"--- [ARBITRARY DEMO PAIRING #{pairing_count}] {img_path.name} + {call_path.name} {disagree_flag} ---")
            print(f"Report ID: {fused_report.report_id}")
            print(f"  Visual Verdict: {fused_report.verdict} (Score: {accident_event.score:.4f})")
            print(f"  Location      : {fused_report.location or 'Not determined'} (Source: {fused_report.location_source})")
            print(f"  Vehicles      : Vision={fused_report.vision_vehicle_count} vs Caller={fused_report.caller_vehicle_count}")
            print(f"  People        : Vision={fused_report.vision_people_count} vs Caller={fused_report.caller_people_count}")
            print(f"  Hazards       : {fused_report.hazards if fused_report.hazards else 'None'} (Source: {fused_report.hazards_source})")
            print(f"  Injuries      : {fused_report.injuries if fused_report.injuries else 'None'} (Source: {fused_report.injuries_source})")

            if fused_report.disagreements:
                print("  Disagreements Flagged:")
                for d in fused_report.disagreements:
                    print(f"    * {d}")
            print()

    print("==================================================================")
    print("PHASE 4 FUSION SUMMARY:")
    print(f"  Total Pairings Evaluated       : {pairing_count}")
    print(f"  Pairings with Disagreements    : {total_disagreements}")
    print(f"  Pairings without Disagreements : {pairing_count - total_disagreements}")
    print("  All fused incident records saved to database: db/third_eye.db (table 'incident_reports')")
    print("==================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
