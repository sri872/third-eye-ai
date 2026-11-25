#!/usr/bin/env python3
"""CLI runner for Project Third-Eye vehicle report generation.

Processes all images found in data/samples/ in a zero-typing workflow.
Generates VehicleReport records, extracts dominant colors and license plates,
saves annotated crops to data/outputs/crops/, and persists records to SQLite.
"""

from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.cv.vehicle_report import build_vehicle_reports

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> int:
    samples_dir = REPO_ROOT / "data" / "samples"
    outputs_dir = REPO_ROOT / "data" / "outputs" / "crops"

    samples_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    sample_images = sorted([
        f for f in samples_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    print("==================================================================")
    print("Project Third-Eye | Vehicle Report Generator (Phase 2)")
    print("Models: Faster R-CNN ResNet-50 FPN + EasyOCR (English)")
    print("UI Principle: Zero typing, automated batch execution")
    print("==================================================================")

    if not sample_images:
        print(f"\n[INFO] No sample images found in: {samples_dir}")
        print("Please place real test photo(s) in data/samples/ to run vehicle reporting.")
        print("==================================================================")
        return 0

    print(f"\nDiscovered {len(sample_images)} image(s) in {samples_dir}.\n")

    total_vehicles = 0
    total_readable_plates = 0
    total_unreadable_plates = 0

    for idx, img_path in enumerate(sample_images, 1):
        print(f"[{idx}/{len(sample_images)}] Processing: {img_path.name}")
        reports, crop_paths = build_vehicle_reports(img_path, conf_threshold=0.5, save_to_db=True)
        print(f"  Extracted {len(reports)} vehicle report(s):")

        for r in reports:
            total_vehicles += 1
            det_conf_str = f"{r.confidence:.4f}" if r.confidence is not None else "N/A"
            color_str = f"{r.color} (share: {r.color_share:.2f})" if r.color else "N/A"

            if r.license_plate:
                total_readable_plates += 1
                plate_str = f"'{r.license_plate}' (conf: {r.plate_confidence:.4f})"
            else:
                total_unreadable_plates += 1
                plate_str = "not readable"

            print(f"    - ID: {r.vehicle_id:<28} | Type: {r.vehicle_type:<10} | Color: {color_str:<22} | Plate: {plate_str:<32} | DetConf: {det_conf_str}")

        print(f"  Saved {len(crop_paths)} annotated crop(s) to data/outputs/crops/\n")

    print("==================================================================")
    print("PHASE 2 VEHICLE REPORT SUMMARY (HONEST COUNT):")
    print(f"  Total Images Processed   : {len(sample_images)}")
    print(f"  Total Vehicles Detected  : {total_vehicles}")
    print(f"  Vehicles with READABLE Plate   : {total_readable_plates}")
    print(f"  Vehicles with NOT READABLE Plate: {total_unreadable_plates}")
    print(f"  Annotated Crops Directory: {outputs_dir.relative_to(REPO_ROOT)}")
    print(f"  Database Records Stored  : db/third_eye.db -> table 'vehicle_reports'")
    print("==================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
