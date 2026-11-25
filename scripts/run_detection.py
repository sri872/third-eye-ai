#!/usr/bin/env python3
"""CLI runner for Project Third-Eye computer vision detection.

Adheres strictly to the dropdown-free, no-typing rule:
Automatically scans data/samples/ for all test images, runs Faster R-CNN detection,
and writes annotated output files to data/outputs/.
Requires zero interactive typing or command line arguments.
"""

from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.cv.detector import detect
from core.cv.draw import draw_detections

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> int:
    samples_dir = REPO_ROOT / "data" / "samples"
    outputs_dir = REPO_ROOT / "data" / "outputs"

    samples_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Discover all image files without typing
    sample_images = sorted([
        f for f in samples_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    print("==================================================================")
    print("Project Third-Eye | Computer Vision Detector (Phase 1)")
    print("Architecture: Faster R-CNN ResNet-50 FPN (COCO Pretrained)")
    print("UI Principle: Zero typing, automated batch execution")
    print("==================================================================")

    if not sample_images:
        print(f"\n[INFO] No sample images found in: {samples_dir}")
        print("Please place real test photo(s) in data/samples/ to run live detection.")
        print("Example: cp /path/to/traffic_accident.jpg data/samples/")
        print("==================================================================")
        return 0

    print(f"\nFound {len(sample_images)} image(s) in {samples_dir}. Processing automatically...\n")

    total_detections = 0
    for idx, img_path in enumerate(sample_images, 1):
        print(f"[{idx}/{len(sample_images)}] Processing: {img_path.name}")
        detections = detect(img_path, conf_threshold=0.5)
        total_detections += len(detections)

        if detections:
            print(f"  Detected {len(detections)} object(s):")
            for det in detections:
                conf_str = f"{det.confidence:.4f}" if det.confidence is not None else "N/A"
                print(f"    - Class: {det.label:<10} | Confidence: {conf_str} | BBox: {det.bbox} | Source: {det.data_source}")
        else:
            print("  No target objects detected above confidence threshold (0.50).")

        _, saved_path = draw_detections(img_path, detections)
        print(f"  Saved annotated output: {saved_path.relative_to(REPO_ROOT)}\n")

    print(f"Done. Processed {len(sample_images)} image(s) with {total_detections} total detection(s).")
    print(f"Annotated outputs saved in: {outputs_dir.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
