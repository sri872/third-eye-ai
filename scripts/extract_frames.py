#!/usr/bin/env python3
"""Frame extraction script for Nexar video dataset.

Reads up to 200 videos per class in place (positive and negative).
Extracts 3 frames per video at identical relative timestamps (60%, 75%, 90%).
Resizes to 224x224 and saves as JPEG in data/frames/<class>/.
Saves metadata mapping each frame to its source video id.
"""

from pathlib import Path
import csv
import cv2

DATASET_ROOT = Path("/Users/srishanthbiradar/Datasets/ThirdEye/Nexar/test-public")
OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "data" / "frames"

RELATIVE_POSITIONS = [0.60, 0.75, 0.90]
MAX_VIDEOS_PER_CLASS = 200


def extract_class_frames(class_name: str, label_val: int, metadata_writer):
    video_dir = DATASET_ROOT / class_name
    class_out_dir = OUTPUT_ROOT / class_name
    class_out_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(list(video_dir.glob("*.mp4")))[:MAX_VIDEOS_PER_CLASS]
    print(f"Processing {len(videos)} videos for class '{class_name}'...")

    total_frames = 0
    for v_idx, v_path in enumerate(videos, 1):
        cap = cv2.VideoCapture(str(v_path))
        if not cap.isOpened():
            print(f"  [WARN] Failed to open {v_path.name}")
            continue

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            continue

        video_id = v_path.stem

        for pos in RELATIVE_POSITIONS:
            target_idx = int(round(pos * (frame_count - 1)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            # Resize to 224x224 for ResNet-18
            resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            frame_id = f"{video_id}_pos{int(pos*100)}_{target_idx}"
            frame_filename = f"{frame_id}.jpg"
            save_path = class_out_dir / frame_filename

            cv2.imwrite(str(save_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
            total_frames += 1

            metadata_writer.writerow([
                frame_id,
                video_id,
                class_name,
                label_val,
                pos,
                target_idx,
                str(save_path.relative_to(OUTPUT_ROOT.parent.parent)),
            ])

        cap.release()
        if v_idx % 50 == 0 or v_idx == len(videos):
            print(f"  Processed {v_idx}/{len(videos)} videos ({total_frames} frames saved)")

    return len(videos), total_frames


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    metadata_csv_path = OUTPUT_ROOT / "metadata.csv"

    print("==================================================================")
    print("Project Third-Eye | Nexar Dataset Frame Extraction")
    print(f"Sampling positions: {RELATIVE_POSITIONS}")
    print(f"Target size: 224x224")
    print("==================================================================")

    with open(metadata_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "video_id", "class_name", "label", "relative_pos", "frame_idx", "rel_path"])

        pos_vids, pos_frames = extract_class_frames("positive", 1, writer)
        neg_vids, neg_frames = extract_class_frames("negative", 0, writer)

    total_size = sum(f.stat().st_size for f in OUTPUT_ROOT.rglob("*.jpg")) / (1024 * 1024)

    print("\nExtraction Summary:")
    print(f"  Positive: {pos_vids} videos, {pos_frames} frames")
    print(f"  Negative: {neg_vids} videos, {neg_frames} frames")
    print(f"  Total frames: {pos_frames + neg_frames}")
    print(f"  Total disk space used: {total_size:.2f} MB")
    print(f"  Metadata written to: {metadata_csv_path}")
    print("==================================================================")


if __name__ == "__main__":
    main()
