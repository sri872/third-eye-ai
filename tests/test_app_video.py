"""Phase 8a-2 video tests using a real Nexar clip cut into a temporary file.

No synthetic frames are generated: the temporary clip contains frames copied
verbatim from a real video under ~/Datasets/ThirdEye/Nexar.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from app.video_utils import DEFAULT_SAMPLE_COUNT, probe_video, sample_video_frames

NEXAR_ROOT = Path.home() / "Datasets" / "ThirdEye" / "Nexar" / "test-public"
NEXAR_SUBDIRS = ("positive", "negative")


def _real_source_video() -> Path:
    for subdir in NEXAR_SUBDIRS:
        root = NEXAR_ROOT / subdir
        try:
            if not root.exists():
                continue
            candidates = sorted(root.glob("*.mp4"))
            if candidates:
                return candidates[0]
        except (PermissionError, OSError):
            continue
    pytest.skip(
        f"Real Nexar dataset not accessible under {NEXAR_ROOT} ({', '.join(NEXAR_SUBDIRS)}); "
        "skipping video test rather than generating fake video."
    )


def _cut_real_clip(tmp_path: Path, seconds: float = 2.0) -> Path:
    """Copy 1–2 seconds of real frames into a temp MP4 using OpenCV."""
    source = _real_source_video()
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        pytest.skip(f"Real Nexar video could not be opened: {source}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        capture.release()
        pytest.skip(f"Real Nexar video has no usable resolution: {source}")

    output = tmp_path / f"real_clip_{source.stem}.mp4"
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        pytest.skip("OpenCV MP4 writer unavailable; cannot cut a real clip.")

    target_frames = max(1, min(int(round(fps * seconds)), 90))
    written = 0
    try:
        for _ in range(target_frames):
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            writer.write(frame)
            written += 1
    finally:
        writer.release()
        capture.release()

    if written == 0 or not output.exists() or output.stat().st_size == 0:
        pytest.skip(f"Could not cut a readable real clip from {source}")
    return output


def test_probe_video_reads_real_metadata(tmp_path: Path) -> None:
    clip = _cut_real_clip(tmp_path)
    metadata = probe_video(clip)

    assert metadata.frame_count > 0
    assert metadata.fps is not None and metadata.fps > 0
    assert metadata.width > 0 and metadata.height > 0
    assert metadata.duration_s is not None and metadata.duration_s > 0
    assert metadata.duration_s <= 3.0, (
        "The temporary clip should be 1–2 seconds of real video, got "
        f"{metadata.duration_s}s"
    )


def test_frame_sampler_covers_whole_real_clip(tmp_path: Path) -> None:
    clip = _cut_real_clip(tmp_path)
    metadata = probe_video(clip)
    sampled = sample_video_frames(clip, sample_count=DEFAULT_SAMPLE_COUNT)

    assert len(sampled.frames) == DEFAULT_SAMPLE_COUNT
    assert sampled.metadata == metadata

    indices = [frame.index for frame in sampled.frames]
    assert indices == sorted(indices)
    assert len(set(indices)) == DEFAULT_SAMPLE_COUNT
    assert indices[0] == 0
    assert indices[-1] == metadata.frame_count - 1

    for frame in sampled.frames:
        assert frame.image.size == (metadata.width, metadata.height)
        assert frame.timestamp_s is not None
        assert 0 <= frame.timestamp_s <= metadata.duration_s

    timestamps = [frame.timestamp_s for frame in sampled.frames]
    assert timestamps == sorted(timestamps)
    assert all(current < following for current, following in zip(timestamps, timestamps[1:]))


def test_frame_sampler_honors_configured_count(tmp_path: Path) -> None:
    clip = _cut_real_clip(tmp_path)
    sampled = sample_video_frames(clip, sample_count=5)

    indices = [frame.index for frame in sampled.frames]
    assert len(indices) == 5
    assert len(set(indices)) == 5
    assert indices == sorted(indices)
    assert indices[0] == 0
    assert indices[-1] == sampled.metadata.frame_count - 1
