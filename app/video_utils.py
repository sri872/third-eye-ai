"""Video helpers for Phase 8a manual-upload dashboard.

Only reads video bytes already supplied by the user (written to a temporary file by
the caller). Nothing here writes into the project, touches SQLite, or fabricates
frame data. Frames are real decoded frames from the uploaded clip.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image

DEFAULT_SAMPLE_COUNT = 12


@dataclass(frozen=True)
class VideoMetadata:
    """Real container metadata reported by OpenCV."""

    frame_count: int
    fps: float | None
    width: int
    height: int
    duration_s: float | None


@dataclass(frozen=True)
class SampledFrame:
    """A real decoded frame and its position in the source clip."""

    index: int
    timestamp_s: float | None
    image: Image.Image


@dataclass(frozen=True)
class SampledVideo:
    metadata: VideoMetadata
    frames: tuple[SampledFrame, ...]


def _open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(
            f"Could not open video '{path.name}'. The file may be corrupt, empty, "
            "or encoded with a codec OpenCV cannot read."
        )
    return cap


def _count_frames(cap: cv2.VideoCapture) -> int:
    """Fallback for containers that do not report CAP_PROP_FRAME_COUNT."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    count = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        count += 1
    return count


def probe_video(path: Path) -> VideoMetadata:
    """Read real duration/resolution/fps metadata without decoding all frames."""
    cap = _open_video(path)
    try:
        reported_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_count = reported_count if reported_count > 0 else _count_frames(cap)

        fps_value = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        fps = fps_value if np.isfinite(fps_value) and fps_value > 0 else None
        duration = round(frame_count / fps, 3) if fps and frame_count > 0 else None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            if ok and frame is not None:
                height, width = frame.shape[:2]

        return VideoMetadata(
            frame_count=frame_count,
            fps=round(fps, 3) if fps else None,
            width=width,
            height=height,
            duration_s=duration,
        )
    finally:
        cap.release()


def _frame_indices(frame_count: int, sample_count: int) -> list[int]:
    """Choose evenly spaced indices across the whole clip, including both ends."""
    if frame_count <= 0:
        return []
    count = max(1, min(sample_count, frame_count))
    if count == 1:
        return [0]
    return [int(round(i * (frame_count - 1) / (count - 1))) for i in range(count)]


def sample_video_frames(
    path: Path,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    progress_callback: Callable[[float], None] | None = None,
) -> SampledVideo:
    """Sample real frames evenly across the entire video.

    Seeking is preferred; if a container cannot seek, the helper falls back to a
    sequential decode up to the requested index. Only the sampled frames are kept.
    """
    metadata = probe_video(path)
    indices = _frame_indices(metadata.frame_count, sample_count)
    if not indices:
        raise ValueError(f"Video '{path.name}' has no readable frames.")

    cap = _open_video(path)
    frames: list[SampledFrame] = []
    try:
        cursor = 0
        for position, target_index in enumerate(indices, start=1):
            if target_index < cursor:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                cursor = 0

            if target_index > cursor:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_index)
                cursor = target_index

            ok, bgr = cap.read()
            if not ok or bgr is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                cursor = 0
                while cursor < target_index:
                    ok, bgr = cap.read()
                    if not ok or bgr is None:
                        raise ValueError(
                            f"Video '{path.name}' stopped decoding before frame {target_index}."
                        )
                    cursor += 1
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    raise ValueError(
                        f"Video '{path.name}' stopped decoding before frame {target_index}."
                    )
            cursor += 1

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(
                SampledFrame(
                    index=target_index,
                    timestamp_s=round(target_index / metadata.fps, 3)
                    if metadata.fps
                    else None,
                    image=Image.fromarray(rgb),
                )
            )
            if progress_callback is not None:
                progress_callback(position / len(indices))
    finally:
        cap.release()

    return SampledVideo(metadata=metadata, frames=tuple(frames))
