"""Regression tests for the demo-input defects found during live verification.

These are logic-level tests using in-memory images and injected model outputs.
They complement the real-file diagnostics recorded in DEMO_READINESS_REPORT.md.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from PIL import Image

import app.main as main
import core.cv.detector as detector
import core.cv.plate_detector as plate_detector
import core.ocr.plate_reader as plate_reader
from app.video_utils import SampledFrame, SampledVideo, VideoMetadata
from core.cv.plate_detector import (
    OCR_FALLBACK_METHOD,
    detect_plate_regions_with_ocr_fallback,
)
from core.nlp.caller_extraction import extract_caller_info
from core.schemas import Detection


class _FakeReader:
    def __init__(self, text: str, confidence: float) -> None:
        self.text = text
        self.confidence = confidence

    def readtext(self, crop: np.ndarray):
        return [([(10, 5), (70, 5), (70, 15), (10, 15)], self.text, self.confidence)]


def test_detect_with_orientation_recovers_zero_detection_photo(monkeypatch) -> None:
    image = Image.new("RGB", (40, 20), (110, 110, 110))
    seen: list[tuple[int, int]] = []
    detection = Detection(
        detection_id="det-test",
        label="car",
        bbox=[0, 0, 10, 10],
        confidence=0.92,
        data_source="REAL",
    )

    def fake_detect(img: Image.Image, conf_threshold: float = 0.5):
        seen.append(img.size)
        return [] if img.size == (40, 20) else [detection]

    monkeypatch.setattr(detector, "detect", fake_detect)
    detections, oriented, info = detector.detect_with_orientation(image)

    assert detections == [detection]
    assert seen[0] == (40, 20)
    assert oriented.size == (20, 40)
    assert info["rotation_applied"] is True
    assert info["rotation_degrees"] == 90
    assert info["attempted_rotations_deg"][0] == 90
    assert info["source_size_px"] == [40, 20]
    assert info["analyze_size_px"] == [20, 40]


def test_detect_with_orientation_keeps_source_orientation_when_detected(monkeypatch) -> None:
    image = Image.new("RGB", (40, 20), (110, 110, 110))
    detection = Detection(
        detection_id="det-test",
        label="person",
        bbox=[0, 0, 5, 10],
        confidence=0.8,
        data_source="REAL",
    )
    called: list[tuple[int, int]] = []

    def fake_detect(img: Image.Image, conf_threshold: float = 0.5):
        called.append(img.size)
        return [detection]

    monkeypatch.setattr(detector, "detect", fake_detect)
    detections, oriented, info = detector.detect_with_orientation(image)

    assert detections == [detection]
    assert called == [(40, 20)]
    assert oriented.size == (40, 20)
    assert info["rotation_applied"] is False
    assert info["rotation_degrees"] == 0


def test_ocr_fallback_proposes_padded_plate_like_region(monkeypatch) -> None:
    crop = np.full((60, 120, 3), 180, dtype=np.uint8)
    monkeypatch.setattr(plate_detector, "detect_plate_regions", lambda *a, **k: ())
    monkeypatch.setattr(
        plate_reader, "get_ocr_reader", lambda: (_FakeReader("MH12AB1234", 0.62), "cpu")
    )

    regions = detect_plate_regions_with_ocr_fallback(crop, (100, 200, 220, 260))

    assert len(regions) == 1
    region = regions[0]
    assert region.method == OCR_FALLBACK_METHOD
    assert region.bbox_local == (2, 0, 78, 23)
    assert region.bbox_original == (102, 200, 178, 223)
    assert region.ocr_confidence == 0.62
    assert region.heuristic_score == 0.0


def test_ocr_fallback_rejects_short_non_plate_read(monkeypatch) -> None:
    crop = np.full((60, 120, 3), 180, dtype=np.uint8)
    monkeypatch.setattr(plate_detector, "detect_plate_regions", lambda *a, **k: ())
    monkeypatch.setattr(plate_reader, "get_ocr_reader", lambda: (_FakeReader("can09g", 0.9), "cpu"))

    assert detect_plate_regions_with_ocr_fallback(crop) == ()


def test_mandatory_caller_descriptions_extract_only_stated_facts() -> None:
    collision = extract_caller_info(
        "Two cars collided near NH-44. One person may be injured. "
        "Please send emergency assistance.",
        call_id="collision",
    )
    assert collision.location == "NH-44"
    assert collision.vehicle_count == 2
    assert collision.vehicle_types == ["car"]
    assert collision.people_involved == 1
    assert collision.injury_keywords == ["injured"]
    assert collision.urgency_keywords == ["emergency assistance"]

    traffic = extract_caller_info(
        "There are many cars and motorcycles near the main road. "
        "Several pedestrians are crossing. No collision has been reported.",
        call_id="traffic",
    )
    assert traffic.location == "the main road"
    assert traffic.vehicle_count is None
    assert traffic.vehicle_types == ["car", "motorcycle"]
    assert traffic.injury_keywords == []
    assert traffic.hazard_keywords == []

    uncertain = extract_caller_info(
        "I saw a vehicle stopped on the road. I am not sure whether an accident happened.",
        call_id="uncertain",
    )
    assert uncertain.vehicle_count == 1
    assert uncertain.people_involved is None
    assert uncertain.injury_keywords == []
    assert "not sure whether an accident happened" in uncertain.raw_text


def test_process_video_forwards_requested_sample_count(monkeypatch) -> None:
    captured: dict[str, int] = {}
    image = Image.new("RGB", (8, 8), (120, 120, 120))
    frames = tuple(
        SampledFrame(index=index, timestamp_s=float(index), image=image)
        for index in range(4)
    )

    def fake_sample(path, *, sample_count, progress_callback=None):
        captured["sample_count"] = sample_count
        return SampledVideo(
            VideoMetadata(
                frame_count=4, fps=1.0, width=8, height=8, duration_s=4.0
            ),
            frames,
        )

    monkeypatch.setattr(main, "sample_video_frames", fake_sample)
    monkeypatch.setattr(main, "detect", lambda image: [])
    monkeypatch.setattr(main, "classify_frame_probability", lambda image: 0.5)
    monkeypatch.setattr(
        main.st,
        "progress",
        lambda *a, **k: SimpleNamespace(progress=lambda *a, **k: None),
    )

    result = main._process_video(
        b"placeholder", "tiny.mp4", {"latitude": 12.97, "longitude": 77.59}, sample_count=7
    )

    assert captured["sample_count"] == 7
    assert len(result["frame_rows"]) == 4
    assert result["media_type"] == "video"
