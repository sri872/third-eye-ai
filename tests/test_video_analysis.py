"""Phase 14 tests for sampled-frame temporal evidence (logic only, no fake media)."""

from __future__ import annotations

from app.video_analysis import analyze_frame_scores, build_video_assessment
from app.video_utils import VideoMetadata


def _rows(scores: list[float]) -> list[dict]:
    return [
        {
            "Frame": position,
            "Source frame": position * 10,
            "Timestamp (s)": position / 2.5,
            "Score": score,
            "Verdict": "UNCERTAIN",
            "Severity": "uncertain",
            "Vehicles": 1,
            "Persons": 0,
        }
        for position, score in enumerate(scores, start=1)
    ]


def test_sustained_high_signal_counts_longest_run() -> None:
    temporal = analyze_frame_scores(_rows([0.10, 0.65, 0.72, 0.20]))

    assert temporal["temporal_signal"] == "SUSTAINED HIGH SIGNAL IN SAMPLED FRAMES"
    assert temporal["sustained_high_run"] == 2
    assert temporal["high_frame_count"] == 2
    assert temporal["peak_source_frame"] == 30
    assert temporal["peak_position"] == 3


def test_isolated_high_frame_is_not_claimed_as_sustained() -> None:
    temporal = analyze_frame_scores(_rows([0.10, 0.61, 0.20, 0.18]))

    assert temporal["temporal_signal"] == "ISOLATED HIGH FRAME IN SAMPLE"
    assert temporal["sustained_high_run"] == 1


def test_no_signal_sample_has_explicit_label() -> None:
    temporal = analyze_frame_scores(_rows([0.10, 0.20, 0.30]))

    assert temporal["temporal_signal"] == "NO HIGH OR UNCERTAIN FRAME IN SAMPLE"
    assert temporal["no_signal_frame_count"] == 3


def test_video_assessment_keeps_authoritative_verdict_and_limitations() -> None:
    rows = _rows([0.20, 0.65])
    temporal = analyze_frame_scores(rows)
    metadata = VideoMetadata(
        frame_count=100,
        fps=25.0,
        width=640,
        height=480,
        duration_s=4.0,
    )

    assessment = build_video_assessment(metadata, rows, temporal, "UNCERTAIN")

    assert assessment["authoritative_verdict"] == "UNCERTAIN"
    assert "does not override" in assessment["assessment_text"]
    assert "not an accident confirmation" in assessment["assessment_text"]
    assert assessment["nominal_sampling_interval_s"] == 4.0
    assert assessment["frame_rows"] == rows
