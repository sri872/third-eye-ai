"""Frame-level and temporal evidence analysis for sampled uploaded videos.

This module works only on classifier outputs already produced by the app. It
does not decode video, retrain a model, or turn a small sample into a confirmed
accident. Thresholds mirror the existing deployment bands and remain explanatory.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Sequence

HIGH_THRESHOLD = 0.60
UNCERTAIN_THRESHOLD = 0.40
MIN_SUSTAINED_FRAMES = 2


def _longest_true_run(values: Sequence[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def analyze_frame_scores(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarise temporal signal from chronological frame-level classifier rows."""
    scores = [float(row["Score"]) for row in rows]
    if not scores:
        return {
            "sampled_frames": 0,
            "min_score": None,
            "max_score": None,
            "mean_score": None,
            "high_frame_count": 0,
            "uncertain_frame_count": 0,
            "no_signal_frame_count": 0,
            "sustained_high_run": 0,
            "sustained_uncertain_or_high_run": 0,
            "temporal_signal": "NO FRAMES ANALYSED",
            "peak_position": None,
            "peak_source_frame": None,
            "peak_timestamp_s": None,
        }

    peak_position = max(range(len(scores)), key=lambda index: scores[index])
    peak_row = rows[peak_position]
    high = [score >= HIGH_THRESHOLD for score in scores]
    uncertain = [score >= UNCERTAIN_THRESHOLD for score in scores]
    high_count = int(sum(high))
    uncertain_count = int(sum(score >= UNCERTAIN_THRESHOLD and score < HIGH_THRESHOLD for score in scores))
    no_signal_count = int(sum(score < UNCERTAIN_THRESHOLD for score in scores))
    sustained_high = _longest_true_run(high)
    sustained_uncertain = _longest_true_run(uncertain)

    if sustained_high >= MIN_SUSTAINED_FRAMES:
        signal = "SUSTAINED HIGH SIGNAL IN SAMPLED FRAMES"
    elif high_count >= 1:
        signal = "ISOLATED HIGH FRAME IN SAMPLE"
    elif sustained_uncertain >= MIN_SUSTAINED_FRAMES:
        signal = "SUSTAINED UNCERTAIN SIGNAL IN SAMPLED FRAMES"
    elif uncertain_count >= 1:
        signal = "ISOLATED UNCERTAIN FRAME IN SAMPLE"
    else:
        signal = "NO HIGH OR UNCERTAIN FRAME IN SAMPLE"

    return {
        "sampled_frames": len(scores),
        "min_score": round(min(scores), 4),
        "max_score": round(max(scores), 4),
        "mean_score": round(sum(scores) / len(scores), 4),
        "high_frame_count": high_count,
        "uncertain_frame_count": uncertain_count,
        "no_signal_frame_count": no_signal_count,
        "sustained_high_run": sustained_high,
        "sustained_uncertain_or_high_run": sustained_uncertain,
        "temporal_signal": signal,
        "peak_position": peak_position + 1,
        "peak_source_frame": peak_row.get("Source frame"),
        "peak_timestamp_s": peak_row.get("Timestamp (s)"),
    }


def build_video_assessment(
    metadata: Any,
    frame_rows: Sequence[Mapping[str, Any]],
    temporal: Mapping[str, Any],
    authoritative_verdict: str,
) -> dict[str, Any]:
    """Combine real metadata, frame-level rows, and explanatory temporal signal."""
    duration = float(metadata.duration_s) if getattr(metadata, "duration_s", None) else None
    sampled = len(frame_rows)
    interval = (
        round(duration / (sampled - 1), 3)
        if duration and sampled > 1
        else None
    )
    signal = str(temporal.get("temporal_signal", "NO FRAMES ANALYSED"))
    assessment = (
        f"The trained model verdict remains {authoritative_verdict}, taken from the "
        "highest-scoring sampled frame. Temporal evidence is explanatory only and does "
        f"not override it: {signal}. This is an even-sample review, not full-clip "
        "inference, and it is not an accident confirmation."
    )
    return {
        "source": "REAL uploaded video + sampled-frame classifier output",
        "metadata": asdict(metadata),
        "frame_rows": list(frame_rows),
        "temporal": dict(temporal),
        "authoritative_verdict": authoritative_verdict,
        "nominal_sampling_interval_s": interval,
        "assessment_text": assessment,
        "limitation": (
            "Evenly spaced frames only; no full-clip inference, no OCR on every frame, "
            "and no temporal ground truth. Simulated caller evidence never overrides "
            "the REAL vision verdict."
        ),
    }
