"""End-to-end Streamlit AppTest verification of the professor demonstration.

Runs the real app with real uploaded bytes (no monkeypatched models) for the
three mandatory images and the road video, checks all six tabs, and writes a
compact evidence summary to data/outputs/streamlit_workflow.json.

Run from the repository root:

    .venv/bin/python scripts/verify_streamlit_workflow.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

APP_FILE = REPO_ROOT / "app" / "main.py"
DEMO_DIR = REPO_ROOT / "data" / "samples" / "demo"
CALL_OPTION = "call_demo_01_collision.txt | SIMULATED SAMPLE CALL"

DEMO_FILES = (
    ("single_car_plate.jpg", "image/jpeg", None),
    ("two_car_collision.jpg", "image/jpeg", CALL_OPTION),
    ("heavy_traffic.jpg", "image/jpeg", None),
    ("road vdo.mp4", "video/mp4", None),
)


def tab_text(at: AppTest, index: int) -> str:
    tab = at.tabs[index]
    return " ".join(
        str(node.value)
        for kind in ("markdown", "caption", "text", "info", "warning", "error")
        for node in tab.get(kind)
    )


def run_demo(name: str, mime: str, call_option: str | None) -> dict:
    path = DEMO_DIR / name
    assert path.is_file(), path
    raw = path.read_bytes()
    at = AppTest.from_file(str(APP_FILE), default_timeout=900).run()
    assert not at.exception, [item.value for item in at.exception]

    at.sidebar.file_uploader[0].upload(name, raw, mime)
    if call_option:
        at.sidebar.selectbox[0].set_value(call_option)
    at.run()
    assert not at.exception, [item.value for item in at.exception]
    assert not at.error, [item.value for item in at.error]
    assert len(at.tabs) == 6

    result = dict(at.session_state["uploaded_result"])
    scene, vehicles, accident, report, hospitals, dispatch = (
        tab_text(at, index) for index in range(6)
    )
    assert "Media fingerprint (SHA-256)" in scene
    assert "Vehicle count (detected)" in scene
    assert result["media_sha256"] == hashlib.sha256(raw).hexdigest()
    if result["media_type"] == "image":
        assert all(
            expected in accident
            for expected in ("Assessment", "Raw model output", "Uncalibrated model score")
        )
        assert result["verdict"] == "NOT DETERMINED (out-of-domain still image)"
        assert result["raw_model_verdict"] in {
            "LIKELY ACCIDENT", "UNCERTAIN", "NO ACCIDENT SIGNAL"
        }
    else:
        assert all(
            expected in accident
            for expected in ("Model verdict", "Uncalibrated model score")
        )
    assert "Incident report" in report or "SOURCE" in report.upper()
    assert "Hospitals" in hospitals or "OSM" in hospitals
    assert "SIMULATED" in dispatch

    summary: dict = {
        "file": name,
        "bytes": len(raw),
        "sha256": result["media_sha256"],
        "media_type": result["media_type"],
        "score": result["score"],
        "verdict": result["verdict"],
        "scene_status": result["status"],
        "vehicle_count": sum(
            1 for item in result["detections"] if item.label in {
                "car", "truck", "bus", "motorcycle", "bicycle"
            }
        ),
        "person_count": sum(1 for item in result["detections"] if item.label == "person"),
        "vehicles_table_rows": len(
            [node for node in at.tabs[1].table]
        ),
        "tabs_rendered": len(at.tabs),
        "ocr_evidence": [
            {
                "plate": evidence.plate,
                "plate_confidence": evidence.plate_confidence,
                "raw_text": evidence.raw_text,
                "raw_confidence": evidence.raw_confidence,
                "plate_detection_status": evidence.plate_detection_status,
            }
            for evidence in result["ocr_evidence"].values()
        ],
        "report_has_source_separation": "CALLER_SIMULATED" in report or "VISION" in report,
        "hospital_tab_mentions_osm": "OSM" in hospitals,
        "dispatch_tab_is_simulated": "SIMULATED" in dispatch,
    }
    if result["media_type"] == "video":
        summary["video_metadata"] = {
            "frame_count": result["video_metadata"].frame_count,
            "fps": result["video_metadata"].fps,
            "duration_s": result["video_metadata"].duration_s,
            "width": result["video_metadata"].width,
            "height": result["video_metadata"].height,
        }
        summary["sampled_frame_rows"] = result["frame_rows"]
        summary["temporal_signal"] = result["video_temporal_signal"]
        summary["video_assessment"] = result["video_assessment"]
        assert f"Frames sampled" in accident
        assert all("Timestamp (s)" in str(row) for row in [summary["sampled_frame_rows"][0]])
    return summary


def main() -> None:
    findings = [run_demo(name, mime, call) for name, mime, call in DEMO_FILES]

    # Repeated switching: no stale result may leak from one upload to the next.
    at = AppTest.from_file(str(APP_FILE), default_timeout=900).run()
    first = DEMO_DIR / "two_car_collision.jpg"
    second = DEMO_DIR / "single_car_plate.jpg"
    at.sidebar.file_uploader[0].upload(first.name, first.read_bytes(), "image/jpeg").run()
    assert not at.exception, [item.value for item in at.exception]
    first_sha = at.session_state["uploaded_result"]["media_sha256"]
    at.sidebar.file_uploader[0].upload(second.name, second.read_bytes(), "image/jpeg").run()
    assert not at.exception, [item.value for item in at.exception]
    second_sha = at.session_state["uploaded_result"]["media_sha256"]
    at.sidebar.file_uploader[0].upload(first.name, first.read_bytes(), "image/jpeg").run()
    assert not at.exception, [item.value for item in at.exception]
    again_sha = at.session_state["uploaded_result"]["media_sha256"]
    assert first_sha == again_sha != second_sha
    switching = {
        "first_sha256": first_sha,
        "second_sha256": second_sha,
        "third_sha256": again_sha,
        "no_stale_result": first_sha == again_sha != second_sha,
    }

    output = REPO_ROOT / "data" / "outputs" / "streamlit_workflow.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"demos": findings, "input_switching": switching}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"demos": findings, "input_switching": switching}, indent=2))
    print(f"Wrote Streamlit workflow evidence to {output.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
