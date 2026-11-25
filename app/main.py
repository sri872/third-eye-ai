"""Third Eye AI — six-tab Streamlit demonstration dashboard.

One Streamlit page. Nothing runs on data/samples automatically: the user must
upload one image or video. SQLite is always opened read-only (mode=ro); images
stay in memory and videos use an auto-deleted temporary file outside the project.
"""

from __future__ import annotations

import hashlib
import html
import io
import json
import sqlite3
import sys
import tempfile
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.incident_report import build_report_sections, narrative_markdown
from app.dispatch_ui import (
    AMBULANCE_SELECTION_CRITERIA,
    FALLBACK_TILE_NAME,
    GOOGLE_MAPS_LABEL,
    GPS_NOTICE,
    LIVE_TRAFFIC_NOTICE,
    NOT_DETERMINED,
    SIMULATION_NOTICE,
    alternative_rows,
    ambulance_selection_reason,
    build_dispatch_map,
    build_google_maps_url,
    configured_summary_path,
    coord_text,
    coordinate_pair,
    display,
    emergency_plain_text,
    emergency_status_from_capability,
    fallback_geometry_check,
    hospital_alternative_rows,
    hospital_capability_text,
    hospital_eligibility,
    load_dispatch_summary,
    navigation_ready,
    location_audit,
    number_text,
    optional_value,
    replay_follows_route,
    replay_labels,
    replay_phase,
    replay_position,
    route_audit_lines,
    route_verification,
    routing_check_lines,
    scenarios,
    selection_consistency,
    with_unit,
)
from core.routing.dispatch import HOSPITAL_RANKING_PRIORITY
from app.hospital_ui import (
    GENERAL_NUMBERS_NOTE,
    HOSPITAL_NUMBERS_NOTE,
    LIMITATION_LINES,
    general_emergency_rows,
    hospital_number_lines,
    recommendation_rows,
)
from app.video_analysis import analyze_frame_scores, build_video_assessment
from app.video_utils import sample_video_frames
from core.cv.detector import detect, detect_with_orientation, get_target_device
from core.cv.plate_detector import (
    OCR_FALLBACK_METHOD,
    detect_plate_regions_with_ocr_fallback,
)
from core.cv.vehicle_attributes import crop_vehicle, dominant_color, get_vehicle_type
from core.ml.accident_analyzer import (
    classify_frame_probability,
    compute_persons_near_vehicles,
    determine_verdict,
    get_device,
    load_model_and_extractor,
)
from core.ml.accident_features import extract_accident_features
from core.nlp.caller_extraction import extract_caller_info
from core.nlp.fusion import fuse_incident_evidence
from core.nlp.incident_text import generate_incident_summary
from core.nlp.occupant_report import build_occupant_reports
from core.ocr.plate_reader import (
    PlateOCREvidence,
    preprocess_plate_region,
    read_plate_evidence,
)
from core.rag.emergency_numbers import get_emergency_numbers
from core.rag.facility_classification import dispatch_eligible_facilities
from core.rag.hospital_data import fetch_or_load_hospitals
from core.rag.index import HospitalIndex
from core.rag.location_consistency import compare_locations
from core.rag.retriever import HospitalRetriever, formulate_retrieval_query
from core.schemas import AccidentEvent, Detection, VehicleReport

DB_PATH = REPO_ROOT / "db" / "third_eye.db"
CALLS_DIR = REPO_ROOT / "data" / "sample_calls"
ROUTES_IMG_DIR = REPO_ROOT / "data" / "outputs" / "routes"
METRICS_PATH = REPO_ROOT / "models" / "accident_classifier_metrics.json"
LOCATIONS_PATH = REPO_ROOT / "config" / "demo_incident_locations.json"
SUMMARY_PATH = configured_summary_path()

UPLOAD_TYPES = ["jpg", "jpeg", "png", "mp4", "mov", "avi"]
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}
NO_CALL_LABEL = "No call (vision only)"
PLATE_TRUST_THRESHOLD = 0.7
NOT_READABLE = "NOT READABLE (UNVERIFIED)"
PLATE_DETECTED = "PLATE REGION DETECTED"
NO_PLATE_REGION = "NO PLATE REGION DETECTED"
STILL_IMAGE_STATUS = "NOT DETERMINED (out-of-domain still image)"
DISPATCH_MISSING_MESSAGE = "Dispatch results not available yet"
SAMPLE_COUNT = 12
MAX_SAMPLE_COUNT = 60
LONG_VIDEO_SECONDS = 180.0

PROTOTYPE_STATEMENT = (
    "This is a prototype frame classifier trained on dashcam clips "
    "(held-out ROC-AUC 0.7452); a single high frame is not proof of an accident."
)

# Verbatim limitation text from docs/model_card_accident.md, section 5.
ACCIDENT_LIMITATIONS = (
    "Dataset Bias & Generalization: The model was trained on forward-facing dashcam "
    "footage from the Nexar dataset. It may not generalize reliably to high-angle CCTV "
    "or overhead surveillance angles.",
    "Imminent vs Post-Impact: Nexar positive clips terminate within 500ms–1500ms of "
    "collision. Frames capture imminent impact trajectories; post-impact wreckage without "
    "dynamic approach motion may have lower sensitivity.",
    "Uncalibrated Probabilities: Outputs must be interpreted as ranking scores rather than "
    "absolute true-world accident probabilities.",
)

st.set_page_config(
    page_title="Third Eye AI | Six-Tab Demonstration",
    page_icon="🚑",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Badges and display helpers
# ---------------------------------------------------------------------------
def _badge(source: str) -> None:
    """Render the mandatory REAL / SIMULATED provenance badge."""
    normalized = str(source).upper()
    if normalized == "REAL":
        st.badge("REAL", color="green")
    elif normalized == "SIMULATED":
        st.badge("SIMULATED", color="orange")
    else:
        st.badge(str(source), color="gray")


def tagged(label: str, value: Any, source: str) -> None:
    """Display one labelled value with its provenance badge."""
    left, right = st.columns([6, 1])
    with left:
        st.markdown(f"**{label}:** {html.escape(str(value))}")
    with right:
        _badge(source)


def _text(value: Any) -> str:
    if value is None or value == "":
        return "not determined"
    return str(value)


def tagged_plain(label: str, value: Any, source: str) -> None:
    """Render a value as plain text (never a phone link) with a provenance badge."""
    left, right = st.columns([6, 1])
    with left:
        st.text(f"{label}: {value}")
    with right:
        _badge(source)


def render_emergency_numbers(urgency: Any, hazards: Any) -> None:
    """Emergency numbers from core/rag/emergency_numbers.py, shown as plain text."""
    st.text("This app does not place calls; dial from your phone.")
    for contact in get_emergency_numbers(urgency=urgency, hazards=hazards):
        tagged_plain(
            display(contact.get("service")),
            emergency_plain_text(contact),
            "REAL",
        )


# ---------------------------------------------------------------------------
# Static, read-only helpers (never load a model on the no-upload path)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _db_tables() -> tuple[str, ...]:
    if not DB_PATH.exists():
        return ()
    con: sqlite3.Connection | None = None
    try:
        con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        rows = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        return tuple(str(row[0]) for row in rows)
    except sqlite3.Error:
        return ()
    finally:
        if con is not None:
            con.close()


@st.cache_data(show_spinner=False)
def _db_rows(table: str) -> tuple[tuple[str, ...], tuple[tuple[Any, ...], ...]]:
    if table not in _db_tables():
        return (), ()
    con: sqlite3.Connection | None = None
    try:
        con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        cursor = con.execute(f'SELECT * FROM "{table}"')
        columns = tuple(str(desc[0]) for desc in (cursor.description or []))
        return columns, tuple(tuple(row) for row in cursor.fetchall())
    except sqlite3.Error:
        return (), ()
    finally:
        if con is not None:
            con.close()


@st.cache_data(show_spinner=False)
def _held_out_metrics() -> dict[str, Any]:
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def _demo_locations() -> tuple[dict[str, Any], ...]:
    if not LOCATIONS_PATH.exists():
        return ()
    return tuple(json.loads(LOCATIONS_PATH.read_text(encoding="utf-8")))


@st.cache_data(show_spinner=False)
def _call_names() -> tuple[str, ...]:
    return tuple(sorted(path.name for path in CALLS_DIR.glob("call_*.txt")))


@st.cache_data(show_spinner=False)
def _call_text(name: str) -> str:
    return (CALLS_DIR / name).read_text(encoding="utf-8")


@st.cache_data(show_spinner=False)
def _dispatch_summary(path: str) -> dict[str, Any]:
    """Read only the real Phase 6d summary; the path is injectable for tests."""
    return load_dispatch_summary(path)


@st.cache_resource(show_spinner=False)
def _detection_model():
    from core.cv.detector import get_detector_model

    return get_detector_model()


@st.cache_resource(show_spinner=False)
def _accident_model():
    return load_model_and_extractor()


@st.cache_resource(show_spinner=False)
def _ocr_reader():
    from core.ocr.plate_reader import get_ocr_reader

    return get_ocr_reader()


@st.cache_resource(show_spinner=False)
def _hospital_resources() -> tuple[HospitalRetriever | None, dict[str, Any], str, int, int]:
    """Load OSM records, then index only dispatch-eligible hospitals/trauma centres."""
    hospitals, stats, source = fetch_or_load_hospitals(save_to_db=False)
    eligible = dispatch_eligible_facilities(hospitals)
    retriever = HospitalRetriever(HospitalIndex(eligible)) if eligible else None
    return retriever, stats, source, len(hospitals), len(eligible)


# ---------------------------------------------------------------------------
# Phase 8b dispatch tab (summary file only, no invented values)
# ---------------------------------------------------------------------------
def _render_dispatch_tab(
    summary: dict[str, Any],
    urgency: Any,
    hazards: Any,
    report: dict[str, Any] | None = None,
    selected_location: dict[str, Any] | None = None,
) -> None:
    st.subheader("Dispatch: simulated demo placement, real OSM facilities")
    st.warning(SIMULATION_NOTICE)
    _badge("SIMULATED")
    st.caption(
        "This tab ranks records from the cached dispatch summary. It does not contact "
        "ambulances, verify vehicle GPS, check live traffic, or verify hospital capability."
    )

    demo_scenarios = scenarios(summary)
    disclaimers = summary.get("disclaimers")
    if not demo_scenarios:
        st.info(DISPATCH_MISSING_MESSAGE)
        tagged("Summary path checked", str(SUMMARY_PATH), "REAL")
        tagged("Dispatch data source", NOT_DETERMINED, "REAL")
        tagged("GPS verification", GPS_NOTICE, "REAL")
        st.markdown("**Emergency contacts (plain text):**")
        render_emergency_numbers(urgency, hazards)
        return

    options = [
        f"{optional_value(item.get('location_name'), NOT_DETERMINED)} | SIMULATED LOCATION"
        for item in demo_scenarios
    ]
    choice = st.selectbox(
        "Simulated demo incident",
        options=options,
        index=0,
        key="dispatch_scenario",
        help="SIMULATED demo location from the real dispatch summary file.",
    )
    scenario = demo_scenarios[options.index(choice)]

    incident = scenario.get("incident") or {}
    ambulance = scenario.get("ambulance") or {}
    hospital = scenario.get("hospital") or {}
    incident_pair = coordinate_pair(incident, "lat", "lon")
    base_pair = coordinate_pair(ambulance, "base_lat", "base_lon")

    # 1. Incident / dispatch location -------------------------------------
    st.markdown("### 1. Incident / dispatch location")
    tagged("Severity", "not determined (demo placement)", "SIMULATED")
    tagged("Location ID", display(scenario.get("location_id")), "SIMULATED")
    tagged("Location name", display(scenario.get("location_name")), "SIMULATED")
    tagged(
        "Dispatch coordinates used for both legs",
        ", ".join(number_text(value) for value in incident_pair) if incident_pair else NOT_DETERMINED,
        "SIMULATED",
    )
    tagged("Incident data source", display(incident.get("data_source")), "SIMULATED")

    audit = location_audit(report, selected_location, scenario)
    st.table(audit["rows"])
    for discrepancy in audit["discrepancies"]:
        st.warning(discrepancy)
    tagged("GPS verification", audit["gps_status"], "REAL")
    with st.expander("Location source details and provenance"):
        st.caption(
            "Three sources are kept separate. Only the SIMULATED dispatch demo location "
            "from the summary is used for Tab 6 routing. The caller-reported location and "
            "the sidebar hospital-search location are shown for comparison, never substituted."
        )
        tagged("Caller-reported comparison", audit["dispatch_status"], "SIMULATED")
        tagged("Hospital-search comparison", audit["sidebar_status"], "SIMULATED")
        tagged("Coordinates used", audit["dispatch_coordinates_used"], "SIMULATED")
        tagged("GPS verification", GPS_NOTICE, "REAL")

    # 2. Ambulance and alternatives ----------------------------------------
    st.markdown("### 2. Ambulance and alternatives (SIMULATED)")
    tagged(
        "Chosen ambulance (SIMULATED)",
        " / ".join(
            display(ambulance.get(key))
            for key in ("id", "callsign", "base_name")
        ),
        "SIMULATED",
    )
    tagged(
        "Ambulance base coordinates (SIMULATED)",
        ", ".join(number_text(value) for value in base_pair) if base_pair else NOT_DETERMINED,
        "SIMULATED",
    )
    tagged("Ambulance ETA", with_unit(ambulance.get("eta_seconds"), "s"), "SIMULATED")
    tagged(
        "Ambulance road distance",
        with_unit(ambulance.get("driving_distance_km"), "km"),
        "SIMULATED",
    )
    tagged(
        "Equipment level (configuration only)",
        optional_value(ambulance.get("equipment_level"), NOT_DETERMINED),
        "SIMULATED",
    )
    tagged("Selection reason", ambulance_selection_reason(scenario), "SIMULATED")
    tagged(
        "Selection criteria",
        display(scenario.get("selection_criteria") or AMBULANCE_SELECTION_CRITERIA),
        "SIMULATED",
    )
    st.caption(
        "No live GPS, vehicle availability, operational status or real ambulance contact "
        "is used. All fleet values are from simulated configuration."
    )
    ambulance_rows = alternative_rows(scenario.get("ambulance_alternatives"), "SIMULATED")
    if ambulance_rows:
        st.table(ambulance_rows)
    else:
        tagged("Ambulance alternatives", NOT_DETERMINED, "SIMULATED")
    with st.expander("Ambulance source details"):
        tagged("Fleet data source", display(ambulance.get("data_source")), "SIMULATED")
        tagged("Live GPS / operational status", "NOT AVAILABLE", "REAL")
        tagged("Selection criteria", AMBULANCE_SELECTION_CRITERIA, "REAL")

    # 3. Hospital and alternatives -----------------------------------------
    st.markdown("### 3. Hospital and alternatives (REAL OSM)")
    eligibility = hospital_eligibility(scenario)
    tagged("Tab 5 eligibility check", eligibility["status"], "SIMULATED")
    tagged("Facility category (OSM name/tags)", eligibility["category"], "REAL")
    tagged("Eligibility detail", eligibility["detail"], "REAL")
    if not eligibility["dispatch_eligible"]:
        st.error(
            "The stored destination is NOT a dispatch-eligible hospital or trauma centre "
            "under the Tab 5 classifier. Do not use it as an emergency destination."
        )
    tagged("Chosen hospital (REAL OSM)", display(hospital.get("name")), "REAL")
    tagged("Hospital OSM ID", display(hospital.get("osm_id")), "REAL")
    tagged_plain("Hospital phone (OSM)", optional_value(hospital.get("phone"), "not listed"), "REAL")
    tagged(
        "Emergency department tag (OSM; unverified)",
        optional_value(hospital.get("emergency_tag"), "not specified"),
        "REAL",
    )
    tagged(
        "Emergency status (Tab 5 rule)",
        emergency_status_from_capability(
            hospital.get("emergency_capability") or hospital.get("emergency_tag")
        ),
        "REAL",
    )
    tagged(
        "Specialties (OSM)",
        optional_value(hospital.get("specialties"), "not specified in OpenStreetMap"),
        "REAL",
    )
    tagged(
        "Emergency capability (OSM tags only)",
        hospital_capability_text(hospital),
        "REAL",
    )
    tagged("Hospital data source", display(hospital.get("data_source")), "REAL")
    tagged(
        "Driving distance (incident to hospital)",
        with_unit(hospital.get("driving_km"), "km"),
        "SIMULATED",
    )
    tagged(
        "Straight-line distance (incident to hospital)",
        with_unit(hospital.get("straight_line_km"), "km"),
        "SIMULATED",
    )
    tagged("Hospital transit", with_unit(hospital.get("transit_seconds"), "s"), "SIMULATED")
    tagged(
        "Ranking formula (ranking heuristic, not a clinical rating)",
        display(scenario.get("ranking_formula")),
        "SIMULATED",
    )
    tagged(
        "Hospital ranking priority (same as Tab 5)",
        display(scenario.get("ranking_priority") or HOSPITAL_RANKING_PRIORITY),
        "SIMULATED",
    )
    tagged("Bed availability", NOT_DETERMINED, "REAL")
    hospital_rows = hospital_alternative_rows(scenario)
    if hospital_rows:
        st.table(hospital_rows)
    else:
        tagged("Hospital alternatives", NOT_DETERMINED, "REAL")
    with st.expander("Hospital source details and limitations"):
        tagged("Map / data attribution", "© OpenStreetMap contributors", "REAL")
        tagged("OSM emergency tag", display(hospital.get("emergency_tag")), "REAL")
        tagged("OSM specialties", display(hospital.get("specialties")), "REAL")
        tagged(
            "Verified emergency department / trauma centre / specialties / beds",
            NOT_DETERMINED,
            "REAL",
        )
        st.caption(
            "Real OSM records are preserved with their original tags. Missing tags stay "
            "unspecified; proximity or a ranking score is not clinical verification."
        )

    # 4. Map, ETA and emergency contacts -----------------------------------
    st.markdown("### 4. Map, ETA and emergency contacts")
    st.caption(
        f"OpenStreetMap tiles are primary; switch to {FALLBACK_TILE_NAME} if tiles load "
        "partially. The static PNG fallback is derived from the same stored route legs."
    )
    replay_options = replay_labels(scenario)
    if replay_options:
        replay_choice = st.selectbox(
            "Simulated ambulance replay step",
            options=replay_options,
            index=0,
            key="dispatch_replay_step",
            help="SIMULATED replay only; no timer, no autoplay, no buttons.",
        )
        tagged("Replay step", replay_choice, "SIMULATED")
        tagged("Replay phase", replay_phase(scenario, replay_choice), "SIMULATED")
        follows = replay_follows_route(scenario, replay_choice)
        tagged(
            "Replay marker on stored route",
            "yes" if follows is True else ("no" if follows is False else NOT_DETERMINED),
            "SIMULATED",
        )
    else:
        replay_choice = None
        tagged("Replay trajectory", NOT_DETERMINED, "SIMULATED")

    tagged("Total distance", with_unit(scenario.get("total_km"), "km"), "SIMULATED")
    tagged("Total time", with_unit(scenario.get("total_seconds"), "s"), "SIMULATED")
    tagged("Travel-time basis", LIVE_TRAFFIC_NOTICE, "REAL")
    tagged("Simulation notice", SIMULATION_NOTICE, "SIMULATED")
    tagged("Map / data attribution", "© OpenStreetMap contributors", "REAL")

    st.markdown("**Data disclaimers:**")
    if isinstance(disclaimers, list) and disclaimers:
        for item in disclaimers:
            tagged("Disclaimer", display(item), "REAL")
    else:
        tagged("Disclaimers", NOT_DETERMINED, "REAL")

    from streamlit_folium import st_folium

    st_folium(
        build_dispatch_map(scenario, replay_choice),
        height=430,
        use_container_width=True,
        returned_objects=[],
    )
    with st.expander("Static route map fallback (use if OSM tiles cannot load)"):
        png_value = display(scenario.get("map_png"))
        if png_value == NOT_DETERMINED:
            tagged("Static map PNG", NOT_DETERMINED, "REAL")
        else:
            png_path = Path(png_value)
            if not png_path.is_absolute():
                png_path = REPO_ROOT / png_path
            if png_path.is_file():
                st.image(str(png_path), caption=f"Static fallback: {png_value}", width="stretch")
            else:
                tagged("Static map PNG", f"{png_value} (file not found)", "REAL")
        check = fallback_geometry_check(scenario)
        tagged("Static PNG available", "yes" if check["png_available"] else "no", "REAL")
        tagged("Fallback GeoJSON available", "yes" if check["geojson_available"] else "no", "REAL")
        tagged("Fallback geometry detail", check["detail"], "REAL")

    st.markdown("**External Google Maps links:**")
    incident_text = coord_text(incident_pair)
    base_text = coord_text(base_pair)
    hospital_text = coord_text(coordinate_pair(hospital, "lat", "lon"))
    if (
        navigation_ready(incident_pair)
        and navigation_ready(base_pair)
        and navigation_ready(coordinate_pair(hospital, "lat", "lon"))
        and incident_text
        and base_text
        and hospital_text
    ):
        base_via_hospital = build_google_maps_url(
            base_text, hospital_text, [incident_text]
        )
        incident_to_hospital = build_google_maps_url(incident_text, hospital_text)
        st.markdown(
            f"- [{GOOGLE_MAPS_LABEL} — ambulance base → incident → hospital]"
            f"({base_via_hospital})"
        )
        st.markdown(
            f"- [{GOOGLE_MAPS_LABEL} — incident → hospital]({incident_to_hospital})"
        )
    else:
        tagged("Google Maps links", NOT_DETERMINED, "REAL")

    st.markdown("**Emergency numbers (plain text):**")
    render_emergency_numbers(urgency, hazards)

    # Technical diagnostics ------------------------------------------------
    with st.expander("Technical diagnostics — route legs, distances and replay"):
        selection = selection_consistency(scenario)
        tagged(
            "Ambulance selection matches recorded ETAs",
            "yes" if selection["ambulance_lowest_eta"] else "no / not determined",
            "SIMULATED",
        )
        tagged(
            "Hospital selection matches Tab 5 priority",
            "yes" if selection["hospital_priority_ok"] else "no / not determined",
            "SIMULATED",
        )
        for line in selection["details"]:
            st.text(line)
        for line in route_audit_lines(scenario):
            st.text(line)
        for line in routing_check_lines(scenario):
            st.text(line)
        verification = route_verification(scenario)
        tagged(
            "Legs join at simulated incident",
            "yes" if verification.get("legs_join_at_incident") is True else "no / not determined",
            "SIMULATED",
        )
        tagged(
            "Repeated / shared route segments",
            (
                f"{display(verification.get('repeated_segments_leg1'))} / "
                f"{display(verification.get('repeated_segments_leg2'))} / "
                f"{display(verification.get('shared_segments_between_legs'))}"
            ),
            "SIMULATED",
        )
    with st.expander("Technical diagnostics — summary and source details"):
        tagged("Summary path", str(SUMMARY_PATH), "REAL")
        tagged("Summary generated at", display(summary.get("generated_at")), "REAL")
        tagged("Summary GPS status", display(summary.get("gps_status") or GPS_NOTICE), "REAL")
        tagged("Summary traffic note", display(summary.get("traffic_note") or LIVE_TRAFFIC_NOTICE), "REAL")


# ---------------------------------------------------------------------------
# In-memory vision pipeline (no DB writes, no project files written)
# ---------------------------------------------------------------------------
def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def _build_vehicle_reports(
    image: Image.Image, detections: list[Detection], stem: str
) -> tuple[list[VehicleReport], dict[str, PlateOCREvidence]]:
    """Compose Phase 2 reports in memory from in-memory core helpers.

    core.cv.vehicle_report.build_vehicle_reports is deliberately NOT used: it always
    creates annotated crop files on disk. These lower-level core functions return
    their results in memory and perform no I/O. Raw OCR evidence is returned beside
    the reports so the UI can show recognized text without inventing a plate.
    """
    bgr = _pil_to_bgr(image)
    reports: list[VehicleReport] = []
    ocr_evidence: dict[str, PlateOCREvidence] = {}
    vehicle_index = 0
    for detection in detections:
        if get_vehicle_type(detection.label) is None:
            continue
        vehicle_index += 1
        try:
            crop = crop_vehicle(bgr, detection)
        except (ValueError, TypeError):
            continue
        color, color_share = dominant_color(crop)
        crop_x0 = max(0, min(int(round(detection.bbox[0])), bgr.shape[1] - 1))
        crop_y0 = max(0, min(int(round(detection.bbox[1])), bgr.shape[0] - 1))
        vehicle_bbox = (
            crop_x0,
            crop_y0,
            crop_x0 + crop.shape[1],
            crop_y0 + crop.shape[0],
        )
        regions = detect_plate_regions_with_ocr_fallback(crop, vehicle_bbox)
        if regions:
            region = regions[0]
            lx0, ly0, lx1, ly1 = region.bbox_local
            plate_crop = crop[ly0:ly1, lx0:lx1]
            if region.method == OCR_FALLBACK_METHOD:
                detector_label = (
                    f"{region.method} (whole-crop OCR confidence "
                    f"{float(region.ocr_confidence or 0.0):.2f})"
                )
            else:
                detector_label = (
                    f"{region.method} (detector score {region.heuristic_score:.2f})"
                )
            try:
                processed = preprocess_plate_region(plate_crop)
                ev_proc = read_plate_evidence(processed)
                ev_raw = read_plate_evidence(plate_crop)
                ev_best = (
                    ev_raw
                    if (ev_raw.plate and (not ev_proc.plate or (ev_raw.plate_confidence or 0) > (ev_proc.plate_confidence or 0)))
                    else ev_proc
                )
                evidence = replace(
                    ev_best,
                    plate_detection_status=PLATE_DETECTED,
                    plate_bbox=region.bbox_original,
                    plate_detector=detector_label,
                )
            except Exception:
                evidence = replace(
                    PlateOCREvidence(),
                    plate_detection_status=PLATE_DETECTED,
                    plate_bbox=region.bbox_original,
                    plate_detector=region.method,
                    ocr_status="PLATE PREPROCESSING FAILURE",
                )
        else:
            evidence = replace(
                PlateOCREvidence(),
                plate_detection_status=NO_PLATE_REGION,
                ocr_status="NOT RUN (NO PLATE REGION)",
            )
        vehicle_id = f"veh-{stem}-{vehicle_index:02d}-{uuid.uuid4().hex[:4]}"
        ocr_evidence[vehicle_id] = evidence
        reports.append(
            VehicleReport(
                vehicle_id=vehicle_id,
                license_plate=evidence.plate,
                plate_confidence=evidence.plate_confidence,
                plate_detection_status=evidence.plate_detection_status,
                plate_bbox=evidence.plate_bbox,
                plate_detector=evidence.plate_detector,
                vehicle_type=get_vehicle_type(detection.label),
                color=color,
                color_share=color_share,
                damage_severity=None,
                confidence=detection.confidence,
                timestamp=detection.timestamp,
                data_source=detection.data_source,
            )
        )
    return reports, ocr_evidence


def _vision_result(
    image: Image.Image,
    detections: list[Detection],
    score: float,
    stem: str,
    location: dict[str, Any],
    media_type: str,
    media_sha256: str,
    orientation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw_model_verdict, raw_model_severity, model_status = determine_verdict(score)
    # The classifier was trained on dashcam video frames, not standalone photos.
    # Never present its still-image output as a final accident/no-accident decision.
    # Preserve the raw classifier result for transparent diagnostics only.
    is_out_of_domain_still = media_type == "image"
    verdict = STILL_IMAGE_STATUS if is_out_of_domain_still else raw_model_verdict
    severity = "uncertain" if is_out_of_domain_still else raw_model_severity
    status = STILL_IMAGE_STATUS if is_out_of_domain_still else model_status
    vehicle_reports, ocr_evidence = _build_vehicle_reports(image, detections, stem)
    occupant_reports, occupant_counts = build_occupant_reports(detections, vehicle_reports)
    features = extract_accident_features(
        detections, image_width=image.width, image_height=image.height
    )
    reasons = (
        f"Media type: {media_type}; "
        f"Raw classifier score: {score:.4f} -> Model output: {raw_model_verdict}; "
        f"Scene assessment: {verdict}; "
        f"Vehicles detected: {len([d for d in detections if get_vehicle_type(d.label)])}; "
        f"Persons near vehicles: {compute_persons_near_vehicles(detections)}; "
        f"Geometric overlap context (explanatory only, not a verdict): "
        f"max vehicle IoU={features['max_vehicle_iou']:.2f}, "
        f"heavily overlapping pairs={features['heavily_overlapping_pairs']}"
    )
    accident_event = AccidentEvent(
        event_id=f"acc-{stem}-{uuid.uuid4().hex[:6]}",
        timestamp=datetime.now(timezone.utc),
        location_name="Uploaded media (simulated demo location used only for hospital lookup)",
        latitude=float(location["latitude"]),
        longitude=float(location["longitude"]),
        severity=severity,
        vehicles_involved=len([d for d in detections if get_vehicle_type(d.label)]),
        casualties_estimated=occupant_counts["persons_near_vehicles_count"],
        status=status,
        confidence=round(score, 4),
        method="TRAINED_MODEL",
        score=round(score, 4),
        verdict=verdict,
        persons_near_vehicles_count=occupant_counts["persons_near_vehicles_count"],
        reasons=reasons,
        data_source="REAL",
    )
    return {
        "image": image,
        "detections": detections,
        "score": round(score, 4),
        "verdict": verdict,
        "raw_model_verdict": raw_model_verdict,
        "raw_model_severity": raw_model_severity,
        "severity": severity,
        "status": status,
        "method": "TRAINED_MODEL",
        "media_type": media_type,
        "media_sha256": media_sha256,
        "detection_conf_threshold": 0.5,
        "orientation": orientation
        or {
            "rotation_applied": False,
            "rotation_degrees": 0,
            "source_size_px": [image.width, image.height],
            "analyze_size_px": [image.width, image.height],
            "attempted_rotations_deg": [],
            "reason": "video frame orientation retained",
        },
        "features": features,
        "ocr_evidence": ocr_evidence,
        "vehicle_reports": vehicle_reports,
        "occupant_reports": occupant_reports,
        "occupant_counts": occupant_counts,
        "accident_event": accident_event,
        "vision_only_text": generate_incident_summary(
            accident_event, vehicle_reports, occupant_reports, media_type=media_type
        ),
    }


def _process_image(
    data: bytes, filename: str, location: dict[str, Any]
) -> dict[str, Any]:
    with Image.open(io.BytesIO(data)) as opened:
        source_image = opened.convert("RGB")
    # Real photos can be stored physically rotated without an EXIF tag. The
    # detector retries quarter turns only when the source has zero detections.
    detections, image, orientation = detect_with_orientation(source_image)
    score = classify_frame_probability(image)
    return _vision_result(
        image,
        detections,
        score,
        Path(filename).stem,
        location,
        media_type="image",
        media_sha256=hashlib.sha256(data).hexdigest(),
        orientation=orientation,
    )


def _process_video(
    data: bytes,
    filename: str,
    location: dict[str, Any],
    sample_count: int = SAMPLE_COUNT,
) -> dict[str, Any]:
    """Sample, classify, and inspect an uploaded video using a temporary file."""
    sample_count = max(2, min(int(sample_count), MAX_SAMPLE_COUNT))
    safe_name = Path(filename).name or "upload.mp4"
    with tempfile.TemporaryDirectory(prefix="third_eye_upload_") as temp_dir:
        video_path = Path(temp_dir) / safe_name
        video_path.write_bytes(data)

        progress = st.progress(0.0, text="Reading uploaded video…")
        sampled = sample_video_frames(
            video_path,
            sample_count=sample_count,
            progress_callback=lambda fraction: progress.progress(
                fraction * 0.3, text="Reading uploaded video…"
            ),
        )

        frame_rows: list[dict[str, Any]] = []
        detections_by_frame: list[list[Detection]] = []
        for position, frame in enumerate(sampled.frames, start=1):
            detections = detect(frame.image)
            detections_by_frame.append(detections)
            score = classify_frame_probability(frame.image)
            verdict, severity, status = determine_verdict(score)
            frame_rows.append(
                {
                    "Frame": position,
                    "Source frame": frame.index,
                    "Timestamp (s)": _text(frame.timestamp_s),
                    "Score": round(score, 4),
                    "Verdict": verdict,
                    "Severity": severity,
                    "Vehicles": sum(
                        1 for item in detections if get_vehicle_type(item.label)
                    ),
                    "Persons": sum(1 for item in detections if item.label == "person"),
                }
            )
            progress.progress(
                0.3 + 0.7 * position / len(sampled.frames),
                text=f"Classifying frame {position}/{len(sampled.frames)}…",
            )

        top_row = max(frame_rows, key=lambda row: row["Score"])
        top_position = int(top_row["Frame"]) - 1
        top_frame = sampled.frames[top_position]
        top_detections = detections_by_frame[top_position]
        score = float(top_row["Score"])
        temporal = analyze_frame_scores(frame_rows)
        video_evidence = build_video_assessment(
            sampled.metadata, frame_rows, temporal, str(top_row["Verdict"])
        )
        result = _vision_result(
            top_frame.image,
            top_detections,
            score,
            f"{Path(filename).stem}-frame{top_frame.index}",
            location,
            media_type="video",
            media_sha256=hashlib.sha256(data).hexdigest(),
        )
        result.update(
            {
                "video_metadata": sampled.metadata,
                "frame_rows": frame_rows,
                "top_frame": top_frame,
                "top_timestamp": top_frame.timestamp_s,
                "high_frame_count": temporal["high_frame_count"],
                "temporal_evidence": temporal,
                "video_evidence": video_evidence,
                "video_assessment": video_evidence["assessment_text"],
                "video_temporal_signal": temporal["temporal_signal"],
                "is_long_video": bool(
                    sampled.metadata.duration_s
                    and sampled.metadata.duration_s > LONG_VIDEO_SECONDS
                ),
            }
        )
        return result


# ---------------------------------------------------------------------------
# Page setup — the only interactive controls are one uploader and two dropdowns
# ---------------------------------------------------------------------------
st.title("🚑 Third Eye AI")
st.caption(
    "Six-tab multimodal demonstration | Upload one image or video | "
    "No automatic sample processing | SQLite opened read-only (mode=ro)"
)

uploaded = st.sidebar.file_uploader(
    "Upload one image or video (CV INPUT)",
    type=UPLOAD_TYPES,
    accept_multiple_files=False,
    help="JPG/JPEG/PNG images or MP4/MOV/AVI videos. Nothing is saved to the project or database.",
)
video_sample_count = st.sidebar.number_input(
    "Video frames to sample",
    min_value=4,
    max_value=MAX_SAMPLE_COUNT,
    value=SAMPLE_COUNT,
    step=1,
    help=(
        "Evenly spaced real frames across the whole clip. More frames cost more "
        "inference time; this is still a sample, never full-clip analysis."
    ),
)

calls = _call_names()
call_options = [NO_CALL_LABEL] + [
    f"{name} | SIMULATED SAMPLE CALL" for name in calls
]
call_choice = st.sidebar.selectbox(
    "Emergency call",
    options=call_options,
    index=0,
    help="No call means vision-only. Sample calls are SIMULATED and are arbitrarily paired with your media.",
)

locations = _demo_locations()
location_options = [
    f"{item.get('name', 'not determined')} | SIMULATED LOCATION" for item in locations
] or ["No demo locations available | SIMULATED LOCATION"]
location_choice = st.sidebar.selectbox(
    "Incident location (for hospitals only)",
    options=location_options,
    index=0,
    help="SIMULATED demo location; never inferred from your uploaded media.",
)
location = locations[location_options.index(location_choice)] if locations else {
    "name": "not determined",
    "latitude": None,
    "longitude": None,
    "data_source": "SIMULATED",
}

metrics = _held_out_metrics()
summary = _dispatch_summary(str(SUMMARY_PATH))
result: dict[str, Any] | None = None

if uploaded is None:
    st.session_state.pop("uploaded_result", None)
    st.session_state.pop("uploaded_result_key", None)
    st.info(
        "Upload one image or video in the sidebar to begin. No model runs and no "
        "sample file is read until you do."
    )
else:
    data = uploaded.getvalue()
    content_sha256 = hashlib.sha256(data).hexdigest()
    # The selected demo location is embedded in the AccidentEvent produced by the
    # vision pipeline. Include it in the cache key so changing location cannot leave
    # old coordinates in Tab 4 while Tab 5/6 use the new selection.
    upload_key = (
        f"{uploaded.name}:{len(data)}:{content_sha256}:"
        f"video_sample_count={int(video_sample_count)}:"
        f"location={json.dumps(location, sort_keys=True, default=str)}"
    )
    cached_result = st.session_state.get("uploaded_result")
    if (
        st.session_state.get("uploaded_result_key") != upload_key
        or not isinstance(cached_result, dict)
        or cached_result.get("media_sha256") != content_sha256
    ):
        suffix = Path(uploaded.name).suffix.lower()
        try:
            if suffix in VIDEO_EXTENSIONS:
                with st.spinner("Processing uploaded video in memory…"):
                    processed = _process_video(
                        data,
                        uploaded.name,
                        location,
                        sample_count=int(video_sample_count),
                    )
            else:
                with st.spinner("Processing uploaded image in memory…"):
                    processed = _process_image(data, uploaded.name, location)
            st.session_state["uploaded_result"] = processed
            st.session_state["uploaded_result_key"] = upload_key
        except Exception as exc:  # unreadable media must be a clear message, not a crash
            st.session_state.pop("uploaded_result", None)
            st.session_state.pop("uploaded_result_key", None)
            st.error(
                f"Could not process '{uploaded.name}': {exc}. "
                "Please upload a readable JPG/JPEG/PNG image or MP4/MOV/AVI video."
            )
    result = st.session_state.get("uploaded_result")

scene_tab, vehicles_tab, accident_tab, report_tab, hospitals_tab, dispatch_tab = st.tabs(
    [
        "1. Scene",
        "2. Vehicles",
        "3. Accident check",
        "4. Incident report",
        "5. Hospitals & emergency numbers",
        "6. Dispatch",
    ]
)

# ---------------------------------------------------------------------------
# Shared report construction (in memory, save_to_db=False for fusion)
# ---------------------------------------------------------------------------
report: dict[str, Any] | None = None
needs: dict[str, Any] = {"urgency": None, "hazards": [], "injuries": []}
if result is not None:
    if call_choice == NO_CALL_LABEL:
        event = result["accident_event"]
        vision_only_text = result["vision_only_text"]
        if result["media_type"] == "video":
            vision_only_text += (
                "\n\nREAL VIDEO FRAME/TEMPORAL EVIDENCE (Source: REAL):\n"
                + result["video_assessment"]
                + "\nTemporal evidence is explanatory and never overrides the trained "
                "model verdict. No accident is confirmed from sampled frames alone."
            )
        report = {
            "report_id": f"vision-only-{event.event_id}",
            "image_name": uploaded.name,
            "call_id": None,
            "verdict": result["verdict"],
            "verdict_source": "VISION",
            "location": None,
            "location_source": "MISSING",
            "vision_vehicle_count": len(result["vehicle_reports"]),
            "caller_vehicle_count": None,
            "vehicle_count_disagreement": False,
            "vision_people_count": result["occupant_counts"]["persons_visible_count"],
            "caller_people_count": None,
            "people_count_disagreement": False,
            "hazards": [],
            "hazards_source": "MISSING",
            "injuries": [],
            "injuries_source": "MISSING",
            "urgency": None,
            "urgency_source": "MISSING",
            "summary_text": vision_only_text,
            "disagreements": [],
            "data_source": "REAL",
            "video_evidence": result.get("video_evidence")
            if result["media_type"] == "video"
            else None,
            "video_assessment": result.get("video_assessment")
            if result["media_type"] == "video"
            else None,
            "video_temporal_signal": result.get("video_temporal_signal")
            if result["media_type"] == "video"
            else None,
            # Source-separated fields (no call selected, demo location only for
            # hospital lookup; no location is inferred from image pixels).
            "image_location": location.get("name"),
            "image_location_source": "SIMULATED",
            "caller_location": None,
            "location_mismatch": False,
            "vision_score": result["score"],
            "source_disagreement": False,
            "source_disagreements": [],
        }
        needs = {"urgency": None, "hazards": [], "injuries": []}
    else:
        call_index = call_options.index(call_choice) - 1
        call_name = calls[call_index]
        caller = extract_caller_info(_call_text(call_name), call_id=Path(call_name).stem)
        fused = fuse_incident_evidence(
            accident_event=result["accident_event"],
            vehicle_reports=result["vehicle_reports"],
            occupant_reports=result["occupant_reports"],
            caller_extraction=caller,
            image_name=uploaded.name,
            save_to_db=False,
            media_type=result["media_type"],
            # The sidebar demo location is associated with the upload for
            # hospital lookup only. It is kept separate from the caller's own
            # location and is never presented as a location seen in the image.
            image_location=location.get("name"),
            image_location_source="SIMULATED",
            video_evidence=result.get("video_evidence")
            if result["media_type"] == "video"
            else None,
        )
        report = fused.model_dump(mode="json")
        needs = {
            "urgency": report.get("urgency"),
            "hazards": report.get("hazards") or [],
            "injuries": report.get("injuries") or [],
        }

# ---------------------------------------------------------------------------
# Tab 1 — Scene
# ---------------------------------------------------------------------------
with scene_tab:
    st.subheader("Scene: real detector boxes on your upload")
    if result is None:
        tagged("Uploaded media", "not determined — no upload", "REAL")
    else:
        if result["media_type"] == "video":
            tagged("Media type", "uploaded video (sampled frames)", "REAL")
            tagged(
                "Displayed frame",
                f"highest-scoring sampled frame (source frame {result['top_frame'].index})",
                "REAL",
            )
        else:
            tagged("Media type", "uploaded image", "REAL")
        tagged("Filename", uploaded.name, "REAL")
        tagged("Media fingerprint (SHA-256)", result["media_sha256"], "REAL")
        tagged(
            "Detection confidence threshold",
            f"{float(result['detection_conf_threshold']):.2f}",
            "REAL",
        )
        orientation = result.get("orientation", {})
        tagged(
            "Bounding-box coordinates",
            "displayed analysis-image pixels, [xmin, ymin, xmax, ymax]",
            "REAL",
        )
        tagged(
            "Image orientation",
            (
                f"{int(orientation.get('rotation_degrees', 0))} degree rotation applied "
                f"({orientation.get('reason', 'not determined')})"
                if orientation.get("rotation_applied")
                else "source orientation retained (no rotation applied)"
            ),
            "REAL",
        )
        if orientation.get("rotation_applied"):
            st.caption(
                "The uploaded file was physically rotated with no EXIF tag; boxes, "
                "classifier and OCR use the displayed corrected orientation. The "
                "original file and its bytes were not modified."
            )
        colors = {
            "car": "#19c919",
            "truck": "#ffa500",
            "bus": "#00bfff",
            "motorcycle": "#ff00ff",
            "bicycle": "#ffff00",
            "person": "#ff3333",
        }
        annotated = result["image"].convert("RGB")
        drawer = ImageDraw.Draw(annotated)
        for detection in result["detections"]:
            color = colors.get(detection.label, "#19c919")
            box = tuple(int(round(value)) for value in detection.bbox)
            drawer.rectangle(box, outline=color, width=3)
            drawer.text(
                (box[0], max(0, box[1] - 14)),
                f"{detection.label} {float(detection.confidence or 0):.2f}",
                fill=color,
            )
        st.image(annotated, caption=f"{uploaded.name} — REAL Faster R-CNN boxes", width="stretch")
        tagged(
            "Vehicle count (detected)",
            sum(1 for d in result["detections"] if get_vehicle_type(d.label)),
            "REAL",
        )
        tagged(
            "Person count (visible)",
            sum(1 for d in result["detections"] if d.label == "person"),
            "REAL",
        )
        tagged(
            "Overlapping vehicle boxes (occlusion/duplicate check)",
            f"{result['features']['heavily_overlapping_pairs']} heavily overlapping pair(s) "
            f"among {result['features']['total_vehicle_pairs']} pair(s); raw detector output, no NMS applied",
            "REAL",
        )
        tagged("Fused-call selection", call_choice, "SIMULATED" if report and report.get("call_id") else "REAL")
        if report and report.get("call_id"):
            tagged("Pairing label", "ARBITRARY DEMO PAIRING", "SIMULATED")
        tagged(
            "Limitation",
            "Visible exterior only. Occupants inside vehicles cannot be reliably counted "
            "from a single image. Unknown facts are strictly marked 'not determined'.",
            "REAL",
        )

# ---------------------------------------------------------------------------
# Tab 2 — Vehicles
# ---------------------------------------------------------------------------
with vehicles_tab:
    st.subheader("Vehicles: in-memory forensic attributes from your upload")
    if result is None:
        tagged("Vehicle attributes", "not determined — no upload", "REAL")
    else:
        tagged("Media fingerprint (SHA-256)", result["media_sha256"], "REAL")
        if result["media_type"] == "video":
            tagged(
                "OCR scope",
                "Vehicle attributes and plate OCR were run ONLY on the highest-scoring "
                "sampled frame, because OCR on every frame is too slow. Other sampled "
                "frames use detector counts only.",
                "REAL",
            )
        rows = []
        for vehicle in result["vehicle_reports"]:
            share = vehicle.color_share
            color_value = (
                f"{_text(vehicle.color)} ({float(share):.1%} share of pixels)"
                if share is not None
                else "not determined"
            )
            confidence = vehicle.plate_confidence
            verified = confidence is not None and float(confidence) >= PLATE_TRUST_THRESHOLD
            # Only a candidate at or above the fixed 0.70 verification threshold is
            # shown as a plate. Below it, keep the raw OCR candidate in a separate
            # column but never promote it to a verified registration number.
            if not verified:
                plate = NOT_READABLE
                plate_status = NOT_READABLE
            else:
                plate = str(vehicle.license_plate)
                plate_status = "verified"

            # Raw EasyOCR text is shown as recognized text, never as a verified plate.
            evidence = result["ocr_evidence"].get(vehicle.vehicle_id, PlateOCREvidence())
            raw_text = (
                f"{evidence.raw_text} ({float(evidence.raw_confidence):.2f})"
                if evidence.raw_text is not None and evidence.raw_confidence is not None
                else "not determined"
            )
            candidate = (
                f"{vehicle.license_plate} ({float(confidence):.2f})"
                if vehicle.license_plate is not None and confidence is not None
                else "not determined"
            )
            detection_status = evidence.plate_detection_status or NO_PLATE_REGION
            if detection_status == PLATE_DETECTED:
                if evidence.ocr_status in {
                    "OCR INITIALIZATION FAILURE",
                    "OCR INFERENCE FAILURE",
                    "PLATE PREPROCESSING FAILURE",
                }:
                    detection_status = evidence.ocr_status
                elif evidence.ocr_status == "NO OCR RESULTS":
                    detection_status = "PLATE REGION DETECTED — UNREADABLE"
                elif vehicle.plate_confidence is not None and float(vehicle.plate_confidence) >= PLATE_TRUST_THRESHOLD:
                    detection_status = "PLATE REGION DETECTED — READABLE"
                elif evidence.plate is not None:
                    detection_status = "PLATE REGION DETECTED — UNREADABLE (LOW OCR CONFIDENCE)"
            bbox = evidence.plate_bbox
            bbox_text = (
                ",".join(str(int(value)) for value in bbox)
                if bbox is not None
                else "not determined"
            )
            rows.append(
                {
                    "Vehicle ID": vehicle.vehicle_id,
                    "Type": _text(vehicle.vehicle_type),
                    "Color (share of pixels)": color_value,
                    "Plate detection status": detection_status,
                    "Plate bbox (source px)": bbox_text,
                    "Plate": plate,
                    "OCR confidence": f"{float(confidence):.2f}" if confidence is not None else "not determined",
                    "Unverified OCR candidate": candidate if not verified else "not applicable (verified plate above)",
                    "Raw OCR text (unverified)": raw_text,
                    "OCR status": evidence.ocr_status,
                    "Plate status": plate_status,
                    "Badge": vehicle.data_source,
                }
            )
        if rows:
            st.table(rows)
        else:
            tagged("Vehicle attributes", "not determined", "REAL")
        tagged(
            "Plate localisation method",
            "OpenCV Zoo LPD-YuNet ONNX (CPU, SHA-256 verified), a deterministic "
            "OpenCV heuristic, then a whole-crop EasyOCR fallback; the bbox is a "
            "candidate region, not proof of a registration number.",
            "REAL",
        )
        tagged(
            "Limitation",
            "EasyOCR runs on detected plate regions. YuNet was trained on Chinese "
            "plates and may miss Indian or unusual plates; if both YuNet and the "
            "heuristic return nothing, a plate-like whole-crop EasyOCR text box is "
            "used to propose a candidate region, then that crop is re-read. OCR "
            "evidence is preserved verbatim; any plate candidate below "
            "0.7 OCR confidence is shown as 'NOT READABLE (UNVERIFIED)' in the Plate "
            "column, with its raw text and candidate retained in separate columns, and "
            "no registration number is invented. "
            "Missing reads are 'NOT READABLE (UNVERIFIED)'. Damage severity is not "
            "determined.",
            "REAL",
        )

# ---------------------------------------------------------------------------
# Tab 3 — Accident check
# ---------------------------------------------------------------------------
with accident_tab:
    st.subheader("Accident check: trained model verdict and held-out test metrics")
    if result is None:
        tagged("Verdict", "not determined — no upload", "REAL")
    else:
        if result["media_type"] == "video":
            meta = result["video_metadata"]
            st.markdown("**Video overview (REAL container metadata):**")
            _badge("REAL")
            tagged("Duration", _text(meta.duration_s) + (" s" if meta.duration_s else ""), "REAL")
            tagged("Resolution", f"{meta.width} x {meta.height}", "REAL")
            tagged("FPS", _text(meta.fps), "REAL")
            tagged("Decoded frames", meta.frame_count, "REAL")
            tagged("Frames sampled", len(result["frame_rows"]), "REAL")
            tagged(
                "Sampling method",
                f"Evenly spaced real frames across the full clip (configured in sidebar, "
                f"max {MAX_SAMPLE_COUNT}); no full-clip inference.",
                "REAL",
            )
            if result["is_long_video"]:
                tagged(
                    "Work cap",
                    f"Video longer than 3 minutes: still only {len(result['frame_rows'])} "
                    "evenly sampled frames; no full-clip inference and no OCR on every frame.",
                    "REAL",
                )
            st.markdown("**Classifier score per sampled frame:**")
            _badge("REAL")
            st.table(result["frame_rows"])
            chart_data = pd.DataFrame(result["frame_rows"])
            st.line_chart(
                chart_data,
                x="Frame",
                y="Score",
                x_label="Sampled frame (chronological)",
                y_label="Classifier score (uncalibrated)",
            )
            temporal = result["temporal_evidence"]
            st.markdown("**Frame-level temporal evidence (explanatory):**")
            _badge("REAL")
            tagged("Temporal signal", temporal["temporal_signal"], "REAL")
            tagged(
                "Score range (sampled)",
                f"{temporal['min_score']} – {temporal['max_score']} "
                f"(mean {temporal['mean_score']})",
                "REAL",
            )
            tagged(
                "Score band counts",
                f">=0.60: {temporal['high_frame_count']}; "
                f"0.40–0.60: {temporal['uncertain_frame_count']}; "
                f"<0.40: {temporal['no_signal_frame_count']}",
                "REAL",
            )
            tagged(
                "Longest consecutive runs",
                f">=0.60: {temporal['sustained_high_run']} sampled frame(s); "
                f">=0.40: {temporal['sustained_uncertain_or_high_run']} sampled frame(s)",
                "REAL",
            )
            tagged("Video-level assessment", result["video_assessment"], "REAL")
            st.markdown("**Overall video result:**")
            _badge("REAL")
            tagged(
                "Highest-scoring frame",
                f"frame {result['top_frame'].index} at timestamp {_text(result['top_timestamp'])} s "
                f"(score {result['score']:.4f})",
                "REAL",
            )
            tagged(
                "Sampled frames with score >= 0.60",
                result["high_frame_count"],
                "REAL",
            )
            st.markdown("**Model output (from the highest-scoring sampled frame):**")
            _badge("REAL")
            tagged("Model verdict", result["verdict"], "REAL")
            tagged("Uncalibrated model score", f"{result['score']:.4f}", "REAL")
            tagged("Method", result["method"], "REAL")
            tagged("Scene status", result["status"], "REAL")
            st.markdown(PROTOTYPE_STATEMENT)
        else:
            st.warning(
                "The accident classifier was trained on dashcam video frames "
                "(held-out ROC-AUC 0.7452 on 60 videos). This uploaded still image is "
                "outside that training domain: the score below is an uncalibrated ranking "
                "signal, not proof of an accident or of normal traffic. Human review is required."
            )
            st.markdown("**Scene assessment (primary result):**")
            _badge("REAL")
            tagged("Assessment", result["verdict"], "REAL")
            tagged("Scene status", result["status"], "REAL")
            tagged("Severity", result["severity"], "REAL")
            st.markdown("**Raw classifier output (not reliable for a still-image decision):**")
            _badge("REAL")
            tagged("Raw model output", result["raw_model_verdict"], "REAL")
            tagged("Uncalibrated model score", f"{result['score']:.4f}", "REAL")
            tagged("Method", result["method"], "REAL")
            tagged("Reasoning context", result["accident_event"].reasons, "REAL")
    st.caption("Held-out test metrics from models/accident_classifier_metrics.json")
    tagged(
        "Test accuracy",
        f"{float(metrics['test_accuracy']) * 100:.2f}%" if "test_accuracy" in metrics else "not determined",
        "REAL",
    )
    tagged(
        "Test F1",
        f"{float(metrics['test_f1']):.4f}" if "test_f1" in metrics else "not determined",
        "REAL",
    )
    tagged(
        "Test ROC-AUC",
        f"{float(metrics['test_roc_auc']):.4f}" if "test_roc_auc" in metrics else "not determined",
        "REAL",
    )
    tagged(
        "Baseline accuracy",
        f"{float(metrics['baseline_accuracy']) * 100:.2f}%" if "baseline_accuracy" in metrics else "not determined",
        "REAL",
    )
    tagged("Held-out test set", "60 videos / 180 frames (video-level split)", "REAL")
    st.markdown("**Plain limitation text (docs/model_card_accident.md, §5):**")
    for item in ACCIDENT_LIMITATIONS:
        tagged("Limitation", item, "REAL")

# ---------------------------------------------------------------------------
# Tab 4 — Incident report
# ---------------------------------------------------------------------------
with report_tab:
    st.subheader("Incident report: source-separated evidence (in memory)")
    if report is None:
        tagged("Incident report", "not determined — no upload", "REAL")
    else:
        st.caption(
            "REAL vision evidence comes only from your upload. SIMULATED caller evidence "
            "comes from an arbitrarily paired sample call. Each table has its own Source column."
        )
        for section in build_report_sections(report):
            st.markdown(f"### {section.heading}")
            st.table(section.as_table())
        with st.expander("Full generated narrative (plain text, wraps to window width)"):
            st.markdown(narrative_markdown(report.get("summary_text")))
        st.caption(
            "No report is written to SQLite; fusion is called with save_to_db=False. "
            "Dispatcher verification is required before acting on simulated urgency."
        )

# ---------------------------------------------------------------------------
# Tab 5 — Hospitals & emergency numbers
# ---------------------------------------------------------------------------
with hospitals_tab:
    st.subheader("Hospitals & emergency numbers: real OSM facilities, simulated incident location")
    tagged(
        "Assigned demo location (SIMULATED; used for hospital lookup)",
        location.get("name"),
        "SIMULATED",
    )
    if location.get("latitude") is None:
        tagged("Demo coordinates", "not determined", "SIMULATED")
    else:
        tagged(
            "Demo coordinates",
            f"{location.get('latitude')}, {location.get('longitude')}",
            "SIMULATED",
        )

    # Tab 4's caller-reported location is preserved separately; it never replaces
    # the selected demo coordinates used for the hospital search.
    caller_location = (report or {}).get("location")
    consistency = compare_locations(location, caller_location)
    tagged(
        "Caller-reported location (Tab 4; preserved)",
        consistency.caller_reported,
        "SIMULATED",
    )
    tagged("Location consistency", consistency.status, "SIMULATED")
    tagged("Location detail", consistency.detail, "SIMULATED")
    tagged("GPS verification", consistency.gps_status, "REAL")
    if consistency.status == "LOCATION MISMATCH":
        st.warning(
            "Location discrepancy: the caller-reported location differs from the "
            "selected demo location. Distances below use only the selected demo "
            "coordinates; the caller location was not substituted."
        )

    retrieval: dict[str, Any] | None = None
    if result is None or location.get("latitude") is None:
        tagged("Hospital suggestions", "not determined — no upload", "SIMULATED")
    else:
        with st.spinner("Loading cached hospital index…"):
            retriever, hospital_stats, hospital_source, raw_count, eligible_count = _hospital_resources()
            if retriever is not None:
                query = formulate_retrieval_query(
                    urgency=needs["urgency"],
                    hazards=needs["hazards"],
                    injuries=needs["injuries"],
                    verdict=result["verdict"],
                )
                retrieval = retriever.retrieve_suggestions(
                    incident_lat=float(location["latitude"]),
                    incident_lon=float(location["longitude"]),
                    query_override=query,
                    final_top_k=3,
                )
            else:
                query = "not determined — no dispatch-eligible facility records"
        tagged("Raw OSM facility records loaded", raw_count, "REAL")
        tagged("Dispatch-eligible hospitals / trauma centres", eligible_count, "REAL")
        tagged("Irrelevant or non-hospital records excluded", raw_count - eligible_count, "REAL")
        tagged("OSM cache source", hospital_source, "REAL")
        tagged("Rule-based retrieval query", query, "SIMULATED")

        if retrieval is None:
            st.info("No dispatch-eligible healthcare facility is available in the cached OSM data.")
        else:
            suggestions = retrieval["suggestions"]
            confirmed = [item for item in suggestions if item["recommended_for_emergency"]]
            unverified = [item for item in suggestions if not item["recommended_for_emergency"]]

            st.markdown(
                f"**Source-confirmed emergency facilities ({len(confirmed)} of "
                f"{len(suggestions)} ranked; ranking heuristic, not medical verification):**"
            )
            if confirmed:
                for rank, item in enumerate(confirmed, start=1):
                    with st.container(border=True):
                        st.markdown(
                            f"#### {rank}. {html.escape(item['hospital'].name)} — REAL OSM facility"
                        )
                        st.caption("Location context: SIMULATED demo location; no GPS verification")
                        _badge("REAL")
                        st.table(recommendation_rows([item]))
                        st.caption(f"Why ranked here: {item['why_suggested']}")
            else:
                st.info(
                    "No ranked facility has a source-reported emergency department. "
                    "Use the general emergency numbers below; do not treat nearby facilities as verified."
                )

            if unverified:
                st.markdown("**Nearby facilities NOT recommended without verification:**")
                st.table(recommendation_rows(unverified))
                st.caption(
                    "These records are real OSM facilities, but the emergency department is "
                    "unspecified or reported absent in the source. Proximity alone is not suitability."
                )

    st.markdown("**Emergency contacts (plain text):**")
    st.text("This app does not place calls; dial from your phone.")
    st.markdown(f"**General — {GENERAL_NUMBERS_NOTE}**")
    st.table(general_emergency_rows(get_emergency_numbers(needs["urgency"], needs["hazards"])))
    st.markdown(f"**Hospital-specific — {HOSPITAL_NUMBERS_NOTE}**")
    number_lines = hospital_number_lines(retrieval["suggestions"]) if retrieval else []
    if number_lines:
        for line in number_lines:
            st.text(line)
    else:
        tagged("Hospital-specific numbers", "not determined — no upload", "SIMULATED")

    st.markdown("**Data, location and routing limitations:**")
    for line in LIMITATION_LINES:
        tagged("Limitation", line, "REAL")
    if retrieval is not None:
        tagged("Attribution", retrieval["attribution"], "REAL")

# ---------------------------------------------------------------------------
# Tab 6 — Dispatch (Phase 8b: summary file only, never guessed)
# ---------------------------------------------------------------------------
with dispatch_tab:
    _render_dispatch_tab(summary, needs["urgency"], needs["hazards"], report, location)

# Device note is intentionally shown only after a real upload, so the no-upload
# path cannot touch any model loader.
if result is not None:
    st.markdown("**Environment / device note**")
    tagged("Vision detector device", get_target_device().type.upper(), "REAL")
    tagged("Accident extractor device", get_device().type.upper(), "REAL")
    tagged("Hospital ranking method", "deterministic source-tag, facility-category and Haversine-distance rules (no embeddings)", "REAL")
    tagged("SQLite access", "read-only URI (mode=ro)", "REAL")
    tagged("Upload handling", "in-memory bytes; video uses an auto-deleted temp file", "REAL")
