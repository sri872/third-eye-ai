"""SOS incident location inputs with explicit source provenance."""
from __future__ import annotations

import json
import streamlit as st

from app.components.sos_gps_component import capture_device_location
from core.sos.location import (
    SOURCE_DEVICE_GEOLOCATION,
    SOURCE_OPERATOR_MANUAL,
    SOURCE_SIMULATED_DEMO,
    SOURCE_SOS_JSON_IMPORT,
    SOSLocation,
    create_sos_location,
    parse_sos_payload,
)


DEMO_INCIDENTS = {
    "NH-44 GT Road near Jagatjit Sugar Mill": (31.2142, 75.7785),
    "Banga Road commercial intersection": (31.2185, 75.7852),
    "GT Road near LPU, Chaheru": (31.2532, 75.7035),
    "Phagwara-Hoshiarpur Road": (31.2455, 75.7802),
}


def _remember(location: SOSLocation, mode: str) -> None:
    st.session_state["third_eye_sos_location"] = location.to_dict()
    st.session_state["third_eye_sos_mode"] = mode


def _clear_saved() -> None:
    st.session_state.pop("third_eye_sos_location", None)
    st.session_state.pop("third_eye_sos_mode", None)


def _restore(mode: str) -> SOSLocation | None:
    saved = st.session_state.get("third_eye_sos_location")
    if not isinstance(saved, dict) or st.session_state.get("third_eye_sos_mode") != mode:
        return None
    location, _ = create_sos_location(
        saved.get("latitude"),
        saved.get("longitude"),
        source=saved.get("source"),
        accuracy_meters=saved.get("accuracy_meters"),
        timestamp=saved.get("timestamp"),
        location_name=saved.get("location_name"),
    )
    return location


def render_sos_location_component() -> SOSLocation | None:
    """Capture an SOS payload, actual browser GPS, or operator/demo coordinates."""
    st.subheader("Incident SOS location")
    st.caption(
        "The selected accident coordinates are the route origin. A dispatcher’s "
        "computer location is never substituted automatically."
    )
    mode = st.radio(
        "Incident location source",
        [
            "Incoming SOS event JSON (sender not authenticated here)",
            "Browser GPS (device must be at the incident)",
            "Operator-verified coordinates",
            "Phagwara demo scenario (SIMULATED)",
        ],
        key="sos-source-mode",
    )
    location = None
    error = None
    allow_restore = True

    if mode == "Incoming SOS event JSON (sender not authenticated here)":
        allow_restore = False
        st.caption(
            "Upload/paste an event exported by an authorized SOS provider. This prototype "
            "does not expose an authenticated live webhook; verify the sender and event out of band."
        )
        upload = st.file_uploader("SOS event JSON file", type=["json"], key="sos-json-upload")
        payload_text = st.text_area(
            "Or paste SOS JSON",
            key="sos-json-text",
            placeholder='{"latitude": 31.2142, "longitude": 75.7785, "accuracy_meters": 12, "timestamp": "2026-11-26T08:00:00Z"}',
        )
        if upload is not None or payload_text.strip():
            try:
                raw = upload.getvalue().decode("utf-8") if upload is not None else payload_text
                payload = json.loads(raw)
                location, error = parse_sos_payload(
                    payload,
                    source=SOURCE_SOS_JSON_IMPORT,
                )
                if location is not None:
                    _remember(location, mode)
                else:
                    _clear_saved()
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                error = f"Invalid SOS JSON: {exc}"
                _clear_saved()

    elif mode == "Browser GPS (device must be at the incident)":
        st.info(
            "Use browser GPS only when this device is physically with the accident. "
            "The browser asks permission only after you press the capture button."
        )
        result = capture_device_location()
        if isinstance(result, dict):
            if result.get("error"):
                error = str(result["error"])
                _clear_saved()
            else:
                location, error = parse_sos_payload(
                    result,
                    source=SOURCE_DEVICE_GEOLOCATION,
                )
                if location is not None:
                    _remember(location, mode)
                else:
                    _clear_saved()

    elif mode == "Operator-verified coordinates":
        allow_restore = True
        with st.form("operator-sos-location"):
            lat_text = st.text_input("Accident latitude", placeholder="Enter verified latitude")
            lon_text = st.text_input("Accident longitude", placeholder="Enter verified longitude")
            accuracy_text = st.text_input("Accuracy metres (optional)", placeholder="Leave blank if unknown")
            name = st.text_input("Location description (optional)", placeholder="Road, landmark or address")
            submitted = st.form_submit_button("Validate incident coordinates")
        if submitted:
            location, error = create_sos_location(
                lat_text,
                lon_text,
                source=SOURCE_OPERATOR_MANUAL,
                accuracy_meters=accuracy_text or None,
                location_name=name or None,
            )
            if location is not None:
                _remember(location, mode)
            else:
                _clear_saved()

    else:
        choice = st.selectbox("Choose simulated incident location", list(DEMO_INCIDENTS))
        if st.button("Use simulated demo location", key="choose-demo-sos"):
            lat, lon = DEMO_INCIDENTS[choice]
            location, error = create_sos_location(
                lat,
                lon,
                source=SOURCE_SIMULATED_DEMO,
                location_name=choice,
            )
            if location is not None:
                _remember(location, mode)

    if location is None and allow_restore:
        location = _restore(mode)
    if error:
        st.error(error)
    if location is not None:
        if location.is_stale:
            st.error("This SOS timestamp is older than 15 minutes. Capture or verify a fresh incident location before routing.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Latitude", f"{location.latitude:.6f}")
        c2.metric("Longitude", f"{location.longitude:.6f}")
        c3.metric(
            "GPS accuracy",
            f"±{location.accuracy_meters:.0f} m"
            if location.accuracy_meters is not None else "Not supplied",
        )
        st.caption(f"Source: {location.source} | Captured: {location.timestamp}")
        if location.source == SOURCE_SIMULATED_DEMO:
            st.warning("SIMULATED location for demonstration only. It is not a live SOS signal.")
        elif location.source == SOURCE_OPERATOR_MANUAL:
            st.info("Operator-entered coordinates; verify the incident pin before using the route.")
        elif location.source == SOURCE_SOS_JSON_IMPORT:
            st.warning("Imported SOS JSON is not cryptographically authenticated by this application.")
        elif location.accuracy_meters is not None and location.accuracy_meters > 100:
            st.warning("GPS accuracy exceeds 100 m. Confirm the incident location before relying on this route.")
    else:
        st.info("No incident coordinates have been confirmed. Hospital routing remains disabled.")
    return location
