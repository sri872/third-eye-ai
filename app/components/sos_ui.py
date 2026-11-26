"""Streamlit UI Component for SOS Location Capture and Verification.

Integrates browser Geolocation API with explicit user permission,
preset accident SOS events, coordinate validation, and source provenance tracking.
"""

import os
import json
from datetime import datetime, timezone
import streamlit as st
import streamlit.components.v1 as components

from core.sos.location import (
    SOSLocation,
    validate_coordinates,
    create_sos_location,
    SOURCE_GPS_SOS_PAYLOAD,
    SOURCE_DEVICE_GEOLOCATION,
    SOURCE_OPERATOR_MANUAL,
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRESETS_PATH = os.path.join(BASE_DIR, "config", "demo_incident_locations.json")


def render_sos_location_component() -> SOSLocation:
    """Render the SOS location management interface and return validated SOSLocation."""
    st.subheader("📍 Incident SOS Location & GPS Verification")
    st.markdown(
        "Validated emergency coordinates determine hospital suitability and route origin. "
        "**Dispatcher device location never silently overrides incident SOS coordinates.**"
    )

    # Load presets
    presets = []
    if os.path.exists(PRESETS_PATH):
        with open(PRESETS_PATH, "r", encoding="utf-8") as f:
            presets = json.load(f)

    # 3-Way Location Input Selection
    loc_mode = st.radio(
        "Location Input Mode",
        [
            "1. Verified SOS Payload / Scenario (Priority 1)",
            "2. Device Geolocation (At Scene - Priority 2)",
            "3. Operator Manual Verification (Priority 3)",
        ],
        index=0,
    )

    selected_sos: SOSLocation = None

    if "1." in loc_mode:
        preset_options = {p["name"]: p for p in presets}
        chosen_preset_name = st.selectbox("Select Verified Accident SOS Payload", list(preset_options.keys()))
        preset_data = preset_options[chosen_preset_name]

        lat = preset_data["latitude"]
        lon = preset_data["longitude"]
        name = preset_data["name"]

        selected_sos, msg = create_sos_location(
            lat=lat,
            lon=lon,
            source=SOURCE_GPS_SOS_PAYLOAD,
            accuracy_meters=5.0,  # High accuracy payload
            location_name=name,
        )

    elif "2." in loc_mode:
        st.info("Browser GPS requires explicit device permission. Click below to capture.")
        # HTML5 Geolocation bridge
        gps_html = """
        <div style="font-family: sans-serif; padding: 10px; background: #f0f2f6; border-radius: 8px;">
            <p><strong>Device Geolocation API (User Consent Required)</strong></p>
            <button onclick="getLocation()" style="padding: 8px 16px; background: #ff4b4b; color: white; border: none; border-radius: 4px; cursor: pointer;">
                📍 Grant Permission & Capture Device GPS
            </button>
            <p id="status" style="font-size: 13px; color: #555; margin-top: 8px;">Waiting for user trigger...</p>
        </div>
        <script>
            function getLocation() {
                var status = document.getElementById("status");
                if (navigator.geolocation) {
                    status.innerText = "Requesting browser permission...";
                    navigator.geolocation.getCurrentPosition(showPosition, showError, {timeout: 10000, enableHighAccuracy: true});
                } else {
                    status.innerText = "Geolocation is not supported by this browser.";
                }
            }
            function showPosition(position) {
                var status = document.getElementById("status");
                status.innerText = "Captured: Lat " + position.coords.latitude.toFixed(5) + ", Lon " + position.coords.longitude.toFixed(5) + " (±" + Math.round(position.coords.accuracy) + "m)";
            }
            function showError(error) {
                var status = document.getElementById("status");
                switch(error.code) {
                    case error.PERMISSION_DENIED:
                        status.innerText = "Permission Denied: User refused GPS access.";
                        break;
                    case error.POSITION_UNAVAILABLE:
                        status.innerText = "Position Unavailable: GPS signal not acquired.";
                        break;
                    case error.TIMEOUT:
                        status.innerText = "Timeout: Location request timed out.";
                        break;
                    default:
                        status.innerText = "Unknown geolocation error occurred.";
                }
            }
        </script>
        """
        components.html(gps_html, height=130)

        # Allow entering captured device coords
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            dev_lat = st.number_input("Captured Latitude", value=31.2185, format="%.5f")
        with col_c2:
            dev_lon = st.number_input("Captured Longitude", value=75.7852, format="%.5f")
        with col_c3:
            dev_acc = st.number_input("GPS Accuracy (m)", value=8.0, min_value=1.0, max_value=500.0)

        selected_sos, msg = create_sos_location(
            lat=dev_lat,
            lon=dev_lon,
            source=SOURCE_DEVICE_GEOLOCATION,
            accuracy_meters=dev_acc,
            location_name="Device GPS (Physically at Incident)",
        )

    else:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            man_lat = st.number_input("Operator Latitude", value=31.2142, format="%.5f")
        with col_m2:
            man_lon = st.number_input("Operator Longitude", value=75.7785, format="%.5f")

        man_name = st.text_input("Verified Location Description", value="Phagwara NH-44 Highway Milepost")
        selected_sos, msg = create_sos_location(
            lat=man_lat,
            lon=man_lon,
            source=SOURCE_OPERATOR_MANUAL,
            accuracy_meters=25.0,
            location_name=man_name,
        )

    # Display validated location metrics
    if selected_sos:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latitude", f"{selected_sos.latitude:.5f}°")
        c2.metric("Longitude", f"{selected_sos.longitude:.5f}°")
        c3.metric("GPS Accuracy", f"±{selected_sos.accuracy_meters:.1f} m" if selected_sos.accuracy_meters else "N/A")
        c4.metric("Location Source", selected_sos.source)

    return selected_sos
