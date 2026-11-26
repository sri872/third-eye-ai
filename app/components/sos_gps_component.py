"""Actual browser geolocation bridge returning coordinates to Streamlit."""
from pathlib import Path
import streamlit.components.v1 as components

_component = components.declare_component(
    "third_eye_sos_gps",
    path=str(Path(__file__).resolve().parent / "sos_gps_frontend"),
)


def capture_device_location():
    """Ask for browser GPS only after the user clicks the in-frame capture button."""
    return _component(key="third-eye-device-gps", default=None)
