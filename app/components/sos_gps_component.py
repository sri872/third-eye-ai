"""Lazy browser geolocation bridge returning coordinates to Streamlit."""
from functools import lru_cache
from pathlib import Path

from streamlit.components.v1 import declare_component


@lru_cache(maxsize=1)
def _get_component():
    """Register the component in the active Streamlit script context on first use."""
    return declare_component(
        "third_eye_sos_gps",
        path=str(Path(__file__).resolve().parent / "sos_gps_frontend"),
    )


def capture_device_location():
    """Ask for browser GPS only after the user clicks the in-frame capture button."""
    return _get_component()(key="third-eye-device-gps", default=None)
