"""SOS Incident Location and Geolocation Module.

Validates and manages emergency GPS coordinates, browser geolocation capture,
coordinate integrity checks, and source provenance tracking.
"""

from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone


# Supported Location Sources in strict priority order
SOURCE_GPS_SOS_PAYLOAD = "GPS_SOS_PAYLOAD"  # Priority 1: Genuine incoming vehicle / device SOS payload
SOURCE_DEVICE_GEOLOCATION = "DEVICE_GEOLOCATION"  # Priority 2: Explicitly captured device location at scene
SOURCE_OPERATOR_MANUAL = "OPERATOR_MANUAL_SELECTION"  # Priority 3: Operator verified incident location

# Geofence boundary for the demo region (Punjab / Phagwara corridor)
MIN_LAT, MAX_LAT = 30.5, 32.0
MIN_LON, MAX_LON = 75.0, 76.5


class SOSLocation:
    """Represents a validated incident location with explicit provenance and accuracy."""

    def __init__(
        self,
        latitude: float,
        longitude: float,
        source: str = SOURCE_GPS_SOS_PAYLOAD,
        accuracy_meters: Optional[float] = None,
        timestamp: Optional[str] = None,
        location_name: Optional[str] = None,
    ):
        self.latitude = float(latitude)
        self.longitude = float(longitude)
        self.source = source
        self.accuracy_meters = float(accuracy_meters) if accuracy_meters is not None else None
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        self.location_name = location_name or f"{latitude:.4f}°N, {longitude:.4f}°E"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "source": self.source,
            "accuracy_meters": self.accuracy_meters,
            "timestamp": self.timestamp,
            "location_name": self.location_name,
            "is_valid": True,
        }


def validate_coordinates(
    lat: Any,
    lon: Any,
    enforce_regional_bounds: bool = False,
) -> Tuple[bool, str]:
    """Validate latitude and longitude values.

    Returns:
        (is_valid, error_message)
    """
    try:
        f_lat = float(lat)
        f_lon = float(lon)
    except (ValueError, TypeError):
        return False, "Coordinates must be valid numeric values."

    # General global latitude and longitude boundaries
    if f_lat < -90.0 or f_lat > 90.0:
        return False, f"Latitude {f_lat} is outside valid range [-90, +90]."
    if f_lon < -180.0 or f_lon > 180.0:
        return False, f"Longitude {f_lon} is outside valid range [-180, +180]."

    # Reject null island / default 0,0
    if abs(f_lat) < 0.0001 and abs(f_lon) < 0.0001:
        return False, "Rejecting invalid default coordinates (0.0, 0.0)."

    # Regional bounds check for corridor routing
    if enforce_regional_bounds:
        if not (MIN_LAT <= f_lat <= MAX_LAT and MIN_LON <= f_lon <= MAX_LON):
            return False, f"Coordinates ({f_lat}, {f_lon}) are outside the active road corridor [{MIN_LAT}-{MAX_LAT}°N, {MIN_LON}-{MAX_LON}°E]."

    return True, "Coordinates are valid."


def create_sos_location(
    lat: float,
    lon: float,
    source: str = SOURCE_GPS_SOS_PAYLOAD,
    accuracy_meters: Optional[float] = None,
    timestamp: Optional[str] = None,
    location_name: Optional[str] = None,
    enforce_regional_bounds: bool = False,
) -> Tuple[Optional[SOSLocation], str]:
    """Factory to create a validated SOSLocation instance.

    Rejects invalid coordinates and ensures source provenance is preserved.
    """
    valid, msg = validate_coordinates(lat, lon, enforce_regional_bounds=enforce_regional_bounds)
    if not valid:
        return None, msg

    valid_sources = (SOURCE_GPS_SOS_PAYLOAD, SOURCE_DEVICE_GEOLOCATION, SOURCE_OPERATOR_MANUAL)
    if source not in valid_sources:
        source = SOURCE_OPERATOR_MANUAL

    loc = SOSLocation(
        latitude=lat,
        longitude=lon,
        source=source,
        accuracy_meters=accuracy_meters,
        timestamp=timestamp,
        location_name=location_name,
    )
    return loc, "Valid location created."
