"""SOS incident location validation, parsing and provenance tracking."""
from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Mapping, Optional, Tuple

SOURCE_GPS_SOS_PAYLOAD = "GPS_SOS_PAYLOAD"  # Reserved for authenticated SOS service adapters.
SOURCE_SOS_JSON_IMPORT = "SOS_JSON_IMPORT"  # Uploaded/pasted; sender is not authenticated by this app.
SOURCE_DEVICE_GEOLOCATION = "DEVICE_GEOLOCATION"
SOURCE_OPERATOR_MANUAL = "OPERATOR_MANUAL_SELECTION"
SOURCE_SIMULATED_DEMO = "SIMULATED_DEMO"

MIN_LAT, MAX_LAT = 30.5, 32.0
MIN_LON, MAX_LON = 75.0, 76.5
MAX_LOCATION_AGE_SECONDS = 15 * 60


def _parse_timestamp(value: Any) -> datetime:
    if value in (None, ""):
        return datetime.now(timezone.utc)
    try:
        if isinstance(value, (int, float)):
            seconds = float(value)
            if seconds > 10_000_000_000:
                seconds /= 1000.0
            parsed = datetime.fromtimestamp(seconds, tz=timezone.utc)
        else:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            parsed = parsed.astimezone(timezone.utc)
        return parsed
    except (TypeError, ValueError, OverflowError, OSError):
        raise ValueError("Location timestamp must be ISO-8601 or a Unix timestamp.") from None


class SOSLocation:
    """Validated incident coordinates with source, accuracy and timestamp."""

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
        self.timestamp = _parse_timestamp(timestamp).isoformat()
        self.location_name = location_name or f"{self.latitude:.6f}, {self.longitude:.6f}"

    @property
    def age_seconds(self) -> float:
        stamp = _parse_timestamp(self.timestamp)
        return max(0.0, (datetime.now(timezone.utc) - stamp).total_seconds())

    @property
    def is_stale(self) -> bool:
        return self.age_seconds > MAX_LOCATION_AGE_SECONDS

    def to_dict(self) -> dict[str, Any]:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "source": self.source,
            "accuracy_meters": self.accuracy_meters,
            "timestamp": self.timestamp,
            "location_name": self.location_name,
            "age_seconds": round(self.age_seconds, 1),
            "is_stale": self.is_stale,
            "is_valid": True,
        }


def validate_coordinates(
    lat: Any,
    lon: Any,
    enforce_regional_bounds: bool = False,
) -> Tuple[bool, str]:
    """Validate finite geographic coordinates and optionally the Phagwara region."""
    try:
        f_lat = float(lat)
        f_lon = float(lon)
    except (ValueError, TypeError):
        return False, "Coordinates must be valid numeric values."
    if not math.isfinite(f_lat) or not math.isfinite(f_lon):
        return False, "Coordinates must be finite numbers."
    if not -90.0 <= f_lat <= 90.0:
        return False, f"Latitude {f_lat} is outside valid range [-90, +90]."
    if not -180.0 <= f_lon <= 180.0:
        return False, f"Longitude {f_lon} is outside valid range [-180, +180]."
    if abs(f_lat) < 0.0001 and abs(f_lon) < 0.0001:
        return False, "Rejecting invalid default coordinates (0.0, 0.0)."
    if enforce_regional_bounds and not (
        MIN_LAT <= f_lat <= MAX_LAT and MIN_LON <= f_lon <= MAX_LON
    ):
        return False, (
            f"Coordinates ({f_lat}, {f_lon}) are outside the supported "
            f"Phagwara-region bounds [{MIN_LAT}-{MAX_LAT}°N, {MIN_LON}-{MAX_LON}°E]."
        )
    return True, "Coordinates are valid."


def create_sos_location(
    lat: Any,
    lon: Any,
    source: str = SOURCE_GPS_SOS_PAYLOAD,
    accuracy_meters: Optional[Any] = None,
    timestamp: Optional[Any] = None,
    location_name: Optional[str] = None,
    enforce_regional_bounds: bool = False,
) -> Tuple[Optional[SOSLocation], str]:
    """Create an SOS location without inventing accuracy or source provenance."""
    valid, message = validate_coordinates(
        lat, lon, enforce_regional_bounds=enforce_regional_bounds
    )
    if not valid:
        return None, message
    valid_sources = (
        SOURCE_GPS_SOS_PAYLOAD,
        SOURCE_SOS_JSON_IMPORT,
        SOURCE_DEVICE_GEOLOCATION,
        SOURCE_OPERATOR_MANUAL,
        SOURCE_SIMULATED_DEMO,
    )
    if source not in valid_sources:
        return None, f"Unsupported location source: {source}"

    accuracy = None
    if accuracy_meters not in (None, ""):
        try:
            accuracy = float(accuracy_meters)
        except (TypeError, ValueError):
            return None, "GPS accuracy must be numeric."
        if not math.isfinite(accuracy) or accuracy < 0:
            return None, "GPS accuracy must be a finite non-negative value."

    try:
        parsed_timestamp = _parse_timestamp(timestamp)
        location = SOSLocation(
            latitude=float(lat),
            longitude=float(lon),
            source=source,
            accuracy_meters=accuracy,
            timestamp=parsed_timestamp.isoformat(),
            location_name=location_name,
        )
    except ValueError as exc:
        return None, str(exc)
    return location, "Valid location created."


def parse_sos_payload(
    payload: Mapping[str, Any],
    source: str = SOURCE_SOS_JSON_IMPORT,
) -> Tuple[Optional[SOSLocation], str]:
    """Parse flat, nested or GeoJSON SOS JSON; require its event timestamp."""
    if not isinstance(payload, Mapping):
        return None, "SOS event must be a JSON object."
    data: Mapping[str, Any] = payload
    if isinstance(data.get("sos"), Mapping):
        data = data["sos"]
    if isinstance(data.get("location"), Mapping):
        data = data["location"]

    if isinstance(data.get("coordinates"), (list, tuple)):
        coords = data["coordinates"]
        if len(coords) < 2:
            return None, "SOS coordinates must contain longitude and latitude."
        lon, lat = coords[0], coords[1]
    else:
        lat = data.get("latitude", data.get("lat"))
        lon = data.get("longitude", data.get("lon", data.get("lng")))

    accuracy = data.get("accuracy_meters", data.get("accuracy"))
    timestamp = data.get("timestamp", data.get("timestamp_utc", data.get("created_at")))
    if timestamp in (None, ""):
        return None, "SOS event timestamp is required so stale coordinates can be detected."
    name = data.get("location_name", data.get("name"))
    return create_sos_location(
        lat,
        lon,
        source=source,
        accuracy_meters=accuracy,
        timestamp=timestamp,
        location_name=str(name) if name else None,
    )
