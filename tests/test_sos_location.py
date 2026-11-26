"""SOS coordinate parsing, validation, provenance and staleness tests."""
from datetime import datetime, timedelta, timezone

import pytest

from core.sos.location import (
    SOSLocation,
    SOURCE_DEVICE_GEOLOCATION,
    SOURCE_GPS_SOS_PAYLOAD,
    SOURCE_OPERATOR_MANUAL,
    SOURCE_SIMULATED_DEMO,
    SOURCE_SOS_JSON_IMPORT,
    create_sos_location,
    parse_sos_payload,
    validate_coordinates,
)


def _now():
    return datetime.now(timezone.utc).isoformat()


def test_validate_coordinates_valid():
    valid, message = validate_coordinates(31.2142, 75.7785)
    assert valid is True
    assert "valid" in message.lower()


def test_validate_coordinates_invalid_bounds_and_nonfinite():
    assert validate_coordinates(95.0, 75.0)[0] is False
    assert validate_coordinates(31.0, 195.0)[0] is False
    assert validate_coordinates(float("nan"), 75.0)[0] is False
    assert validate_coordinates("not-a-number", 75.0)[0] is False


def test_validate_coordinates_rejects_null_island_and_supports_region_check():
    assert validate_coordinates(0.0, 0.0)[0] is False
    assert validate_coordinates(31.2142, 75.7785, enforce_regional_bounds=True)[0] is True
    assert validate_coordinates(51.5074, -0.1278, enforce_regional_bounds=True)[0] is False


def test_create_location_preserves_provenance_without_inventing_accuracy():
    location, message = create_sos_location(
        31.2185, 75.7852,
        source=SOURCE_OPERATOR_MANUAL,
        location_name="Operator verified Banga Road",
    )
    assert location is not None, message
    assert location.source == SOURCE_OPERATOR_MANUAL
    assert location.accuracy_meters is None
    assert location.to_dict()["accuracy_meters"] is None


def test_create_location_rejects_unknown_source_and_invalid_accuracy():
    assert create_sos_location(31.22, 75.77, source="UNKNOWN")[0] is None
    assert create_sos_location(31.22, 75.77, accuracy_meters=-2)[0] is None
    assert create_sos_location(31.22, 75.77, accuracy_meters=float("inf"))[0] is None


def test_parse_flat_sos_payload_and_preserve_timestamp_accuracy():
    location, message = parse_sos_payload(
        {
            "latitude": 31.2142,
            "longitude": 75.7785,
            "accuracy_meters": 11.5,
            "timestamp": _now(),
            "location_name": "NH-44",
        },
        source=SOURCE_SOS_JSON_IMPORT,
    )
    assert location is not None, message
    assert location.latitude == pytest.approx(31.2142)
    assert location.longitude == pytest.approx(75.7785)
    assert location.accuracy_meters == pytest.approx(11.5)
    assert location.source == SOURCE_SOS_JSON_IMPORT


def test_parse_nested_and_geojson_sos_payloads():
    nested, message = parse_sos_payload(
        {"sos": {"location": {
            "latitude": 31.22, "longitude": 75.77, "timestamp": _now()
        }}}
    )
    assert nested is not None, message
    geojson, message = parse_sos_payload(
        {"coordinates": [75.77, 31.22], "timestamp": _now()}
    )
    assert geojson is not None, message
    assert geojson.latitude == pytest.approx(31.22)
    assert geojson.longitude == pytest.approx(75.77)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"latitude": 91, "longitude": 75, "timestamp": _now()},
        {"latitude": 31, "longitude": 181, "timestamp": _now()},
        {"latitude": float("nan"), "longitude": 75, "timestamp": _now()},
        {"latitude": 0, "longitude": 0, "timestamp": _now()},
        {"coordinates": [75], "timestamp": _now()},
        {"latitude": 31, "longitude": 75},
        {"latitude": 31, "longitude": 75, "timestamp": "not-a-time"},
    ],
)
def test_parse_sos_payload_rejects_invalid_inputs(payload):
    location, message = parse_sos_payload(payload)
    assert location is None
    assert message


def test_stale_sos_location_is_flagged():
    old_timestamp = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()
    location, message = create_sos_location(
        31.22, 75.77, timestamp=old_timestamp, source=SOURCE_DEVICE_GEOLOCATION
    )
    assert location is not None, message
    assert location.is_stale is True


def test_demo_source_remains_explicitly_simulated():
    location, message = create_sos_location(
        31.2142, 75.7785, source=SOURCE_SIMULATED_DEMO
    )
    assert location is not None, message
    assert location.source == SOURCE_SIMULATED_DEMO
    assert location.accuracy_meters is None


def test_reserved_authenticated_sos_payload_source_is_available():
    location, message = create_sos_location(
        31.2142, 75.7785, source=SOURCE_GPS_SOS_PAYLOAD
    )
    assert location is not None, message
    assert location.source == SOURCE_GPS_SOS_PAYLOAD


def test_sos_location_class_retains_legacy_constructor():
    location = SOSLocation(31.2142, 75.7785, source=SOURCE_GPS_SOS_PAYLOAD)
    assert location.latitude == pytest.approx(31.2142)
    assert location.longitude == pytest.approx(75.7785)
