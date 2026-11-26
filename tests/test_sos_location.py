"""Tests for SOS Location and Geolocation Module.

Verifies coordinate validation, boundary checking, source prioritization,
and provenance integrity for incident locations.
"""

import pytest
from core.sos.location import (
    SOSLocation,
    validate_coordinates,
    create_sos_location,
    SOURCE_GPS_SOS_PAYLOAD,
    SOURCE_DEVICE_GEOLOCATION,
    SOURCE_OPERATOR_MANUAL,
)


def test_validate_coordinates_valid():
    """Test valid coordinates are accepted."""
    valid, msg = validate_coordinates(31.2142, 75.7785)
    assert valid is True
    assert "valid" in msg.lower()


def test_validate_coordinates_invalid_bounds():
    """Test out of range coordinates are rejected."""
    # Latitude out of bounds
    v1, msg1 = validate_coordinates(95.0, 75.0)
    assert v1 is False
    assert "Latitude" in msg1

    # Longitude out of bounds
    v2, msg2 = validate_coordinates(31.0, 195.0)
    assert v2 is False
    assert "Longitude" in msg2


def test_validate_coordinates_null_island():
    """Test null island (0.0, 0.0) is rejected as invalid default."""
    valid, msg = validate_coordinates(0.0, 0.0)
    assert valid is False
    assert "Rejecting" in msg


def test_validate_coordinates_non_numeric():
    """Test non-numeric coordinate inputs are rejected."""
    valid, msg = validate_coordinates("invalid_lat", "invalid_lon")
    assert valid is False
    assert "numeric" in msg


def test_create_sos_location_provenance():
    """Test creation and serialization of SOSLocation with source tracking."""
    loc, msg = create_sos_location(
        lat=31.2185,
        lon=75.7852,
        source=SOURCE_GPS_SOS_PAYLOAD,
        accuracy_meters=4.5,
        location_name="Banga Road Intersection",
    )
    assert loc is not None
    assert loc.latitude == 31.2185
    assert loc.longitude == 75.7852
    assert loc.source == SOURCE_GPS_SOS_PAYLOAD
    assert loc.accuracy_meters == 4.5
    assert loc.location_name == "Banga Road Intersection"

    data = loc.to_dict()
    assert data["latitude"] == 31.2185
    assert data["source"] == SOURCE_GPS_SOS_PAYLOAD
    assert data["accuracy_meters"] == 4.5
    assert "timestamp" in data


def test_regional_geofence_validation():
    """Test regional corridor geofence enforcement."""
    # Phagwara coordinate: inside
    v_in, _ = validate_coordinates(31.2142, 75.7785, enforce_regional_bounds=True)
    assert v_in is True

    # London coordinate: outside active corridor
    v_out, msg_out = validate_coordinates(51.5074, -0.1278, enforce_regional_bounds=True)
    assert v_out is False
    assert "corridor" in msg_out.lower()
