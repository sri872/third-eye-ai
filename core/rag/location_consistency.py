"""Location provenance and mismatch detection for Tab 5.

The app has two different locations:

* the selected SIMULATED demo location, whose coordinates are used for hospital
  lookup and straight-line distance; and
* the caller-reported location from the SIMULATED sample call in Tab 4.

This module keeps them separate, reports a discrepancy instead of silently
substituting one for the other, and never claims device GPS verification.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, Optional

NOT_REPORTED = "not reported"

_STOP_TOKENS = {
    "near",
    "road",
    "rd",
    "street",
    "st",
    "highway",
    "nh",
    "nh44",
    "gt",
    "mg",
    "junction",
    "chowk",
    "intersection",
    "crossing",
    "metro",
    "station",
    "the",
    "and",
    "of",
    "at",
    "by",
    "simulated",
    "location",
}

GPS_STATUS = "NOT AVAILABLE — no device GPS verification was performed"


@dataclass(frozen=True)
class LocationConsistency:
    """Source-separated view of the selected and caller-reported locations."""

    selected_name: str
    selected_coordinates: str
    caller_reported: str
    status: str
    detail: str
    gps_status: str = GPS_STATUS

    def as_dict(self) -> Dict[str, str]:
        return {
            "selected_name": self.selected_name,
            "selected_coordinates": self.selected_coordinates,
            "caller_reported": self.caller_reported,
            "status": self.status,
            "detail": self.detail,
            "gps_status": self.gps_status,
        }


def _display(value: Any, missing: str = NOT_REPORTED) -> str:
    if value is None:
        return missing
    text = str(value).strip()
    return text if text and text.lower() not in {"none", "not determined"} else missing


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {word for word in words if word not in _STOP_TOKENS and len(word) > 1}


def compare_locations(
    selected: Optional[Dict[str, Any]],
    caller_reported: Optional[str],
) -> LocationConsistency:
    """Compare (but never merge) the selected demo location and caller report.

    The comparison is text-only. A high token overlap is reported as a possible
    text match, not a verified location; anything else is a discrepancy.
    """
    selected = selected or {}
    name = _display(
        selected.get("name") or selected.get("location_name"),
        "not determined",
    )
    lat = selected.get("latitude")
    lon = selected.get("longitude")
    if lat is None or lon is None:
        coordinates = "not determined"
    else:
        coordinates = f"{lat}, {lon} (SIMULATED demo coordinates)"

    caller = _display(caller_reported)
    if name == "not determined" or lat is None or lon is None:
        status = "SELECTED LOCATION MISSING"
        detail = "No selected demo coordinates; hospital lookup cannot be performed."
    elif caller == NOT_REPORTED:
        status = "CALLER LOCATION NOT REPORTED"
        detail = (
            f"Distances use only the selected demo location '{name}'. "
            "No caller location was reported to compare with it."
        )
    else:
        selected_tokens = _tokens(name)
        caller_tokens = _tokens(caller)
        shared = selected_tokens & caller_tokens
        smaller = min(len(selected_tokens), len(caller_tokens))
        overlap = len(shared) / smaller if smaller else 0.0
        if len(shared) >= 2 and overlap >= 0.5:
            status = "POSSIBLE TEXT MATCH (NOT GPS-VERIFIED)"
            detail = (
                f"Caller said '{caller}'; selected demo location is '{name}'. "
                "Text tokens overlap, but this is not GPS verification."
            )
        else:
            status = "LOCATION MISMATCH"
            detail = (
                f"Caller reported '{caller}', while the selected demo location is '{name}'. "
                "The caller-reported location was preserved and was NOT substituted. "
                f"All distances use only the selected demo coordinates ({lat}, {lon})."
            )

    return LocationConsistency(
        selected_name=name,
        selected_coordinates=coordinates,
        caller_reported=caller,
        status=status,
        detail=detail,
    )
