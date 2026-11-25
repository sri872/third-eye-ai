"""Pure formatting helpers for the Hospitals & emergency numbers tab.

No Streamlit calls live here so the rendering rules can be unit tested cheaply.
All facility values come from the ranking result; no value is invented.
"""

from __future__ import annotations

from typing import Any, Dict, List

from app.dispatch_ui import display, emergency_status
from core.hospitals.ranking import DISTANCE_LABEL, ROAD_ROUTE_UNAVAILABLE


GENERAL_NUMBERS_NOTE = (
    "General government emergency numbers (REAL, verified in the source data); "
    "these are NOT hospital-specific numbers."
)
HOSPITAL_NUMBERS_NOTE = (
    "Hospital-specific numbers come only from OSM tags and may be missing, "
    "unverified or stale; the app does not place calls."
)
LIMITATION_LINES = (
    "No reliable road-routing provider is configured for this tab: road distance "
    "and estimated driving time are unavailable.",
    "Straight-line distances are computed from the SIMULATED demo location; "
    "no device GPS verification is available.",
    "Emergency department, specialty and phone fields are only as reported by "
    "OpenStreetMap; missing fields are shown as unavailable, never assumed.",
    "Bed availability, waiting times, 24/7 operation and ambulance dispatch "
    "status are not determined from OpenStreetMap.",
    "Ranking scores are transparent heuristics, not verified medical suitability.",
)


def rank_basis(item: Dict[str, Any]) -> str:
    """Short, source-labelled reason shown on a recommendation card."""
    parts = [f"category: {item.get('facility_category', 'not determined')}"]
    parts.append(
        "OSM emergency=yes (source-reported)"
        if item.get("emergency_tier") == 0
        else "emergency department not source-confirmed"
    )
    parts.append(f"{item.get('distance_km')} km straight-line (SIMULATED location)")
    parts.append("heuristic score only")
    return "; ".join(parts)


def recommendation_rows(suggestions: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Compact, readable rows for one recommendation card or the fallback table."""
    rows = []
    for rank, item in enumerate(suggestions, start=1):
        suitability = (
            "SOURCE-CONFIRMED EMERGENCY TAG (not independently verified)"
            if item.get("recommended_for_emergency")
            else "NOT RECOMMENDED WITHOUT VERIFICATION — emergency department unavailable/unspecified"
        )
        rows.append(
            {
                "Rank": str(rank),
                "Facility": display(item["hospital"].name),
                "Category": display(item.get("facility_category")),
                "Straight-line distance": (
                    f"{display(item.get('distance_km'))} km "
                    f"({item.get('distance_label') or DISTANCE_LABEL})"
                ),
                "Phone (hospital-specific, OSM)": (
                    f"{display(item.get('phone'))} — {display(item.get('phone_status'))}"
                ),
                "Emergency department (OSM)": display(item.get("emergency_suitability")),
                "Emergency suitability": suitability,
                "Specialties (OSM)": display(item.get("specialties")),
                "Source specialty categories": (
                    ", ".join(item.get("specialty_categories") or [])
                    or "not specified in OpenStreetMap"
                ),
                "Road distance / driving time": ROAD_ROUTE_UNAVAILABLE,
                "Rank basis": rank_basis(item),
            }
        )
    return rows


def general_emergency_rows(contacts: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Render the verified general emergency contacts without changing their data."""
    return [
        {
            "Service": display(contact.get("service")),
            "Number": display(contact.get("number")),
            "Status": emergency_status(contact),
            "Official source": display(contact.get("official_source")),
        }
        for contact in contacts
    ]


def hospital_number_lines(suggestions: List[Dict[str, Any]]) -> List[str]:
    """Source-labelled hospital-specific numbers, including explicit gaps."""
    return [
        f"{item['hospital'].name}: {display(item.get('phone'))} "
        f"({display(item.get('phone_status'))})"
        for item in suggestions
    ]
