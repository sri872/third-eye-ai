"""Conservative suitability-first ranking of road-routable hospitals."""
from __future__ import annotations

from typing import Any, Optional

from core.sos.location import SOSLocation
from core.hospitals.catalog import HospitalCatalog
from core.routing.router import RoadNetworkRouter, RouteUnavailableError


_SPECIALTY_TRIGGERS = {
    "eye": ("eye", "vision", "cornea", "ophthalm"),
    "dental": ("dental", "tooth", "teeth", "jaw"),
    "maternity": ("pregnant", "pregnancy", "maternity", "labor", "delivery"),
    "children": ("child", "children", "infant", "baby", "pediatric", "kid"),
    "cardiac": ("heart", "cardiac", "chest pain", "angina"),
}
_SPECIALTY_MARKERS = {
    "eye": ("eye", "ophthalm", "netra"),
    "dental": ("dental", "dentist"),
    "maternity": ("maternity", "gynaec", "gynec", "obstetric"),
    "children": ("children", "paediatric", "pediatric", "neonatal"),
    "cardiac": ("heart care", "cardiac", "cardiology"),
}


class HospitalRanker:
    """Ranks source-tagged hospitals using documented suitability then road ETA."""

    def __init__(
        self,
        catalog: Optional[HospitalCatalog] = None,
        router: Optional[RoadNetworkRouter] = None,
    ):
        self.catalog = catalog or HospitalCatalog()
        self.router = router or RoadNetworkRouter()

    @staticmethod
    def _specialty_penalty(facility: dict[str, Any], incident_text: str) -> tuple[int, str]:
        specialty = str(facility.get("speciality") or "").lower()
        name = str(facility.get("name") or "").lower()
        hint = specialty + " " + name
        incident = (incident_text or "").lower()
        for category, markers in _SPECIALTY_MARKERS.items():
            if any(marker in hint for marker in markers):
                if any(trigger in incident for trigger in _SPECIALTY_TRIGGERS[category]):
                    return 0, f"Possible '{category}' name/source hint matches the caller report; capability is not verified."
                return 1, (
                    f"Down-ranked: name/source contains a possible '{category}' specialty hint "
                    "not mentioned in the caller description. This is not confirmation of capability."
                )
        return 0, "No single-specialty name/source hint conflicts with the caller description."

    def rank_hospitals(
        self,
        sos_location: SOSLocation,
        urgency_level: str = "HIGH",
        max_recommendations: int = 5,
        incident_text: str = "",
    ) -> dict[str, Any]:
        """Return nearest routable source-tagged hospitals; no straight-line fallback."""
        if sos_location is None:
            return {
                "error": "A validated incident SOS location is required.",
                "primary_recommendation": None,
                "alternatives": [],
            }
        try:
            lat, lon = float(sos_location.latitude), float(sos_location.longitude)
        except (TypeError, ValueError):
            return {
                "error": "SOS coordinates are invalid.",
                "primary_recommendation": None,
                "alternatives": [],
            }
        if not (-90 <= lat <= 90 and -180 <= lon <= 180) or (abs(lat) < 0.0001 and abs(lon) < 0.0001):
            return {
                "error": "SOS coordinates are outside geographic limits or equal to invalid default 0,0.",
                "primary_recommendation": None,
                "alternatives": [],
            }
        if sos_location.is_stale:
            return {
                "error": "SOS location is older than 15 minutes. Capture or confirm a fresh accident location.",
                "primary_recommendation": None,
                "alternatives": [],
            }

        candidates = self.catalog.get_eligible_facilities(require_hospital_tier=True)
        candidates = [
            item for item in candidates
            if item.get("category") == "hospital"
            and item.get("name")
            and item.get("latitude") is not None
            and item.get("longitude") is not None
        ]
        if not candidates:
            return {
                "error": "The OSM snapshot contains no source-tagged hospital records.",
                "primary_recommendation": None,
                "alternatives": [],
            }

        ranked: list[dict[str, Any]] = []
        unroutable = []
        for facility in candidates:
            try:
                route = self.router.calculate_route(
                    lat, lon, facility["latitude"], facility["longitude"]
                )
            except (RouteUnavailableError, ValueError) as exc:
                unroutable.append({"facility": facility["name"], "reason": str(exc)})
                continue
            if not route.get("is_road_network"):
                unroutable.append({
                    "facility": facility["name"],
                    "reason": "A verified road-network route was not returned.",
                })
                continue
            specialty_penalty, specialty_reason = self._specialty_penalty(
                facility, incident_text
            )
            emergency_tag_priority = 0 if facility.get("has_documented_emergency") else 1
            reasons = [
                "Source-tagged hospital (OpenStreetMap)",
                specialty_reason,
                (
                    "OSM emergency=yes tag present (hours and readiness unverified)"
                    if facility.get("has_documented_emergency")
                    else "Emergency service capability undocumented in OSM"
                ),
                f"Directed road-network distance {route['distance_km']:.2f} km; "
                f"static estimated travel time {route['estimated_time_minutes']:.1f} min",
            ]
            ranked.append({
                "facility": facility,
                "route": route,
                "specialty_penalty": specialty_penalty,
                "emergency_tag_priority": emergency_tag_priority,
                "ranking_reason": " | ".join(reasons),
            })

        # For a critical incident, source-tagged emergency capability is
        # the first available ranking signal; otherwise use specialty-name
        # hint compatibility first. Neither OSM tag proves clinical readiness.
        urgency = str(urgency_level or "HIGH").upper()
        if urgency == "CRITICAL":
            ranked.sort(key=lambda item: (
                item["emergency_tag_priority"],
                item["specialty_penalty"],
                item["route"]["estimated_time_minutes"],
                item["route"]["distance_km"],
                item["facility"]["name"].lower(),
            ))
        else:
            ranked.sort(key=lambda item: (
                item["specialty_penalty"],
                item["emergency_tag_priority"],
                item["route"]["estimated_time_minutes"],
                item["route"]["distance_km"],
                item["facility"]["name"].lower(),
            ))
        if not ranked:
            return {
                "error": (
                    "No hospital is reachable from this SOS location on the saved directed "
                    "road graph. No straight-line estimate has been represented as a road route."
                ),
                "primary_recommendation": None,
                "alternatives": [],
                "unroutable_hospitals": unroutable,
            }

        count = max(1, int(max_recommendations))
        return {
            "sos_location": sos_location.to_dict(),
            "primary_recommendation": ranked[0],
            "alternatives": ranked[1:count],
            "total_candidates_evaluated": len(ranked),
            "total_hospitals_in_snapshot": len(candidates),
            "unroutable_hospitals": unroutable,
            "urgency_level": urgency_level,
            "ranking_method": (
                (
                    "CRITICAL: documented emergency tag priority, specialty-name hint compatibility, "
                    "then static road-network time; "
                    if urgency == "CRITICAL"
                    else "Specialty-name hint compatibility, documented emergency tag, "
                         "then static road-network time; "
                )
                + "all emergency/capability tags are unverified source metadata."
            ),
            "limitations": (
                "OpenStreetMap data may be incomplete. Emergency department capability, "
                "staffing, opening hours, hospital acceptance and bed availability are not verified. "
                "Route times use static graph estimates, not live traffic."
            ),
        }
