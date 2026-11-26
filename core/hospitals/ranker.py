"""Intelligent Hospital Selection and Multi-Tier Ranking.

Selects and ranks nearest suitable hospitals from validated SOS coordinates
based on documented capabilities and road-network travel time.
"""

from typing import List, Dict, Any, Optional
from core.sos.location import SOSLocation
from core.hospitals.catalog import HospitalCatalog
from core.routing.router import RoadNetworkRouter


class HospitalRanker:
    """Ranks hospitals primarily by documented suitability and secondary by road travel time."""

    def __init__(
        self,
        catalog: Optional[HospitalCatalog] = None,
        router: Optional[RoadNetworkRouter] = None,
    ):
        self.catalog = catalog or HospitalCatalog()
        self.router = router or RoadNetworkRouter()

    def rank_hospitals(
        self,
        sos_location: SOSLocation,
        urgency_level: str = "HIGH",
        max_recommendations: int = 5,
    ) -> Dict[str, Any]:
        """Rank eligible facilities from validated SOS location.

        Args:
            sos_location: Validated incident SOSLocation.
            urgency_level: 'CRITICAL', 'HIGH', 'MEDIUM', or 'LOW'.
            max_recommendations: Number of ranked facilities to return.

        Returns:
            Dictionary with primary recommendation, alternative hospitals, and ranking metadata.
        """
        if not sos_location or not sos_location.latitude or not sos_location.longitude:
            return {
                "error": "No valid accident SOS location provided. Coordinates are required for hospital routing.",
                "primary_recommendation": None,
                "alternatives": [],
            }

        start_lat = sos_location.latitude
        start_lon = sos_location.longitude

        # For CRITICAL or HIGH urgency, prioritize hospital category
        require_hospital = urgency_level in ("CRITICAL", "HIGH")
        candidates = self.catalog.get_eligible_facilities(require_hospital_tier=require_hospital)
        if not candidates:
            # Fallback to all eligible if no strict hospitals available
            candidates = self.catalog.get_eligible_facilities(require_hospital_tier=False)

        scored_hospitals: List[Dict[str, Any]] = []

        for cand in candidates:
            c_lat = cand["latitude"]
            c_lon = cand["longitude"]

            # Calculate road route starting strictly from SOS coordinates
            route_res = self.router.calculate_route(start_lat, start_lon, c_lat, c_lon)

            # Scoring:
            # Tier 1 (Suitability): documented emergency tag 'yes' gives priority bonus
            # Tier 2 (Category): hospital (0) vs clinic (+5 min penalty)
            # Tier 3 (Road Time): travel time in minutes
            suitability_score = 0.0
            if cand.get("has_documented_emergency"):
                suitability_score += 100.0  # Documented emergency tag priority
            if cand.get("category") == "hospital":
                suitability_score += 50.0  # Full hospital priority

            # Adjusted cost = Road travel time - suitability bonus weight
            # Lower adjusted cost is better
            travel_time = route_res["estimated_time_minutes"]
            adjusted_cost = travel_time - (suitability_score * 0.1)

            # Build ranking reason
            reasons = []
            if cand.get("has_documented_emergency"):
                reasons.append("Documented 24/7 Emergency Services")
            if cand.get("category") == "hospital":
                reasons.append("Full Hospital Tier")
            else:
                reasons.append("Clinic / Healthcare Center")
            reasons.append(f"Road Distance: {route_res['distance_km']} km (~{travel_time} mins)")

            scored_hospitals.append({
                "facility": cand,
                "route": route_res,
                "suitability_score": suitability_score,
                "adjusted_cost": round(adjusted_cost, 2),
                "ranking_reason": " | ".join(reasons),
            })

        # Sort: Highest suitability score first, then lowest travel time
        scored_hospitals.sort(
            key=lambda x: (-x["suitability_score"], x["route"]["estimated_time_minutes"])
        )

        if not scored_hospitals:
            return {
                "error": "No eligible hospitals found within corridor.",
                "primary_recommendation": None,
                "alternatives": [],
            }

        primary = scored_hospitals[0]
        alternatives = scored_hospitals[1:max_recommendations]

        return {
            "sos_location": sos_location.to_dict(),
            "primary_recommendation": primary,
            "alternatives": alternatives,
            "total_candidates_evaluated": len(scored_hospitals),
            "urgency_level": urgency_level,
        }
