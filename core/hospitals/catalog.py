"""Hospital Catalog and Documented Capability Extraction.

Loads real OpenStreetMap healthcare facilities from data/hospitals_osm.json
and parses documented facility categories and emergency tags without inventing clinical capabilities.
"""

import os
import json
from typing import List, Dict, Any, Optional


class HospitalCatalog:
    """Manages cached OpenStreetMap hospital records for the active corridor."""

    def __init__(self, data_path: Optional[str] = None):
        if data_path is None:
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.abspath(
                os.path.join(curr_dir, "..", "..", "data", "hospitals_osm.json")
            )
        self.data_path = data_path
        self.facilities: List[Dict[str, Any]] = []
        self._load_facilities()

    def _load_facilities(self) -> None:
        if not os.path.exists(self.data_path):
            return

        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        elements = data.get("elements", [])
        for el in elements:
            tags = el.get("tags", {})
            name = tags.get("name")
            if not name:
                continue

            lat = el.get("lat")
            lon = el.get("lon")
            if lat is None or lon is None:
                continue

            # Determine facility category from OSM tags
            healthcare = tags.get("healthcare", "")
            amenity = tags.get("amenity", "")

            if healthcare == "hospital" or amenity == "hospital":
                category = "hospital"
            elif healthcare in ("clinic", "centre") or amenity in ("clinic", "doctors"):
                category = "clinic"
            else:
                category = "healthcare_facility"

            # Documented emergency capabilities - NEVER invent trauma or ICU from name!
            emergency_tag = tags.get("emergency", "no").lower()
            has_documented_emergency = (emergency_tag == "yes")

            speciality = tags.get("healthcare:speciality", "General / Undocumented")
            phone = tags.get("phone") or tags.get("contact:phone") or tags.get("mobile")

            addr_dict = el.get("address", {})
            full_addr_parts = [
                addr_dict.get("road"),
                addr_dict.get("town") or addr_dict.get("city") or addr_dict.get("hamlet"),
                addr_dict.get("county"),
                addr_dict.get("state"),
                addr_dict.get("postcode"),
            ]
            full_addr = ", ".join([p for p in full_addr_parts if p]) or "Phagwara Corridor, Punjab, India"

            self.facilities.append({
                "id": el.get("id"),
                "name": name,
                "latitude": float(lat),
                "longitude": float(lon),
                "category": category,
                "emergency_tag": emergency_tag,
                "has_documented_emergency": has_documented_emergency,
                "speciality": speciality,
                "phone": phone,
                "address": full_addr,
                "source": "OpenStreetMap contributors (ODbL)",
                "data_freshness": "Cached OSM corridor extract",
            })

    def get_eligible_facilities(self, require_hospital_tier: bool = False) -> List[Dict[str, Any]]:
        """Filter facilities eligible for emergency ambulance routing."""
        if require_hospital_tier:
            return [f for f in self.facilities if f["category"] == "hospital"]
        return [f for f in self.facilities if f["category"] in ("hospital", "clinic")]
