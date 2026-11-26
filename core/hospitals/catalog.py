"""Hospital catalog from the dated OpenStreetMap snapshot.

Only source-tagged hospitals are eligible emergency destinations. Clinics,
doctors, dentists, pharmacies and other non-hospital services are not returned.
No clinical capabilities are inferred from facility names.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HOSPITAL_DATA = PROJECT_ROOT / "data" / "hospitals_osm.json"

_NAME_SPECIALTIES = {
    "eye": ("eye", "ophthalm", "netra"),
    "dental": ("dental", "dentist"),
    "maternity": ("maternity", "gynaec", "gynec", "obstetric"),
    "children": ("children", "paediatric", "pediatric", "neonatal"),
    "cardiac": ("heart care", "cardiac", "cardiology"),
}
_OUTPATIENT_NAME_MARKERS = (
    "clinic", "dentist", "dental clinic", "pharmacy", "chemist",
    "laboratory", "pathology", "veterinary",
)
_NONMEDICAL_MARKERS = (
    "architecture", "architectural", "design", "deisgn", "university",
    "school", "hostel", "campus", "office", "showroom", "parking", "library",
)


def _coords(element: dict[str, Any]) -> tuple[float, float] | None:
    lat, lon = element.get("lat"), element.get("lon")
    if lat is None or lon is None:
        centre = element.get("center") or {}
        lat, lon = centre.get("lat"), centre.get("lon")
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return lat, lon


def _specialty(name: str, tags: dict[str, Any]) -> Optional[str]:
    raw = tags.get("healthcare:speciality") or tags.get("speciality")
    if raw:
        return str(raw)
    name_lower = name.lower()
    for category, markers in _NAME_SPECIALTIES.items():
        if any(marker in name_lower for marker in markers):
            return f"Name hint only: {category}"
    return None


class HospitalCatalog:
    """Loads real, source-tagged hospitals with honest data freshness metadata."""

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = Path(data_path) if data_path else DEFAULT_HOSPITAL_DATA
        self.metadata: dict[str, Any] = {}
        self.facilities: list[dict[str, Any]] = []
        self._load_facilities()

    def _load_facilities(self) -> None:
        if not self.data_path.is_file():
            raise FileNotFoundError(f"Hospital snapshot not found: {self.data_path}")
        try:
            payload = json.loads(self.data_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Hospital snapshot could not be read: {exc}") from exc
        self.metadata = {
            "fetch_timestamp": payload.get("fetch_timestamp"),
            "source": payload.get("source", "OpenStreetMap"),
            "attribution": payload.get("attribution", "© OpenStreetMap contributors"),
            "license": payload.get("license", "ODbL 1.0"),
            "region": payload.get("region", {}),
        }
        seen: set[str] = set()
        for element in payload.get("elements", []):
            tags = element.get("tags") or {}
            name = str(tags.get("name") or tags.get("name:en") or "").strip()
            if not name:
                continue
            coords = _coords(element)
            if coords is None:
                continue
            amenity = str(tags.get("amenity") or "").lower()
            healthcare = str(tags.get("healthcare") or "").lower()
            if amenity in {"clinic", "doctors", "dentist", "pharmacy"}:
                continue
            if healthcare in {"clinic", "doctor", "dentist", "pharmacy", "laboratory", "veterinary"}:
                continue
            if any(re.search(r"\b" + re.escape(term) + r"\b", name.lower()) for term in _NONMEDICAL_MARKERS):
                continue
            if any(re.search(r"\b" + re.escape(term) + r"\b", name.lower()) for term in _OUTPATIENT_NAME_MARKERS):
                if "hospital" not in name.lower():
                    continue
            if amenity != "hospital" and healthcare != "hospital":
                continue

            specialty = _specialty(name, tags)
            address_data = element.get("address") or {}
            address_parts = [
                address_data.get("road"),
                address_data.get("suburb") or address_data.get("hamlet") or address_data.get("village"),
                address_data.get("town") or address_data.get("city"),
                address_data.get("postcode"),
                address_data.get("state"),
                address_data.get("country") or "India",
            ]
            address = ", ".join(str(part) for part in address_parts if part) or "Address not listed in OpenStreetMap"
            phone = (
                tags.get("phone") or tags.get("contact:phone")
                or tags.get("contact:mobile") or tags.get("telephone")
            )
            emergency_tag = str(tags.get("emergency") or "").strip().lower() or None
            osm_type = str(element.get("type") or "node")
            osm_id = element.get("id")
            facility_id = f"osm-{osm_type}-{osm_id}"
            if facility_id in seen:
                continue
            seen.add(facility_id)
            self.facilities.append({
                "id": facility_id,
                "name": name,
                "latitude": coords[0],
                "longitude": coords[1],
                "category": "hospital",
                "emergency_tag": emergency_tag,
                "has_documented_emergency": emergency_tag == "yes",
                "speciality": specialty,
                "phone": str(phone).strip() if phone else None,
                "address": address,
                "source": "OpenStreetMap contributors (ODbL)",
                "source_url": f"https://www.openstreetmap.org/{osm_type}/{osm_id}",
                "data_freshness": self.metadata.get("fetch_timestamp") or "Unknown snapshot date",
                "emergency_capability_note": (
                    "OSM emergency=yes tag is present; type, hours, staffing and current readiness are not verified."
                    if emergency_tag == "yes"
                    else "Emergency capability is undocumented in this OSM record."
                ),
            })

    def get_eligible_facilities(self, require_hospital_tier: bool = True) -> list[dict[str, Any]]:
        """Return only source-tagged hospital facilities; never substitute clinics."""
        return [facility for facility in self.facilities if facility["category"] == "hospital"]
