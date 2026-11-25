"""OpenStreetMap hospital ingestion, cache management, and SQLite persistence.

Fetches real healthcare facilities from OpenStreetMap (Overpass API / Nominatim),
converts them into strict Hospital schema objects, provides honest data coverage
statistics, and falls back gracefully to cached data if external APIs are unreachable.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
import urllib.parse
import urllib.request

from core.rag.facility_classification import classify_facility
from core.schemas import Hospital
from db.database import DEFAULT_DB_PATH, init_db, save_hospitals

logger = logging.getLogger(__name__)

DEFAULT_REGION_CONFIG = Path(__file__).resolve().parent.parent.parent / "config" / "demo_region.json"
DEFAULT_CACHE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "hospitals_osm.json"

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
]

DISPATCH_HOSPITAL_KEYWORDS = [
    "hospital", "medical college", "trauma", "nursing home",
    "civil hospital", "health centre", "health center",
]

DISPATCH_NON_MEDICAL_KEYWORDS = [
    "college", "university", "department", "design", "deisgn",
    "school", "store", "shop", "rack",
]

DISPATCH_OUTPATIENT_KEYWORDS = [
    "clinic", "dentist", "dental", "pharmacy", "chemist",
    "laboratory", "lab", "veterinary", "vet",
]


def validate_hospital_for_dispatch(
    hospital: Hospital,
    raw_tags: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Optional[str]]:
    """Determine if a facility is a valid emergency ambulance destination.

    Delegates to the conservative facility classifier. Only an unambiguous
    hospital or trauma centre is eligible: outpatients, clinics, dentists,
    pharmacies, labs, vets, and non-medical buildings (universities, design
    studios, offices, shops) are excluded even when OSM mis-tags them.
    Proximity to an incident is never part of this decision.

    Returns:
        (is_valid: bool, exclusion_reason: Optional[str])
    """
    classification = classify_facility(hospital.name, raw_tags)
    return classification.dispatch_eligible, classification.reason


def load_demo_region(config_path: Optional[Path | str] = None) -> Dict[str, Any]:
    """Load demo geographic region parameters from config."""
    path = Path(config_path) if config_path else DEFAULT_REGION_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Demo region config not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_overpass_query(lat: float, lon: float, radius_km: float) -> str:
    """Build Overpass QL query for healthcare facilities within radius (meters)."""
    radius_meters = int(radius_km * 1000)
    query = f"""
[out:json][timeout:25];
(
  node["amenity"="hospital"](around:{radius_meters},{lat},{lon});
  way["amenity"="hospital"](around:{radius_meters},{lat},{lon});
  node["amenity"="clinic"](around:{radius_meters},{lat},{lon});
  way["amenity"="clinic"](around:{radius_meters},{lat},{lon});
  node["healthcare"="hospital"](around:{radius_meters},{lat},{lon});
  way["healthcare"="hospital"](around:{radius_meters},{lat},{lon});
);
out center tags;
"""
    return query.strip()


def parse_osm_element_to_hospital(elem: Dict[str, Any]) -> Optional[Hospital]:
    """Convert raw OpenStreetMap element into validated Hospital schema."""
    tags = elem.get("tags", {})
    name = tags.get("name") or tags.get("name:en")
    if not name:
        # Require hospital/clinic name; unnamed nodes are not usable routing targets
        return None

    # Handle coordinates (node has lat/lon; way has center.lat/center.lon or lat/lon)
    lat = elem.get("lat")
    lon = elem.get("lon")
    if lat is None or lon is None:
        center = elem.get("center", {})
        lat = center.get("lat")
        lon = center.get("lon")

    if lat is None or lon is None:
        return None

    elem_type = elem.get("type", "node")
    elem_id = elem.get("id")
    hospital_id = f"osm-{elem_type}-{elem_id}"

    # Phone contact
    phone = (
        tags.get("phone")
        or tags.get("contact:phone")
        or tags.get("contact:mobile")
        or tags.get("telephone")
    )
    if phone:
        phone = str(phone).strip()

    # Address extraction
    address_str = None
    if isinstance(elem.get("address"), dict):
        addr = elem["address"]
        parts = [
            addr.get("road"),
            addr.get("suburb") or addr.get("hamlet"),
            addr.get("town") or addr.get("city"),
            addr.get("postcode"),
            addr.get("state"),
        ]
        address_str = ", ".join(p for p in parts if p)
    elif "addr:full" in tags:
        address_str = tags["addr:full"]
    else:
        addr_parts = [
            tags.get("addr:housenumber"),
            tags.get("addr:street"),
            tags.get("addr:suburb"),
            tags.get("addr:city") or tags.get("addr:town"),
            tags.get("addr:postcode"),
        ]
        filtered = [p for p in addr_parts if p]
        if filtered:
            address_str = ", ".join(filtered)

    # Emergency tag
    emergency = tags.get("emergency")
    if emergency:
        emergency = emergency.strip().lower()

    # Specialties
    specialties = (
        tags.get("healthcare:speciality")
        or tags.get("speciality")
        or tags.get("medical_system")
    )
    if specialties:
        specialties = specialties.strip()

    hosp = Hospital(
        hospital_id=hospital_id,
        name=name.strip(),
        latitude=float(lat),
        longitude=float(lon),
        total_beds=None,            # Explicitly unknown in OSM: never invent
        available_beds=None,        # Explicitly unknown in OSM: never invent
        icu_beds_available=None,    # Explicitly unknown in OSM: never invent
        trauma_center_level=None,
        contact_number=phone,
        address=address_str,
        emergency=emergency,
        specialties=specialties,
        attribution="© OpenStreetMap contributors",
        confidence=1.0,             # Confirmed real geographic entity
        data_source="REAL",
    )
    is_valid, reason = validate_hospital_for_dispatch(hosp, tags)
    hosp.excluded_from_dispatch_reason = reason
    return hosp


def normalize_hospital_name(name: str) -> set:
    """Normalize facility name into core informative tokens for duplicate detection."""
    s = name.lower()
    s = s.replace("civilhospital", "civil hospital")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    tokens = set(s.split())
    # Remove purely generic terms
    tokens.difference_update({
        "hospital", "centre", "center", "clinic", "ltd", "pvt", "p",
        "and", "the", "care", "service", "services", "home", "nursing",
    })
    return tokens


def deduplicate_hospitals(
    hospitals: List[Hospital],
    distance_threshold_km: float = 0.150,
) -> Tuple[List[Hospital], int]:
    """Merge duplicate hospital records within distance threshold whose normalized names are similar.
    
    Preserves all real tags from both records (contact, emergency, specialties, address).
    Returns (deduplicated_hospitals, merged_count).
    """
    from core.rag.retriever import haversine_distance_km

    deduped: List[Hospital] = []
    to_skip = set()
    merged_count = 0

    for i in range(len(hospitals)):
        if i in to_skip:
            continue
        h1 = hospitals[i]
        merged_hosp = h1
        for j in range(i + 1, len(hospitals)):
            if j in to_skip:
                continue
            h2 = hospitals[j]
            dist = haversine_distance_km(h1.latitude, h1.longitude, h2.latitude, h2.longitude)
            if dist <= distance_threshold_km:
                t1 = normalize_hospital_name(h1.name)
                t2 = normalize_hospital_name(h2.name)
                # Check for significant token overlap or subset
                if t1 and t2 and (t1 == t2 or t1.issubset(t2) or t2.issubset(t1)):
                    # Merge records: prefer longer/more informative name
                    chosen_name = h1.name if len(h1.name) >= len(h2.name) else h2.name
                    chosen_id = h1.hospital_id if len(h1.name) >= len(h2.name) else h2.hospital_id
                    chosen_phone = h1.contact_number or h2.contact_number
                    chosen_emergency = (
                        h1.emergency if h1.emergency in ("yes", "no")
                        else (h2.emergency if h2.emergency in ("yes", "no") else None)
                    )
                    # Merge specialties
                    specs = []
                    if h1.specialties:
                        specs.append(h1.specialties)
                    if h2.specialties and h2.specialties not in specs:
                        specs.append(h2.specialties)
                    merged_specs = "; ".join(specs) if specs else None

                    # Merge address
                    chosen_addr = h1.address or h2.address

                    # Choose coordinates from facility with phone or longer name
                    chosen_lat = h1.latitude if h1.contact_number else h2.latitude
                    chosen_lon = h1.longitude if h1.contact_number else h2.longitude
                    chosen_reason = h1.excluded_from_dispatch_reason if len(h1.name) >= len(h2.name) else h2.excluded_from_dispatch_reason

                    merged_hosp = Hospital(
                        hospital_id=chosen_id,
                        name=chosen_name,
                        latitude=chosen_lat,
                        longitude=chosen_lon,
                        total_beds=None,
                        available_beds=None,
                        icu_beds_available=None,
                        trauma_center_level=None,
                        contact_number=chosen_phone,
                        address=chosen_addr,
                        emergency=chosen_emergency,
                        specialties=merged_specs,
                        attribution="© OpenStreetMap contributors",
                        confidence=1.0,
                        data_source="REAL",
                        excluded_from_dispatch_reason=chosen_reason,
                    )
                    to_skip.add(j)
                    merged_count += 1
                    logger.info("Merged duplicate hospital: '%s' and '%s' (%d m apart)", h1.name, h2.name, int(dist * 1000))
        deduped.append(merged_hosp)

    return deduped, merged_count


def compute_coverage_statistics(hospitals: List[Hospital]) -> Dict[str, Any]:
    """Compute and report honest data coverage statistics for retrieved hospitals."""
    total = len(hospitals)
    if total == 0:
        return {
            "total_count": 0,
            "phone_count": 0,
            "phone_percentage": 0.0,
            "emergency_count": 0,
            "emergency_percentage": 0.0,
            "specialties_count": 0,
            "specialties_percentage": 0.0,
        }

    phone_count = sum(1 for h in hospitals if h.contact_number is not None)
    emergency_count = sum(1 for h in hospitals if h.emergency is not None)
    specialties_count = sum(1 for h in hospitals if h.specialties is not None)

    return {
        "total_count": total,
        "phone_count": phone_count,
        "phone_percentage": round((phone_count / total) * 100, 2),
        "emergency_count": emergency_count,
        "emergency_percentage": round((emergency_count / total) * 100, 2),
        "specialties_count": specialties_count,
        "specialties_percentage": round((specialties_count / total) * 100, 2),
    }


def fetch_from_overpass(lat: float, lon: float, radius_km: float) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Attempt to fetch healthcare data from public Overpass API endpoints."""
    query = build_overpass_query(lat, lon, radius_km)
    data = urllib.parse.urlencode({"data": query}).encode("utf-8")
    headers = {
        "User-Agent": "ProjectThirdEye/1.0 (AmbulanceRoutingResearch; contact@thirdeye.edu)",
        "Accept": "application/json",
    }

    last_error = None
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            req = urllib.request.Request(endpoint, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=12) as resp:
                if resp.status == 200:
                    raw_text = resp.read().decode("utf-8")
                    return json.loads(raw_text), None
        except Exception as e:
            last_error = f"{endpoint}: {type(e).__name__} - {e}"
            logger.warning("Overpass fetch attempt failed on %s: %s", endpoint, e)

    return None, last_error


def fetch_or_load_hospitals(
    force_refresh: bool = False,
    cache_path: Optional[Path | str] = None,
    region_config_path: Optional[Path | str] = None,
    save_to_db: bool = True,
    db_path: Optional[Path | str] = None,
) -> Tuple[List[Hospital], Dict[str, Any], str]:
    """Retrieve hospitals from Overpass API or fallback gracefully to local cache.
    
    Returns:
        (hospitals, statistics, status_message)
    """
    cpath = Path(cache_path) if cache_path else DEFAULT_CACHE_PATH
    region_cfg = load_demo_region(region_config_path)
    lat = region_cfg.get("center_latitude", region_cfg.get("latitude"))
    lon = region_cfg.get("center_longitude", region_cfg.get("longitude"))
    radius = float(region_cfg.get("radius_km", region_cfg.get("demo_radius_km", 25.0)))

    raw_data = None
    source_used = ""

    if force_refresh:
        raw_data, error_msg = fetch_from_overpass(lat, lon, radius)
        if raw_data:
            source_used = "LIVE_OVERPASS_API"
            # Cache the newly fetched data
            cpath.parent.mkdir(parents=True, exist_ok=True)
            with open(cpath, "w", encoding="utf-8") as f:
                json.dump(raw_data, f, indent=2)
        else:
            source_used = f"CACHE_FALLBACK (Overpass error: {error_msg})"
            print(f"[WARN] Live Overpass fetch failed ({error_msg}). Falling back to cached {cpath.name}.")

    if raw_data is None:
        if not cpath.exists():
            # If cache does not exist, try fetching
            raw_data, error_msg = fetch_from_overpass(lat, lon, radius)
            if raw_data:
                source_used = "LIVE_OVERPASS_API"
                cpath.parent.mkdir(parents=True, exist_ok=True)
                with open(cpath, "w", encoding="utf-8") as f:
                    json.dump(raw_data, f, indent=2)
            else:
                raise RuntimeError(
                    f"Overpass API unavailable ({error_msg}) and local cache {cpath} not found. Cannot invent hospitals."
                )
        else:
            if not source_used:
                source_used = "LOCAL_CACHE"
            with open(cpath, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

    # Parse elements
    elements = raw_data.get("elements", [])
    hospitals: List[Hospital] = []
    seen_ids = set()

    for elem in elements:
        hosp = parse_osm_element_to_hospital(elem)
        if hosp and hosp.hospital_id not in seen_ids:
            seen_ids.add(hosp.hospital_id)
            hospitals.append(hosp)

    # Deduplicate hospital records within 150m with similar names
    hospitals, merged_count = deduplicate_hospitals(hospitals, distance_threshold_km=0.150)
    stats = compute_coverage_statistics(hospitals)
    stats["merged_duplicates_count"] = merged_count

    if save_to_db and hospitals:
        init_db(db_path)
        save_hospitals(hospitals, db_path=db_path)

    return hospitals, stats, source_used
