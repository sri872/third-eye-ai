#!/usr/bin/env python3
"""Zero-typing deterministic hospital suggestions runner for Project Third-Eye (Phase 5).

Automated workflow:
1. Loads real OpenStreetMap healthcare facilities (116 hospitals/clinics within 25 km of Phagwara).
2. Orders eligible facilities by source-reported emergency-tag tier, facility category, then a
   secondary straight-line proximity heuristic.
3. Assigns Phase 4 still-image incidents (road_1, road_2, road_3) to simulated demo locations.
4. Formulates caller-keyword needs and orders listed candidates using the deterministic priority order.
5. Surfaces top 3 suggested hospitals with transparent rationale, verified emergency dispatch numbers,
   and strict clinical honesty notes ("bed availability: not determined").
"""

import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.cv.detector import detect
from core.cv.vehicle_report import build_vehicle_reports
from core.ml.accident_analyzer import analyze
from core.nlp.caller_extraction import extract_caller_info
from core.nlp.fusion import fuse_incident_evidence
from core.nlp.occupant_report import build_occupant_reports
from core.hospitals.catalog import HospitalCatalog
from core.hospitals.data import fetch_or_load_hospitals
from core.hospitals.emergency_numbers import get_emergency_numbers
from core.hospitals.ranking import HospitalRanker


def load_demo_locations() -> list:
    locs_path = REPO_ROOT / "config" / "demo_incident_locations.json"
    with open(locs_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    print("==================================================================")
    print("Project Third-Eye | Hospital Suggestions & Emergency Dispatch (Phase 5)")
    print("Ranking Stack    : deterministic source-tag, category and Haversine-distance rules")
    print("Geographic Data  : OpenStreetMap Overpass (25 km radius around Phagwara, Punjab)")
    print("Attribution      : © OpenStreetMap contributors (ODbL 1.0)")
    print("Emergency Stack  : Verified Govt. of India Emergency Services (112, 108, 101, 100)")
    print("UI Principle     : Zero-typing dropdown scenario selection, auto-run")
    print("==================================================================\n")

    # 1. Load real OSM hospital dataset
    print("[1/4] Ingesting OpenStreetMap healthcare facility network...")
    hospitals, stats, source = fetch_or_load_hospitals(save_to_db=True)
    print(f"      Facility Source       : {source}")
    print(f"      Total Facilities      : {stats['total_count']}")
    print(f"      Facilities with Phone : {stats['phone_count']} ({stats['phone_percentage']}%)")
    print(f"      With Emergency Tag    : {stats['emergency_count']} ({stats['emergency_percentage']}%)")
    print(f"      With Specialties Tag  : {stats['specialties_count']} ({stats['specialties_percentage']}%)")

    # 2. Prepare deterministic hospital ranking
    print("\n[2/4] Preparing deterministic hospital ranking...")
    catalog = HospitalCatalog(hospitals)
    ranker = HospitalRanker(catalog)
    print(f"      Candidate records ready: {len(catalog)} (no model-based text search or vector index).")

    # 3. Load demo locations
    print("\n[3/4] Loading simulated demo incident locations from dropdown config...")
    demo_locations = load_demo_locations()
    for idx, loc in enumerate(demo_locations, 1):
        print(f"      Option {idx}: {loc['name']} ({loc['latitude']}, {loc['longitude']}) - [{loc['data_source']}]")

    # 4. Process Phase 4 incidents
    samples_dir = REPO_ROOT / "data" / "samples"
    calls_dir = REPO_ROOT / "data" / "sample_calls"

    test_triplets = [
        ("road_1.jpg", "call_01.txt", demo_locations[0]),
        ("road_2.jpg", "call_02.txt", demo_locations[1]),
        ("road_3.jpg", "call_03.txt", demo_locations[2]),
    ]

    print("\n[4/4] Executing zero-typing end-to-end hospital suggestion pipeline...\n")

    for idx, (img_name, call_name, location) in enumerate(test_triplets, 1):
        img_path = samples_dir / img_name
        call_path = calls_dir / call_name

        if not img_path.exists() or not call_path.exists():
            print(f"[SKIP] Missing asset for {img_name} or {call_name}")
            continue

        # Extract multimodal evidence
        detections = detect(img_path, conf_threshold=0.5)
        vehicle_reports, _ = build_vehicle_reports(img_path, save_to_db=False)
        accident_event = analyze(img_path, location_name=f"Sample: {img_name}", save_to_db=False)
        occupant_reports, occ_summary = build_occupant_reports(detections, vehicle_reports)
        call_text = call_path.read_text(encoding="utf-8")
        caller_ext = extract_caller_info(call_text, call_id=call_path.stem)

        # These demo assets are still photos, while the accident classifier was trained
        # on dashcam video frames. Keep its raw output for audit, but do not promote
        # NO ACCIDENT SIGNAL to a final scene assessment for a still image.
        accident_event_for_fusion = accident_event.model_copy(update={
            "severity": "uncertain",
            "status": "NOT DETERMINED (out-of-domain still image)",
            "verdict": "NOT DETERMINED (out-of-domain still image)",
            "reasons": (
                "Still image is outside the dashcam-video training domain. "
                f"Raw classifier output: {accident_event.verdict}; "
                f"uncalibrated score: {accident_event.score}. "
                "This is not proof of an accident or of normal traffic; human review is required."
            ),
        })
        fused_report = fuse_incident_evidence(
            accident_event=accident_event_for_fusion,
            vehicle_reports=vehicle_reports,
            occupant_reports=occupant_reports,
            caller_extraction=caller_ext,
            image_name=img_name,
            save_to_db=False,
        )

        # Rank hospital suggestions
        hospital_result = ranker.rank_suggestions(
            incident_lat=location["latitude"],
            incident_lon=location["longitude"],
            incident=fused_report,
            final_top_k=3,
        )

        # Emergency dispatch numbers
        emergency_contacts = get_emergency_numbers(
            urgency=fused_report.urgency,
            hazards=fused_report.hazards,
        )

        print("******************************************************************")
        print(f"INCIDENT SCENARIO #{idx}: {img_name} + {call_name}")
        print("******************************************************************")
        print(f"Assigned Location : {location['name']} [{location['data_source']}]")
        print(f"GPS Coordinates   : Latitude {location['latitude']:.4f}, Longitude {location['longitude']:.4f}")
        print(f"Corridor Profile  : {location['environment']} | Traffic: {location['typical_traffic']}")
        print(f"Nearby Landmarks  : {', '.join(location['nearby_landmarks'])}")
        print("------------------------------------------------------------------")
        print(f"Visual Assessment : {fused_report.verdict}")
        print(
            f"Raw Model Output  : {accident_event.verdict} "
            f"(uncalibrated score: {accident_event.score:.4f}; not a reliable still-image decision)"
        )
        print(f"Vehicles Involved : Vision={fused_report.vision_vehicle_count} vs Caller={fused_report.caller_vehicle_count}")
        print(f"People Visible    : Vision={fused_report.vision_people_count} vs Caller={fused_report.caller_people_count}")
        print(f"Identified Hazards: {fused_report.hazards if fused_report.hazards else 'None reported'} (Source: {fused_report.hazards_source})")
        print(f"Reported Injuries : {fused_report.injuries if fused_report.injuries else 'None reported'} (Source: {fused_report.injuries_source})")
        print(f"Triage Urgency    : {fused_report.urgency.upper()} (Source: {fused_report.urgency_source})")
        print("------------------------------------------------------------------")
        print("VERIFIED EMERGENCY DISPATCH CONTACTS (Govt. of India):")
        for contact in emergency_contacts:
            print(f"  * {contact['number']}: {contact['service']} - {contact['official_source']} ({contact['source_url']})")
        print("------------------------------------------------------------------")
        print(f"Assumed Needs (rule-based from caller keywords): \"{hospital_result['assumed_needs']}\"")
        print("TOP 3 SUGGESTED HEALTHCARE FACILITIES:")
        print(f"Ordering Basis : {hospital_result['ranking_method']}")
        for rank, sug in enumerate(hospital_result["suggestions"], 1):
            hosp = sug["hospital"]
            print(f"  [{rank}] {hosp.name}")
            print(f"      Distance        : {sug['distance_km']} km ({sug['distance_label']})")
            print(f"      Phone           : {sug['phone']}")
            print(f"      Emergency Dept  : {sug['emergency']}")
            print(f"      Specialties     : {sug['specialties']}")
            print(f"      Match Rationale : {sug['why_suggested']}")
            print(
                "      Secondary Heuristic : {:.4f} "
                "(proximity: {:.4f}; applied only after emergency-tag and category priority)".format(
                    sug["combined_score"], sug["proximity_score"]
                )
            )
            print(f"      Bed Status      : bed availability: {sug['bed_availability']}")
            print(f"      Attribution     : {sug['attribution']}")
        print()

    print("==================================================================")
    print("OPERATIONAL LIMITATIONS & CLINICAL DISCLAIMER:")
    print("1. Distance calculation uses straight-line Haversine geometry. Real driving")
    print("   distance and transit time will be higher based on road network conditions.")
    print("2. OpenStreetMap does not provide real-time hospital bed availability or")
    print("   live ambulance telemetry. 'bed availability: not determined' is enforced.")
    print("3. When OpenStreetMap specialty tags are sparse, deterministic ranking uses")
    print("   source-tagged emergency status, facility category, and geographic proximity.")
    print("==================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
