#!/usr/bin/env python3
"""Smoke evaluation for deterministic hospital ranking using cached OSM records."""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.rag.facility_classification import dispatch_eligible_facilities
from core.rag.hospital_data import fetch_or_load_hospitals
from core.rag.index import HospitalIndex
from core.rag.retriever import HospitalRetriever

CASES = [
    ("General emergency / trauma", 31.2142, 75.7785),
    ("Pediatric caller-reported need", 31.2185, 75.7852),
    ("Burn caller-reported need", 31.2207, 75.7696),
    ("Highway multi-vehicle incident", 31.2532, 75.7035),
]

def main():
    hospitals, stats, source = fetch_or_load_hospitals(save_to_db=False)
    eligible = dispatch_eligible_facilities(hospitals)
    retriever = HospitalRetriever(HospitalIndex(eligible)) if eligible else None
    print("Third Eye AI | Deterministic Hospital Ranking Smoke Evaluation")
    print(f"Source: {source}")
    print(f"OSM facilities loaded: {len(hospitals)}; dispatch-eligible: {len(eligible)}")
    print("Ranking inputs: facility eligibility, source-reported emergency tag, facility category and straight-line distance.")
    print("No embeddings, vector store, or semantic retrieval are used.")
    if retriever is None:
        print("No eligible facilities are present in the current OSM cache.")
        return
    for label, latitude, longitude in CASES:
        result = retriever.retrieve_suggestions(latitude, longitude, query_override=label, final_top_k=3)
        print("\n" + label)
        for rank, item in enumerate(result["suggestions"], start=1):
            print(f'{rank}. {item["hospital"].name} | {item["distance_km"]} km straight-line | '
                  f'emergency={item["emergency"]} | heuristic={item["combined_score"]}')
    print("\nThis is a ranking sanity check, not clinical validation or hospital-capability verification.")

if __name__ == "__main__":
    main()
