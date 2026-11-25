# Phase 10 Final Acceptance Report — Review Fixes, System Card, Launcher, Dependencies, Demo, Final Gate

> **Historical report (superseded 2026-11-25):** This report records the earlier FAISS-era CLI output and test names. FAISS and SentenceTransformers have since been removed; current hospital ranking is deterministic in `core/hospitals/`. Historical evidence below does not describe the current branch. See [SYSTEM_CARD.md](SYSTEM_CARD.md).

**Project**: Project Third-Eye (`third-eye-v2`)  
**Date**: 2026-11-24  
**Environment**: Apple Silicon macOS, Python 3.12.3, local `.venv`  
**Status**: COMPLETE — all six Phase 9/10 tasks executed and evidenced below.

All command outputs in this report are **verbatim raw output** captured from a normal Terminal shell (outside any sandbox). Nothing has been reconstructed, tidied or retyped from memory. No file under `app/` was edited. No file was deleted.

---

## 0. Disk space

Command: `df -h ~`

```text
Filesystem      Size    Used   Avail Capacity iused ifree %iused  Mounted on
/dev/disk3s5   228Gi   159Gi    40Gi    80%    1.5M  423M    0%   /System/Volumes/Data
```

---

## 1. Task 0 — Dispatch seconds/minutes inconsistency

### Root cause (real observation, not the quoted example)

The user's quoted `234.0` is **not** present in the current repository. The real affected case was the 4th scenario, `loc-phagwara-nh44-sugarmill`: before the fix the JSON had `total_seconds: 231.6`, `ambulance.eta_seconds: 105.0`, `hospital.transit_seconds: 126.6`, while `hospital_alternatives` for the same `Parhar Hospital` said `transit_seconds: 126.8`.

Cause: `plan_emergency_dispatch()` stored the three minute fields rounded to 2 decimals (`round(x / 60.0, 2)`), whereas `chosen_ambulance.eta_seconds`, `chosen_hospital.transit_seconds` and the alternatives table store authoritative seconds rounded to 1 decimal. `export_dispatch_summary()` then recomputed seconds as `DB minutes × 60`, producing a round-trip drift. This also made `total_seconds` disagree with `eta_seconds + transit_seconds`.

### Files edited (only these two)

- `core/routing/dispatch.py`
  - Persist `ambulance_eta_minutes = ambulance_eta_sec / 60.0`, `hospital_transit_minutes = hospital_transit_sec / 60.0`, and `estimated_time_minutes = (round(eta,1) + round(transit,1)) / 60.0` (full-precision minutes derived from the authoritative seconds), instead of pre-rounded 2-decimal minutes.
  - Export all three JSON second fields from one pair of authoritative leg seconds: `eta_sec = round(row["ambulance_eta_minutes"] * 60.0, 1)`, `transit_sec = round(row["hospital_transit_minutes"] * 60.0, 1)`, `total_sec = round(eta_sec + transit_sec, 1)`.
- `scripts/run_dispatch.py`
  - Terminal output and the in-memory summary now print the same authoritative seconds from `chosen_ambulance.eta_seconds` / `chosen_hospital.transit_seconds` instead of multiplying rounded minute fields.

The JSON **structure was not changed**: same keys, same nesting, same scenario count. No test file was edited.

### Rerun and JSON-vs-database verification

Command: `.venv/bin/python scripts/run_dispatch.py > /tmp/phase10_dispatch_rerun.txt 2>&1; echo "exit=$?"; cat /tmp/phase10_dispatch_rerun.txt`

```text
exit=0
==============================================================================
PROJECT THIRD-EYE: MULTIMODAL EMERGENCY DISPATCH & ROUTING ENGINE
==============================================================================
[INIT] Loaded 6 simulated ambulances into SQLite.
[GRAPH] Loading road network graph...
  Nodes: 3247, Edges: 8582, Total Road Length: 1466.63 km
  Attribution: © OpenStreetMap contributors
  Disclaimer: Estimated from OSM speed data, no live traffic
  Coverage: Routing covers this corridor only (Phagwara corridor: 75.69-75.80°E, 31.20-31.27°N)

------------------------------------------------------------------------------
SCENARIO 1/4: NH-44 GT Road near Jagatjit Sugar Mill
Location ID: loc-phagwara-nh44-sugarmill | GPS: (31.2142, 75.7785) [SIMULATED]
Environment: Highway corridor
Severity: not determined (demo placement)
Candidate Selection Rule: All valid corridor hospitals (non-medical places, outpatient clinics, pharmacies excluded; corridor bounds: 31.20-31.27°N, 75.69-75.80°E)

  Available Ambulances Evaluation (all 6 fleet units):
    Callsign         | Base Hospital                    | ETA (min / sec)  | Dist     | Chosen
    ----------------------------------------------------------------------------------------
    AMB-FOXTROT-06   | Anoop Hospital And Heart Care Ce |  1.8m (  105s)   |  0.57km | YES (FASTEST)
    AMB-DELTA-04     | Gee Bee Hospital and Maternity H |  2.0m (  122s)   |  0.71km | No
    AMB-CHARLIE-03   | Parhar Hospital                  |  2.1m (  127s)   |  0.76km | No
    AMB-BRAVO-02     | Gandhi Hospital (P) Ltd          |  2.2m (  135s)   |  0.79km | No
    AMB-ECHO-05      | Sharma Children Hospital         |  2.6m (  158s)   |  0.98km | No
    AMB-ALPHA-01     | Civil Hospital Phagwara          |  5.9m (  356s)   |  3.27km | No

  Candidate Hospitals Evaluation (all 8 corridor facilities):
    Hospital Name                | Driving  | Time (min/s)   | Penalty | Adj Cost  | Considered | Decision Note
    -------------------------------------------------------------------------------------------------------------------
    Parhar Hospital              |  0.76km  |  2.1m ( 127s) |  0.00   |  126.8s   | YES        | Chosen: lowest adjusted travel time (126.8s) among trauma-suited corridor facilities
    Gandhi Hospital (P) Ltd      |  0.79km  |  2.2m ( 135s) |  0.00   |  134.8s   | YES        | Not chosen: driving time (134.8s) is longer than chosen hospital (126.8s)
    Mitra Eye Hospital           |  0.47km  |  1.5m (  93s) |  0.30   |  148.1s   | YES        | Not chosen: down-ranked by suitability rule (Down-ranked by suitability rule: facility specialized in eye, but incident has no reported eye requirement.)
    Anoop Hospital And Heart Car |  0.57km  |  1.8m ( 105s) |  0.30   |  168.1s   | YES        | Not chosen: down-ranked by suitability rule (Down-ranked by suitability rule: facility specialized in heart/cardiac, but incident has no reported heart/cardiac requirement.)
    Gee Bee Hospital and Materni |  0.71km  |  2.0m ( 122s) |  0.30   |  194.8s   | YES        | Not chosen: down-ranked by suitability rule (Down-ranked by suitability rule: facility specialized in maternity, but incident has no reported maternity requirement.)
    Sharma Children Hospital     |  0.98km  |  2.6m ( 158s) |  0.30   |  252.0s   | YES        | Not chosen: down-ranked by suitability rule (Down-ranked by suitability rule: facility specialized in children, but incident has no reported children requirement.)
    Virk Hospital And Maternity  |  1.04km  |  2.8m ( 166s) |  0.30   |  264.9s   | YES        | Not chosen: down-ranked by suitability rule (Down-ranked by suitability rule: facility specialized in maternity, but incident has no reported maternity requirement.)
    Civil Hospital Phagwara      |  3.46km  |  6.3m ( 378s) |  0.00   |  378.5s   | YES        | Not chosen: driving time (378.5s) is longer than chosen hospital (126.8s)

  [DISPATCH] Ambulance: amb-phagwara-06 (AMB-FOXTROT-06) [SIMULATED]
             Station Base: Anoop Hospital And Heart Care Centre
             Ambulance ETA: 1.75 min (105.0 sec)
  [HOSPITAL] Destination: osm-node-7105996895 (Parhar Hospital) [REAL OSM]
             Facility type from name/tag: hospital
             Emergency tag in OpenStreetMap: not specified
             Specialties in OpenStreetMap: not specified in OSM
             Ranking Heuristic (not a clinical rating): travel time with penalty weighting for unsuited single-specialties
             Formula: adjusted_cost_sec = travel_time_sec * (1.0 + (suitability_penalty * 2.0))
             Adjusted Cost: 126.8 s
             Hospital Transit: 2.11 min (126.8 sec)
             Driving Distance: 0.76 km
             Straight-line Distance: 0.27 km
             Driving / Straight-Line Ratio: 2.82x
  [MISSION]  Total Driving: 1.33 km, Estimated Total Time: 3.86 min
  [COVERAGE] Routing covers this corridor only (Phagwara corridor: 75.69-75.80°E, 31.20-31.27°N)
  [SIMULATION] Trajectory: 9 waypoint steps generated at 15s intervals.
  [ARTIFACTS] Map: loc-phagwara-nh44-sugarmill_route.png | GeoJSON: loc-phagwara-nh44-sugarmill_route.geojson
  [DATABASE] Stored route record: route-loc-phagwara-nh44-sugarmill
------------------------------------------------------------------------------
SCENARIO 2/4: Banga Road Commercial Intersection
Location ID: loc-phagwara-banga-road | GPS: (31.2185, 75.7852) [SIMULATED]
Environment: Suburban arterial intersection
Severity: not determined (demo placement)
Candidate Selection Rule: All valid corridor hospitals (non-medical places, outpatient clinics, pharmacies excluded; corridor bounds: 31.20-31.27°N, 75.69-75.80°E)

  Available Ambulances Evaluation (all 6 fleet units):
    Callsign         | Base Hospital                    | ETA (min / sec)  | Dist     | Chosen
    ----------------------------------------------------------------------------------------
    AMB-DELTA-04     | Gee Bee Hospital and Maternity H |  9.1m (  546s)   |  3.98km | YES (FASTEST)
    AMB-ALPHA-01     | Civil Hospital Phagwara          |  9.1m (  547s)   |  4.25km | No
    AMB-ECHO-05      | Sharma Children Hospital         | 10.0m (  602s)   |  4.40km | No
    AMB-BRAVO-02     | Gandhi Hospital (P) Ltd          | 10.6m (  635s)   |  4.63km | No
    AMB-FOXTROT-06   | Anoop Hospital And Heart Care Ce | 10.7m (  640s)   |  4.74km | No
    AMB-CHARLIE-03   | Parhar Hospital                  | 11.0m (  657s)   |  4.78km | No

  Candidate Hospitals Evaluation (all 8 corridor facilities):
    Hospital Name                | Driving  | Time (min/s)   | Penalty | Adj Cost  | Considered | Decision Note
    -------------------------------------------------------------------------------------------------------------------
    Civil Hospital Phagwara      |  4.25km  |  9.1m ( 547s) |  0.00   |  546.8s   | YES        | Chosen: lowest adjusted travel time (546.8s) among trauma-suited corridor facilities
    Gandhi Hospital (P) Ltd      |  4.63km  | 10.6m ( 635s) |  0.00   |  635.4s   | YES        | Not chosen: driving time (635.4s) is longer than chosen hospital (546.8s)
    Parhar Hospital              |  5.58km  | 10.7m ( 640s) |  0.00   |  639.8s   | YES        | Not chosen: driving time (639.8s) is longer than chosen hospital (546.8s)
    Gee Bee Hospital and Materni |  3.98km  |  9.1m ( 546s) |  0.30   |  872.8s   | YES        | Not chosen: down-ranked by suitability rule (Down-ranked by suitability rule: facility specialized in maternity, but incident has no reported maternity requirement.)
    Sharma Children Hospital     |  4.40km  | 10.0m ( 602s) |  0.30   |  963.1s   | YES        | Not chosen: down-ranked by suitability rule (Down-ranked by suitability rule: facility specialized in children, but incident has no reported children requirement.)
    Mitra Eye Hospital           |  4.64km  | 10.5m ( 628s) |  0.30   | 1004.3s   | YES        | Not chosen: down-ranked by suitability rule (Down-ranked by suitability rule: facility specialized in eye, but incident has no reported eye requirement.)
    Anoop Hospital And Heart Car |  4.74km  | 10.7m ( 640s) |  0.30   | 1024.2s   | YES        | Not chosen: down-ranked by suitability rule (Down-ranked by suitability rule: facility specialized in heart/cardiac, but incident has no reported heart/cardiac requirement.)
    Virk Hospital And Maternity  |  5.68km  | 10.8m ( 651s) |  0.30   | 1041.6s   | YES        | Not chosen: down-ranked by suitability rule (Down-ranked by suitability rule: facility specialized in maternity, but incident has no reported maternity requirement.)

  [DISPATCH] Ambulance: amb-phagwara-04 (AMB-DELTA-04) [SIMULATED]
             Station Base: Gee Bee Hospital and Maternity Home
             Ambulance ETA: 9.09 min (545.5 sec)
  [HOSPITAL] Destination: osm-node-7150766771 (Civil Hospital Phagwara) [REAL OSM]
             Facility type from name/tag: hospital
             Emergency tag in OpenStreetMap: not specified
             Specialties in OpenStreetMap: not specified in OSM
             Ranking Heuristic (not a clinical rating): travel time with penalty weighting for unsuited single-specialties
             Formula: adjusted_cost_sec = travel_time_sec * (1.0 + (suitability_penalty * 2.0))
             Adjusted Cost: 546.8 s
             Hospital Transit: 9.11 min (546.8 sec)
             Driving Distance: 4.25 km
             Straight-line Distance: 2.51 km
             Driving / Straight-Line Ratio: 1.69x
  [MISSION]  Total Driving: 8.23 km, Estimated Total Time: 18.20 min
  [COVERAGE] Routing covers this corridor only (Phagwara corridor: 75.69-75.80°E, 31.20-31.27°N)
  [SIMULATION] Trajectory: 38 waypoint steps generated at 15s intervals.
  [ARTIFACTS] Map: loc-phagwara-banga-road_route.png | GeoJSON: loc-phagwara-banga-road_route.geojson
  [DATABASE] Stored route record: route-loc-phagwara-banga-road
------------------------------------------------------------------------------
SCENARIO 3/4: GT Road near Lovely Professional University (Chaheru)
Location ID: loc-phagwara-lpu-chaheru | GPS: (31.2532, 75.7035) [SIMULATED]
Environment: University campus highway corridor
Severity: not determined (demo placement)
Candidate Selection Rule: All valid corridor hospitals (non-medical places, outpatient clinics, pharmacies excluded; corridor bounds: 31.20-31.27°N, 75.69-75.80°E)

  Available Ambulances Evaluation (all 6 fleet units):
    Callsign         | Base Hospital                    | ETA (min / sec)  | Dist     | Chosen
    ----------------------------------------------------------------------------------------
    AMB-ALPHA-01     | Civil Hospital Phagwara          | 11.6m (  696s)   |  8.43km | YES (FASTEST)
    AMB-CHARLIE-03   | Parhar Hospital                  | 12.8m (  771s)   | 10.58km | No
    AMB-DELTA-04     | Gee Bee Hospital and Maternity H | 13.1m (  786s)   | 10.91km | No
    AMB-ECHO-05      | Sharma Children Hospital         | 13.2m (  791s)   | 10.84km | No
    AMB-FOXTROT-06   | Anoop Hospital And Heart Care Ce | 13.2m (  792s)   | 10.76km | No
    AMB-BRAVO-02     | Gandhi Hospital (P) Ltd          | 13.3m (  801s)   | 10.79km | No

  Candidate Hospitals Evaluation (all 8 corridor facilities):
    Hospital Name                | Driving  | Time (min/s)   | Penalty | Adj Cost  | Considered | Decision Note
    -------------------------------------------------------------------------------------------------------------------
    Civil Hospital Phagwara      |  8.45km  | 11.6m ( 698s) |  0.00   |  698.4s   | YES        | Chosen: lowest adjusted travel time (698.4s) among trauma-suited corridor facilities
    Parhar Hospital              | 10.61km  | 12.9m ( 777s) |  0.00   |  776.8s   | YES        | Not chosen: driving time (776.8s) is longer than chosen hospital (698.4s)
    Gandhi Hospital (P) Ltd      | 10.82km  | 13.4m ( 807s) |  0.00   |  807.0s   | YES        | Not chosen: driving time (807.0s) is longer than chosen hospital (698.4s)
    Gee Bee Hospital and Materni | 10.54km  | 12.9m ( 772s) |  0.30   | 1236.0s   | YES        | Not chosen: down-ranked by suitability rule (Down-ranked by suitability rule: facility specialized in maternity, but incident has no reported maternity requirement.)
    Sharma Children Hospital     | 10.47km  | 13.0m ( 777s) |  0.30   | 1243.9s   | YES        | Not chosen: down-ranked by suitability rule (Down-ranked by suitability rule: facility specialized in children, but incident has no reported children requirement.)
    Virk Hospital And Maternity  | 10.71km  | 13.1m ( 788s) |  0.30   | 1260.7s   | YES        | Not chosen: down-ranked by suitability rule (Down-ranked by suitability rule: facility specialized in maternity, but incident has no reported maternity requirement.)
    Anoop Hospital And Heart Car | 10.79km  | 13.3m ( 798s) |  0.30   | 1277.7s   | YES        | Not chosen: down-ranked by suitability rule (Down-ranked by suitability rule: facility specialized in heart/cardiac, but incident has no reported heart/cardiac requirement.)
    Mitra Eye Hospital           | 10.71km  | 13.4m ( 803s) |  0.30   | 1285.1s   | YES        | Not chosen: down-ranked by suitability rule (Down-ranked by suitability rule: facility specialized in eye, but incident has no reported eye requirement.)

  [DISPATCH] Ambulance: amb-phagwara-01 (AMB-ALPHA-01) [SIMULATED]
             Station Base: Civil Hospital Phagwara
             Ambulance ETA: 11.61 min (696.5 sec)
  [HOSPITAL] Destination: osm-node-7150766771 (Civil Hospital Phagwara) [REAL OSM]
             Facility type from name/tag: hospital
             Emergency tag in OpenStreetMap: not specified
             Specialties in OpenStreetMap: not specified in OSM
             Ranking Heuristic (not a clinical rating): travel time with penalty weighting for unsuited single-specialties
             Formula: adjusted_cost_sec = travel_time_sec * (1.0 + (suitability_penalty * 2.0))
             Adjusted Cost: 698.4 s
             Hospital Transit: 11.64 min (698.4 sec)
             Driving Distance: 8.45 km
             Straight-line Distance: 6.18 km
             Driving / Straight-Line Ratio: 1.37x
  [MISSION]  Total Driving: 16.88 km, Estimated Total Time: 23.25 min
  [COVERAGE] Routing covers this corridor only (Phagwara corridor: 75.69-75.80°E, 31.20-31.27°N)
  [SIMULATION] Trajectory: 48 waypoint steps generated at 15s intervals.
  [ARTIFACTS] Map: loc-phagwara-lpu-chaheru_route.png | GeoJSON: loc-phagwara-lpu-chaheru_route.geojson
  [DATABASE] Stored route record: route-loc-phagwara-lpu-chaheru
------------------------------------------------------------------------------
SCENARIO 4/4: Phagwara-Hoshiarpur State Highway (Rural Link)
Location ID: loc-phagwara-hoshiarpur-road | GPS: (31.2455, 75.7802) [SIMULATED]
Environment: Semi-rural two-lane connector
Severity: not determined (demo placement)
Candidate Selection Rule: All valid corridor hospitals (non-medical places, outpatient clinics, pharmacies excluded; corridor bounds: 31.20-31.27°N, 75.69-75.80°E)

  Available Ambulances Evaluation (all 6 fleet units):
    Callsign         | Base Hospital                    | ETA (min / sec)  | Dist     | Chosen
    ----------------------------------------------------------------------------------------
    AMB-ALPHA-01     | Civil Hospital Phagwara          |  6.4m (  382s)   |  3.60km | YES (FASTEST)
    AMB-DELTA-04     | Gee Bee Hospital and Maternity H |  8.1m (  484s)   |  3.60km | No
    AMB-ECHO-05      | Sharma Children Hospital         |  9.0m (  540s)   |  4.01km | No
    AMB-BRAVO-02     | Gandhi Hospital (P) Ltd          |  9.6m (  573s)   |  4.24km | No
    AMB-FOXTROT-06   | Anoop Hospital And Heart Care Ce |  9.6m (  578s)   |  4.36km | No
    AMB-CHARLIE-03   | Parhar Hospital                  |  9.9m (  595s)   |  4.40km | No

  Candidate Hospitals Evaluation (all 8 corridor facilities):
    Hospital Name                | Driving  | Time (min/s)   | Penalty | Adj Cost  | Considered | Decision Note
    -------------------------------------------------------------------------------------------------------------------
    Civil Hospital Phagwara      |  3.60km  |  6.4m ( 382s) |  0.00   |  382.3s   | YES        | Chosen: lowest adjusted travel time (382.3s) among trauma-suited corridor facilities
    Parhar Hospital              |  4.93km  |  7.9m ( 475s) |  0.00   |  475.3s   | YES        | Not chosen: driving time (475.3s) is longer than chosen hospital (382.3s)
    Gandhi Hospital (P) Ltd      |  4.83km  |  8.4m ( 506s) |  0.00   |  505.5s   | YES        | Not chosen: driving time (505.5s) is longer than chosen hospital (382.3s)
    Gee Bee Hospital and Materni |  4.67km  |  7.8m ( 467s) |  0.30   |  747.4s   | YES        | Not chosen: down-ranked by suitability rule (Down-ranked by suitability rule: facility specialized in maternity, but incident has no reported maternity requirement.)
    Sharma Children Hospital     |  4.60km  |  7.9m ( 472s) |  0.30   |  755.3s   | YES        | Not chosen: down-ranked by suitability rule (Down-ranked by suitability rule: facility specialized in children, but incident has no reported children requirement.)
    Virk Hospital And Maternity  |  5.03km  |  8.1m ( 486s) |  0.30   |  778.4s   | YES        | Not chosen: down-ranked by suitability rule (Down-ranked by suitability rule: facility specialized in maternity, but incident has no reported maternity requirement.)
    Anoop Hospital And Heart Car |  5.11km  |  8.3m ( 497s) |  0.30   |  795.4s   | YES        | Not chosen: down-ranked by suitability rule (Down-ranked by suitability rule: facility specialized in heart/cardiac, but incident has no reported heart/cardiac requirement.)
    Mitra Eye Hospital           |  4.84km  |  8.3m ( 498s) |  0.30   |  796.5s   | YES        | Not chosen: down-ranked by suitability rule (Down-ranked by suitability rule: facility specialized in eye, but incident has no reported eye requirement.)

  [DISPATCH] Ambulance: amb-phagwara-01 (AMB-ALPHA-01) [SIMULATED]
             Station Base: Civil Hospital Phagwara
             Ambulance ETA: 6.37 min (382.2 sec)
  [HOSPITAL] Destination: osm-node-7150766771 (Civil Hospital Phagwara) [REAL OSM]
             Facility type from name/tag: hospital
             Emergency tag in OpenStreetMap: not specified
             Specialties in OpenStreetMap: not specified in OSM
             Ranking Heuristic (not a clinical rating): travel time with penalty weighting for unsuited single-specialties
             Formula: adjusted_cost_sec = travel_time_sec * (1.0 + (suitability_penalty * 2.0))
             Adjusted Cost: 382.3 s
             Hospital Transit: 6.37 min (382.3 sec)
             Driving Distance: 3.60 km
             Straight-line Distance: 2.35 km
             Driving / Straight-Line Ratio: 1.54x
  [MISSION]  Total Driving: 7.20 km, Estimated Total Time: 12.74 min
  [COVERAGE] Routing covers this corridor only (Phagwara corridor: 75.69-75.80°E, 31.20-31.27°N)
  [SIMULATION] Trajectory: 27 waypoint steps generated at 15s intervals.
  [ARTIFACTS] Map: loc-phagwara-hoshiarpur-road_route.png | GeoJSON: loc-phagwara-hoshiarpur-road_route.geojson
  [DATABASE] Stored route record: route-loc-phagwara-hoshiarpur-road

==============================================================================
DISPATCH SUMMARY TABLE
==============================================================================
No  | Incident Location              | Ambulance       | ETA     | Target Hospital           | Total
---------------------------------------------------------------------------------------------------------
1   | NH-44 GT Road near Jagatjit Su | AMB-FOXTROT-06  | 1.8  m | Parhar Hospital           | 3.9  m
2   | Banga Road Commercial Intersec | AMB-DELTA-04    | 9.1  m | Civil Hospital Phagwara   | 18.2 m
3   | GT Road near Lovely Profession | AMB-ALPHA-01    | 11.6 m | Civil Hospital Phagwara   | 23.2 m
4   | Phagwara-Hoshiarpur State High | AMB-ALPHA-01    | 6.4  m | Civil Hospital Phagwara   | 12.7 m
---------------------------------------------------------------------------------------------------------
Attribution: © OpenStreetMap contributors
Disclaimer: Estimated from OSM speed data, no live traffic
Coverage: Routing covers this corridor only (Phagwara corridor: 75.69-75.80°E, 31.20-31.27°N)
Database Persistence Verified: 4 total records in route_results table.
Exported dispatch summary JSON: data/outputs/dispatch_summary.json (4 scenarios)
```

JSON-vs-database check (read-only Python using `db.database.get_route_results`):

```text
loc-phagwara-banga-road: eta=545.5, transit=546.8, alt_transit=546.8, total=1092.3, sum=1092.3
  DB minutes*60: eta=545.5, transit=546.8, total=1092.3
loc-phagwara-hoshiarpur-road: eta=382.2, transit=382.3, alt_transit=382.3, total=764.5, sum=764.5
  DB minutes*60: eta=382.2, transit=382.3, total=764.5
loc-phagwara-lpu-chaheru: eta=696.5, transit=698.4, alt_transit=698.4, total=1394.9, sum=1394.9
  DB minutes*60: eta=696.5, transit=698.4, total=1394.9
loc-phagwara-nh44-sugarmill: eta=105.0, transit=126.8, alt_transit=126.8, total=231.8, sum=231.8
  DB minutes*60: eta=105.0, transit=126.8, total=231.8
```

All four scenarios now satisfy: `eta_seconds + transit_seconds == total_seconds`, `chosen_hospital.transit_seconds == its hospital_alternatives entry`, and `DB minute columns × 60 == JSON seconds`. `tests/test_routing.py::TestPhase6dDispatchSummaryAndAlternatives::test_dispatch_summary_json_matches_database` passes (see §6).

---

## 2. Task 1 — Honest `docs/SYSTEM_CARD.md`

Created `docs/SYSTEM_CARD.md`. It documents capabilities, the REAL/SIMULATED boundary, classifier metrics, OSM tag sparsity, and seven explicit limitations. No classifier number was recalled from memory: the values were read from `models/accident_classifier_metrics.json` with this command:

```bash
.venv/bin/python - <<'PY'
import json
m = json.load(open('models/accident_classifier_metrics.json'))
print('accuracy', m['test_accuracy'], 'roc_auc', m['test_roc_auc'], 'test_videos', m['splits']['test_videos'])
print('75 in test ids:', 75 in m['test_video_ids'], '1023 in test ids:', 1023 in m['test_video_ids'])
for frame in m['test_frames']:
    if frame['video_id'] in ('75', '1023'):
        print(frame)
PY
```

Raw output:

```text
accuracy 0.6778 roc_auc 0.7452 test_videos 60
75 in test ids: True 1023 in test ids: True
{'frame_id': '00075_pos60_173', 'video_id': '75', 'label': 1, 'prob': 0.5965451002120972}
{'frame_id': '00075_pos75_216', 'video_id': '75', 'label': 1, 'prob': 0.6754897236824036}
{'frame_id': '00075_pos90_259', 'video_id': '75', 'label': 1, 'prob': 0.8242461085319519}
{'frame_id': '01023_pos60_170', 'video_id': '1023', 'label': 1, 'prob': 0.196553573012352}
{'frame_id': '01023_pos75_212', 'video_id': '1023', 'label': 1, 'prob': 0.252641499042511}
{'frame_id': '01023_pos90_255', 'video_id': '1023', 'label': 1, 'prob': 0.31014132499694824}
```

Full metric values copied from that file into the card: accuracy 0.6778, precision 0.66, recall 0.7333, F1 0.6947, ROC-AUC 0.7452, `best_C` 0.01, baseline accuracy 0.5, confusion matrix `[[56, 34], [24, 66]]`, splits 280/60/60 videos and 840/180/180 frames.

---

## 3. Task 2 — Launcher verification and README quick start

`launch.command` is unchanged and contains `exec streamlit run app/main.py` (with venv activation). Double-click-style run:

```text
$ nohup bash launch.command > /tmp/phase10_launch.log 2>&1 < /dev/null & echo $!
3620
$ curl --retry 15 --retry-delay 1 --retry-connrefused -sS -w "\nHTTP %{http_code}\n" http://localhost:8501/_stcore/health
ok
HTTP 200
$ ps -p 3620 -o pid=,command=
 3620 /Library/Frameworks/Python.framework/Versions/3.12/Resources/Python.app/Contents/MacOS/Python /Users/srishanthbiradar/Projects/third-eye-v2/.venv/bin/streamlit run app/main.py
$ kill 3620; ps -p 3620 -o pid=,command= || true; lsof -nP -iTCP:8501 -sTCP:LISTEN || true
 3620 /Library/Frameworks/Python.framework/Versions/3.12/Resources/Python.app/Contents/MacOS/Python /Users/srishanthbiradar/Projects/third-eye-v2/.venv/bin/streamlit run app/main.py
COMMAND  PID             USER   FD   TYPE             DEVICE SIZE/OFF NODE NAME
Python  3620 srishanthbiradar    6u  IPv6 0x30b6536d7d0df6fc      0t0  TCP *:8501 (LISTEN)
$ ps -p 3620 -o pid=,command= || true; lsof -nP -iTCP:8501 -sTCP:LISTEN || true; cat /tmp/phase10_launch.log
Activating Python virtual environment (.venv)...
Launching Project Third-Eye Streamlit Interface...
2026-11-24 16:32:09.801 Uvicorn server started on :::8501

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://10.35.232.48:8501

  Help agents write better Streamlit apps?
  Install the official Streamlit skills by running streamlit skills in your terminal.

  For better performance, install the Watchdog module:

  $ xcode-select --install
  $ pip install watchdog
            
  Stopping...
```

Only the process started by this verification (PID 3620) was signalled. A subsequent `ps` and `lsof` returned no output, confirming the port was released and no other Streamlit process was touched.

`README.md` gained this 10-line quick start (verbatim):

```text
### Quick start (10 lines)
```bash
cd ~/Projects/third-eye-v2
source .venv/bin/activate
.venv/bin/pytest --cache-clear -v    # run the test suite
bash launch.command                  # double-click equivalent; starts Streamlit
# open http://localhost:8501 in your browser
# sidebar: upload ONE image or video (e.g. data/samples/road_1.jpg)
# Scene tab: detector boxes, vehicle/person counts
# Accident check tab: trained-model score + per-frame scores for video
# Dispatch tab: 4 scenarios; source is data/outputs/dispatch_summary.json
# Stop with Ctrl-C; outputs: data/outputs/ (JSON, crops, routes/ PNG+GeoJSON)
```
```

---

## 4. Task 3 — Dependency check

### `pip check` (verbatim)

```text
$ .venv/bin/pip check
No broken requirements found.
```

### `pip freeze` (117 lines, verbatim)

Command: `.venv/bin/pip freeze > /tmp/phase10_pip_freeze.txt; wc -l /tmp/phase10_pip_freeze.txt; comm -23 <(sort -f /tmp/phase10_pip_freeze.txt) <(sort -f requirements.txt)`

```text
     117 /tmp/phase10_pip_freeze.txt
altair==6.3.0
annotated-doc==0.0.7
annotated-types==0.8.0
anyio==4.16.0
attrs==26.1.0
blis==1.3.4
branca==0.8.2
catalogue==2.0.10
certifi==2026.11.20
charset-normalizer==3.5.3
click==8.5.1
cloudpathlib==0.26.0
cloudpickle==3.1.3
confection==1.3.4
contourpy==1.4.0
cycler==0.12.1
cymem==2.0.16
easyocr==1.7.2
en_core_web_sm @ file:///tmp/en_core_web_sm-3.8.0-py3-none-any.whl#sha256=1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85
faiss-cpu==1.16.0
filelock==4.3.1
folium==1.0.0
fonttools==4.68.0
fsspec==2026.9.0
geopandas==1.2.0
h11==0.16.0
hf-xet==1.8.0
httpcore==1.0.9
httpcore2==2.16.2
httpx==0.28.1
httpx2==2.16.2
huggingface_hub==2.4.0
idna==3.20
ImageIO==2.38.2
iniconfig==2.3.1
itsdangerous==2.2.0
Jinja2==3.1.6
joblib==1.6.0
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
kiwisolver==1.5.1
lazy-loader==0.6
markdown-it-py==4.3.0
MarkupSafe==3.0.4
matplotlib==3.11.2
mdurl==0.1.2
mpmath==1.3.0
murmurhash==1.0.16
narwhals==2.29.0
networkx==3.7
ninja==1.13.2
numpy==2.5.5
opencv-python==5.1.0.95
opencv-python-headless==5.1.0.95
osmnx==2.2.0
packaging==26.4
pandas==3.1.1
pillow==13.0.0
pluggy==1.7.0
preshed==3.0.15
protobuf==7.36.2
pyarrow==26.0.1
pyclipper==1.4.0
pydantic==2.14.4
pydantic_core==2.50.5
pydeck==0.9.3
Pygments==2.21.0
pyogrio==0.13.0
pyparsing==3.3.3
pyproj==3.8.0
pytest==9.2.0
python-bidi==0.6.12
python-dateutil==4.0.0
python-multipart==0.0.34
PyYAML==6.0.4
referencing==0.37.0
regex==2026.11.23
requests==2.34.2
rich==15.0.0
rpds-py==2026.9.1
safetensors==0.8.0
scikit-image==0.26.0
scikit-learn==1.9.1
scipy==1.18.1
sentence-transformers==6.3.0
setuptools==84.2.1
shapely==2.2.0
shellingham==1.5.4
smart_open==10.0.0
spacy==3.8.18
spacy-legacy==3.0.12
spacy-loggers==1.0.5
srsly==2.5.6
starlette==1.10.0
streamlit==1.67.0
streamlit-folium==0.28.0
sympy==1.14.0
thinc==8.3.14
threadpoolctl==3.8.0
tifffile==2026.11.16
tokenizers==0.23.3
torch==2.15.0
torchvision==0.29.2
tqdm==4.70.1
transformers==5.24.0
truststore==0.10.5
typer==0.29.2
typing-inspection==0.4.5
typing_extensions==4.16.0
tzdata==2026.6
urllib3==2.8.0
uvicorn==0.54.0
wasabi==1.1.3
weasel==1.0.1
websockets==17.4
wrapt==2.5.2
xyzservices==2026.9.1
```

Most of the remaining freeze lines are transitive dependencies. Directly imported packages missing from `requirements.txt` were identified with `grep -RhoE '^(import|from) [A-Za-z0-9_]+' app core db scripts tests | sort -u`; only `joblib` and `pandas` were both directly imported and unpinned. All example pins named in the task (folium, streamlit-folium, branca, xyzservices, easyocr, spacy, the spaCy model wheel, sentence-transformers, faiss-cpu, osmnx, geopandas) were **already present** and were not removed or changed.

### Diff after adding missing pins (verbatim)

```text
$ diff -u /tmp/phase10_requirements_before.txt requirements.txt
--- /tmp/phase10_requirements_before.txt	2026-11-24 16:31:17
+++ requirements.txt	2026-11-24 16:31:28
@@ -1,6 +1,8 @@
 pytest==9.2.0
 pydantic==2.14.4
 numpy==2.5.5
+joblib==1.6.0
+pandas==3.1.1
 streamlit==1.67.0
 torch==2.15.0
 torchvision==0.29.2
```

Exit code 1 from `diff` simply means the files differ.

### Pin-vs-installed verification (verbatim)

```text
pytest: pinned=9.2.0 installed=9.2.0 OK
pydantic: pinned=2.14.4 installed=2.14.4 OK
numpy: pinned=2.5.5 installed=2.5.5 OK
joblib: pinned=1.6.0 installed=1.6.0 OK
pandas: pinned=3.1.1 installed=3.1.1 OK
streamlit: pinned=1.67.0 installed=1.67.0 OK
torch: pinned=2.15.0 installed=2.15.0 OK
torchvision: pinned=0.29.2 installed=0.29.2 OK
opencv-python: pinned=5.1.0.95 installed=5.1.0.95 OK
pillow: pinned=13.0.0 installed=13.0.0 OK
easyocr: pinned=1.7.2 installed=1.7.2 OK
scikit-learn: pinned=1.9.1 installed=1.9.1 OK
spacy: pinned=3.8.18 installed=3.8.18 OK
model wheel line present: https://github.com/explosion/spacy-models/releases/download/en_core_web_...
sentence-transformers: pinned=6.3.0 installed=6.3.0 OK
faiss-cpu: pinned=1.16.0 installed=1.16.0 OK
osmnx: pinned=2.2.0 installed=2.2.0 OK
networkx: pinned=3.7 installed=3.7 OK
matplotlib: pinned=3.11.2 installed=3.11.2 OK
geopandas: pinned=1.2.0 installed=1.2.0 OK
branca: pinned=0.8.2 installed=0.8.2 OK
folium: pinned=1.0.0 installed=1.0.0 OK
streamlit-folium: pinned=0.28.0 installed=0.28.0 OK
xyzservices: pinned=2026.9.1 installed=2026.9.1 OK
```

**A full fresh-environment reinstall from `requirements.txt` was NOT performed.** No package was installed, upgraded, downgraded or removed during Phase 10; only `requirements.txt` was amended with the two missing pins.

---

## 5. Task 4 — Five-minute demo script with real evidence

Created `docs/DEMO_SCRIPT.md`. It uses one image from `data/samples/` (`road_1.jpg`) and two positive-labelled clips from the held-out TEST split of `~/Datasets/ThirdEye/Nexar/test-public/`: `00075.mp4` (model-correct) and `01023.mp4` (model-wrong). It prints absolute paths and real per-frame probabilities, uses `app/video_utils.py` read-only (the same 12-frame sampler as the dashboard), and contains five honest viva questions (YOLO policy, simulated ambulances, 68% accuracy, OSM tag sparsity, difference from the earlier report).

Real pipeline output (`/tmp/phase10_demo_evidence.txt`, verbatim):

```text
/Users/srishanthbiradar/Projects/third-eye-v2/.venv/lib/python3.12/site-packages/torch/ao/nn/quantized/dynamic/modules/rnn.py:162: UserWarning: torch.quantize_per_tensor, torch.quantize_per_channel and other quantized tensor creation functions that produce tensors with dtype torch.quint8, torch.qint8, and torch.qint32 are deprecated and will be removed in a future PyTorch release. Please see https://github.com/pytorch/pytorch/issues/184982 for more information. (Triggered internally at /Users/runner/work/pytorch/pytorch/aten/src/ATen/quantized/Quantizer.cpp:116.)
  w_ih = torch.quantize_per_tensor(
/Users/srishanthbiradar/Projects/third-eye-v2/.venv/lib/python3.12/site-packages/torch/utils/data/dataloader.py:759: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, device pinned memory won't be used.
  super().__init__(loader)
=== DEMO IMAGE ===
path: /Users/srishanthbiradar/Projects/third-eye-v2/data/samples/road_1.jpg
size: (1771, 1237)
[Third-Eye CV] Initializing Faster R-CNN ResNet-50 FPN on device: MPS
detections: 8 vehicles: 6 persons: 2
[Third-Eye Accident Analyzer] Loading frozen ResNet-18 on device: MPS
classifier_score: 0.31794 verdict: ('NO ACCIDENT SIGNAL', 'none', 'normal_flow')
[Third-Eye OCR] Initializing EasyOCR Reader on device: CPU
vehicle 1: type=car color=red share=0.4439 det_conf=0.999 plate=CAN09G plate_conf=0.5412
vehicle 2: type=car color=green share=0.2985 det_conf=0.9988 plate=None plate_conf=None
vehicle 3: type=car color=red share=0.5779 det_conf=0.9912 plate=None plate_conf=None
vehicle 4: type=car color=grey share=0.7336 det_conf=0.9155 plate=None plate_conf=None
vehicle 5: type=car color=grey share=0.5938 det_conf=0.6039 plate=None plate_conf=None
vehicle 6: type=car color=green share=0.4912 det_conf=0.5256 plate=None plate_conf=None

=== DEMO VIDEO ===
path: /Users/srishanthbiradar/Datasets/ThirdEye/Nexar/test-public/positive/00075.mp4
metadata: frames=289 fps=30.0 width=1280 height=720 duration_s=9.633
frame= 1 source_frame=   0 t=  0.000s score=0.539433 verdict=UNCERTAIN vehicles=6 persons=0
frame= 2 source_frame=  26 t=  0.867s score=0.649817 verdict=LIKELY ACCIDENT vehicles=4 persons=0
frame= 3 source_frame=  52 t=  1.733s score=0.835425 verdict=LIKELY ACCIDENT vehicles=3 persons=0
frame= 4 source_frame=  79 t=  2.633s score=0.669323 verdict=LIKELY ACCIDENT vehicles=1 persons=0
frame= 5 source_frame= 105 t=  3.500s score=0.449300 verdict=UNCERTAIN vehicles=3 persons=0
frame= 6 source_frame= 131 t=  4.367s score=0.577594 verdict=UNCERTAIN vehicles=5 persons=0
frame= 7 source_frame= 157 t=  5.233s score=0.410082 verdict=UNCERTAIN vehicles=6 persons=0
frame= 8 source_frame= 183 t=  6.100s score=0.724450 verdict=LIKELY ACCIDENT vehicles=11 persons=1
frame= 9 source_frame= 209 t=  6.967s score=0.782801 verdict=LIKELY ACCIDENT vehicles=5 persons=0
frame=10 source_frame= 236 t=  7.867s score=0.737955 verdict=LIKELY ACCIDENT vehicles=4 persons=3
frame=11 source_frame= 262 t=  8.733s score=0.868467 verdict=LIKELY ACCIDENT vehicles=10 persons=0
frame=12 source_frame= 288 t=  9.600s score=0.875386 verdict=LIKELY ACCIDENT vehicles=11 persons=0
top_frame=12 source_frame=288 t=9.6s top_score=0.875386 frames>=0.60=8

=== DEMO VIDEO ===
path: /Users/srishanthbiradar/Datasets/ThirdEye/Nexar/test-public/positive/01023.mp4
metadata: frames=284 fps=30.0 width=1280 height=720 duration_s=9.467
frame= 1 source_frame=   0 t=  0.000s score=0.207900 verdict=NO ACCIDENT SIGNAL vehicles=17 persons=0
frame= 2 source_frame=  26 t=  0.867s score=0.248185 verdict=NO ACCIDENT SIGNAL vehicles=20 persons=0
frame= 3 source_frame=  51 t=  1.700s score=0.139082 verdict=NO ACCIDENT SIGNAL vehicles=18 persons=0
frame= 4 source_frame=  77 t=  2.567s score=0.116356 verdict=NO ACCIDENT SIGNAL vehicles=21 persons=0
frame= 5 source_frame= 103 t=  3.433s score=0.197019 verdict=NO ACCIDENT SIGNAL vehicles=18 persons=0
frame= 6 source_frame= 129 t=  4.300s score=0.151156 verdict=NO ACCIDENT SIGNAL vehicles=20 persons=0
frame= 7 source_frame= 154 t=  5.133s score=0.231850 verdict=NO ACCIDENT SIGNAL vehicles=24 persons=0
frame= 8 source_frame= 180 t=  6.000s score=0.271732 verdict=NO ACCIDENT SIGNAL vehicles=19 persons=0
frame= 9 source_frame= 206 t=  6.867s score=0.341071 verdict=NO ACCIDENT SIGNAL vehicles=22 persons=0
frame=10 source_frame= 232 t=  7.733s score=0.418328 verdict=UNCERTAIN vehicles=22 persons=0
frame=11 source_frame= 257 t=  8.567s score=0.445759 verdict=UNCERTAIN vehicles=29 persons=0
frame=12 source_frame= 283 t=  9.433s score=0.502151 verdict=UNCERTAIN vehicles=24 persons=0
top_frame=12 source_frame=283 t=9.433s top_score=0.502151 frames>=0.60=0
```

The honest failure (`01023.mp4`, a positive-labelled held-out clip with `frames>=0.60=0`) is shown explicitly in the demo, alongside the metric-file probabilities for video 1023 (0.1966 / 0.2526 / 0.3101) quoted in §2.

---

## 6. Task 5 — Final gate

### Full pytest suite (verbatim)

Command: `.venv/bin/pytest --cache-clear -v`

```text
============================= test session starts ==============================
platform darwin -- Python 3.12.3, pytest-9.2.0, pluggy-1.7.0 -- /Users/srishanthbiradar/Projects/third-eye-v2/.venv/bin/python3.12
cachedir: .pytest_cache
rootdir: /Users/srishanthbiradar/Projects/third-eye-v2
configfile: pytest.ini
testpaths: tests
plugins: anyio-4.16.0
collecting ... collected 116 items

tests/test_accident_analysis.py::TestAccidentFeaturesMocked::test_compute_iou_zero_overlap PASSED [  0%]
tests/test_accident_analysis.py::TestAccidentFeaturesMocked::test_compute_iou_heavy_overlap PASSED [  1%]
tests/test_accident_analysis.py::TestAccidentFeaturesMocked::test_extract_features_no_overlap PASSED [  2%]
tests/test_accident_analysis.py::TestAccidentFeaturesMocked::test_extract_features_collision_and_pedestrian PASSED [  3%]
tests/test_accident_analysis.py::TestAccidentRulesMocked::test_rules_no_accident_scenario PASSED [  4%]
tests/test_accident_analysis.py::TestAccidentRulesMocked::test_rules_severe_collision_scenario PASSED [  5%]
tests/test_accident_analysis.py::TestPersonsNearVehicles::test_no_persons_or_vehicles PASSED [  6%]
tests/test_accident_analysis.py::TestPersonsNearVehicles::test_disjoint_person_and_vehicle PASSED [  6%]
tests/test_accident_analysis.py::TestPersonsNearVehicles::test_overlapping_person_and_vehicle PASSED [  7%]
tests/test_accident_analysis.py::TestVerdictThresholds::test_verdict_high_probability PASSED [  8%]
tests/test_accident_analysis.py::TestVerdictThresholds::test_verdict_uncertain PASSED [  9%]
tests/test_accident_analysis.py::TestVerdictThresholds::test_verdict_no_accident PASSED [ 10%]
tests/test_accident_analysis.py::TestModelSplitZeroLeakage::test_zero_leakage_in_splits PASSED [ 11%]
tests/test_accident_analysis.py::TestAccidentEventSchemaAndDatabase::test_accident_event_valid_with_phase3b_fields PASSED [ 12%]
tests/test_accident_analysis.py::TestAccidentEventSchemaAndDatabase::test_accident_event_rejects_invalid_score PASSED [ 12%]
tests/test_accident_analysis.py::TestAccidentEventSchemaAndDatabase::test_save_and_retrieve_accident_event PASSED [ 13%]
tests/test_accident_analysis.py::TestRegressionNormalTrafficSamplePhotos::test_normal_traffic_not_all_likely_accident PASSED [ 14%]
tests/test_app_dispatch.py::test_google_maps_url_builder_uses_official_scheme PASSED [ 15%]
tests/test_app_dispatch.py::test_map_builder_returns_folium_map_with_expected_markers PASSED [ 16%]
tests/test_app_dispatch.py::test_replay_step_selection_changes_simulated_marker_position PASSED [ 17%]
tests/test_app_dispatch.py::test_app_scan_finds_no_forbidden_widgets_or_phone_links PASSED [ 18%]
tests/test_app_dispatch.py::test_dispatch_tab_renders_real_summary_fields PASSED [ 18%]
tests/test_app_dispatch.py::test_missing_and_null_fields_render_not_determined PASSED [ 19%]
tests/test_app_dispatch.py::test_emergency_numbers_are_plain_text_and_status_comes_from_data PASSED [ 20%]
tests/test_app_ui.py::test_widget_policy_allows_only_the_cv_uploader PASSED [ 21%]
tests/test_app_ui.py::test_no_upload_runs_no_model PASSED                [ 22%]
tests/test_app_ui.py::test_real_image_upload_runs_and_shows_vision_only_report PASSED [ 23%]
tests/test_app_ui.py::test_dispatch_tab_shows_not_available_with_injected_missing_summary PASSED [ 24%]
tests/test_app_video.py::test_probe_video_reads_real_metadata PASSED     [ 25%]
tests/test_app_video.py::test_frame_sampler_covers_whole_real_clip PASSED [ 25%]
tests/test_caller_extraction.py::TestCallerExtraction::test_evaluate_against_expected_json PASSED [ 26%]
tests/test_caller_extraction.py::TestCallerExtraction::test_negation_awareness PASSED [ 27%]
tests/test_caller_extraction.py::TestCallerExtraction::test_vague_call_returns_none PASSED [ 28%]
tests/test_database.py::TestDatabaseSetup::test_init_db_creates_all_tables PASSED [ 29%]
tests/test_database.py::TestDatabaseSetup::test_foreign_keys_and_pragma_enabled PASSED [ 30%]
tests/test_database.py::TestDatabaseSetup::test_insert_and_query_all_records PASSED [ 31%]
tests/test_database.py::TestDatabaseSetup::test_save_vehicle_reports_helper PASSED [ 31%]
tests/test_database.py::TestDatabaseSetup::test_database_enforces_data_source_constraint PASSED [ 32%]
tests/test_database.py::TestDatabaseSetup::test_database_enforces_bed_capacity_constraint PASSED [ 33%]
tests/test_detector.py::TestDetectorLogicMocked::test_filter_detections_class_filtering PASSED [ 34%]
tests/test_detector.py::TestDetectorLogicMocked::test_filter_detections_confidence_threshold PASSED [ 35%]
tests/test_detector.py::TestDetectorLogicMocked::test_filter_detections_reject_out_of_bounds_and_degenerate_boxes PASSED [ 36%]
tests/test_detector.py::TestDetectorLogicMocked::test_device_selection_mps_or_cpu PASSED [ 37%]
tests/test_detector.py::TestDetectorLogicMocked::test_draw_detections_renders_and_saves PASSED [ 37%]
tests/test_detector.py::TestRealModelIntegration::test_live_detector_on_sample_image PASSED [ 38%]
tests/test_fusion.py::TestFusionDiscrepancyAndDatabase::test_fusion_flags_disagreement_and_preserves_both PASSED [ 39%]
tests/test_fusion.py::TestFusionDiscrepancyAndDatabase::test_fusion_aligned_sources PASSED [ 40%]
tests/test_fusion.py::TestFusionDiscrepancyAndDatabase::test_fused_report_matches_exact_vehicle_report_attributes PASSED [ 41%]
tests/test_incident_text.py::TestIncidentTextSummary::test_summary_no_accident_verdict PASSED [ 42%]
tests/test_incident_text.py::TestIncidentTextSummary::test_summary_likely_accident_verdict PASSED [ 43%]
tests/test_incident_text.py::TestIncidentTextSummary::test_summary_uncertain_verdict PASSED [ 43%]
tests/test_incident_text.py::TestIncidentTextSummary::test_summary_plate_handling_none_and_unverified PASSED [ 44%]
tests/test_occupant_report.py::TestOccupantReportMocked::test_box_intersection_ratio PASSED [ 45%]
tests/test_occupant_report.py::TestOccupantReportMocked::test_mocked_no_persons_detected PASSED [ 46%]
tests/test_occupant_report.py::TestOccupantReportMocked::test_mocked_disjoint_pedestrian PASSED [ 47%]
tests/test_occupant_report.py::TestOccupantReportMocked::test_mocked_person_overlapping_vehicle PASSED [ 48%]
tests/test_occupant_report.py::TestOccupantReportLive::test_live_occupant_generation PASSED [ 49%]
tests/test_plate_reader.py::TestPlateReaderMocked::test_normalize_plate_text PASSED [ 50%]
tests/test_plate_reader.py::TestPlateReaderMocked::test_read_plate_with_mocked_success PASSED [ 50%]
tests/test_plate_reader.py::TestPlateReaderMocked::test_read_plate_with_mocked_unreadable PASSED [ 51%]
tests/test_plate_reader.py::TestPlateReaderMocked::test_read_plate_selects_highest_confidence PASSED [ 52%]
tests/test_plate_reader.py::TestPlateReaderLive::test_live_ocr_on_real_sample_photo PASSED [ 53%]
tests/test_rag.py::test_haversine_distance_accuracy PASSED               [ 54%]
tests/test_rag.py::test_reranking_proximity_boost PASSED                 [ 55%]
tests/test_rag.py::test_hospital_schema_validation PASSED                [ 56%]
tests/test_rag.py::test_save_and_retrieve_hospitals PASSED               [ 56%]
tests/test_rag.py::test_missing_phone_reports_not_listed PASSED          [ 57%]
tests/test_rag.py::test_bed_availability_honesty PASSED                  [ 58%]
tests/test_rag.py::test_overpass_failure_fallback PASSED                 [ 59%]
tests/test_rag.py::test_emergency_numbers_verified PASSED                [ 60%]
tests/test_routing.py::TestAStarRouting::test_astar_matches_networkx_on_20_random_pairs PASSED [ 61%]
tests/test_routing.py::TestAStarRouting::test_route_continuity PASSED    [ 62%]
tests/test_routing.py::TestAStarRouting::test_astar_identical_source_and_target PASSED [ 62%]
tests/test_routing.py::TestAStarRouting::test_astar_disconnected_returns_empty PASSED [ 63%]
tests/test_routing.py::TestRoadNetworkProperties::test_driving_distance_greater_than_or_equal_straight_line PASSED [ 64%]
tests/test_routing.py::TestRoadNetworkProperties::test_graph_stats_metadata PASSED [ 65%]
tests/test_routing.py::TestDispatchEngine::test_nearest_ambulance_selection_on_mock_graph PASSED [ 66%]
tests/test_routing.py::TestDispatchEngine::test_full_dispatch_on_demo_incident PASSED [ 67%]
tests/test_routing.py::TestSimulationTrajectory::test_simulate_leg_trajectory PASSED [ 68%]
tests/test_routing.py::TestTask0ReviewFixes::test_hospital_deduplication PASSED [ 68%]
tests/test_routing.py::TestTask0ReviewFixes::test_suitability_filter_downranks_unrelated_specialty PASSED [ 69%]
tests/test_routing.py::TestTask0ReviewFixes::test_retrieval_query_assumed_needs_format PASSED [ 70%]
tests/test_routing.py::TestDatabaseRoutingPersistence::test_save_and_retrieve_ambulances PASSED [ 71%]
tests/test_routing.py::TestDatabaseRoutingPersistence::test_save_and_retrieve_route_result PASSED [ 72%]
tests/test_routing.py::TestPhase6bMedicalValidityRule::test_regression_lpu_architecture_never_selected PASSED [ 73%]
tests/test_routing.py::TestPhase6bMedicalValidityRule::test_exclusion_reasons_and_categories PASSED [ 74%]
tests/test_routing.py::TestPhase6bMedicalValidityRule::test_graph_corridor_hospital_coverage PASSED [ 75%]
tests/test_routing.py::TestPhase6bMedicalValidityRule::test_route_results_replace_old_rows_without_duplicates PASSED [ 75%]
tests/test_routing.py::TestPhase6dDispatchSummaryAndAlternatives::test_all_corridor_hospitals_considered_with_reasons PASSED [ 76%]
tests/test_routing.py::TestPhase6dDispatchSummaryAndAlternatives::test_dispatch_summary_json_matches_database PASSED [ 77%]
tests/test_schemas.py::TestSchemasValidation::test_detection_valid PASSED [ 78%]
tests/test_schemas.py::TestSchemasValidation::test_vehicle_report_valid PASSED [ 79%]
tests/test_schemas.py::TestSchemasValidation::test_occupant_report_valid PASSED [ 80%]
tests/test_schemas.py::TestSchemasValidation::test_accident_event_valid PASSED [ 81%]
tests/test_schemas.py::TestSchemasValidation::test_hospital_valid PASSED [ 81%]
tests/test_schemas.py::TestSchemasValidation::test_ambulance_valid PASSED [ 82%]
tests/test_schemas.py::TestSchemasValidation::test_route_result_valid PASSED [ 83%]
tests/test_schemas.py::TestSchemasValidation::test_audit_report_valid PASSED [ 84%]
tests/test_schemas.py::TestSchemasRejection::test_reject_invalid_data_source[Detection-sample_kwargs0] PASSED [ 85%]
tests/test_schemas.py::TestSchemasRejection::test_reject_invalid_data_source[VehicleReport-sample_kwargs1] PASSED [ 86%]
tests/test_schemas.py::TestSchemasRejection::test_reject_invalid_data_source[OccupantReport-sample_kwargs2] PASSED [ 87%]
tests/test_schemas.py::TestSchemasRejection::test_reject_invalid_data_source[AccidentEvent-sample_kwargs3] PASSED [ 87%]
tests/test_schemas.py::TestSchemasRejection::test_reject_invalid_data_source[Hospital-sample_kwargs4] PASSED [ 88%]
tests/test_schemas.py::TestSchemasRejection::test_reject_invalid_data_source[Ambulance-sample_kwargs5] PASSED [ 89%]
tests/test_schemas.py::TestSchemasRejection::test_reject_invalid_data_source[RouteResult-sample_kwargs6] PASSED [ 90%]
tests/test_schemas.py::TestSchemasRejection::test_reject_invalid_data_source[AuditReport-sample_kwargs7] PASSED [ 91%]
tests/test_schemas.py::TestSchemasRejection::test_reject_invalid_confidence PASSED [ 92%]
tests/test_schemas.py::TestSchemasRejection::test_reject_malformed_bbox PASSED [ 93%]
tests/test_schemas.py::TestSchemasRejection::test_reject_hospital_bed_inconsistency PASSED [ 93%]
tests/test_schemas.py::TestSchemasRejection::test_reject_invalid_coordinates PASSED [ 94%]
tests/test_vehicle_attributes.py::TestVehicleCropping::test_crop_vehicle_clamping PASSED [ 95%]
tests/test_vehicle_attributes.py::TestVehicleCropping::test_crop_vehicle_invalid_box_raises PASSED [ 96%]
tests/test_vehicle_attributes.py::TestVehicleTypeMapping::test_valid_vehicle_types PASSED [ 97%]
tests/test_vehicle_attributes.py::TestVehicleTypeMapping::test_non_vehicle_labels_rejected PASSED [ 98%]
tests/test_vehicle_attributes.py::TestDominantColorExtraction::test_classify_pure_bgr_colors PASSED [ 99%]
tests/test_vehicle_attributes.py::TestDominantColorExtraction::test_dominant_color_on_real_sample_crop PASSED [100%]

=============================== warnings summary ===============================
tests/test_app_ui.py::test_real_image_upload_runs_and_shows_vision_only_report
  /Users/srishanthbiradar/Projects/third-eye-v2/.venv/lib/python3.12/site-packages/easyocr/detection.py:81: DeprecationWarning: torch.ao.quantization is deprecated and will be removed in 2.10. 
  For migrations of users: 
  1. Eager mode quantization (torch.ao.quantization.quantize, torch.ao.quantization.quantize_dynamic), please migrate to use torchao eager mode quantize_ API instead 
  2. FX graph mode quantization (torch.ao.quantization.quantize_fx.prepare_fx,torch.ao.quantization.quantize_fx.convert_fx, please migrate to use torchao pt2e quantization API instead (prepare_pt2e, convert_pt2e) 
  3. pt2e quantization has been migrated to torchao (https://github.com/pytorch/ao/tree/main/torchao/quantization/pt2e) 
  see https://github.com/pytorch/ao/issues/2259 for more details
    torch.quantization.quantize_dynamic(net, dtype=torch.qint8, inplace=True)

tests/test_app_ui.py::test_real_image_upload_runs_and_shows_vision_only_report
tests/test_app_ui.py::test_real_image_upload_runs_and_shows_vision_only_report
  /Users/srishanthbiradar/Projects/third-eye-v2/.venv/lib/python3.12/site-packages/torch/ao/quantization/quantize.py:570: DeprecationWarning: torch.ao.quantization is deprecated and will be removed in 2.10. 
  For migrations of users: 
  1. Eager mode quantization (torch.ao.quantization.quantize, torch.ao.quantization.quantize_dynamic), please migrate to use torchao eager mode quantize_ API instead 
  2. FX graph mode quantization (torch.ao.quantization.quantize_fx.prepare_fx,torch.ao.quantization.quantize_fx.convert_fx, please migrate to use torchao pt2e quantization API instead (prepare_pt2e, convert_pt2e) 
  3. pt2e quantization has been migrated to torchao (https://github.com/pytorch/ao/tree/main/torchao/quantization/pt2e) 
  see https://github.com/pytorch/ao/issues/2259 for more details
    convert(model, mapping, inplace=True)

tests/test_app_ui.py::test_real_image_upload_runs_and_shows_vision_only_report
  /Users/srishanthbiradar/Projects/third-eye-v2/.venv/lib/python3.12/site-packages/easyocr/recognition.py:177: DeprecationWarning: torch.ao.quantization is deprecated and will be removed in 2.10. 
  For migrations of users: 
  1. Eager mode quantization (torch.ao.quantization.quantize, torch.ao.quantization.quantize_dynamic), please migrate to use torchao eager mode quantize_ API instead 
  2. FX graph mode quantization (torch.ao.quantization.quantize_fx.prepare_fx,torch.ao.quantization.quantize_fx.convert_fx, please migrate to use torchao pt2e quantization API instead (prepare_pt2e, convert_pt2e) 
  3. pt2e quantization has been migrated to torchao (https://github.com/pytorch/ao/tree/main/torchao/quantization/pt2e) 
  see https://github.com/pytorch/ao/issues/2259 for more details
    torch.quantization.quantize_dynamic(model, dtype=torch.qint8, inplace=True)

tests/test_app_ui.py::test_real_image_upload_runs_and_shows_vision_only_report
  /Users/srishanthbiradar/Projects/third-eye-v2/.venv/lib/python3.12/site-packages/torch/ao/nn/quantized/dynamic/modules/rnn.py:162: UserWarning: torch.quantize_per_tensor, torch.quantize_per_channel and other quantized tensor creation functions that produce tensors with dtype torch.quint8, torch.qint8, and torch.qint32 are deprecated and will be removed in a future PyTorch release. Please see https://github.com/pytorch/pytorch/issues/184982 for more information. (Triggered internally at /Users/runner/work/pytorch/pytorch/aten/src/ATen/quantized/Quantizer.cpp:116.)
    w_ih = torch.quantize_per_tensor(

tests/test_app_ui.py::test_real_image_upload_runs_and_shows_vision_only_report
tests/test_fusion.py::TestFusionDiscrepancyAndDatabase::test_fused_report_matches_exact_vehicle_report_attributes
tests/test_occupant_report.py::TestOccupantReportLive::test_live_occupant_generation
tests/test_plate_reader.py::TestPlateReaderLive::test_live_ocr_on_real_sample_photo
  /Users/srishanthbiradar/Projects/third-eye-v2/.venv/lib/python3.12/site-packages/torch/utils/data/dataloader.py:759: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, device pinned memory won't be used.
    super().__init__(loader)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================= 116 passed, 9 warnings in 26.65s =======================
```

Wins: 116 passed, 0 failed. Warnings are upstream PyTorch/EasyOCR deprecation and MPS pin-memory notices only.

### Six pipeline scripts — one run each, exit status + one-line real result

Command: `for s in run_detection run_vehicle_reports run_accident_analysis run_incident_reports run_hospital_suggestions run_dispatch; do .venv/bin/python scripts/$s.py > /tmp/phase10_$s.txt 2>&1; echo "$s.py exit=$?"; done`

```text
run_detection.py exit=0
run_vehicle_reports.py exit=0
run_accident_analysis.py exit=0
run_incident_reports.py exit=0
run_hospital_suggestions.py exit=0
run_dispatch.py exit=0
```

One-line real result from each captured output file (verbatim lines):

| Script | Exit | Verbatim one-line result |
| :--- | :---: | :--- |
| `run_detection.py` | 0 | `Done. Processed 3 image(s) with 37 total detection(s).` |
| `run_vehicle_reports.py` | 0 | `  Total Vehicles Detected  : 30` |
| `run_accident_analysis.py` | 0 | `    - NO ACCIDENT SIGNAL: 3` |
| `run_incident_reports.py` | 0 | `  Total Pairings Evaluated       : 24` |
| `run_hospital_suggestions.py` | 0 | `      FAISS index active with 115 facility profiles (dim=384).` |
| `run_dispatch.py` | 0 | `Exported dispatch summary JSON: data/outputs/dispatch_summary.json (4 scenarios)` |

### Stray files (listed, not deleted)

```text
./data/third_eye.db                   0 bytes, SHA-256 e3b0c442... (empty; canonical DB is db/third_eye.db)
./data/.DS_Store                      6148 bytes (macOS Finder metadata)
./data/outputs/.DS_Store              6148 bytes (macOS Finder metadata)
./data/hospitals_osm_raw.json         265 bytes (leftover raw Overpass snippet; data/hospitals_osm.json is the real cache)
./data/embeddings.npz                 2219392 bytes (gitignored training cache from earlier phases)
./data/frames/metadata.csv            gitignored extracted-frame cache from earlier phases
./.pytest_cache/                      generated pytest cache
__pycache__/ directories under app/, core/, db/, tests/    generated Python bytecode caches
```

Generated-but-expected pipeline outputs in `data/outputs/` (annotated JPGs, `crops/`, `routes/*.png`, `routes/*.geojson`, `dispatch_summary.json`) were regenerated by the six scripts and are intentionally retained. Nothing above was deleted.

---

## 7. Summary of files changed in Phase 10

| File | Change |
| :--- | :--- |
| `core/routing/dispatch.py` | Task 0 full-precision minutes and single-source seconds in JSON export |
| `scripts/run_dispatch.py` | Task 0 terminal output / in-memory summary use authoritative seconds |
| `requirements.txt` | Task 3 added `joblib==1.6.0`, `pandas==3.1.1` (no pins removed) |
| `README.md` | Task 2 added 10-line quick start |
| `docs/SYSTEM_CARD.md` | Task 1 new honest system card |
| `docs/DEMO_SCRIPT.md` | Task 4 new 5-minute demo script |
| `docs/phase10_report.md` | Task 6 this report |

No file under `app/` was edited. No test file was edited. No file was deleted. No package was installed or removed. No full fresh-environment reinstall was performed.

**Phase 10 is complete.**
