# Deterministic NLP Documentation — Incident Information Extraction

## 1. Overview
The Natural Language Processing (NLP) module analyzes unstructured emergency call transcripts, bystander reports, or incident descriptions using transparent Python regular expressions and deterministic keyword dictionaries.

No trained spaCy models, BERT, Transformers, LLMs, or vector embeddings are used.

---

## 2. Text Normalization
Cleans transcripts before parsing:
- Lowercase conversion and whitespace collapsing.
- Punctuation stripping.
- Dictionary-based phonetic typo correction for common transcription errors:
  - `acident` / `accidnt` -> `accident`
  - `vehical` / `vehicl` -> `vehicle`
  - `ambulence` -> `ambulance`
  - `colision` -> `collision`
  - `injuired` -> `injured`
  - `unconcious` -> `unconscious`
  - `overturnd` -> `overturned`

---

## 3. Entity & Keyword Extraction
1. **Vehicle Extraction**:
   - Matches vehicle types: `car`, `sedan`, `suv`, `hatchback`, `truck`, `lorry`, `bus`, `motorcycle`, `bike`, `scooter`, `auto`, `van`, `bicycle`.
   - Extracts quantities (digits or number words: `two cars`, `3 trucks`).
   - Extracts modifier colors (`white`, `red`, `black`, `blue`, `silver`, etc.).
2. **Location Extraction**:
   - Matches known regional corridor landmarks: `Phagwara`, `GT Road`, `Banga Road`, `NH44`, `Sugar Mill`, `Railway Crossing`, `Bus Stand`, `Civil Hospital`, etc.
   - Matches prepositional phrases: `near ...`, `at ...`, `on ...`.
3. **Injury Keywords**:
   - Categories: `unconscious`, `fatality`, `critical_condition`, `serious_injury`, `bleeding`, `fracture`, `head_injury`, `trapped_occupant`, `minor_injury`.
   - **Negation Handling**: Inspects for preceding negative tokens (`no injuries`, `without hurt`, `zero casualties`) to prevent false alarm triggers.
4. **Hazard Keywords**:
   - Categories: `fire`, `smoke`, `fuel_leak`, `gas_leak`, `oil_spill`, `overturned_vehicle`, `road_obstruction`, `electrical_hazard`.
   - **Negation Handling**: Evaluates `no fire`, `without smoke`, etc.

---

## 4. Urgency Classification Rules
Urgency is determined by an explicit, transparent rule hierarchy:

| Level | Criteria | Example Triggers |
|---|---|---|
| **CRITICAL** | Life-safety threats | Unconscious victim, fatality, active vehicle fire, trapped occupant, severe head trauma |
| **HIGH** | Serious injury or major hazard | Fractures, heavy bleeding, overturned vehicle, fuel leak on highway |
| **MEDIUM** | Moderate injury or obstruction | Minor injuries, general road obstruction, property damage |
| **LOW** | Minor incident, no injuries | Minor bumper scratch, fender bender, no injuries explicitly confirmed |

---

## 5. Contradiction Detection
Flags logical inconsistencies in caller statements:
- Caller states `no injuries` but mentions `unconscious` or `bleeding` victim.
- Caller mentions single vehicle but describes `head-on collision` with another vehicle.
