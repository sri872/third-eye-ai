"""Emergency caller description information extraction using spaCy and Rule Matchers.

Extracts:
- Location mentions (roadways, landmarks, intersections)
- Vehicle count and vehicle types
- People involved count
- Injury keywords (with negation awareness e.g. 'no injuries')
- Fire / hazard keywords (with negation awareness e.g. 'no smoke')
- Urgency keywords

Rules:
- Strictly extracts what the text says. Missing fields remain None.
- Negated terms ('no fire', 'nobody is hurt') are explicitly excluded.
- Evaluates against data/sample_calls/expected.json.
- Data source is tagged as CALLER_SIMULATED.
"""

import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import spacy
from spacy.matcher import Matcher, PhraseMatcher

from core.schemas import CallerExtraction

logger = logging.getLogger("third_eye.nlp.caller_extraction")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_SPACY_CACHE = {}

NUMBER_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "pair": 2,
    "couple": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

VEHICLE_KEYWORDS = [
    "car", "cars", "sedan", "sedans", "suv", "suvs", "truck", "trucks",
    "bus", "buses", "motorcycle", "motorcycles", "bike", "bikes",
    "bicycle", "bicycles", "auto", "rickshaw", "van", "vans"
]

INJURY_KEYWORDS = [
    "bleeding", "unconscious", "broken", "pain", "injured", "trapped",
    "burned", "dying", "motionless", "head injury", "hurt", "fracture", "wound"
]

HAZARD_KEYWORDS = [
    "smoke", "fuel spill", "fire", "fuel leak", "burning", "flames", "gas leak"
]

URGENCY_KEYWORDS = [
    "hurry", "immediately", "quickly", "critical", "now", "rush", "fast", "urgent", "asap",
    "emergency assistance",
]

ROAD_KEYWORDS = {
    "road", "street", "highway", "junction", "intersection", "flyover",
    "cross", "circle", "avenue", "lane", "metro", "board", "stand",
    "bus stand", "bus stop", "station", "market", "bypass", "crossing",
    "toll plaza", "bridge"
}

NEGATION_TOKENS = {"no", "not", "without", "zero", "none", "neither", "nor", "n't", "nobody", "never", "nothing"}


def get_spacy_nlp():
    """Lazy load spaCy en_core_web_sm pipeline."""
    if "nlp" not in _SPACY_CACHE:
        _SPACY_CACHE["nlp"] = spacy.load("en_core_web_sm")
    return _SPACY_CACHE["nlp"]


def is_negated(token) -> bool:
    """Check if token or its immediate left context contains a negation word."""
    # Check left siblings/children
    for left in token.lefts:
        if left.lower_ in NEGATION_TOKENS or left.dep_ == "neg":
            return True
    # Check 3 preceding tokens in sentence
    doc = token.doc
    idx = token.i
    start_idx = max(0, idx - 3)
    for i in range(start_idx, idx):
        if doc[i].lower_ in NEGATION_TOKENS:
            return True
    return False


def extract_location(doc) -> Optional[str]:
    """Extract roadway, intersection, or landmark location."""
    text = doc.text.strip()
    if not text:
        return None

    # Pattern 1: Regex for standard Indian urban road addresses following on/at/near
    loc_pattern = re.compile(
        r"(?:on|at|near)\s+([A-Z0-9][A-Za-z0-9\s]*(?:Road|Street|Highway|Junction|Flyover|Metro|Board|Ring Road|Intersection|Cross|Circle|Avenue)[A-Za-z0-9\s]*)",
        re.IGNORECASE,
    )
    match = loc_pattern.search(text)
    if match:
        loc = match.group(1).strip()
        # Clean trailing punctuation
        loc = re.sub(r"[.,;!?\"']+$", "", loc).strip()
        # Truncate if too long (e.g. bleed into next sentence)
        for stop_word in [" Two", " A ", " Four", " Five", " One", " Huge"]:
            if stop_word in loc:
                loc = loc.split(stop_word)[0].strip()
        if len(loc.split()) >= 2:
            return loc

    # Pattern 1b: Indian national/state highway route numbers ("near NH-44", "on SH-1").
    route_match = re.search(r"\b(?:NH|SH)[-\s]?\d{1,3}[A-Za-z]?\b", text, re.IGNORECASE)
    if route_match:
        return re.sub(r"[.,;!?\"']+$", "", route_match.group(0).strip()).upper()

    # Pattern 1c: Common Indian transit landmarks following on/at/near: e.g. "near the bus stand"
    landmark_match = re.search(
        r"(?:on|at|near)\s+(?:the\s+)?([A-Za-z0-9\s]+?(?:bus\s+stand|bus\s+stop|metro\s+station|railway\s+station|market|bypass|crossing|toll\s+plaza|bridge|junction|flyover))\b",
        text,
        re.IGNORECASE,
    )
    if landmark_match:
        lm = landmark_match.group(1).strip()
        lm = re.sub(r"[.,;!?\"']+$", "", lm).strip()
        if lm:
            return lm

    # Pattern 2: spaCy named entities with GPE, LOC, FAC
    for ent in doc.ents:
        if ent.label_ in {"GPE", "LOC", "FAC"}:
            if any(term in ent.text.lower() for term in ROAD_KEYWORDS) or len(ent.text.split()) >= 2:
                return ent.text.strip()

    return None


def detect_contradictions(text: str) -> List[str]:
    """Identify contradictory statements in the caller description."""
    contradictions: List[str] = []
    text_lower = text.lower()

    # 1. Conflicting vehicle counts in transcript
    counts_found = []
    for word, num in NUMBER_WORDS.items():
        if re.search(
            rf"\b{word}\s+(?:(?:damaged|colliding|overturned|speeding|stopped|moving|passenger)\s+)?(?:cars|vehicles|trucks|buses|sedans|suvs)\b",
            text_lower,
        ):
            counts_found.append(num)
        elif re.search(rf"\b{word}\s+car\b", text_lower):
            counts_found.append(num)
    for digit in [1, 2, 3, 4, 5, 6]:
        if re.search(
            rf"\b{digit}\s+(?:(?:damaged|colliding|overturned|speeding|stopped|moving|passenger)\s+)?(?:cars|vehicles|trucks|buses|sedans|suvs)\b",
            text_lower,
        ):
            counts_found.append(digit)
        elif re.search(rf"\b{digit}\s+car\b", text_lower):
            counts_found.append(digit)

    unique_counts = sorted(list(set(counts_found)))
    if len(unique_counts) >= 2:
        if any(
            kw in text_lower
            for kw in [
                "first",
                "then says",
                "later says",
                "only one",
                "actually",
                "instead",
                "contradiction",
                "subsequently",
                "reports",
            ]
        ):
            contradictions.append(
                f"Caller vehicle count contradiction: caller mentions conflicting counts "
                f"({' vs '.join(str(c) for c in unique_counts)} vehicle(s))."
            )

    # 2. Conflicting injury reporting
    has_injury_mention = any(
        kw in text_lower
        for kw in ["injury", "injured", "hurt", "bleeding", "wound", "someone was hurt"]
    )
    has_injury_negation = any(
        neg in text_lower
        for neg in [
            "nobody was injured",
            "no one was hurt",
            "nobody was hurt",
            "no injuries",
            "nobody is hurt",
        ]
    )
    if has_injury_mention and has_injury_negation:
        if any(
            kw in text_lower
            for kw in ["subsequently", "later", "then says", "actually", "initially", "but"]
        ):
            contradictions.append(
                "Caller injury contradiction: caller reports both presence and absence of injuries."
            )

    return contradictions


def extract_vehicle_count(doc, contradictions: Optional[List[str]] = None) -> Optional[int]:
    """Extract total vehicle count mentioned in emergency text."""
    text_lower = doc.text.lower()

    # If an intra-call contradiction on vehicle count was detected, do not silently resolve
    if contradictions and any("vehicle count contradiction" in c for c in contradictions):
        return None

    # Check for indefinite/vague quantifiers: "many cars", "several vehicles" -> not determined
    # Do not invent counts from vague words
    has_indefinite = bool(
        re.search(r"\b(?:many|several|multiple|a\s+lot\s+of)\s+(?:cars|vehicles|trucks|motorcycles)\b", text_lower)
    )

    # Pattern: explicit count e.g. "Four vehicles", "Two cars", "Five cars", "couple of cars", "two damaged cars"
    explicit_count = None
    for word, num in NUMBER_WORDS.items():
        if re.search(
            rf"\b{word}\s+(?:(?:damaged|colliding|overturned|speeding|stopped|moving|passing|passenger|commercial)\s+)?(?:vehicles|cars|trucks|buses|motorcycles|sedans|suvs)\b",
            text_lower,
        ):
            explicit_count = num
            break

    if explicit_count is not None:
        return explicit_count

    # Check for "couple of cars" or "pair of"
    if "couple of cars" in text_lower or "pair of cars" in text_lower:
        return 2

    # Check for two distinct single vehicle mentions (e.g. "a bus hit a motorcycle", "a car and a motorcycle")
    single_vehicles = 0
    if re.search(r"\b(?:a|one)\s+(?:bmtc\s+)?(?:bus|truck|car|sedan|suv|motorcycle|bike|auto)\b", text_lower):
        single_vehicles += 1
    if re.search(r"\b(?:hit|and|into)\s+(?:a|an)\s+(?:bus|truck|car|sedan|suv|motorcycle|bike|auto)\b", text_lower):
        single_vehicles += 1
    if single_vehicles >= 2:
        return single_vehicles

    # If vague quantifiers are present and no explicit numeric count was stated, return None
    if has_indefinite:
        return None

    # Check if a single vehicle is mentioned (e.g. "a truck overturned", "a speeding car", "one car stopped")
    if re.search(r"\b(?:a|an|one)\s+(?:truck|car|bus|motorcycle|vehicle)\b", text_lower):
        return 1

    return None


def extract_vehicle_types(doc) -> List[str]:
    """Extract unique vehicle types mentioned."""
    found: Set[str] = set()
    text_lower = doc.text.lower()

    mapping = {
        "car": ["car", "cars"],
        "sedan": ["sedan", "sedans"],
        "suv": ["suv", "suvs"],
        "truck": ["truck", "trucks"],
        "bus": ["bus", "buses"],
        "motorcycle": ["motorcycle", "motorcycles", "bike", "bikes"],
    }

    for standard_type, variations in mapping.items():
        for var in variations:
            if re.search(rf"\b{var}\b", text_lower):
                found.add(standard_type)
                break

    # If generic "vehicle" mentioned and no specific type was identified
    if not found and re.search(r"\bvehicles?\b", text_lower):
        found.add("vehicle")

    return sorted(list(found))


def extract_people_count(doc) -> Optional[int]:
    """Extract count of people/victims involved."""
    text_lower = doc.text.lower()

    # Do not invent count from indefinite quantifiers like "several pedestrians" or "many people"
    if re.search(r"\b(?:several|many|multiple)\s+(?:people|passengers|victims|pedestrians)\b", text_lower):
        # If there is also an explicit number elsewhere, let that take precedence, otherwise None
        pass

    # Pattern: e.g. "three people", "two passengers", "at least three people"
    for word, num in NUMBER_WORDS.items():
        if re.search(rf"\b(?:at least\s+)?{word}\s+(?:people|passengers|victims|pedestrians|drivers)\b", text_lower):
            return num

    # Single person indicators: "One driver", "The rider", "The victim", "A pedestrian"
    if any(p in text_lower for p in ["one driver", "the rider", "the victim", "a pedestrian"]):
        return 1
    if re.search(r"\b(?:one|a|an)\s+person\b", text_lower):
        return 1

    return None


def extract_injury_status(text: str, doc) -> Tuple[str, List[str]]:
    """Determine injury status and keywords with negation and uncertainty awareness."""
    text_lower = text.lower()

    # 1. Check explicit negation
    negation_patterns = [
        r"\bnobody\s+(?:was|is)\s+(?:injured|hurt|bleeding)\b",
        r"\bno\s+one\s+(?:was|is)\s+(?:injured|hurt|bleeding)\b",
        r"\bno\s+injuries\b",
        r"\bnot\s+injured\b",
        r"\bno\s+visible\s+(?:injuries|injury)\b",
        r"\bno\s+casualties\b",
    ]
    if any(re.search(pat, text_lower) for pat in negation_patterns):
        return "explicitly_absent", []

    # 2. Check uncertainty / possible injury
    uncertainty_patterns = [
        r"\b(?:may|might|could)\s+be\s+(?:injured|hurt|bleeding|unconscious)\b",
        r"\bthinks?\s+(?:someone|person|somebody)\s+(?:may|might|could)?\s*(?:be\s+)?(?:hurt|injured)\b",
        r"\blooks\s+like\s+(?:someone|a\s+person)\s+(?:is|may\s+be)\s+(?:injured|hurt)\b",
        r"\bnot\s+sure\s+if\s+(?:anyone|someone|anybody)\s+(?:is|was)\s+(?:injured|hurt)\b",
        r"\bpossible\s+(?:injuries|injury)\b",
        r"\bpossibly\s+injured\b",
    ]
    if any(re.search(pat, text_lower) for pat in uncertainty_patterns):
        return "possible", ["injured"]

    # 3. Check confirmed caller-reported injury keywords
    injury_kws = extract_keyword_matches(doc, INJURY_KEYWORDS)
    if injury_kws:
        return "confirmed_caller_reported", injury_kws

    return "not determined", []


def extract_accident_status(text: str) -> str:
    """Determine accident status: reported, uncertain, explicitly_not_reported, or not determined."""
    text_lower = text.lower()

    # 1. Negation
    neg_patterns = [
        r"\bno\s+(?:collision|accident|crash)\s+(?:has\s+been|was)?\s*reported\b",
        r"\bno\s+(?:collision|accident|crash)\s+occurred\b",
        r"\bno\s+visible\s+accident\b",
        r"\bnot\s+an\s+accident\b",
        r"\bno\s+collision\b",
    ]
    if any(re.search(pat, text_lower) for pat in neg_patterns):
        return "explicitly_not_reported"

    # 2. Uncertainty
    unc_patterns = [
        r"\bnot\s+sure\s+whether\s+(?:an?\s+)?(?:accident|collision|crash)\s+happened\b",
        r"\bnot\s+sure\s+if\s+(?:an?\s+)?(?:accident|collision|crash)\s+occurred\b",
        r"\buncertain\s+(?:whether|if)\s+(?:an?\s+)?(?:accident|collision)\b",
        r"\bunclear\s+if\s+(?:an?\s+)?(?:accident|collision)\b",
    ]
    if any(re.search(pat, text_lower) for pat in unc_patterns):
        return "uncertain"

    # 3. Reported
    rep_patterns = [
        r"\b(?:collided|collision|crashed|crash|accident|hit\s+a|overturned)\b",
    ]
    if any(re.search(pat, text_lower) for pat in rep_patterns):
        return "reported"

    return "not determined"


def extract_ambulance_request(text: str) -> Optional[bool]:
    """Check if ambulance is requested, explicitly not requested, or unmentioned."""
    text_lower = text.lower()
    if re.search(
        r"\b(?:no\s+ambulance\s+(?:was\s+)?(?:requested|needed)|did\s+not\s+request\s+(?:an\s+)?ambulance|don't\s+need\s+(?:an\s+)?ambulance)\b",
        text_lower,
    ):
        return False
    if re.search(
        r"\b(?:ambulance\s+(?:is\s+)?needed|send\s+(?:an\s+)?ambulance|need\s+(?:an\s+)?ambulance|call\s+(?:an\s+)?ambulance|requesting\s+(?:an\s+)?ambulance)\b",
        text_lower,
    ):
        return True
    return None


def extract_emergency_assistance_request(text: str) -> Optional[bool]:
    """Check if emergency assistance was requested or negated."""
    text_lower = text.lower()
    if re.search(
        r"\b(?:no\s+emergency\s+assistance|do\s+not\s+send\s+(?:emergency\s+)?assistance|no\s+help\s+needed)\b",
        text_lower,
    ):
        return False
    if re.search(
        r"\b(?:emergency\s+assistance|please\s+send\s+emergency\s+assistance|send\s+help\s+immediately|need\s+emergency\s+assistance)\b",
        text_lower,
    ):
        return True
    return None


def extract_keyword_matches(doc, keywords: List[str]) -> List[str]:
    """Extract keywords with negation filtering."""
    matched = set()
    text_lower = doc.text.lower()

    for kw in keywords:
        pattern = rf"\b{re.escape(kw)}\b"
        for m in re.finditer(pattern, text_lower):
            start_char = m.start()
            # Find token corresponding to start_char
            token = None
            for t in doc:
                if t.idx <= start_char < t.idx + len(t.text):
                    token = t
                    break
            if token and not is_negated(token):
                # Double-check preceding words in text slice
                preceding = text_lower[max(0, start_char - 20):start_char]
                if not any(neg in preceding.split() for neg in NEGATION_TOKENS):
                    matched.add(kw)

    return sorted(list(matched))


def extract_caller_info(text: str, call_id: str = "call_sim") -> CallerExtraction:
    """Run full extraction pipeline on emergency caller description."""
    # Handle empty, None, or whitespace input safely
    if not text or not text.strip():
        return CallerExtraction(
            call_id=call_id,
            location=None,
            vehicle_count=None,
            vehicle_types=[],
            people_involved=None,
            injury_keywords=[],
            hazard_keywords=[],
            urgency_keywords=[],
            raw_text="",
            data_source="SIMULATED",
            injury_status="not determined",
            accident_status="not determined",
            ambulance_requested=None,
            emergency_assistance_requested=None,
            pedestrians_mentioned=False,
            passengers_mentioned=False,
            contradictions=[],
            uncertainty_detected=False,
            negation_detected=False,
        )

    nlp = get_spacy_nlp()
    # Strip call transcript header if present
    content = text
    if "Caller:" in text:
        content = text.split("Caller:", 1)[1].strip()
    content = content.strip("\"' \n")

    doc = nlp(content)

    # Check contradictions
    contradictions = detect_contradictions(content)

    location = extract_location(doc)
    vehicle_count = extract_vehicle_count(doc, contradictions=contradictions)
    vehicle_types = extract_vehicle_types(doc)
    people_involved = extract_people_count(doc)

    # Injury status and keywords
    injury_status, injury_keywords = extract_injury_status(content, doc)

    hazard_keywords = extract_keyword_matches(doc, HAZARD_KEYWORDS)
    urgency_keywords = extract_keyword_matches(doc, URGENCY_KEYWORDS)

    accident_status = extract_accident_status(content)
    ambulance_requested = extract_ambulance_request(content)
    emergency_assistance_requested = extract_emergency_assistance_request(content)

    text_lower = content.lower()
    pedestrians_mentioned = bool(re.search(r"\bpedestrians?\b", text_lower))
    passengers_mentioned = bool(re.search(r"\bpassengers?\b", text_lower))

    uncertainty_detected = bool(
        injury_status == "possible"
        or accident_status == "uncertain"
        or any(term in text_lower for term in ["not sure", "uncertain", "looks like", "might be", "may be", "possibly", "thinks"])
    )

    negation_detected = bool(
        injury_status == "explicitly_absent"
        or accident_status == "explicitly_not_reported"
        or ambulance_requested is False
        or emergency_assistance_requested is False
        or any(tok in text_lower.split() for tok in NEGATION_TOKENS)
    )

    return CallerExtraction(
        call_id=call_id,
        location=location,
        vehicle_count=vehicle_count,
        vehicle_types=vehicle_types,
        people_involved=people_involved,
        injury_keywords=injury_keywords,
        hazard_keywords=hazard_keywords,
        urgency_keywords=urgency_keywords,
        raw_text=content,
        data_source="SIMULATED",
        injury_status=injury_status,
        accident_status=accident_status,
        ambulance_requested=ambulance_requested,
        emergency_assistance_requested=emergency_assistance_requested,
        pedestrians_mentioned=pedestrians_mentioned,
        passengers_mentioned=passengers_mentioned,
        contradictions=contradictions,
        uncertainty_detected=uncertainty_detected,
        negation_detected=negation_detected,
    )


def evaluate_caller_extraction(
    samples_dir: Optional[Path] = None,
    expected_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Evaluate extraction accuracy against expected.json answer key.

    Returns:
        Dict of per-field precision, recall, and F1 metrics.
    """
    s_dir = samples_dir or (REPO_ROOT / "data" / "sample_calls")
    exp_file = expected_path or (s_dir / "expected.json")

    if not exp_file.exists():
        raise FileNotFoundError(f"Expected answer key not found: {exp_file}")

    with open(exp_file) as f:
        expected_all = json.load(f)

    field_stats = {
        "location": {"tp": 0, "fp": 0, "fn": 0},
        "vehicle_count": {"tp": 0, "fp": 0, "fn": 0},
        "vehicle_types": {"tp": 0, "fp": 0, "fn": 0},
        "people_involved": {"tp": 0, "fp": 0, "fn": 0},
        "injury_keywords": {"tp": 0, "fp": 0, "fn": 0},
        "hazard_keywords": {"tp": 0, "fp": 0, "fn": 0},
        "urgency_keywords": {"tp": 0, "fp": 0, "fn": 0},
    }

    sample_results = []

    for call_id, expected in expected_all.items():
        call_file = s_dir / f"{call_id}.txt"
        if not call_file.exists():
            continue

        text = call_file.read_text(encoding="utf-8")
        extracted = extract_caller_info(text, call_id=call_id)

        sample_eval = {"call_id": call_id, "fields": {}}

        # 1. Location (partial/substring or exact)
        exp_loc = expected.get("location")
        act_loc = extracted.location
        if exp_loc is None and act_loc is None:
            field_stats["location"]["tp"] += 1
            loc_status = "CORRECT_NONE"
        elif exp_loc and act_loc and (act_loc.lower() in exp_loc.lower() or exp_loc.lower() in act_loc.lower()):
            field_stats["location"]["tp"] += 1
            loc_status = "TP"
        elif act_loc and not exp_loc:
            field_stats["location"]["fp"] += 1
            loc_status = "FP"
        elif exp_loc and not act_loc:
            field_stats["location"]["fn"] += 1
            loc_status = "FN"
        else:
            field_stats["location"]["fp"] += 1
            loc_status = "MISMATCH"
        sample_eval["fields"]["location"] = {"exp": exp_loc, "act": act_loc, "status": loc_status}

        # 2. Vehicle count
        exp_vc = expected.get("vehicle_count")
        act_vc = extracted.vehicle_count
        if exp_vc == act_vc:
            field_stats["vehicle_count"]["tp"] += 1
            vc_status = "TP"
        elif act_vc is not None and exp_vc is None:
            field_stats["vehicle_count"]["fp"] += 1
            vc_status = "FP"
        elif act_vc is None and exp_vc is not None:
            field_stats["vehicle_count"]["fn"] += 1
            vc_status = "FN"
        else:
            field_stats["vehicle_count"]["fp"] += 1
            vc_status = "MISMATCH"
        sample_eval["fields"]["vehicle_count"] = {"exp": exp_vc, "act": act_vc, "status": vc_status}

        # 3. People count
        exp_pc = expected.get("people_involved")
        act_pc = extracted.people_involved
        if exp_pc == act_pc:
            field_stats["people_involved"]["tp"] += 1
            pc_status = "TP"
        elif act_pc is not None and exp_pc is None:
            field_stats["people_involved"]["fp"] += 1
            pc_status = "FP"
        elif act_pc is None and exp_pc is not None:
            field_stats["people_involved"]["fn"] += 1
            pc_status = "FN"
        else:
            field_stats["people_involved"]["fp"] += 1
            pc_status = "MISMATCH"
        sample_eval["fields"]["people_involved"] = {"exp": exp_pc, "act": act_pc, "status": pc_status}

        # 4. Set fields: vehicle_types, injury_keywords, hazard_keywords, urgency_keywords
        for set_field in ["vehicle_types", "injury_keywords", "hazard_keywords", "urgency_keywords"]:
            exp_set = set(expected.get(set_field, []))
            act_set = set(getattr(extracted, set_field, []))

            tp = len(exp_set & act_set)
            fp = len(act_set - exp_set)
            fn = len(exp_set - act_set)

            # If both empty, perfect match
            if not exp_set and not act_set:
                tp = 1

            field_stats[set_field]["tp"] += tp
            field_stats[set_field]["fp"] += fp
            field_stats[set_field]["fn"] += fn

            sample_eval["fields"][set_field] = {
                "exp": sorted(list(exp_set)),
                "act": sorted(list(act_set)),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }

        sample_results.append(sample_eval)

    # Compute Precision, Recall, F1
    metrics = {}
    for field, counts in field_stats.items():
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        metrics[field] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }

    return {
        "metrics": metrics,
        "sample_evaluations": sample_results,
        "sample_count": len(sample_results),
        "note": "Evaluation conducted on 8 simulated emergency call texts. Tiny sample size for initial verification.",
    }
