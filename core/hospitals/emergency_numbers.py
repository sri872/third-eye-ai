"""Verified emergency services and national contact numbers for Project Third-Eye.

All emergency contacts are verified against official Government of India portals.
Never invents, alters, or simulates emergency dispatch telephone numbers.
"""

from typing import Dict, List, Optional


EMERGENCY_SERVICES: List[Dict[str, str]] = [
    {
        "number": "112",
        "service": "Universal Emergency Response (ERSS)",
        "description": "Unified single emergency response number across India for Police, Fire, and Ambulance.",
        "jurisdiction": "National (All States & UTs)",
        "official_source": "Emergency Response Support System (ERSS), Ministry of Home Affairs, Govt. of India",
        "source_url": "https://112.gov.in",
        "page_fetched_url": "https://www.india.gov.in/directory/helpline",
        "quoted_line": "Integrated Helpline (Emergency Response Support System (ERSS) Police, Fire & Rescue, Health services etc)",
        "verification_status": "VERIFIED",
        "verification_note": "Direct portal https://112.gov.in uses NIC self-signed certificate; verified via National Portal of India directory.",
    },
    {
        "number": "108",
        "service": "Emergency Medical Transport (Ambulance)",
        "description": "Toll-free 24x7 emergency medical transport for trauma, critical care, and accident victims.",
        "jurisdiction": "Punjab & Most Indian States (National Health Mission)",
        "official_source": "National Health Mission, Ministry of Health and Family Welfare, Govt. of India",
        "source_url": "https://nhm.gov.in",
        "page_fetched_url": "https://www.india.gov.in/directory/helpline",
        "quoted_line": "Dial 108 is predominantly an emergency response system primarily designed to attend trauma",
        "verification_status": "VERIFIED",
        "verification_note": "Verified from National Portal of India directory linked official NHM National Ambulance Service portal.",
    },
    {
        "number": "102",
        "service": "Free Patient Transport Ambulance (JSSK)",
        "description": "Free referral and transport for pregnant women, infants, and non-trauma hospital transfers.",
        "jurisdiction": "National Health Mission, Govt. of India",
        "official_source": "Janani Shishu Suraksha Karyakram (JSSK), Ministry of Health and Family Welfare",
        "source_url": "https://nhm.gov.in",
        "page_fetched_url": "https://www.india.gov.in/directory/helpline",
        "quoted_line": "National Ambulance Service (NAS). Emergency Response System / Pregnant Women and Children",
        "verification_status": "VERIFIED",
        "verification_note": "Verified from National Portal of India directory entry for National Ambulance Service (102).",
    },
    {
        "number": "100",
        "service": "Police Emergency",
        "description": "Immediate law enforcement, traffic incident reporting, and accident site control.",
        "jurisdiction": "National (State Police Departments)",
        "official_source": "National Portal of India, Govt. of India",
        "source_url": "https://www.india.gov.in",
        "page_fetched_url": "https://www.india.gov.in/directory/helpline",
        "quoted_line": "Police Helpline",
        "verification_status": "VERIFIED",
        "verification_note": "Verified from National Portal of India directory helpline list.",
    },
    {
        "number": "101",
        "service": "Fire and Rescue",
        "description": "Fire hazard suppression, vehicle extrication, and chemical/fuel spill containment.",
        "jurisdiction": "National (State Fire & Rescue Services)",
        "official_source": "National Portal of India, Govt. of India",
        "source_url": "https://www.india.gov.in",
        "page_fetched_url": "https://www.india.gov.in/directory/helpline",
        "quoted_line": "Fire Helpline",
        "verification_status": "VERIFIED",
        "verification_note": "Verified from National Portal of India directory helpline list.",
    },
]


def get_emergency_numbers(
    urgency: Optional[str] = None,
    hazards: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """Return tailored emergency contact list based on incident triage.
    
    112 and 108 are ALWAYS prioritized for accident response.
    101 is appended if fire/fuel/chemical hazards are identified.
    100 is appended for traffic control and law enforcement.
    """
    selected: List[Dict[str, str]] = []
    
    # Primary medical and universal emergency dispatch
    for service in EMERGENCY_SERVICES:
        if service["number"] in ("112", "108"):
            selected.append(service)

    # Fire emergency if hazards include fire, smoke, fuel spill
    if hazards:
        hazard_text = " ".join(hazards).lower()
        if any(h in hazard_text for h in ["fire", "fuel", "gas", "smoke", "chemical", "spill"]):
            for service in EMERGENCY_SERVICES:
                if service["number"] == "101" and service not in selected:
                    selected.append(service)

    # Police for traffic control / accident management
    for service in EMERGENCY_SERVICES:
        if service["number"] == "100" and service not in selected:
            selected.append(service)

    return selected
