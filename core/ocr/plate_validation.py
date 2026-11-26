"""Indian License Plate Format Validation.

Validates recognized plate candidate strings against official Motor Vehicles Department
standards for Indian vehicle registration plates (HSRP and standard series).
"""

import re
from typing import Dict, Any, Tuple


INDIAN_STATE_CODES = {
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CG": "Chhattisgarh",
    "CH": "Chandigarh",
    "DD": "Daman and Diu",
    "DL": "Delhi",
    "DN": "Dadra and Nagar Haveli",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HP": "Himachal Pradesh",
    "HR": "Haryana",
    "JH": "Jharkhand",
    "JK": "Jammu and Kashmir",
    "KA": "Karnataka",
    "KL": "Kerala",
    "LA": "Ladakh",
    "LD": "Lakshadweep",
    "MH": "Maharashtra",
    "ML": "Meghalaya",
    "MN": "Manipur",
    "MP": "Madhya Pradesh",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OD": "Odisha",
    "OR": "Odisha",
    "PB": "Punjab",
    "PY": "Puducherry",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TN": "Tamil Nadu",
    "TR": "Tripura",
    "TS": "Telangana",
    "UK": "Uttarakhand",
    "UP": "Uttar Pradesh",
    "WB": "West Bengal",
}

# Standard Indian format: 2-letter State + 1-2 digit RTO + 1-3 letter Series + 4 digit number
# e.g., TS07JA0566, KA01AB1234, DL01AB1234, MH12DE1433
STD_INDIAN_PLATE_PATTERN = re.compile(
    r"^([A-Z]{2})\s*([0-9]{1,2})\s*([A-Z]{1,3})\s*([0-9]{4})$"
)

# Bharat Series: 2-digit Year (BH) 4-digit number 1-2 letters
# e.g., 22BH1234AA
BHARAT_SERIES_PATTERN = re.compile(
    r"^([0-9]{2})\s*(BH)\s*([0-9]{4})\s*([A-Z]{1,2})$"
)


def validate_indian_plate(raw_candidate: str) -> Dict[str, Any]:
    """Validate a candidate registration string against Indian registration standards.

    Args:
        raw_candidate: Raw string recognized by template matching.

    Returns:
        Dictionary with validation results:
            is_valid: bool
            normalized: formatted uppercase string without spaces
            format_type: 'STANDARD_INDIAN' or 'BHARAT_SERIES' or 'INVALID'
            state_code: 2-letter state code or 'BH'
            state_name: full state name
            rto_code: district/RTO digits
            series: alphabet series
            registration_number: 4-digit number
            notes: diagnostic text
    """
    if not raw_candidate or not isinstance(raw_candidate, str):
        return {
            "is_valid": False,
            "normalized": "",
            "format_type": "INVALID",
            "state_code": None,
            "state_name": None,
            "rto_code": None,
            "series": None,
            "registration_number": None,
            "notes": "Empty or non-string candidate text.",
        }

    # Clean characters, remove spaces and hyphens, convert to uppercase
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw_candidate).upper()

    # Check Standard Indian Pattern
    match_std = STD_INDIAN_PLATE_PATTERN.match(cleaned)
    if match_std:
        state_code, rto_str, series, num_str = match_std.groups()
        # Format RTO to 2 digits if single digit (e.g. KA1 -> KA01)
        rto_norm = rto_str.zfill(2)
        norm_text = f"{state_code}{rto_norm}{series}{num_str}"
        state_name = INDIAN_STATE_CODES.get(state_code, "Unknown State / Union Territory")
        is_known_state = state_code in INDIAN_STATE_CODES

        return {
            "is_valid": is_known_state,
            "normalized": norm_text,
            "format_type": "STANDARD_INDIAN",
            "state_code": state_code,
            "state_name": state_name,
            "rto_code": rto_norm,
            "series": series,
            "registration_number": num_str,
            "notes": f"Standard Indian registration ({state_name})",
        }

    # Check Bharat Series Pattern
    match_bh = BHARAT_SERIES_PATTERN.match(cleaned)
    if match_bh:
        yr, bh, num_str, series = match_bh.groups()
        norm_text = f"{yr}{bh}{num_str}{series}"
        return {
            "is_valid": True,
            "normalized": norm_text,
            "format_type": "BHARAT_SERIES",
            "state_code": "BH",
            "state_name": "All-India Bharat Series",
            "rto_code": yr,
            "series": series,
            "registration_number": num_str,
            "notes": f"Bharat Series registered year 20{yr}",
        }

    # If it does not match exactly, provide diagnostic notes
    return {
        "is_valid": False,
        "normalized": cleaned,
        "format_type": "INVALID",
        "state_code": None,
        "state_name": None,
        "rto_code": None,
        "series": None,
        "registration_number": None,
        "notes": f"Candidate string '{cleaned}' does not match standard Indian plate syntax.",
    }
