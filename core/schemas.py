"""Shared data models and schemas for Project Third-Eye.

All records MUST specify `data_source` as either "REAL" or "SIMULATED".
Confidence fields are always optional and must never be invented.
"""

from datetime import datetime
from typing import List, Literal, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Strict data source literal type
DataSourceType = Literal["REAL", "SIMULATED"]


class BaseThirdEyeModel(BaseModel):
    """Base model enforcing common configuration and data_source validation."""
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    data_source: DataSourceType = Field(
        ...,
        description="Must be explicitly declared as 'REAL' or 'SIMULATED'."
    )


class Detection(BaseThirdEyeModel):
    """Computer vision bounding box detection schema."""
    detection_id: str = Field(..., min_length=1, description="Unique detection identifier")
    label: str = Field(..., min_length=1, description="Detected object class (e.g., vehicle, person)")
    bbox: List[float] = Field(
        ...,
        description="Bounding box coordinates [xmin, ymin, xmax, ymax]",
        min_length=4,
        max_length=4
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional model confidence score [0.0 - 1.0]. Never fabricate."
    )
    timestamp: Optional[datetime] = Field(
        default=None,
        description="Timestamp of detection"
    )

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v: List[float]) -> List[float]:
        xmin, ymin, xmax, ymax = v
        if xmin > xmax or ymin > ymax:
            raise ValueError(f"Invalid bounding box coordinates: xmin ({xmin}) > xmax ({xmax}) or ymin ({ymin}) > ymax ({ymax})")
        return v


class VehicleReport(BaseThirdEyeModel):
    """Forensic report on an individual vehicle involved in an event."""
    vehicle_id: str = Field(..., min_length=1)
    license_plate: Optional[str] = Field(default=None)
    plate_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional OCR plate reading confidence [0.0 - 1.0]. Never fabricate."
    )
    plate_detection_status: Optional[str] = Field(
        default=None,
        description="Plate-localisation status; a detected region is not a verified registration."
    )
    plate_bbox: Optional[Tuple[int, int, int, int]] = Field(
        default=None,
        description="Detected plate region in original-image pixels, if available."
    )
    plate_detector: Optional[str] = Field(
        default=None,
        description="Method used to localise the plate region."
    )
    vehicle_type: Optional[str] = Field(default=None)
    color: Optional[str] = Field(
        default=None,
        description="Dominant color name: white, black, grey, red, blue, green, yellow, other"
    )
    color_share: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional dominant color pixel share measure [0.0 - 1.0]."
    )
    damage_severity: Optional[str] = Field(
        default=None,
        description="Severity: 'minor', 'moderate', 'severe', 'totaled'"
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional detector confidence score. Never fabricate."
    )
    timestamp: Optional[datetime] = Field(default=None)


class OccupantReport(BaseThirdEyeModel):
    """Assessment of vehicle occupants and visible pedestrians."""
    occupant_id: str = Field(..., min_length=1)
    vehicle_id: Optional[str] = Field(default=None)
    bbox: Optional[List[float]] = Field(default=None, description="[xmin, ymin, xmax, ymax] bounding box")
    estimated_count: int = Field(default=1, ge=0)
    injury_severity: Optional[str] = Field(
        default="not determined",
        description="Triage level: 'uninjured', 'minor', 'critical', 'fatal', 'not determined'"
    )
    consciousness: Optional[str] = Field(
        default="unknown",
        description="State of consciousness: 'conscious', 'unconscious', 'unknown'"
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional confidence score. Never fabricate."
    )
    overlap_ratio: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Overlap intersection ratio with assigned vehicle"
    )
    persons_visible_count: Optional[int] = Field(default=None, ge=0)
    persons_near_vehicles_count: Optional[int] = Field(default=None, ge=0)
    limitation_note: str = Field(
        default="Visible people only. Occupants inside vehicles cannot be reliably counted from a single image."
    )
    timestamp: Optional[datetime] = Field(default=None)


FOUR_TIER_ASSESSMENT_CATEGORIES = (
    "ACCIDENT EVIDENCE DETECTED",
    "POSSIBLE ACCIDENT",
    "INCONCLUSIVE",
    "NO CLEAR ACCIDENT EVIDENCE DETECTED",
)

STILL_IMAGE_SCENE_STATUS = "NOT DETERMINED (out-of-domain still image)"


class EvidenceAssessment(BaseThirdEyeModel):
    """Shared four-tier accident evidence assessment.

    This is the single contract consumed by Accident Check, Incident Report and
    Dispatch. It is deliberately distinct from the raw classifier verdict kept
    in ``AccidentEvent.verdict``: a model prediction is diagnostic output, not a
    confirmed real-world collision. Never fabricate evidence or calibrated
    confidence here.
    """

    assessment_category: Literal[
        "ACCIDENT EVIDENCE DETECTED",
        "POSSIBLE ACCIDENT",
        "INCONCLUSIVE",
        "NO CLEAR ACCIDENT EVIDENCE DETECTED",
    ]
    model_verdict: Optional[str] = Field(
        default=None,
        description="Original unmodified classifier verdict (diagnostic only).",
    )
    model_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Uncalibrated classifier score, never presented as calibrated probability.",
    )
    media_type: Literal["image", "video", "unknown"] = "unknown"
    domain_compatibility: str = "NOT DETERMINED"
    scene_status: str = "NOT DETERMINED"
    evidence: List[str] = Field(default_factory=list)
    contradictory_evidence: List[str] = Field(default_factory=list)
    evidence_sources: List[str] = Field(default_factory=list)
    uncertainty: str = "HIGH"
    human_review_required: bool = True
    reasoning: str = "NOT DETERMINED — insufficient evidence for an evidence assessment."
    temporal_evidence_available: bool = False
    data_source: DataSourceType = "REAL"

    @field_validator("human_review_required")
    @classmethod
    def _review_always_required(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError("human_review_required must remain True for accident assessments")
        return True

    def as_dict(self) -> dict:
        """Return a JSON-safe copy for the report, dispatch and database layers."""
        return self.model_dump(mode="json")


class AccidentEvent(BaseThirdEyeModel):
    """Core accident incident record."""
    event_id: str = Field(..., min_length=1)
    timestamp: datetime = Field(..., description="Timestamp of accident occurrence/report")
    location_name: str = Field(..., min_length=1, description="Intersection or road descriptor")
    latitude: float = Field(..., ge=-90.0, le=90.0, description="WGS84 latitude")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="WGS84 longitude")
    severity: str = Field(..., min_length=1, description="Severity: 'none', 'minor', 'moderate', 'high_probability', 'uncertain', 'critical'")
    vehicles_involved: int = Field(default=1, ge=0)
    casualties_estimated: int = Field(default=0, ge=0)
    status: str = Field(default="reported", description="'reported', 'dispatched', 'en_route', 'resolved', 'normal_flow'")
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional trained model confidence score."
    )
    method: Optional[str] = Field(
        default="TRAINED_MODEL",
        description="'TRAINED_MODEL' or 'RULE_BASED_HEURISTIC'"
    )
    score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Uncalibrated model probability score on this dataset."
    )
    verdict: Optional[str] = Field(
        default=None,
        description="Verdict: 'LIKELY ACCIDENT', 'UNCERTAIN', 'NO ACCIDENT SIGNAL'"
    )
    persons_near_vehicles_count: Optional[int] = Field(
        default=0,
        ge=0,
        description="Plain count of people whose bounding box touches or intersects a vehicle box."
    )
    reasons: Optional[str] = Field(
        default=None,
        description="Forensic explanation of triggered rules or geometric feature highlights."
    )
    evidence_assessment: Optional[EvidenceAssessment] = Field(
        default=None,
        description="Structured four-tier evidence assessment; the raw classifier verdict stays in `verdict`.",
    )


class Hospital(BaseThirdEyeModel):
    """Hospital facility capacity and trauma level schema."""
    hospital_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    total_beds: Optional[int] = Field(default=None, ge=0)
    available_beds: Optional[int] = Field(default=None, ge=0)
    icu_beds_available: Optional[int] = Field(default=None, ge=0)
    trauma_center_level: Optional[int] = Field(default=None, ge=1, le=5)
    contact_number: Optional[str] = Field(default=None)
    address: Optional[str] = Field(default=None)
    emergency: Optional[str] = Field(default=None, description="'yes', 'no', or None")
    specialties: Optional[str] = Field(default=None)
    attribution: str = Field(default="© OpenStreetMap contributors")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    excluded_from_dispatch_reason: Optional[str] = Field(
        default=None,
        description="Reason if excluded from emergency dispatch destination"
    )
    source_facility_category: Optional[str] = Field(
        default=None,
        description="Facility category derived from source OSM tags/name at ingestion time"
    )
    source_dispatch_eligible: Optional[bool] = Field(
        default=None,
        description="Dispatch eligibility derived from source OSM tags/name; None means not classified"
    )

    @model_validator(mode="after")
    def validate_beds(self) -> "Hospital":
        if self.available_beds is not None and self.total_beds is not None:
            if self.available_beds > self.total_beds:
                raise ValueError(
                    f"available_beds ({self.available_beds}) cannot exceed total_beds ({self.total_beds})"
                )
        if self.icu_beds_available is not None and self.available_beds is not None:
            if self.icu_beds_available > self.available_beds:
                raise ValueError(
                    f"icu_beds_available ({self.icu_beds_available}) cannot exceed available_beds ({self.available_beds})"
                )
        return self


class Ambulance(BaseThirdEyeModel):
    """Ambulance unit operational state."""
    ambulance_id: str = Field(..., min_length=1)
    call_sign: str = Field(..., min_length=1)
    status: str = Field(
        default="available",
        description="'available', 'dispatched', 'in_transit', 'busy', 'maintenance'"
    )
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    equipment_level: str = Field(
        default="Advanced",
        description="'Basic', 'Advanced', 'Critical Care'"
    )
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    base_hospital: Optional[str] = Field(default=None, description="Home base hospital name")
    hospital_osm_id: Optional[int] = Field(default=None, description="OSM node ID of base hospital")


class RouteResult(BaseThirdEyeModel):
    """Routing solution from ambulance location to accident scene and target hospital."""
    route_id: str = Field(..., min_length=1)
    ambulance_id: str = Field(..., min_length=1)
    hospital_id: str = Field(..., min_length=1)
    accident_id: str = Field(..., min_length=1)
    distance_km: float = Field(..., ge=0.0)
    estimated_time_minutes: float = Field(..., ge=0.0)
    route_waypoints: Optional[List[List[float]]] = Field(
        default=None,
        description="List of [latitude, longitude] pairs"
    )
    traffic_condition: Optional[str] = Field(
        default=None,
        description="'clear', 'moderate', 'congested'"
    )
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    leg1_waypoints: Optional[List[List[float]]] = Field(
        default=None,
        description="List of [latitude, longitude] pairs for leg 1 (ambulance -> accident)"
    )
    leg2_waypoints: Optional[List[List[float]]] = Field(
        default=None,
        description="List of [latitude, longitude] pairs for leg 2 (accident -> hospital)"
    )
    ambulance_eta_minutes: Optional[float] = Field(default=None, ge=0.0)
    hospital_transit_minutes: Optional[float] = Field(default=None, ge=0.0)
    straight_line_km: Optional[float] = Field(default=None, ge=0.0)
    driving_vs_straight_ratio: Optional[float] = Field(default=None, ge=0.0)
    chosen_ambulance: Optional[dict] = None
    chosen_hospital: Optional[dict] = None
    alternative_ambulances: List[dict] = Field(default_factory=list)
    alternative_hospitals: List[dict] = Field(default_factory=list)
    attribution: str = Field(default="© OpenStreetMap contributors")
    disclaimer: str = Field(default="Estimated from OSM speed data, no live traffic")
    coverage_note: str = Field(
        default="Routing covers this corridor only (Phagwara corridor: 75.69-75.80°E, 31.20-31.27°N)"
    )


class AuditReport(BaseThirdEyeModel):
    """Forensic audit summary generated post-incident."""
    audit_id: str = Field(..., min_length=1)
    accident_id: str = Field(..., min_length=1)
    generated_at: datetime = Field(..., description="Audit generation timestamp")
    collision_analysis: Optional[str] = Field(default=None)
    liability_assessment: Optional[str] = Field(default=None)
    regulatory_compliance: Optional[str] = Field(default=None)
    timeline_summary: Optional[str] = Field(default=None)
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional confidence score. Never fabricate."
    )


class CallerExtraction(BaseThirdEyeModel):
    """Structured entity and keyword extraction from emergency caller text."""
    call_id: str = Field(..., min_length=1)
    location: Optional[str] = Field(default=None, description="Location mentioned by caller, or None if missing")
    vehicle_count: Optional[int] = Field(default=None, ge=0, description="Vehicle count mentioned by caller, or None")
    vehicle_types: List[str] = Field(default_factory=list, description="Vehicle types mentioned")
    people_involved: Optional[int] = Field(default=None, ge=0, description="People count mentioned, or None")
    injury_keywords: List[str] = Field(default_factory=list, description="Injury symptoms/terms extracted")
    hazard_keywords: List[str] = Field(default_factory=list, description="Fire or environmental hazards extracted")
    urgency_keywords: List[str] = Field(default_factory=list, description="Urgency descriptors extracted")
    raw_text: str = Field(default="", description="Original emergency call text")
    data_source: Literal["REAL", "SIMULATED"] = "SIMULATED"
    # Enhanced NLP capability fields (backward-compatible defaults)
    injury_status: Optional[str] = Field(
        default="not determined",
        description="confirmed_caller_reported, possible, explicitly_absent, or not determined",
    )
    accident_status: Optional[str] = Field(
        default="not determined",
        description="reported, uncertain, explicitly_not_reported, not determined",
    )
    ambulance_requested: Optional[bool] = Field(
        default=None,
        description="True if requested, False if explicitly absent, None if not mentioned",
    )
    emergency_assistance_requested: Optional[bool] = Field(
        default=None,
        description="True if requested, False if explicitly absent, None if not mentioned",
    )
    pedestrians_mentioned: bool = Field(
        default=False,
        description="Whether pedestrians were mentioned",
    )
    passengers_mentioned: bool = Field(
        default=False,
        description="Whether passengers were mentioned",
    )
    contradictions: List[str] = Field(
        default_factory=list,
        description="Intra-call contradictory statements",
    )
    uncertainty_detected: bool = Field(
        default=False,
        description="Whether uncertainty/hedging was detected",
    )
    negation_detected: bool = Field(
        default=False,
        description="Whether negation was detected",
    )


class IncidentReport(BaseThirdEyeModel):
    """Fused incident report merging vision analysis and caller emergency text.

    Source values are kept verbatim in dedicated fields (image_location and
    caller_location) so two different locations are never silently merged into a
    single value. The legacy ``location`` / ``location_source`` pair is retained
    for compatibility with existing database records and consumers.
    """
    report_id: str = Field(..., min_length=1)
    image_name: Optional[str] = Field(default=None)
    call_id: Optional[str] = Field(default=None)
    timestamp: datetime = Field(...)
    verdict: str = Field(..., description="'LIKELY ACCIDENT', 'UNCERTAIN', 'NO ACCIDENT SIGNAL'")
    verdict_source: Literal["VISION", "CALLER_SIMULATED", "MISSING", "FUSED"] = "VISION"
    location: Optional[str] = Field(default=None)
    location_source: Literal["VISION", "CALLER_SIMULATED", "MISSING", "FUSED"] = "MISSING"
    vision_vehicle_count: Optional[int] = Field(default=None, ge=0)
    caller_vehicle_count: Optional[int] = Field(default=None, ge=0)
    vehicle_count_disagreement: bool = False
    vision_people_count: Optional[int] = Field(default=None, ge=0)
    caller_people_count: Optional[int] = Field(default=None, ge=0)
    people_count_disagreement: bool = False
    hazards: List[str] = Field(default_factory=list)
    hazards_source: Literal["VISION", "CALLER_SIMULATED", "MISSING", "FUSED"] = "MISSING"
    injuries: List[str] = Field(default_factory=list)
    injuries_source: Literal["VISION", "CALLER_SIMULATED", "MISSING", "FUSED"] = "MISSING"
    urgency: Optional[str] = Field(default=None)
    urgency_source: Literal["VISION", "CALLER_SIMULATED", "MISSING", "FUSED"] = "MISSING"
    summary_text: str = Field(...)
    disagreements: List[str] = Field(default_factory=list)
    data_source: Literal["REAL", "SIMULATED"] = "SIMULATED"
    # Source-separated additions. Defaults keep all older callers valid.
    image_location: Optional[str] = Field(
        default=None,
        description="Location associated with the uploaded media (demo selection for hospital lookup), never inferred from pixels.",
    )
    image_location_source: Literal["VISION", "SIMULATED", "MISSING"] = "MISSING"
    caller_location: Optional[str] = Field(
        default=None,
        description="Caller-reported location, kept verbatim from the simulated call extraction.",
    )
    location_mismatch: bool = False
    vision_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Uncalibrated REAL vision classifier score; never replaced by caller evidence.",
    )
    source_disagreement: bool = False
    source_disagreements: List[str] = Field(default_factory=list)
    # Phase 14: sampled-frame/temporal evidence from uploaded video. Defaults
    # keep every still-image and legacy caller valid.
    video_evidence: Optional[dict] = Field(
        default=None,
        description="Real frame-level and temporal evidence from sampled video frames.",
    )
    video_assessment: Optional[str] = Field(
        default=None,
        description="Explanatory video-level assessment; never overrides the vision verdict.",
    )
    video_temporal_signal: Optional[str] = Field(
        default=None,
        description="Temporal pattern label from sampled classifier scores.",
    )
    # Enhanced NLP capability fields (backward-compatible defaults)
    injury_status: Optional[str] = Field(
        default="not determined",
        description="Caller injury status: confirmed_caller_reported, possible, explicitly_absent, or not determined",
    )
    accident_status: Optional[str] = Field(
        default="not determined",
        description="Caller accident status: reported, uncertain, explicitly_not_reported, not determined",
    )
    ambulance_requested: Optional[bool] = Field(
        default=None,
        description="True if requested, False if explicitly absent, None if not mentioned",
    )
    emergency_assistance_requested: Optional[bool] = Field(
        default=None,
        description="True if requested, False if explicitly absent, None if not mentioned",
    )
    pedestrians_mentioned: bool = Field(
        default=False,
        description="Whether pedestrians were mentioned by caller",
    )
    passengers_mentioned: bool = Field(
        default=False,
        description="Whether passengers were mentioned by caller",
    )
    caller_contradictions: List[str] = Field(
        default_factory=list,
        description="Intra-call contradictory statements identified",
    )
    # The exact structured assessment produced upstream. The report never
    # recalculates the category; a missing value stays explicit.
    evidence_assessment: Optional[dict] = Field(
        default=None,
        description="Verbatim shared four-tier assessment from the accident-analysis pipeline.",
    )
