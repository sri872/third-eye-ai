"""Regression tests for geometry-only vehicle/plate association.

The association API must never select a plate from text or OCR confidence:
candidate regions come from the detector and are matched to a vehicle using
containment and centre geometry only.
"""

from core.cv.plate_detector import (
    AMBIGUOUS,
    ASSOCIATED,
    NO_PLATE_REGION,
    UNASSOCIATED,
    PlateAssociation,
    PlateRegion,
    associate_plate_regions,
)
from core.cv.vehicle_report import plate_detection_status_for_association


def _region(bbox, *, method="TEST", score=0.5, ocr_confidence=None) -> PlateRegion:
    return PlateRegion(
        bbox_local=bbox,
        bbox_original=bbox,
        heuristic_score=score,
        character_candidates=6,
        aspect_ratio=3.0,
        edge_density=0.1,
        method=method,
        ocr_confidence=ocr_confidence,
    )


class TestPlateAssociationGeometry:
    def test_single_contained_region_is_associated(self):
        vehicle = (100, 100, 500, 400)
        region = _region((200, 220, 320, 260))
        association = associate_plate_regions([region], vehicle, 640, 480)

        assert association.status == ASSOCIATED
        assert association.region is region
        assert association.bbox == (200, 220, 320, 260)
        assert association.candidates == (region,)

    def test_overlapping_regions_with_no_clear_winner_are_ambiguous(self):
        vehicle = (100, 100, 500, 400)
        first = _region((200, 220, 320, 260), score=0.9, ocr_confidence=0.05)
        second = _region((205, 225, 325, 265), score=0.1, ocr_confidence=0.99)
        association = associate_plate_regions([first, second], vehicle, 640, 480)

        assert association.status == AMBIGUOUS
        assert association.region is None
        assert association.bbox is None
        assert set(association.candidates) == {first, second}
        assert "no clear" in association.reason.lower()

    def test_region_outside_vehicle_is_unassociated(self):
        vehicle = (100, 100, 500, 400)
        region = _region((10, 10, 100, 50))
        association = associate_plate_regions([region], vehicle, 640, 480)

        assert association.status == UNASSOCIATED
        assert association.region is None

    def test_no_candidates_matches_existing_no_plate_status(self):
        association = associate_plate_regions((), (100, 100, 500, 400), 640, 480)
        assert association.status == NO_PLATE_REGION
        assert association.status == "NO PLATE REGION DETECTED"
        assert association.region is None

    def test_statuses_available_on_frozen_dataclass(self):
        association = PlateAssociation(status=ASSOCIATED)
        assert PlateAssociation.ASSOCIATED == ASSOCIATED
        assert PlateAssociation.AMBIGUOUS == AMBIGUOUS
        assert PlateAssociation.UNASSOCIATED == UNASSOCIATED
        assert PlateAssociation.NO_PLATE_REGION == NO_PLATE_REGION
        assert association.candidates == ()
        assert association.reason == ""

    def test_invalid_vehicle_geometry_never_auto_selects(self):
        region = _region((200, 220, 320, 260))
        assert associate_plate_regions(
            [region], (300, 200, 100, 400), 640, 480
        ).status == UNASSOCIATED
        assert associate_plate_regions([region], None, 640, 480).status == UNASSOCIATED

    def test_association_status_maps_to_existing_string_field(self):
        assert plate_detection_status_for_association(
            associate_plate_regions([_region((200, 220, 320, 260))], (100, 100, 500, 400), 640, 480)
        ) == "PLATE REGION DETECTED"
        assert plate_detection_status_for_association(
            associate_plate_regions(
                [_region((200, 220, 320, 260)), _region((205, 225, 325, 265))],
                (100, 100, 500, 400),
                640,
                480,
            )
        ) == "PLATE REGION DETECTED — AMBIGUOUS ASSOCIATION"
        assert plate_detection_status_for_association(
            associate_plate_regions((), (100, 100, 500, 400), 640, 480)
        ) == "NO PLATE REGION DETECTED"
