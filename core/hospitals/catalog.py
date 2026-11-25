"""In-memory catalog of source-derived hospital candidates.

The catalog only holds records. Ordering is performed by the deterministic rules
in :mod:`core.hospitals.ranking`; no model, score or vector index is built here.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Optional

from core.schemas import Hospital


class HospitalCatalog:
    """A straightforward collection of dispatch-eligible hospital records."""

    def __init__(self, hospitals: Optional[list[Hospital]] = None) -> None:
        self.hospitals: list[Hospital] = list(hospitals or [])

    def replace(self, hospitals: list[Hospital]) -> int:
        """Replace the catalog contents and return the number of records."""
        self.hospitals = list(hospitals)
        return len(self.hospitals)

    def add(self, hospital: Hospital) -> None:
        """Append one record to the catalog."""
        self.hospitals.append(hospital)

    def list_candidates(self, limit: Optional[int] = None) -> list[Hospital]:
        """List candidates in their source order, optionally capped by ``limit``."""
        if limit is None:
            return list(self.hospitals)
        return list(self.hospitals[: max(0, int(limit))])

    def __len__(self) -> int:
        return len(self.hospitals)

    def __iter__(self) -> Iterator[Hospital]:
        return iter(self.hospitals)
