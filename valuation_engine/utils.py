"""Shared utility functions for deterministic valuation lookup."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from difflib import SequenceMatcher
from typing import Iterable


def normalize_text(value: object) -> str:
    """Normalize a lookup key while preserving human-readable source values."""

    if value is None:
        return ""
    return " ".join(str(value).strip().casefold().split())


def closest_match(query: object, choices: Iterable[object]) -> str | None:
    """Return the highest-scoring deterministic fuzzy match from choices."""

    normalized_query = normalize_text(query)
    unique_choices = sorted({str(choice).strip() for choice in choices if str(choice).strip()})
    if not normalized_query or not unique_choices:
        return None

    scored = []
    for choice in unique_choices:
        score = SequenceMatcher(None, normalized_query, normalize_text(choice)).ratio()
        scored.append((score, normalize_text(choice), choice))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][2] if scored else None


def clamp(value: int, minimum: int, maximum: int) -> int:
    """Clamp an integer to the inclusive range."""

    return max(minimum, min(value, maximum))


def round_aed(value: float) -> int:
    """Round a currency value to the nearest AED using half-up rounding."""

    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
