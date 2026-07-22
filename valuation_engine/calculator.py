"""Calculation layer for deterministic vehicle valuation."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any

from .loaders import VehicleDataRepository
from .utils import round_aed

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DecodedVehicle:
    """Attributes produced by the VIN decoder before valuation starts."""

    make: str
    model: str
    trim: str
    model_year: int
    body_type: str
    regional_specs: str


@dataclass(frozen=True)
class ValuationBreakdown:
    """JSON-serializable valuation output."""

    base_price: int
    vehicle_age: int
    age_factor: float
    model_factor: float
    trim_factor: float
    body_factor: float
    regional_spec_factor: float
    market_factor: float
    final_value: int
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary suitable for JSON serialization."""

        data = asdict(self)
        if not self.warnings:
            data.pop("warnings")
        return data


class VehicleValuationCalculator:
    """Coordinates lookups and applies the CSV-defined valuation formula."""

    def __init__(self, repository: VehicleDataRepository, current_year: int | None = None) -> None:
        self.repository = repository
        self.current_year = current_year or date.today().year

    def value(self, vehicle: DecodedVehicle) -> ValuationBreakdown:
        """Calculate a deterministic valuation for decoded VIN attributes."""

        logger.info("Starting valuation for decoded vehicle: %s", vehicle)
        vehicle_age = max(self.current_year - int(vehicle.model_year), 0)
        warnings: list[str] = []

        base_price = self.repository.lookup_base_price(
            make=vehicle.make,
            model=vehicle.model,
            trim=vehicle.trim,
            year=vehicle.model_year,
            body_type=vehicle.body_type,
            regional_specs=vehicle.regional_specs,
        )
        warnings.extend(base_price.warnings)

        age_factor = self.repository.lookup_age_factor(vehicle_age)
        model_factor = self.repository.lookup_model_retention_factor(vehicle.model)
        trim_factor = self.repository.lookup_trim_factor(vehicle.trim)
        body_factor = self.repository.lookup_body_type_factor(vehicle.body_type)
        regional_factor = self.repository.lookup_regional_spec_factor(vehicle.regional_specs)
        market_factor = self.repository.lookup_market_demand_factor(vehicle.model)

        for result in (age_factor, model_factor, trim_factor, body_factor, regional_factor, market_factor):
            warnings.extend(result.warnings)

        raw_value = (
            base_price.base_price
            * age_factor.value
            * model_factor.value
            * trim_factor.value
            * body_factor.value
            * regional_factor.value
            * market_factor.value
        )
        final_value = round_aed(raw_value)
        logger.info("Final value resolved: %s AED", final_value)

        return ValuationBreakdown(
            base_price=round_aed(base_price.base_price),
            vehicle_age=vehicle_age,
            age_factor=age_factor.value,
            model_factor=model_factor.value,
            trim_factor=trim_factor.value,
            body_factor=body_factor.value,
            regional_spec_factor=regional_factor.value,
            market_factor=market_factor.value,
            final_value=final_value,
            warnings=warnings,
        )
