"""Public facade for the Toyota valuation engine."""

from __future__ import annotations

import logging
from typing import Any, Mapping

from .calculator import DecodedVehicle, VehicleValuationCalculator
from .config import DataPaths
from .loaders import VehicleDataRepository

logger = logging.getLogger(__name__)


class ToyotaValuationEngine:
    """Toyota UAE insurance valuation engine backed entirely by CSV files."""

    def __init__(self, data_paths: DataPaths | None = None, current_year: int | None = None) -> None:
        self.data_paths = data_paths or DataPaths.toyota_defaults()
        self.repository = VehicleDataRepository(self.data_paths)
        self.calculator = VehicleValuationCalculator(self.repository, current_year=current_year)

    def value(self, decoded_vehicle: DecodedVehicle | Mapping[str, Any]) -> dict[str, Any]:
        """Value a decoded Toyota VIN payload and return JSON-ready output."""

        vehicle = self._coerce_vehicle(decoded_vehicle)
        result = self.calculator.value(vehicle)
        return result.to_dict()

    @staticmethod
    def _coerce_vehicle(decoded_vehicle: DecodedVehicle | Mapping[str, Any]) -> DecodedVehicle:
        if isinstance(decoded_vehicle, DecodedVehicle):
            return decoded_vehicle

        required = ("make", "model", "trim", "model_year", "body_type", "regional_specs")
        missing = [key for key in required if key not in decoded_vehicle]
        if missing:
            raise ValueError(f"Decoded vehicle payload is missing required keys: {missing}")

        return DecodedVehicle(
            make=str(decoded_vehicle["make"]),
            model=str(decoded_vehicle["model"]),
            trim=str(decoded_vehicle["trim"]),
            model_year=int(decoded_vehicle["model_year"]),
            body_type=str(decoded_vehicle["body_type"]),
            regional_specs=str(decoded_vehicle["regional_specs"]),
        )


def value_vehicle(decoded_vehicle: DecodedVehicle | Mapping[str, Any], current_year: int | None = None) -> dict[str, Any]:
    """Convenience function for one-off Toyota valuation calls."""

    paths = DataPaths.toyota_defaults()
    repository = VehicleDataRepository(paths)
    calculator = VehicleValuationCalculator(repository, current_year=current_year)
    return calculator.value(ToyotaValuationEngine._coerce_vehicle(decoded_vehicle)).to_dict()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example = {
        "make": "Toyota",
        "model": "Land Cruiser",
        "trim": "GXR",
        "model_year": 2022,
        "body_type": "SUV",
        "regional_specs": "GCC",
    }
    print(ToyotaValuationEngine().value(example))
