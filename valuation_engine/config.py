"""Configuration for CSV-backed valuation data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DataPaths:
    """Filesystem locations for one brand's valuation CSVs."""

    matrix_csv: Path
    age_depreciation_csv: Path
    model_factor_csv: Path
    trim_factor_csv: Path
    body_type_factor_csv: Path
    regional_spec_factor_csv: Path
    market_factor_csv: Path

    @classmethod
    def toyota_defaults(cls) -> "DataPaths":
        """Return paths for the Toyota CSV bundle shipped with this package."""

        return cls(
            matrix_csv=PACKAGE_DIR / "matrix.csv",
            age_depreciation_csv=PACKAGE_DIR / "toyota_age_depreciation.csv",
            model_factor_csv=PACKAGE_DIR / "toyota_model_factor.csv",
            trim_factor_csv=PACKAGE_DIR / "toyota_trim_factor.csv",
            body_type_factor_csv=PACKAGE_DIR / "toyota_body_type_factor.csv",
            regional_spec_factor_csv=PACKAGE_DIR / "toyota_regional_spec_factor.csv",
            market_factor_csv=PACKAGE_DIR / "toyota_market_factor.csv",
        )
