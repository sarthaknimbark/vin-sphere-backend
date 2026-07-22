"""CSV loading, validation, and lookup services for vehicle valuation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import DataPaths
from .utils import closest_match, normalize_text

logger = logging.getLogger(__name__)


class ValuationDataError(ValueError):
    """Raised when the valuation CSV bundle is missing required data."""


class MissingBasePriceError(ValuationDataError):
    """Raised when no usable base price can be found."""


@dataclass(frozen=True)
class LookupValue:
    """A numeric lookup result plus any non-fatal fallback warnings."""

    value: float
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BasePriceLookup:
    """Base-price lookup result plus selected row attributes."""

    base_price: float
    matched_trim: str
    matched_year: int
    warnings: list[str] = field(default_factory=list)


class VehicleDataRepository:
    """Loads valuation tables and exposes deterministic lookup methods."""

    MATRIX_COLUMNS = {
        "Make": ("Make",),
        "Model": ("Model",),
        "Trim": ("Trim",),
        "Year": ("Year",),
        "Body Type": ("Body Type", "Body_Type", "body_type"),
        "Regional Specs": ("Regional Specs", "Region Spec", "regional_specs", "regional_spec"),
        "Base Price": ("Base Price", "Avg Price (AED)", "base_price"),
    }

    def __init__(self, data_paths: DataPaths) -> None:
        self.data_paths = data_paths
        self.matrix = self._load_matrix(data_paths.matrix_csv)
        self.age_depreciation = self._load_factor_table(
            data_paths.age_depreciation_csv,
            required_columns=("age", "remaining_value_factor"),
            numeric_columns=("age", "remaining_value_factor"),
        )
        self.model_factors = self._load_factor_table(
            data_paths.model_factor_csv,
            required_columns=("model", "retention_factor", "demand_factor"),
            numeric_columns=("retention_factor", "demand_factor"),
            key_columns=("model",),
        )
        self.trim_factors = self._load_factor_table(
            data_paths.trim_factor_csv,
            required_columns=("trim", "factor"),
            numeric_columns=("factor",),
            key_columns=("trim",),
        )
        self.body_type_factors = self._load_factor_table(
            data_paths.body_type_factor_csv,
            required_columns=("body_type", "factor"),
            numeric_columns=("factor",),
            key_columns=("body_type",),
        )
        self.regional_spec_factors = self._load_factor_table(
            data_paths.regional_spec_factor_csv,
            required_columns=("regional_spec", "factor"),
            numeric_columns=("factor",),
            key_columns=("regional_spec",),
        )
        self.market_factors = self._load_factor_table(
            data_paths.market_factor_csv,
            required_columns=("market_condition", "factor"),
            numeric_columns=("factor",),
            key_columns=("market_condition",),
        )

    def lookup_base_price(
        self,
        *,
        make: str,
        model: str,
        trim: str,
        year: int,
        body_type: str | None = None,
        regional_specs: str | None = None,
    ) -> BasePriceLookup:
        """Lookup a base price using exact keys, then closest trim/year fallbacks."""

        logger.info("Looking up base price for %s %s %s %s", make, model, trim, year)
        warnings: list[str] = []
        frame = self.matrix
        model_rows = frame[
            (frame["_make_key"] == normalize_text(make))
            & (frame["_model_key"] == normalize_text(model))
        ].copy()

        if model_rows.empty:
            message = f"No base-price rows found for make={make!r}, model={model!r}."
            logger.error(message)
            raise MissingBasePriceError(message)

        year_rows = model_rows[model_rows["Year"] == int(year)].copy()
        if year_rows.empty:
            nearest_year = self._nearest_year(year, model_rows["Year"])
            warning = (
                f"Base price for model year {year} is missing; using nearest available "
                f"year {nearest_year} for the same make/model."
            )
            logger.warning(warning)
            warnings.append(warning)
            year_rows = model_rows[model_rows["Year"] == nearest_year].copy()

        exact_trim_rows = year_rows[year_rows["_trim_key"] == normalize_text(trim)].copy()
        if exact_trim_rows.empty:
            matched_trim = closest_match(trim, year_rows["Trim"])
            if matched_trim is None:
                message = (
                    f"No trim candidates found for make={make!r}, model={model!r}, "
                    f"year={year!r}."
                )
                logger.error(message)
                raise MissingBasePriceError(message)
            warning = (
                f"Exact trim {trim!r} is missing; using closest trim {matched_trim!r} "
                "for the same model."
            )
            logger.warning(warning)
            warnings.append(warning)
            candidate_rows = year_rows[year_rows["_trim_key"] == normalize_text(matched_trim)].copy()
        else:
            matched_trim = str(exact_trim_rows.iloc[0]["Trim"])
            candidate_rows = exact_trim_rows

        selected = self._narrow_price_candidates(candidate_rows, body_type, regional_specs, warnings)
        base_price = float(selected["Base Price"].median())
        matched_year = int(selected.iloc[0]["Year"])
        logger.info(
            "Base price resolved: %.2f AED using trim=%s year=%s",
            base_price,
            matched_trim,
            matched_year,
        )
        return BasePriceLookup(base_price=base_price, matched_trim=matched_trim, matched_year=matched_year, warnings=warnings)

    def lookup_age_factor(self, vehicle_age: int) -> LookupValue:
        """Lookup depreciation factor, clamping above the max table age."""

        logger.info("Looking up age depreciation factor for age=%s", vehicle_age)
        table = self.age_depreciation.sort_values("age")
        if table.empty:
            raise ValuationDataError("Age depreciation table is empty.")

        max_age = int(table["age"].max())
        effective_age = min(max(vehicle_age, 0), max_age)
        warnings: list[str] = []
        if vehicle_age > max_age:
            warning = f"Vehicle age {vehicle_age} exceeds depreciation table max {max_age}; using max-age factor."
            logger.warning(warning)
            warnings.append(warning)

        rows = table[table["age"] == effective_age]
        if rows.empty:
            value = float(table["remaining_value_factor"].median())
            warning = f"Depreciation rate for age {effective_age} is missing; using table median factor {value}."
            logger.warning(warning)
            warnings.append(warning)
            return LookupValue(value=value, warnings=warnings)

        value = float(rows.iloc[0]["remaining_value_factor"])
        logger.info("Age factor resolved: %.4f", value)
        return LookupValue(value=value, warnings=warnings)

    def lookup_model_retention_factor(self, model: str) -> LookupValue:
        return self._lookup_factor(self.model_factors, "model", "retention_factor", model, "model retention")

    def lookup_market_demand_factor(self, model: str) -> LookupValue:
        logger.info("Looking up market demand factor from model demand_factor for model=%s", model)
        return self._lookup_factor(self.model_factors, "model", "demand_factor", model, "market demand")

    def lookup_trim_factor(self, trim: str) -> LookupValue:
        return self._lookup_factor(self.trim_factors, "trim", "factor", trim, "trim")

    def lookup_body_type_factor(self, body_type: str) -> LookupValue:
        return self._lookup_factor(self.body_type_factors, "body_type", "factor", body_type, "body type")

    def lookup_regional_spec_factor(self, regional_specs: str) -> LookupValue:
        return self._lookup_factor(self.regional_spec_factors, "regional_spec", "factor", regional_specs, "regional spec")

    def _lookup_factor(self, frame: pd.DataFrame, key_column: str, value_column: str, key: str, label: str) -> LookupValue:
        logger.info("Looking up %s factor for %s=%s", label, key_column, key)
        if frame.empty:
            raise ValuationDataError(f"{label.title()} factor table is empty.")

        key_name = f"_{key_column}_key"
        rows = frame[frame[key_name] == normalize_text(key)]
        if not rows.empty:
            value = float(rows.iloc[0][value_column])
            logger.info("%s factor resolved: %.4f", label.title(), value)
            return LookupValue(value=value)

        matched = closest_match(key, frame[key_column])
        warnings: list[str] = []
        if matched:
            matched_rows = frame[frame[key_name] == normalize_text(matched)]
            if not matched_rows.empty:
                value = float(matched_rows.iloc[0][value_column])
                warning = f"Unknown {label} {key!r}; using closest table entry {matched!r}."
                logger.warning(warning)
                warnings.append(warning)
                return LookupValue(value=value, warnings=warnings)

        value = float(frame[value_column].median())
        warning = f"Unknown {label} {key!r}; using median table factor {value}."
        logger.warning(warning)
        warnings.append(warning)
        return LookupValue(value=value, warnings=warnings)

    def _narrow_price_candidates(
        self,
        rows: pd.DataFrame,
        body_type: str | None,
        regional_specs: str | None,
        warnings: list[str],
    ) -> pd.DataFrame:
        selected = rows
        if body_type:
            body_rows = selected[selected["_body_type_key"] == normalize_text(body_type)]
            if not body_rows.empty:
                selected = body_rows
            else:
                warning = f"Base-price body type {body_type!r} not found among candidates; using available body type rows."
                logger.warning(warning)
                warnings.append(warning)

        if regional_specs:
            regional_rows = selected[selected["_regional_specs_key"] == normalize_text(regional_specs)]
            if not regional_rows.empty:
                selected = regional_rows
            else:
                warning = f"Base-price regional spec {regional_specs!r} not found among candidates; using available regional rows."
                logger.warning(warning)
                warnings.append(warning)

        if len(selected) > 1:
            warning = "Multiple base-price rows matched; using the median base price."
            logger.warning(warning)
            warnings.append(warning)
        return selected

    def _load_matrix(self, path: Path) -> pd.DataFrame:
        frame = self._read_csv(path)
        rename_map = self._column_rename_map(frame.columns, self.MATRIX_COLUMNS, path)
        frame = frame.rename(columns=rename_map)
        frame = frame[list(self.MATRIX_COLUMNS.keys())].copy()
        frame["Year"] = pd.to_numeric(frame["Year"], errors="coerce")
        frame["Base Price"] = pd.to_numeric(frame["Base Price"], errors="coerce")
        frame = frame.dropna(subset=["Make", "Model", "Trim", "Year", "Base Price"])
        frame["Year"] = frame["Year"].astype(int)
        frame["Base Price"] = frame["Base Price"].astype(float)

        for column in ("Make", "Model", "Trim", "Body Type", "Regional Specs"):
            frame[f"_{column.lower().replace(' ', '_')}_key"] = frame[column].map(normalize_text)

        if frame.empty:
            raise ValuationDataError(f"Matrix CSV {path} has no valid price rows.")
        logger.info("Loaded %s valid matrix rows from %s", len(frame), path)
        return frame

    def _load_factor_table(
        self,
        path: Path,
        *,
        required_columns: Iterable[str],
        numeric_columns: Iterable[str],
        key_columns: Iterable[str] = (),
    ) -> pd.DataFrame:
        frame = self._read_csv(path)
        missing = [column for column in required_columns if column not in frame.columns]
        if missing:
            raise ValuationDataError(f"{path} is missing required columns: {missing}")

        frame = frame[list(required_columns)].copy()
        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=list(numeric_columns))

        for column in key_columns:
            frame[f"_{column}_key"] = frame[column].map(normalize_text)

        if frame.empty:
            raise ValuationDataError(f"{path} has no valid rows.")
        logger.info("Loaded %s rows from %s", len(frame), path)
        return frame

    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise ValuationDataError(f"CSV file does not exist: {path}")
        return pd.read_csv(path).drop(columns=lambda column: str(column).startswith("Unnamed"), errors="ignore")

    @staticmethod
    def _column_rename_map(columns: Iterable[str], aliases: dict[str, tuple[str, ...]], path: Path) -> dict[str, str]:
        rename_map: dict[str, str] = {}
        available = {normalize_text(column): column for column in columns}
        for canonical, candidates in aliases.items():
            source = next((available.get(normalize_text(candidate)) for candidate in candidates if normalize_text(candidate) in available), None)
            if source is None:
                raise ValuationDataError(f"{path} is missing required column {canonical!r}.")
            rename_map[source] = canonical
        return rename_map

    @staticmethod
    def _nearest_year(target_year: int, years: pd.Series) -> int:
        available_years = sorted({int(year) for year in years.dropna().tolist()})
        return min(available_years, key=lambda candidate: (abs(candidate - int(target_year)), candidate))
