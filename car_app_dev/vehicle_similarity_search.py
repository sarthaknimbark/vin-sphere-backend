"""
vehicle_similarity_search.py
────────────────────────────
Price prediction via weighted k-nearest-neighbour search.

Given a query vehicle (make, model, year, trim), finds the k most similar
vehicles in the reference dataset and returns a weighted-average price.

Reference dataset : any DataFrame with a `price` column
Target attribute  : price (sole prediction target)
Query fields      : make, model, year, trim
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class VehicleQuery:
    make:  str
    model: Optional[str] = None
    year:  Optional[int] = None
    trim:  Optional[str] = None


@dataclass
class PriceResult:
    price:          Optional[float] = None   # weighted-average predicted price
    price_low:      Optional[float] = None   # min among top-k neighbours
    price_high:     Optional[float] = None   # max among top-k neighbours
    n_neighbors:    int   = 0
    avg_similarity: float = 0.0
    provenance:     str   = "unresolved"     # exact | fuzzy_model | make_fallback | unresolved
    top_neighbors:  list  = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Core engine
# ──────────────────────────────────────────────────────────────────────────────

class VehicleSimilaritySearch:
    """
    Weighted k-nearest-neighbour price predictor.

    Index  : make → model → DataFrame slice  (two-level, O(1) candidate fetch)

    Similarity (convex combination, sums to 1.0):
        model  0.40  — rapidfuzz token_sort_ratio
        year   0.40  — linear decay over YEAR_MAX_DISTANCE (year dominates price)
        trim   0.20  — rapidfuzz token_sort_ratio

    Missing query fields contribute 0.5 (neutral), never 0.

    Fallback chain:
        1. make + model exact        → scored by year + trim
        2. make + model fuzzy (≥80)  → scored by year + trim
        3. make only                 → scored by model + year + trim
        4. unresolved                → PriceResult(provenance="unresolved")
    """

    FEATURE_WEIGHTS = {
        "model": 0.40,
        "year":  0.40,
        "trim":  0.20,
    }

    YEAR_MAX_DISTANCE: int   = 10
    MIN_SIMILARITY:    float = 0.30

    def __init__(self, reference_df: pd.DataFrame) -> None:
        """
        Parameters
        ----------
        reference_df : pd.DataFrame
            Must contain: make, model, price
            Should contain: year, trim
        """
        self._df = reference_df.copy()
        self._validate()
        self._normalise_strings()
        self._build_index()
        logger.info(
            "VehicleSimilaritySearch ready — %d rows, %d makes",
            len(self._df), len(self._make_model_index),
        )

    # ── Construction ──────────────────────────────────────────────────────────

    def _validate(self) -> None:
        required = {"make", "model", "price"}
        missing = required - set(self._df.columns)
        if missing:
            raise ValueError(f"Reference dataset missing columns: {missing}")

    def _normalise_strings(self) -> None:
        for col in ("make", "model", "trim"):
            if col in self._df.columns:
                self._df[col] = (
                    self._df[col]
                    .where(self._df[col].isna(),
                           self._df[col].str.upper().str.strip())
                )

    def _build_index(self) -> None:
        self._make_model_index: dict[str, dict[str, pd.DataFrame]] = {}
        self._make_index:       dict[str, pd.DataFrame]            = {}

        for make, make_group in self._df.groupby("make"):
            self._make_index[make] = make_group
            self._make_model_index[make] = {
                model: grp
                for model, grp in make_group.groupby("model")
            }

    # ── Similarity components ─────────────────────────────────────────────────

    @staticmethod
    def _model_sim(query: str, series: pd.Series) -> np.ndarray:
        def _s(v):
            if pd.isna(v):   return 0.0
            if query == v:   return 1.0
            return fuzz.token_sort_ratio(query, str(v)) / 100.0
        return np.array([_s(v) for v in series], dtype=np.float32)

    def _year_sim(self, query: int, series: pd.Series) -> np.ndarray:
        def _s(v):
            if pd.isna(v): return 0.5
            return max(0.0, 1.0 - abs(query - int(v)) / self.YEAR_MAX_DISTANCE)
        return np.array([_s(v) for v in series], dtype=np.float32)

    @staticmethod
    def _trim_sim(query: str, series: pd.Series) -> np.ndarray:
        def _s(v):
            if pd.isna(v):   return 0.5
            if query == v:   return 1.0
            return fuzz.token_sort_ratio(query, str(v)) / 100.0
        return np.array([_s(v) for v in series], dtype=np.float32)

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score(self, query: VehicleQuery, candidates: pd.DataFrame) -> np.ndarray:
        n = len(candidates)
        w = self.FEATURE_WEIGHTS

        col_model = (
            self._model_sim(query.model.upper(), candidates["model"])
            if query.model is not None and "model" in candidates.columns
            else np.full(n, 0.5, dtype=np.float32)
        )
        col_year = (
            self._year_sim(query.year, candidates["year"])
            if query.year is not None and "year" in candidates.columns
            else np.full(n, 0.5, dtype=np.float32)
        )
        col_trim = (
            self._trim_sim(query.trim.upper(), candidates["trim"])
            if query.trim is not None and "trim" in candidates.columns
            else np.full(n, 0.5, dtype=np.float32)
        )

        return (
            w["model"] * col_model
            + w["year"]  * col_year
            + w["trim"]  * col_trim
        )

    # ── Candidate retrieval ───────────────────────────────────────────────────

    def _get_candidates(self, query: VehicleQuery) -> tuple[pd.DataFrame, str]:
        make  = query.make.upper()  if query.make  else None
        model = query.model.upper() if query.model else None

        # Tier 1 — exact make + model
        if make and model and make in self._make_model_index:
            if model in self._make_model_index[make]:
                return self._make_model_index[make][model], "exact"
            # Tier 2 — fuzzy model within make
            best_model, best_score = None, 0
            for ref_model in self._make_model_index[make]:
                s = fuzz.token_sort_ratio(model, ref_model)
                if s > best_score:
                    best_score, best_model = s, ref_model
            if best_score >= 80:
                return self._make_model_index[make][best_model], "fuzzy_model"

        # Tier 3 — make only
        if make and make in self._make_index:
            return self._make_index[make], "make_fallback"

        return pd.DataFrame(), "unresolved"

    # ── Public API ────────────────────────────────────────────────────────────

    def search(self, query: VehicleQuery, k: int = 10) -> pd.DataFrame:
        """Return top-k neighbours with _similarity and _tier columns."""
        candidates, tier = self._get_candidates(query)
        if candidates.empty:
            return pd.DataFrame()

        scores = self._score(query, candidates)
        result = candidates.copy()
        result["_similarity"] = scores
        result["_tier"]       = tier

        return (
            result[result["_similarity"] >= self.MIN_SIMILARITY]
            .nlargest(k, "_similarity")
            .reset_index(drop=True)
        )

    def predict(self, query: VehicleQuery, k: int = 10) -> PriceResult:
        """
        Weighted-average price prediction.

        Returns
        -------
        PriceResult
            .price          — weighted-average predicted price
            .price_low/high — range across top-k neighbours
            .n_neighbors    — how many neighbours contributed
            .avg_similarity — mean similarity of contributing neighbours
            .provenance     — exact | fuzzy_model | make_fallback | unresolved
            .top_neighbors  — top-3 neighbours for transparency
        """
        neighbors = self.search(query, k=k)

        if neighbors.empty:
            return PriceResult(provenance="unresolved")

        valid = neighbors[neighbors["price"].notna()]
        if valid.empty:
            return PriceResult(
                provenance=neighbors["_tier"].iloc[0],
                n_neighbors=len(neighbors),
            )

        weights = valid["_similarity"].to_numpy(dtype=np.float64)
        prices  = valid["price"].to_numpy(dtype=np.float64)

        summary_cols = [c for c in ("make", "model", "year", "trim", "price", "_similarity")
                        if c in neighbors.columns]

        return PriceResult(
            price          = round(float(np.average(prices, weights=weights)), 2),
            price_low      = float(prices.min()),
            price_high     = float(prices.max()),
            n_neighbors    = len(valid),
            avg_similarity = float(weights.mean()),
            provenance     = valid["_tier"].iloc[0],
            top_neighbors  = neighbors[summary_cols].head(3).to_dict("records"),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Convenience loader
# ──────────────────────────────────────────────────────────────────────────────

def build_price_engine(parquet_path: str) -> VehicleSimilaritySearch:
    """
    Load a csv file and return a ready-to-use price prediction engine.

    The file must have: make, model, price
    Optionally:         year, trim
    """
    df = pd.read_csv('dev_lookup.csv')
    return VehicleSimilaritySearch(df)


# ──────────────────────────────────────────────────────────────────────────────
# Smoke-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    if len(sys.argv) > 1:
        engine = build_price_engine(sys.argv[1])
    else:
        # Synthetic data when no parquet supplied
        df = pd.DataFrame({
            "make":  ["TOYOTA"]*6 + ["NISSAN"]*3,
            "model": ["CAMRY"]*6  + ["PATROL"]*3,
            "year":  [2018, 2019, 2020, 2021, 2020, 2020, 2019, 2020, 2021],
            "trim":  ["GL","GLE","GLE","GLE","GLX","GL", "SE","SE","SE"],
            "price": [62000, 68000, 72000, 75000, 70000, 60000, 155000, 160000, 165000],
        })
        engine = VehicleSimilaritySearch(df)

    queries = [
        VehicleQuery("TOYOTA", "CAMRY",  2020, "GLE"),   # exact match year+trim
        VehicleQuery("TOYOTA", "CAMRY",  2019, "GLE"),   # 1 year off
        VehicleQuery("TOYOTA", "CAMRY",  2020, "GLX"),   # different trim
        VehicleQuery("TOYOTA", "CAMRY",  2020, "GL"),    # base trim
        VehicleQuery("TOYOTA", "CAMRY",  None, None),    # minimal query
        VehicleQuery("TOYOTA", "CAMRY",  2020, "SPORT"), # unseen trim
        VehicleQuery("NISSAN", "SUNNY",  2020, "SV"),    # make fallback
        VehicleQuery("HONDA",  "ACCORD", 2020, "EX"),    # unresolved
    ]

    print("\n" + "=" * 70)
    for q in queries:
        r = engine.predict(q)
        print(f"\nQuery  : {q.make} {q.model} {q.year} {q.trim}")
        print(f"  price       : {r.price:,.0f}" if r.price else "  price       : N/A")
        print(f"  range       : {r.price_low:,.0f} – {r.price_high:,.0f}" if r.price_low else "  range       : N/A")
        print(f"  provenance  : {r.provenance}")
        print(f"  neighbors   : {r.n_neighbors}  avg_sim={r.avg_similarity:.3f}")
    print("=" * 70)
