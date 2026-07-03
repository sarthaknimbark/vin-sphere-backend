from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from rapidfuzz import fuzz


# =============================================================================
# CONFIGURATION
# =============================================================================

# Retrieval
MIN_EXACT_TRIM = 5
MIN_MODEL = 5

# Neighbor selection
MIN_NEIGHBORS = 5
MAX_NEIGHBORS = 15

INITIAL_THRESHOLD = 0.75
MIN_THRESHOLD = 0.55
THRESHOLD_STEP = 0.05

# Similarity
YEAR_DECAY = 3.0

# Tier 2 (make + model)
YEAR_WEIGHT_MODEL = 0.65
TRIM_WEIGHT_MODEL = 0.35

# Tier 3 (make fallback)
MODEL_WEIGHT_MAKE = 0.60
YEAR_WEIGHT_MAKE = 0.40


# =============================================================================
# QUERY
# =============================================================================

@dataclass
class VehicleQuery:
    make: str
    model: Optional[str] = None
    year: Optional[int] = None
    trim: Optional[str] = None


# =============================================================================
# MAIN CLASS
# =============================================================================

class VehicleSimilaritySearch:

    def __init__(self, df: pd.DataFrame):

        self.df = df.copy()

        # Normalize strings once
        for col in ["make", "model", "trim"]:
            if col in self.df.columns:
                self.df[col] = (
                    self.df[col]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .str.upper()
                )

        if "year" in self.df.columns:
            self.df["year"] = pd.to_numeric(
                self.df["year"],
                errors="coerce"
            )

        if "price" in self.df.columns:
            self.df["price"] = pd.to_numeric(
                self.df["price"],
                errors="coerce"
            )
    def _normalize(self, value) -> str:
        if pd.isna(value):
            return ""

        return str(value).strip().upper()

    # ============================================================================
    # RETRIEVAL
    # ============================================================================

    def _retrieve_exact_trim(
        self,
        query: VehicleQuery,
    ) -> pd.DataFrame:

        return self.df[
            (self.df["make"] == self._normalize(query.make))
            &
            (self.df["model"] == self._normalize(query.model))
            &
            (self.df["trim"] == self._normalize(query.trim))
        ].copy()


    def _retrieve_model(
        self,
        query: VehicleQuery,
    ) -> pd.DataFrame:

        return self.df[
            (self.df["make"] == self._normalize(query.make))
            &
            (self.df["model"] == self._normalize(query.model))
        ].copy()
        
    # =========================================================================
    # SIMILARITY FUNCTIONS
    # =========================================================================

    def _year_similarity(
        self,
        query_year: Optional[int],
        candidate_year: Optional[int],
    ) -> float:
        """
        Exponential year decay.

        Same year  -> 1.0
        1 year     -> 0.72
        2 years    -> 0.51
        3 years    -> 0.37
        """

        if pd.isna(query_year) or pd.isna(candidate_year):
            return 0.5

        diff = abs(int(query_year) - int(candidate_year))

        return math.exp(-diff / YEAR_DECAY)


    def _trim_similarity(
        self,
        query_trim: Optional[str],
        candidate_trim: Optional[str],
    ) -> float:

        q = self._normalize(query_trim)
        c = self._normalize(candidate_trim)

        if not q or not c:
            return 0.5

        return fuzz.token_set_ratio(q, c) / 100.0


    def _model_similarity(
        self,
        query_model: Optional[str],
        candidate_model: Optional[str],
    ) -> float:

        q = self._normalize(query_model)
        c = self._normalize(candidate_model)

        if not q or not c:
            return 0.0

        return fuzz.token_set_ratio(q, c) / 100.0
    def _compute_similarity(
        self,
        row: pd.Series,
        query: VehicleQuery,
        tier: str,
    ) -> float:

        # ----------------------------------------------------
        # Tier 1
        # Exact Trim
        # ----------------------------------------------------

        if tier == "exact_trim":

            return self._year_similarity(
                query.year,
                row["year"],
            )

        # ----------------------------------------------------
        # Tier 2
        # Same Model
        # ----------------------------------------------------

        if tier == "model":

            year_score = self._year_similarity(
                query.year,
                row["year"],
            )

            trim_score = self._trim_similarity(
                query.trim,
                row["trim"],
            )

            return (
                YEAR_WEIGHT_MODEL * year_score +
                TRIM_WEIGHT_MODEL * trim_score
            )

        # ----------------------------------------------------
        # Tier 3
        # Make Fallback
        # ----------------------------------------------------

        year_score = self._year_similarity(
            query.year,
            row["year"],
        )

        model_score = self._model_similarity(
            query.model,
            row["model"],
        )

        return (
            MODEL_WEIGHT_MAKE * model_score +
            YEAR_WEIGHT_MAKE * year_score
        )
    # ============================================================================
    # RANKING
    # ============================================================================

    def _rank_exact_trim(
        self,
        df: pd.DataFrame,
        query: VehicleQuery,
    ) -> pd.DataFrame:

        if df.empty:
            return df

        ranked = df.copy()

        ranked["_year_diff"] = (
            ranked["year"] - query.year
        ).abs()

        ranked = ranked.sort_values(
            by="_year_diff",
            ascending=True,
        )

        return ranked
    def _rank_model(
        self,
        df: pd.DataFrame,
        query: VehicleQuery,
    ) -> pd.DataFrame:

        if df.empty:
            return df

        ranked = df.copy()

        ranked["_trim_match"] = (
            ranked["trim"] ==
            self._normalize(query.trim)
        ).astype(int)

        ranked["_year_diff"] = (
            ranked["year"] - query.year
        ).abs()

        ranked = ranked.sort_values(

            by=[
                "_trim_match",
                "_year_diff",
            ],

            ascending=[
                False,
                True,
            ],
        )

        return ranked
    # =========================================================================
    # SEARCH
    # =========================================================================

    def search(
        self,
        query: VehicleQuery,
        k: int = 10,          # kept only for compatibility with LOO
    ) -> pd.DataFrame:

        candidates, tier = self._retrieve_candidates(query)

        # -------------------------------------------------------------
        # No candidates
        # -------------------------------------------------------------

        if candidates.empty:

            candidates["_similarity"] = []
            candidates["_tier"] = []

            return candidates

        # -------------------------------------------------------------
        # Compute similarity
        # -------------------------------------------------------------

        candidates["_similarity"] = candidates.apply(
            lambda row: self._compute_similarity(
                row=row,
                query=query,
                tier=tier,
            ),
            axis=1,
        )

        candidates["_tier"] = tier

        # -------------------------------------------------------------
        # Sort
        # -------------------------------------------------------------

        candidates = candidates.sort_values(
            "_similarity",
            ascending=False,
        ).reset_index(drop=True)

        # -------------------------------------------------------------
        # Adaptive threshold
        # -------------------------------------------------------------

        threshold = INITIAL_THRESHOLD

        selected = pd.DataFrame()

        while threshold >= MIN_THRESHOLD:

            selected = candidates[
                candidates["_similarity"] >= threshold
            ]

            if len(selected) >= MIN_NEIGHBORS:
                break

            threshold -= THRESHOLD_STEP

        # -------------------------------------------------------------
        # Final fallback
        # -------------------------------------------------------------

        if selected.empty:
            selected = candidates.head(MIN_NEIGHBORS)

        # -------------------------------------------------------------
        # Maximum neighbors
        # -------------------------------------------------------------

        selected = selected.head(MAX_NEIGHBORS)

        return selected.reset_index(drop=True)
if __name__ == "__main__":

    df = pd.DataFrame({

        "make": [
            "TOYOTA",
            "TOYOTA",
            "TOYOTA",
            "TOYOTA",
            "TOYOTA",
            "TOYOTA",
        ],

        "model": [
            "CAMRY",
            "CAMRY",
            "CAMRY",
            "CAMRY",
            "CAMRY",
            "COROLLA",
        ],

        "trim": [
            "GLE",
            "GLE",
            "GLE",
            "GL",
            "GLX",
            "GLE",
        ],

        "year": [
            2020,
            2019,
            2021,
            2020,
            2020,
            2020,
        ],

        "price": [
            65000,
            62000,
            68000,
            59000,
            71000,
            55000,
        ],
    })

    engine = VehicleSimilaritySearch(df)

    query = VehicleQuery(
        make="TOYOTA",
        model="CAMRY",
        year=2020,
        trim="GLE",
    )

    result = engine.search(query)

    print(result[[
        "make",
        "model",
        "trim",
        "year",
        "price",
        "_similarity",
        "_tier",
    ]])
        
exact = engine._retrieve_exact_trim(query)

print(exact)       

print(
    engine._rank_exact_trim(
        exact,
        query,
    )[["trim","year"]]
)

model = engine._retrieve_model(query)

print(
    engine._rank_model(
        model,
        query,
    )[["trim","year"]]
)
        
        
        
# import pandas as pd

# df = pd.DataFrame({
#     "make": ["TOYOTA"],
#     "model": ["CAMRY"],
#     "trim": ["GLE"],
#     "year": [2020],
#     "price": [65000]
# })

# query = VehicleQuery(
#     make="TOYOTA",
#     model="CAMRY",
#     year=2020,
#     trim="GLE"
# )

# engine = VehicleSimilaritySearch(df)

# print(engine._year_similarity(2020, 2020))
# print(engine._year_similarity(2020, 2021))
# print(engine._trim_similarity("GLE", "GLE FULL"))
# print(engine._model_similarity("CAMRY", "CAMRY"))