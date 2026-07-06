from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

TARGET_NEIGHBORS = 5

MIN_EXACT_TRIM = 5

MIN_MODEL_ROWS = 5


# =============================================================================
# QUERY
# =============================================================================

@dataclass
class VehicleQuery:
    make: str
    model: str
    year: Optional[int] = None
    trim: Optional[str] = None


# =============================================================================
# SEARCH ENGINE
# =============================================================================

class VehicleSimilaritySearch:

    def __init__(self, df: pd.DataFrame):

        self.df = df.copy()

        for col in ["make", "model", "trim"]:

            self.df[col] = (
                self.df[col]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.upper()
            )

        self.df["year"] = pd.to_numeric(
            self.df["year"],
            errors="coerce",
        )

        self.df["price"] = pd.to_numeric(
            self.df["price"],
            errors="coerce",
        )

    def _normalize(self, value):

        if pd.isna(value):
            return ""

        return str(value).strip().upper()
    # ==========================================================
    # RETRIEVAL
    # ==========================================================

    def _retrieve_exact_trim(
        self,
        query: VehicleQuery,
    ):

        return self.df[
            (self.df.make == self._normalize(query.make))
            &
            (self.df.model == self._normalize(query.model))
            &
            (self.df.trim == self._normalize(query.trim))
        ].copy()
        
    def _retrieve_same_year(
    self,
    query: VehicleQuery,
) -> pd.DataFrame:

        return self.df[
            (self.df["make"] == query.make)
            &
            (self.df["model"] == query.model)
            &
            (self.df["year"] == query.year)
        ].copy()


    def _retrieve_model(
        self,
        query: VehicleQuery,
    ):

        return self.df[
            (self.df.make == self._normalize(query.make))
            &
            (self.df.model == self._normalize(query.model))
            &
            (self.df.trim != self._normalize(query.trim))
        ].copy()
    # ==========================================================
    # RANKING
    # ==========================================================

    def _rank_exact_trim(
        self,
        df: pd.DataFrame,
        query: VehicleQuery,
    ):

        if df.empty:
            return df

        ranked = df.copy()
        
        ranked["_tier"] = "exact_trim"

        ranked["_year_diff"] = (
            ranked["year"] - query.year
        ).abs()

        ranked = ranked.sort_values(
            by="_year_diff",
            ascending=True,
        )
        
        

        return ranked
    
    def _rank_same_year(
        self,
        candidates: pd.DataFrame,
        query: VehicleQuery,
    ) -> pd.DataFrame:

        if candidates.empty:
            return candidates

        ranked = candidates.copy()

        ranked["_tier"] = "same_year"

        ranked["_trim_match"] = (
            ranked["trim"] == query.trim
        ).astype(int)

        ranked = ranked.sort_values(
            "_trim_match",
            ascending=False,
        )

        return ranked


    def _rank_model(
        self,
        df: pd.DataFrame,
        query: VehicleQuery,
    ):

        if df.empty:
            return df

        ranked = df.copy()
        ranked["_tier"] = "model"
        

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
            ]
        )
       
        return ranked

    # ==========================================================
    # COMPARABLE SELECTION
    # ==========================================================

    def _select_neighbors(
        self,
        exact,
        same_year,
        model,
    ):
        
        frames = []

        if not exact.empty:
            frames.append(exact)

        if not same_year.empty:
            frames.append(same_year)

        if not model.empty:
            frames.append(model)
            
        if not frames:
            return pd.DataFrame()

        combined = pd.concat(
            frames,
            ignore_index=False,
        )
         # Remove duplicate vehicles (same row found in multiple tiers)
        combined = combined[~combined.index.duplicated(keep="first")]

        return combined
    # ==========================================================
    # PRICE
    # ==========================================================

    def _predict_price(
        self,
        neighbors: pd.DataFrame,
    ):

        if neighbors.empty:
            return None

        return float(neighbors["price"].mean())
    # ==========================================================
    # SEARCH
    # ==========================================================

    def search(
        self,
        query: VehicleQuery,
    ) -> pd.DataFrame:
    
        exact = self._retrieve_exact_trim(query)
        exact = self._rank_exact_trim(exact, query)

        same_year = self._retrieve_same_year(query)
        same_year = self._rank_same_year(same_year, query)

        model = self._retrieve_model(query)
        model = self._rank_model(model, query)

        return self._select_neighbors(
            exact,
            same_year,
            model,
        )
        
        
    def _filter_price_outliers(
        self,
        neighbors: pd.DataFrame,
    ) -> pd.DataFrame:

        if len(neighbors) < 4:
            return neighbors

        median = neighbors["price"].median()

        lower = median * 0.5
        upper = median * 2.0

        filtered = neighbors[
            neighbors["price"].between(lower, upper)
        ]

        if filtered.empty:
            return neighbors

        return filtered.reset_index(drop=True)
    
    def _filter_neighbors(
        self,
        neighbors: pd.DataFrame,
    ) -> pd.DataFrame:

        if len(neighbors) < 4:
            return neighbors

        prices = neighbors["price"]

        median = prices.median()

        ratio = prices / median

        filtered = neighbors[
            ratio.between(0.4, 2.5)
        ].copy()

        # Don't accidentally remove everything
        if len(filtered) >= 3:
            return filtered
        
        neighbors = neighbors.drop_duplicates(
            subset="chassisNumber",
            keep="first",
        )
        
        neighbors = neighbors.drop(
            columns=[
                "_trim_match",
            ],
            errors="ignore",
        )

        return neighbors
    
    def explain(
        self,
        query: VehicleQuery,
    ):

        neighbors = self.search(query)

        print("=" * 70)
        print("QUERY")
        print(query)

        print()

        print(neighbors[
            [
                "make",
                "model",
                "trim",
                "year",
                "price",
                "_tier",
                "_year_diff",
            ]
        ])

if __name__ == "__main__":

    df = pd.DataFrame({
        "make": ["TOYOTA","TOYOTA","TOYOTA","TOYOTA","TOYOTA"],
        "model": ["CAMRY","CAMRY","CAMRY","CAMRY","CAMRY"],
        "trim": ["GLE","GLE","GL","GLX","GLE"],
        "year": [2020,2019,2020,2020,2021],
        "price": [65000,62000,59000,71000,68000],
    })

    engine = VehicleSimilaritySearch(df)

    q = VehicleQuery(
        make="TOYOTA",
        model="CAMRY",
        year=2020,
        trim="GLE",
    )

    print(engine.search(q))