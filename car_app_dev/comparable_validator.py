from __future__ import annotations

import pandas as pd


class ComparableValidator:

    def __init__(
        self,
        min_neighbors: int = 3,
        lower_ratio: float = 0.40,
        upper_ratio: float = 2.50,
    ):
        self.min_neighbors = min_neighbors
        self.lower_ratio = lower_ratio
        self.upper_ratio = upper_ratio

    def validate(
        self,
        neighbors: pd.DataFrame,
    ) -> pd.DataFrame:

        if neighbors.empty:
            return neighbors

        neighbors = self._remove_duplicate_vins(neighbors)

        neighbors = self._remove_price_outliers(neighbors)

        return neighbors.reset_index(drop=True)
    
    def _remove_duplicate_vins(
        self,
        neighbors: pd.DataFrame,
    ) -> pd.DataFrame:

        if "chassisNumber" not in neighbors.columns:
            return neighbors

        return neighbors.drop_duplicates(
            subset="chassisNumber",
            keep="first",
        )
        
    def _remove_price_outliers(
        self,
        neighbors: pd.DataFrame,
    ) -> pd.DataFrame:

        if len(neighbors) < 4:
            return neighbors

        median_price = neighbors["price"].median()

        if median_price <= 0:
            return neighbors

        ratio = neighbors["price"] / median_price

        filtered = neighbors[
            ratio.between(
                self.lower_ratio,
                self.upper_ratio,
            )
        ].copy()

        # Never become too aggressive.
        if len(filtered) >= self.min_neighbors:
            return filtered

        return neighbors