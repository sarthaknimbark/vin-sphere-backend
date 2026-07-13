from __future__ import annotations

import pandas as pd

from .config import (
    REQUIRED_PREDICTION_COLUMNS,
    REQUIRED_SPLIT_COLUMNS,
)


class DataValidator:
    """
    Validates input DataFrames before analysis.
    """

    @staticmethod
    def _check_required_columns(
        df: pd.DataFrame,
        required: list[str],
        name: str,
    ) -> None:

        missing = sorted(set(required) - set(df.columns))

        if missing:
            raise ValueError(
                f"{name} is missing columns: {missing}"
            )

    @staticmethod
    def validate_predictions(
        df: pd.DataFrame,
    ) -> None:

        if df.empty:
            raise ValueError("Prediction dataframe is empty.")

        DataValidator._check_required_columns(
            df,
            REQUIRED_PREDICTION_COLUMNS,
            "Prediction dataframe",
        )

        if df["predicted_price"].isna().any():
            raise ValueError(
                "Prediction dataframe contains missing predicted_price values."
            )

    @staticmethod
    def validate_split(
        df: pd.DataFrame,
    ) -> None:

        if df.empty:
            raise ValueError("Split dataframe is empty.")

        DataValidator._check_required_columns(
            df,
            REQUIRED_SPLIT_COLUMNS,
            "Split dataframe",
        )