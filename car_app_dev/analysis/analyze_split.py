from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
REQUIRED_COLUMNS = [
    "make",
    "model",
    "trim",
    "year",
]
class SplitAnalyzer:

    def __init__(
        self,
        train_csv,
        test_csv,
        output_csv,
    ):

        self.train_csv = Path(train_csv)
        self.test_csv = Path(test_csv)
        self.output_csv = Path(output_csv)
    def load_data(self):

        print("=" * 70)
        print("LOADING DATA")
        print("=" * 70)

        self.train_df = pd.read_csv(
            self.train_csv
        )

        self.test_df = pd.read_csv(
            self.test_csv
        )

        print(
            f"Train rows : {len(self.train_df):,}"
        )

        print(
            f"Test rows  : {len(self.test_df):,}"
        )

        print()
        
    def validate(self):

        print("=" * 70)
        print("VALIDATING")
        print("=" * 70)

        for col in REQUIRED_COLUMNS:

            if col not in self.train_df.columns:

                raise ValueError(
                    f"Train missing column '{col}'"
                )

            if col not in self.test_df.columns:

                raise ValueError(
                    f"Test missing column '{col}'"
                )

        print("Validation : PASSED")
        print()
        
    def normalize(self):

        print("=" * 70)
        print("NORMALIZING")
        print("=" * 70)

        for df in (
            self.train_df,
            self.test_df,
        ):

            for col in [
                "make",
                "model",
                "trim",
            ]:

                df[col] = (

                    df[col]

                    .fillna("")

                    .astype(str)

                    .str.strip()

                    .str.upper()

                )

        print("Normalization complete")
        print()
        
    def build_lookups(self):

        print("=" * 70)
        print("BUILDING LOOKUPS")
        print("=" * 70)

        self.make_set = set(
            self.train_df["make"]
        )

        self.model_set = set(

            zip(

                self.train_df["make"],

                self.train_df["model"],

            )

        )

        self.trim_set = set(

            zip(

                self.train_df["make"],

                self.train_df["model"],

                self.train_df["trim"],

            )

        )

        self.model_counts = (

            self.train_df

            .groupby(
                [
                    "make",
                    "model",
                ]
            )

            .size()

            .to_dict()

        )

        self.trim_counts = (

            self.train_df

            .groupby(
                [
                    "make",
                    "model",
                    "trim",
                ]
            )

            .size()

            .to_dict()

        )

        print(
            f"Makes : {len(self.make_set):,}"
        )

        print(
            f"Models : {len(self.model_set):,}"
        )

        print(
            f"Trims : {len(self.trim_set):,}"
        )

        print()