from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error

from catboost import CatBoostRegressor

FEATURES = [
    "make",
    "model",
    "trim",
    "year",
]

TARGET = "price"

IGNORE_COLUMNS = [
    "chassisNumber",
    "price_flag",
    "flag_reason",
    "chassis_unreliable",
]

CAT_FEATURES = [
    "make",
    "model",
    "trim",
]
class CatBoostExperiment001:

    def __init__(self, csv_path: str):

        self.csv_path = csv_path

        self.output_dir = Path("experiment001")

        self.output_dir.mkdir(exist_ok=True)
    
    def load_dataset(self):

        print("=" * 70)
        print("LOADING DATASET")
        print("=" * 70)

        self.df = pd.read_csv(self.csv_path)

        print(f"Rows    : {len(self.df):,}")
        print(f"Columns : {len(self.df.columns)}")

        missing = [
            col for col in FEATURES + [TARGET]
            if col not in self.df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        self.df[TARGET] = pd.to_numeric(
            self.df[TARGET],
            errors="coerce",
        )

        self.df = self.df[
            self.df[TARGET].notna()
        ].copy()

        self.df = self.df[
            self.df[TARGET] > 0
        ].copy()

        print()
    
    def validate_dataset(self):

        print("=" * 70)
        print("DATASET VALIDATION")
        print("=" * 70)

        print(f"Rows              : {len(self.df):,}")
        print(f"Duplicate rows    : {self.df.duplicated().sum():,}")

        for col in FEATURES + [TARGET]:

            print(
                f"Null {col:<10}: "
                f"{self.df[col].isna().sum():,}"
            )

        print()

        print(
            f"Unique makes      : {self.df['make'].nunique():,}"
        )

        print(
            f"Unique models     : {self.df['model'].nunique():,}"
        )

        print(
            f"Unique trims      : {self.df['trim'].nunique():,}"
        )

        print()

        print(
            f"Price range       : "
            f"{self.df[TARGET].min():,.0f}"
            f" -> "
            f"{self.df[TARGET].max():,.0f}"
        )

        print()
    def run(self):

        self.load_dataset()

        self.validate_dataset()
    def main():

        parser = argparse.ArgumentParser()

        parser.add_argument(
            "input",
            help="Input CSV",
        )

        args = parser.parse_args()

        experiment = CatBoostExperiment001(
            args.input,
        )

        experiment.run()


    if __name__ == "__main__":
        main()