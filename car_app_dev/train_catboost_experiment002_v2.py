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


def normalize_key(series):
    return (series.fillna("").astype(str).str.strip().str.upper())
class CatBoostExperiment002:

    def __init__(self, csv_path: str):

        self.csv_path = csv_path

        self.output_dir = Path("experiment002")

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

        self.create_train_test_split()

        self.train_model()

        self.predict()

        self.print_summary()

        self.save_predictions()
            
    def create_train_test_split(self):

        print("=" * 70)
        print("CREATING TRAIN / VALIDATION / TEST SPLIT")
        print("=" * 70)

        # ---------------------------------------------------------
        # Match similarity evaluator:
        # remove flagged prices
        # ---------------------------------------------------------

        if "price_flag" in self.df.columns:

            before = len(self.df)

            self.df = self.df[
                ~self.df["price_flag"]
            ].copy()

            print(
                f"Removed flagged rows : "
                f"{before - len(self.df)}"
            )
            

        # ---------------------------------------------------------
        # Missing values
        # ---------------------------------------------------------

        self.df["trim"] = self.df["trim"].fillna("UNKNOWN")

        before = len(self.df)

        self.df = self.df[
            self.df["model"].notna()
        ].copy()

        print(
            f"Dropped rows (missing model): "
            f"{before - len(self.df)}"
        )

        # ---------------------------------------------------------
        # Group key
        # ---------------------------------------------------------

        groups = (

            normalize_key(self.df["make"])
            + "|||"
            + normalize_key(self.df["model"])

        )

        # ---------------------------------------------------------
        # FIRST SPLIT
        #
        # Train+Validation / Test
        # ---------------------------------------------------------

        splitter = GroupShuffleSplit(

            n_splits=1,

            test_size=0.20,

            random_state=42,

        )

        trainval_idx, test_idx = next(

            splitter.split(

                self.df,

                groups=groups,

            )

        )

        trainval_df = self.df.iloc[trainval_idx].copy()

        self.test_df = self.df.iloc[test_idx].copy()

        # ---------------------------------------------------------
        # SECOND SPLIT
        #
        # Train / Validation
        # ---------------------------------------------------------

        trainval_groups = (

            normalize_key(trainval_df["make"])
            + "|||"
            + normalize_key(trainval_df["model"])

        )

        splitter2 = GroupShuffleSplit(

            n_splits=1,

            test_size=0.20,

            random_state=42,

        )

        train_idx, val_idx = next(

            splitter2.split(

                trainval_df,

                groups=trainval_groups,

            )

        )

        self.train_df = trainval_df.iloc[train_idx].copy()

        self.valid_df = trainval_df.iloc[val_idx].copy()
        
        self.train_df.to_csv(
            "experiment002/train_split.csv",
            index=False,
        )

        self.valid_df.to_csv(
            "experiment002/validation_split.csv",
            index=False,
        )

        self.test_df.to_csv(
            "experiment002/test_split.csv",
            index=False,
        )

        print()

        print(f"Train rows      : {len(self.train_df):,}")

        print(f"Validation rows : {len(self.valid_df):,}")

        print(f"Test rows       : {len(self.test_df):,}")

        print()

        train_groups = set(

            self.train_df["make"].astype(str)

            + "|||"

            + self.train_df["model"].astype(str)

        )

        valid_groups = set(

            self.valid_df["make"].astype(str)

            + "|||"

            + self.valid_df["model"].astype(str)

        )

        test_groups = set(

            self.test_df["make"].astype(str)

            + "|||"

            + self.test_df["model"].astype(str)

        )

        print(

            "Train / Validation overlap :",

            len(train_groups & valid_groups),

        )

        print(

            "Train / Test overlap       :",

            len(train_groups & test_groups),

        )

        print(

            "Validation / Test overlap  :",

            len(valid_groups & test_groups),

        )

        print()
        
    def train_model(self):

        print("=" * 70)
        print("TRAINING CATBOOST")
        print("=" * 70)

        X_train = self.train_df[FEATURES].copy()
        X_valid = self.valid_df[FEATURES].copy()

        y_train = np.log1p(self.train_df[TARGET])
        y_valid = np.log1p(self.valid_df[TARGET])

        self.model = CatBoostRegressor(

            iterations=3000,

            learning_rate=0.03,

            depth=8,

            loss_function="RMSE",

            random_seed=42,

            verbose=200,

            od_type="Iter",

            od_wait=200,
        )

        self.model.fit(

            X_train,

            y_train,

            cat_features=CAT_FEATURES,

            eval_set=(X_valid, y_valid),

            use_best_model=True,
        )

        print()
    def predict(self):

        print("=" * 70)
        print("GENERATING PREDICTIONS")
        print("=" * 70)

        X_test = self.test_df[FEATURES].copy()

        pred_log = self.model.predict(X_test)

        pred_price = np.expm1(pred_log)

        self.results = self.test_df.copy()

        self.results["predicted_price"] = pred_price

        self.results["price_diff"] = (
            self.results["predicted_price"]
            - self.results[TARGET]
        )

        self.results["abs_diff"] = (
            self.results["price_diff"]
            .abs()
        )

        self.results["price_diff_pct"] = (
            self.results["price_diff"]
            / self.results[TARGET]
            * 100
        )

        self.results["abs_diff_pct"] = (
            self.results["price_diff_pct"]
            .abs()
        )

        print(
            f"Predictions generated : {len(self.results):,}"
        )

        print()
    
    def print_summary(self):

        r = self.results

        mae = r["abs_diff"].mean()

        median_ae = r["abs_diff"].median()

        mape = r["abs_diff_pct"].mean()

        median_ape = r["abs_diff_pct"].median()

        within5 = (
            r["abs_diff_pct"] <= 5
        ).mean() * 100

        within10 = (
            r["abs_diff_pct"] <= 10
        ).mean() * 100

        within20 = (
            r["abs_diff_pct"] <= 20
        ).mean() * 100

        print("=" * 70)
        print("CATBOOST EXPERIMENT 002")
        print("=" * 70)

        print(f"Rows            : {len(r):,}")

        print()

        print(f"MAE             : {mae:,.0f}")
        print(f"Median AE       : {median_ae:,.0f}")

        print(f"MAPE            : {mape:.2f}%")
        print(f"Median APE      : {median_ape:.2f}%")

        print()

        print(f"Within 5%       : {within5:.2f}%")
        print(f"Within 10%      : {within10:.2f}%")
        print(f"Within 20%      : {within20:.2f}%")

        print()
        
    def save_predictions(self):

        output = (
            self.output_dir
            / "experiment002_predictions.csv"
        )

        self.results.to_csv(
            output,
            index=False,
        )

        print(f"Saved -> {output}")
    
def main():

        parser = argparse.ArgumentParser()

        parser.add_argument(
            "input",
            help="Input CSV",
        )

        args = parser.parse_args()

        experiment = CatBoostExperiment002(
            args.input,
        )

        experiment.run()


if __name__ == "__main__":
    main()