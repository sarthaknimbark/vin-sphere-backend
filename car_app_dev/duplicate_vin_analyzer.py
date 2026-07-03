from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
class DuplicateVINAnalyzer:

    def __init__(self, df):

        self.df = df.copy()

        self.df["price"] = pd.to_numeric(
            self.df["price"],
            errors="coerce",
        )

        self.output_dir = Path("reports")
        self.output_dir.mkdir(exist_ok=True)
    
    def run(self):

        print("=" * 70)
        print("DUPLICATE VIN ANALYSIS")
        print("=" * 70)

        report = self.build_report()

        report.to_csv(

            self.output_dir /
            "duplicate_vin_investigation.csv",

            index=False,

        )

        print()

        print(
            f"Duplicate VIN rows : {len(report):,}"
        )
        print()

        print(report["severity"].value_counts())

        print()

        print(
            "Largest price ratio :",
            report["price_ratio"].max(),
        )

        print(
            "Largest price difference :",
            report["price_difference"].max(),
        )
        
    def build_report(self):

        if "chassisNumber" not in self.df.columns:
            raise ValueError(
                "Missing chassisNumber column."
            )

        # ------------------------------------
        # Find VINs having multiple prices
        # ------------------------------------

        stats = (

            self.df

            .groupby("chassisNumber")

            .agg(

                duplicate_count=("price", "size"),

                unique_prices=("price", "nunique"),

                min_price=("price", "min"),

                max_price=("price", "max"),

                median_price=("price", "median"),

            )

            .reset_index()

        )

        conflicts = stats[
            stats["unique_prices"] > 1
        ].copy()

        if conflicts.empty:
            return pd.DataFrame()

        # ------------------------------------
        # Expand every duplicate VIN
        # ------------------------------------

        report = self.df.merge(

            conflicts,

            on="chassisNumber",

            how="inner",

        )
        report["price_difference"] = (
            report["max_price"]
            -
            report["min_price"]
        )

        report["price_ratio"] = np.where(

            report["min_price"] > 0,

            report["max_price"] /
            report["min_price"],

            np.nan,

        )
        report["severity"] = "MINOR"

        report.loc[
            report["price_ratio"] >= 2,
            "severity",
        ] = "HIGH"

        report.loc[
            report["price_ratio"] >= 10,
            "severity",
        ] = "CRITICAL"

        report["recommended_action"] = "KEEP"

        report.loc[
            report["severity"] == "HIGH",
            "recommended_action",
        ] = "REVIEW"

        report.loc[
            report["severity"] == "CRITICAL",
            "recommended_action",
        ] = "FIX IMMEDIATELY"
        
        report = report.sort_values(

            [
                "price_ratio",
                "price_difference",
            ],

            ascending=[
                False,
                False,
            ],

        )

        return report
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="CSV file",
    )

    args = parser.parse_args()

    print(f"Loading {args.input}...")

    df = pd.read_csv(
        args.input,
        low_memory=False,
    )

    analyzer = DuplicateVINAnalyzer(df)

    analyzer.run()
