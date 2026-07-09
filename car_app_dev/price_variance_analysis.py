from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


class PriceVarianceAnalysis:

    def __init__(
        self,
        df: pd.DataFrame,
    ):

        self.df = df.copy()

        self.df["price"] = pd.to_numeric(
            self.df["price"],
            errors="coerce",
        )

        self.output_dir = Path("reports")
        self.output_dir.mkdir(exist_ok=True)

    def run(self):

        print("=" * 70)
        print("PRICE VARIANCE ANALYSIS")
        print("=" * 70)

        self.validate_dataset()

        self.compute_group_statistics()

        self.compute_group_size_distribution()

        self.compute_variance_statistics()

        self.report_low_variance_groups()

        self.report_high_variance_groups()

        self.print_summary()
    
    def validate_dataset(self):

            required = [
                "make",
                "model",
                "year",
                "trim",
                "price",
            ]

            missing = [
                c
                for c in required
                if c not in self.df.columns
            ]

            if missing:

                raise ValueError(
                    f"Missing columns: {missing}"
                )

            print("Dataset validation : PASSED")

            print(
                f"Rows : {len(self.df):,}"
            )
            
    def compute_group_statistics(self):

            grouped = (

                self.df

                .groupby(
                    [
                        "make",
                        "model",
                        "year",
                        "trim",
                    ],
                    dropna=False,
                )

                .agg(
                    count=("price", "count"),
                    median_price=("price", "median"),
                    mean_price=("price", "mean"),
                    std_price=("price", "std"),
                    min_price=("price", "min"),
                    max_price=("price", "max"),
                    q1=("price", lambda x: x.quantile(0.25)),
                    q3=("price", lambda x: x.quantile(0.75)),
                )

                .reset_index()

            )

            grouped["iqr"] = (
                grouped["q3"]
                -
                grouped["q1"]
            )

            grouped["cv"] = (
                grouped["std_price"]
                /
                grouped["mean_price"]
            )

            self.group_stats = grouped

            grouped.to_csv(

                self.output_dir /
                "group_price_statistics.csv",

                index=False,

            )

            print(
                f"Vehicle groups : {len(grouped):,}"
            )
    def compute_group_size_distribution(self):

        sizes = self.group_stats["count"]

        summary = pd.DataFrame({

            "metric": [

                "Total Groups",

                "Groups >= 2",

                "Groups >= 3",

                "Groups >= 5",

                "Groups >= 10",

                "Groups >= 20",

                "Largest Group",

                "Median Group Size",

            ],

            "value": [

                len(sizes),

                (sizes >= 2).sum(),

                (sizes >= 3).sum(),

                (sizes >= 5).sum(),

                (sizes >= 10).sum(),

                (sizes >= 20).sum(),

                sizes.max(),

                sizes.median(),

            ],

        })

        summary.to_csv(

            self.output_dir /
            "group_size_distribution.csv",

            index=False,

        )

        self.group_size_summary = summary

        print("Group size distribution computed.")
        
    def compute_variance_statistics(self):

        stats = self.group_stats.copy()

        stats = stats[
            stats["count"] >= 5
        ].copy()

        stats["iqr_pct"] = (

            stats["iqr"]

            /

            stats["median_price"]

        ) * 100

        stats["cv_pct"] = (

            stats["cv"]

        ) * 100

        self.variance_stats = stats

        stats.to_csv(

            self.output_dir /
            "variance_statistics.csv",

            index=False,

        )

        print(

            f"Groups with >=5 vehicles : {len(stats):,}"

        ) 
    def report_low_variance_groups(self):

        stats = self.variance_stats.copy()

        stats = stats.sort_values(
            "iqr_pct",
            ascending=True,
        )

        stats.head(100).to_csv(
            self.output_dir / "low_variance_groups.csv",
            index=False,
        )

        print(
            f"Low variance groups : {len(stats.head(100))}"
        )


    def report_high_variance_groups(self):

        stats = self.variance_stats.copy()

        stats = stats.sort_values(
            "iqr_pct",
            ascending=False,
        )

        stats.head(100).to_csv(
            self.output_dir / "high_variance_groups.csv",
            index=False,
        )

        print(
            f"High variance groups : {len(stats.head(100))}"
        )
    def print_summary(self):

        print()
        print("=" * 70)
        print("PRICE VARIANCE ANALYSIS SUMMARY")
        print("=" * 70)

        print(f"Rows                : {len(self.df):,}")
        print(f"Vehicle Groups      : {len(self.group_stats):,}")
        print(f"Groups >=5 vehicles : {len(self.variance_stats):,}")

        print("=" * 70)   

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

    analysis = PriceVarianceAnalysis(df)

    analysis.run()