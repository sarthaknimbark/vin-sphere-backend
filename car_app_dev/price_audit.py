from __future__ import annotations

from pathlib import Path
import time

start = time.time()

import numpy as np
import pandas as pd


class PriceAudit:

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
        print("PRICE AUDIT")
        print("=" * 70)

        self.validate_dataset()

        self.report_impossible_prices()

        self.report_duplicate_vins()

        self.report_group_statistics()

        self.report_statistical_outliers()

        self.report_price_stability()

        self.report_group_size_distribution()

        self.build_resolution_report()

        self.write_summary()

        print("\nFinished.")
    def report_impossible_prices(self):

        bad = self.df[
            (self.df["price"].isna())
            |
            (self.df["price"] <= 100)
        ].copy()

        bad.to_csv(
            self.output_dir / "impossible_prices.csv",
            index=False,
        )

        print(
            f"Impossible prices : {len(bad):,}"
        )
    def report_duplicate_vins(self):

        if "chassisNumber" not in self.df.columns:
            return

        duplicate = (

            self.df

            .groupby("chassisNumber")

            .agg(

                rows=("price", "size"),

                unique_prices=("price", "nunique"),

                min_price=("price", "min"),

                max_price=("price", "max"),

            )

            .reset_index()

        )

        duplicate = duplicate[
            duplicate["unique_prices"] > 1
        ]

        duplicate.to_csv(

            self.output_dir / "duplicate_vins.csv",

            index=False,

        )

        print(

            f"Duplicate VINs : {len(duplicate):,}"

        )
    def report_group_statistics(self):

        required = [
            "make",
            "model",
            "year",
            "trim",
            "price",
        ]

        if not all(col in self.df.columns for col in required):
            return

        stats = (

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

        stats["iqr"] = stats["q3"] - stats["q1"]

        stats["cv"] = (
            stats["std_price"] /
            stats["mean_price"]
        )

        stats = stats.sort_values(
            [
                "cv",
                "count",
            ],
            ascending=[
                False,
                False,
            ],
        )

        stats.to_csv(
            self.output_dir / "high_variance_groups.csv",
            index=False,
        )

        print(
            f"Vehicle groups : {len(stats):,}"
        )

        print(
            f"High variance groups (>30% CV): "
            f"{(stats['cv'] > 0.30).sum():,}"
        )
    def write_summary(self):

        impossible = len(
            self.df[
                self.df["price"].isna()
                |
                (self.df["price"] <= 100)
            ]
        )

        duplicate = 0

        if "chassisNumber" in self.df.columns:

            duplicate = (

                self.df

                .groupby("chassisNumber")["price"]

                .nunique()

                .gt(1)

                .sum()

            )

        grouped = self.df.groupby(
            [
                "make",
                "model",
                "year",
                "trim",
            ],
            dropna=False,
        )

        vehicle_groups = len(grouped)

        summary = f"""
    ==================================================
    PRICE AUDIT SUMMARY
    ==================================================

    Rows                    : {len(self.df):,}

    Vehicle Groups          : {vehicle_groups:,}

    Impossible Prices       : {impossible:,}

    Duplicate VIN Conflicts : {duplicate:,}

    Reports Generated

        impossible_prices.csv
        duplicate_vins.csv
        high_variance_groups.csv
        statistical_outliers.csv
        price_stability.csv
        group_size_distribution.csv
        price_resolution_report.csv

    ==================================================
    """

        print(summary)

        with open(
            self.output_dir / "audit_summary.txt",
            "w",
            encoding="utf-8",
        ) as f:

            f.write(summary)
    def validate_dataset(self):

        required = [
            "chassisNumber",
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
                f"Missing required columns: {missing}"
            )

        print("Dataset validation : PASSED")
    def report_group_size_distribution(self):

        sizes = (

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

            .size()

            .rename("group_size")

            .reset_index()

        )

        distribution = (

            sizes["group_size"]

            .value_counts()

            .sort_index()

            .rename_axis("group_size")

            .reset_index(name="num_groups")

        )

        distribution.to_csv(

            self.output_dir / "group_size_distribution.csv",

            index=False,

        )

        print(

            f"Unique vehicle groups : {len(sizes):,}"

        )

        print(

            f"Largest group size : {sizes['group_size'].max()}"

        )
    def build_resolution_report(self):

        report = self.df.copy()

        report["severity"] = "SAFE"
        report["reason"] = ""
        report["action"] = "KEEP"

        # -------------------------------
        # Impossible prices
        # -------------------------------
        mask = (
            report["price"].isna()
            |
            (report["price"] <= 100)
        )

        report.loc[mask, "severity"] = "CRITICAL"
        report.loc[mask, "reason"] = "Impossible price"
        report.loc[mask, "action"] = "REJECT"

        # -------------------------------
        # Duplicate VIN conflicts
        # -------------------------------
        if "chassisNumber" in report.columns:

            duplicate_vins = (

                report

                .groupby("chassisNumber")["price"]

                .nunique()

            )

            duplicate_vins = duplicate_vins[
                duplicate_vins > 1
            ].index

            mask = report["chassisNumber"].isin(
                duplicate_vins
            )

            report.loc[
                (mask)
                &
                (report["severity"] == "SAFE"),
                "severity",
            ] = "HIGH"

            report.loc[
                (mask)
                &
                (report["reason"] == ""),
                "reason",
            ] = "Duplicate VIN conflict"

            report.loc[
                (mask)
                &
                (report["action"] == "KEEP"),
                "action",
            ] = "REVIEW"

        report.to_csv(
            self.output_dir / "price_resolution_report.csv",
            index=False,
        )

        print(
            f"Resolution report : {len(report):,} rows"
        )
        
    def report_statistical_outliers(self):

        required = [
            "make",
            "model",
            "year",
            "trim",
            "price",
        ]

        if not all(c in self.df.columns for c in required):
            return

        outliers = []

        grouped = self.df.groupby(
            [
                "make",
                "model",
                "year",
                "trim",
            ],
            dropna=False,
        )

        for _, group in grouped:

            # Need enough samples
            if len(group) < 5:
                continue

            q1 = group["price"].quantile(0.25)
            q3 = group["price"].quantile(0.75)

            iqr = q3 - q1

            if iqr == 0:
                continue

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            bad = group[
                (group["price"] < lower)
                |
                (group["price"] > upper)
            ]

            if not bad.empty:

                bad = bad.copy()

                bad["group_median"] = group["price"].median()
                bad["group_q1"] = q1
                bad["group_q3"] = q3
                bad["group_iqr"] = iqr
                bad["lower_limit"] = lower
                bad["upper_limit"] = upper

                outliers.append(bad)

        if outliers:

            outliers = pd.concat(
                outliers,
                ignore_index=True,
            )

        else:

            outliers = pd.DataFrame()

        outliers.to_csv(
            self.output_dir / "statistical_outliers.csv",
            index=False,
        )

        print(
            f"Statistical outliers : {len(outliers):,}"
        )
        
    def report_price_stability(self):

        required = [
            "make",
            "model",
            "year",
            "trim",
            "price",
        ]

        if not all(c in self.df.columns for c in required):
            return

        stats = (

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
            )

            .reset_index()

        )

        stats["cv"] = (
            stats["std_price"] /
            stats["mean_price"]
        ).fillna(0)

        stats["stability_score"] = (
            1 / (1 + stats["cv"])
        )

        stats = stats.sort_values(
            "stability_score",
            ascending=False,
        )

        stats.to_csv(
            self.output_dir / "price_stability.csv",
            index=False,
        )

        print(
            f"Price stability groups : {len(stats):,}"
        )
        
        
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="CSV file",
    )

    args = parser.parse_args()

    print(f"Loading {args.input} ...")

    df = pd.read_csv(
        args.input,
        low_memory=False,
    )

    print(f"Rows : {len(df):,}")

    audit = PriceAudit(df)

    audit.run()
    
elapsed = time.time() - start

print(
    f"Elapsed : {elapsed:.2f} sec"
)