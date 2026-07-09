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

    def analyze(self):

        print("=" * 70)
        print("ANALYZING TEST SPLIT")
        print("=" * 70)

        rows = []

        for _, row in self.test_df.iterrows():

            make = row["make"]
            model = row["model"]
            trim = row["trim"]

            model_key = (make, model)
            trim_key = (make, model, trim)

            known_make = make in self.make_set
            known_model = model_key in self.model_set
            known_trim = trim_key in self.trim_set

            model_count = self.model_counts.get(model_key, 0)
            trim_count = self.trim_counts.get(trim_key, 0)

            if known_trim:
                difficulty = 0
            elif known_model:
                difficulty = 1
            elif known_make:
                difficulty = 2
            else:
                difficulty = 3

            r = row.copy()
            r["known_make"] = known_make
            r["known_model"] = known_model
            r["known_trim"] = known_trim
            r["train_model_count"] = model_count
            r["train_trim_count"] = trim_count
            r["difficulty"] = difficulty

            rows.append(r)

        self.analysis_df = pd.DataFrame(rows)

        # --------------------------------------------------
        # Exact trim matches
        # --------------------------------------------------

        self.trim_matches = self.analysis_df[
            self.analysis_df["known_trim"]
        ].copy()

        self.exact_trim_matches = len(self.trim_matches)

        print("Analysis complete")
        print()

    def print_summary(self):

        df = self.analysis_df
        total = len(df)

        print("=" * 70)
        print("SPLIT SUMMARY")
        print("=" * 70)

        def pct(v):
            return 100.0 * v / total if total else 0.0

        kmake = int(df["known_make"].sum())
        kmodel = int(df["known_model"].sum())
        ktrim = int(df["known_trim"].sum())

        print(f"Total test rows          : {total:,}")
        print()
        print(f"Known make               : {kmake:,} ({pct(kmake):.2f}%)")
        print(f"Known make+model         : {kmodel:,} ({pct(kmodel):.2f}%)")
        print(f"Known make+model+trim    : {ktrim:,} ({pct(ktrim):.2f}%)")
        print(f"Unknown make+model       : {total-kmodel:,} ({pct(total-kmodel):.2f}%)")
        print(f"Known model/new trim     : {kmodel-ktrim:,} ({pct(kmodel-ktrim):.2f}%)")
        print()
        print("Difficulty distribution")
        for d in range(4):
            c = int((df["difficulty"] == d).sum())
            print(f"  {d}: {c:,} ({pct(c):.2f}%)")
        print()
        print(f"Median train rows/model  : {df['train_model_count'].median():.0f}")
        print(f"Mean train rows/model    : {df['train_model_count'].mean():.2f}")
        print(f"Max train rows/model     : {df['train_model_count'].max():.0f}")

        print()

        print("Rows with exactly 1 training example :",
            (df["train_model_count"] == 1).sum())

        print("Rows with >20 training examples      :",
            (df["train_model_count"] > 20).sum())

        print()

        if self.trim_leakage == 0:
            print("Exact trim leakage       : PASS")
        else:
            print(f"Exact trim leakage       : FAIL ({self.trim_leakage:,} rows)")

        print()

        # --------------------------------------------------
        # Known trim breakdown
        # --------------------------------------------------

        if self.trim_leakage > 0:

            print("=" * 70)
            print("KNOWN TRIM BREAKDOWN")
            print("=" * 70)

            print(
                self.trim_matches[
                    [
                        "make",
                        "model",
                        "trim",
                        "train_trim_count",
                    ]
                ]
                .sort_values(
                    "train_trim_count",
                    ascending=False,
                )
                .head(20)
            )

        print()
    def save(self):

        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self.analysis_df.to_csv(self.output_csv, index=False)
        leakage_file = (
            self.output_csv.parent
            / "trim_matches.csv"
        )

        self.trim_matches.to_csv(
            leakage_file,
            index=False,
        )

        print(f"Saved -> {leakage_file}")
        print(f"Saved -> {self.output_csv}")

    def run(self):

        self.load_data()
        self.validate()
        self.normalize()
        self.build_lookups()
        self.analyze()
        self.print_summary()
        self.save()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument(
        "--output",
        default="split_analysis.csv",
    )

    args = parser.parse_args()

    analyzer = SplitAnalyzer(
        args.train,
        args.test,
        args.output,
    )

    analyzer.run()


if __name__ == "__main__":
    main()
