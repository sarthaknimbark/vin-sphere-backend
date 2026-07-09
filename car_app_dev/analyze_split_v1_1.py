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

            difficulty_name = {
                0: "Exact Trim Seen",
                1: "Known Model / New Trim",
                2: "Known Make / New Model",
                3: "Unknown Make",
            }[difficulty]

            r = row.copy()
            r["known_make"] = known_make
            r["known_model"] = known_model
            r["known_trim"] = known_trim
            r["train_model_count"] = model_count
            r["train_trim_count"] = trim_count
            r["difficulty"] = difficulty
            r["difficulty_name"] = difficulty_name

            rows.append(r)

        self.analysis_df = pd.DataFrame(rows)

        # --------------------------------------------------
        # Exact trim matches
        # --------------------------------------------------

        self.exact_trim_matches = self.analysis_df[
            self.analysis_df["known_trim"]
        ].copy()

        self.exact_trim_match_count = len(self.exact_trim_matches)

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

        print("TRAINING COVERAGE HISTOGRAM")
        print("-" * 40)

        buckets = {
            "0 rows": (df["train_model_count"] == 0),
            "1 row": (df["train_model_count"] == 1),
            "2-5 rows": df["train_model_count"].between(2,5),
            "6-20 rows": df["train_model_count"].between(6,20),
            "21-100 rows": df["train_model_count"].between(21,100),
            "100+ rows": (df["train_model_count"] > 100),
        }

        for label, mask in buckets.items():
            print(f"{label:<15}: {int(mask.sum()):,}")

        print()

        if self.exact_trim_match_count == 0:
            print("Exact trim matches       : PASS")
        else:
            print(f"Exact trim matches       : FAIL ({self.exact_trim_match_count:,} rows)")

        print()

        # --------------------------------------------------
        # Known trim breakdown
        # --------------------------------------------------

        if self.exact_trim_match_count > 0:

            print("=" * 70)
            print("EXACT TRIM MATCH BREAKDOWN")
            print("=" * 70)

            print(
                self.exact_trim_matches[
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
            / "exact_trim_matches.csv"
        )

        self.exact_trim_matches.to_csv(
            leakage_file,
            index=False,
        )

        print(f"Saved -> {leakage_file}")
        print(f"Saved -> {self.output_csv}")
    
    def save_summary(self):

        df = self.analysis_df
        total = len(df)

        def pct(x):
            return 100.0 * x / total if total else 0.0

        kmake = int(df["known_make"].sum())
        kmodel = int(df["known_model"].sum())
        ktrim = int(df["known_trim"].sum())

        difficulty_counts = (
            df["difficulty_name"]
            .value_counts()
            .sort_index()
        )

        lines = []

        lines.append("=" * 70)
        lines.append("EXPERIMENT SPLIT ANALYSIS SUMMARY")
        lines.append("=" * 70)
        lines.append("")

        lines.append(f"Train rows : {len(self.train_df):,}")
        lines.append(f"Test rows  : {len(self.test_df):,}")
        lines.append("")

        lines.append("=" * 70)
        lines.append("COVERAGE")
        lines.append("=" * 70)

        lines.append(f"Known make               : {kmake:,} ({pct(kmake):.2f}%)")
        lines.append(f"Known make+model         : {kmodel:,} ({pct(kmodel):.2f}%)")
        lines.append(f"Known make+model+trim    : {ktrim:,} ({pct(ktrim):.2f}%)")
        lines.append(f"Unknown make+model       : {total-kmodel:,} ({pct(total-kmodel):.2f}%)")
        lines.append(f"Known model / new trim   : {kmodel-ktrim:,} ({pct(kmodel-ktrim):.2f}%)")
        lines.append("")

        lines.append("=" * 70)
        lines.append("DIFFICULTY")
        lines.append("=" * 70)

        for name, count in difficulty_counts.items():
            lines.append(
                f"{name:<30} {count:>6,} ({pct(count):6.2f}%)"
            )

        lines.append("")

        lines.append("=" * 70)
        lines.append("TRAINING COVERAGE")
        lines.append("=" * 70)

        lines.append(
            f"Median rows/model : {df['train_model_count'].median():.0f}"
        )

        lines.append(
            f"Mean rows/model   : {df['train_model_count'].mean():.2f}"
        )

        lines.append(
            f"Max rows/model    : {df['train_model_count'].max():.0f}"
        )

        lines.append("")

        buckets = {
            "0 rows": (df["train_model_count"] == 0),
            "1 row": (df["train_model_count"] == 1),
            "2-5 rows": df["train_model_count"].between(2,5),
            "6-20 rows": df["train_model_count"].between(6,20),
            "21-100 rows": df["train_model_count"].between(21,100),
            "100+ rows": (df["train_model_count"] > 100),
        }

        for label, mask in buckets.items():
            lines.append(
                f"{label:<15}: {int(mask.sum()):,}"
            )

        lines.append("")

        lines.append("=" * 70)
        lines.append("EXACT TRIM MATCHES")
        lines.append("=" * 70)

        lines.append(
            f"Exact trim matches : {self.exact_trim_match_count:,}"
        )

        summary_file = (
            self.output_csv.parent /
            "split_analysis_summary.txt"
        )

        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"Saved -> {summary_file}")

    def run(self):

        self.load_data()
        self.validate()
        self.normalize()
        self.build_lookups()
        self.analyze()
        self.print_summary()
        self.save()
        self.save_summary()


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
