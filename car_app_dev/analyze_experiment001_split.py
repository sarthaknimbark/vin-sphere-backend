import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "make",
    "model",
    "trim",
    "year",
]


class Experiment001SplitAnalyzer:

    def __init__(
        self,
        train_csv,
        test_csv,
        output_csv,
    ):

        self.train_csv = Path(train_csv)
        self.test_csv = Path(test_csv)
        self.output_csv = Path(output_csv)

    def run(self):

        self.load_data()

        self.validate()
        
        
        self.build_lookup_tables()
        
        self.analyze_test_rows()
        self.print_summary()


        print()
        print("Ready for lookup analysis.")

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

        print(f"Train rows : {len(self.train_df):,}")
        print(f"Test rows  : {len(self.test_df):,}")

    def validate(self):

        print()
        print("=" * 70)
        print("VALIDATING")
        print("=" * 70)

        for column in REQUIRED_COLUMNS:

            if column not in self.train_df.columns:
                raise ValueError(
                    f"Missing '{column}' in train CSV."
                )

            if column not in self.test_df.columns:
                raise ValueError(
                    f"Missing '{column}' in test CSV."
                )

        print("Validation : PASSED")
        
    def build_lookup_tables(self):

        print()
        print("=" * 70)
        print("BUILDING LOOKUP TABLES")
        print("=" * 70)

        # -------------------------------------------------------
        # Normalize
        # -------------------------------------------------------

        train = self.train_df.copy()

        train["make"] = (
            train["make"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )

        train["model"] = (
            train["model"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )

        train["trim"] = (
            train["trim"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )

        # -------------------------------------------------------
        # Sets
        # -------------------------------------------------------

        self.make_set = set(
            train["make"]
        )

        self.model_set = set(
            zip(
                train["make"],
                train["model"],
            )
        )

        self.trim_set = set(
            zip(
                train["make"],
                train["model"],
                train["trim"],
            )
        )

        # -------------------------------------------------------
        # Counts
        # -------------------------------------------------------

        self.model_counts = (
            train
            .groupby(
                ["make", "model"]
            )
            .size()
            .to_dict()
        )

        self.trim_counts = (
            train
            .groupby(
                ["make", "model", "trim"]
            )
            .size()
            .to_dict()
        )

        print(f"Makes          : {len(self.make_set):,}")
        print(f"Make+Models    : {len(self.model_set):,}")
        print(f"Make+Model+Trim: {len(self.trim_set):,}")
        
    def analyze_test_rows(self):

        print()
        print("=" * 70)
        print("ANALYZING TEST SET")
        print("=" * 70)

        test = self.test_df.copy()

        test["make"] = (
            test["make"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )

        test["model"] = (
            test["model"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )

        test["trim"] = (
            test["trim"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )

        known_make = []
        known_model = []
        known_trim = []

        model_count = []
        trim_count = []

        for _, row in test.iterrows():

            make = row["make"]
            model = row["model"]
            trim = row["trim"]

            make_key = make

            model_key = (
                make,
                model,
            )

            trim_key = (
                make,
                model,
                trim,
            )

            known_make.append(
                make_key in self.make_set
            )

            known_model.append(
                model_key in self.model_set
            )

            known_trim.append(
                trim_key in self.trim_set
            )

            model_count.append(
                self.model_counts.get(
                    model_key,
                    0,
                )
            )

            trim_count.append(
                self.trim_counts.get(
                    trim_key,
                    0,
                )
            )

        test["known_make"] = known_make
        test["known_model"] = known_model
        test["known_trim"] = known_trim

        test["train_model_count"] = model_count
        test["train_trim_count"] = trim_count

        self.analysis_df = test

        print("Analysis complete.")
        
    def print_summary(self):

        print()
        print("=" * 70)
        print("SPLIT SUMMARY")
        print("=" * 70)

        total = len(self.analysis_df)

        known_model = self.analysis_df["known_model"].sum()
        unknown_model = total - known_model

        known_trim = self.analysis_df["known_trim"].sum()

        print(f"Total test rows        : {total:,}")

        print()

        print(
            f"Known make+model       : "
            f"{known_model:,} "
            f"({known_model/total:.1%})"
        )

        print(
            f"Unknown make+model     : "
            f"{unknown_model:,} "
            f"({unknown_model/total:.1%})"
        )

        print()

        print(
            f"Known make+model+trim  : "
            f"{known_trim:,} "
            f"({known_trim/total:.1%})"
        )

        print(
            f"Unknown trim           : "
            f"{total-known_trim:,} "
            f"({(total-known_trim)/total:.1%})"
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        required=True,
    )

    parser.add_argument(
        "--test",
        required=True,
    )

    parser.add_argument(
        "--output",
        default="experiment001_split_analysis.csv",
    )

    args = parser.parse_args()

    analyzer = Experiment001SplitAnalyzer(
        train_csv=args.train,
        test_csv=args.test,
        output_csv=args.output,
    )

    analyzer.run()


if __name__ == "__main__":
    main()