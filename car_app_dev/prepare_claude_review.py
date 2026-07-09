import pandas as pd
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="experiment001_predictions.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    print("=" * 70)
    print("Preparing Claude Review Package")
    print("=" * 70)
    print(f"Rows : {len(df):,}")

    # ---------------------------------------------------------
    # First 10 rows (schema)
    # ---------------------------------------------------------

    df.head(10).to_csv(
        "claude_schema_sample.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Best predictions
    # ---------------------------------------------------------

    (
        df.sort_values("abs_diff_pct")
          .head(20)
          .to_csv(
              "claude_best_predictions.csv",
              index=False,
          )
    )

    # ---------------------------------------------------------
    # Worst predictions
    # ---------------------------------------------------------

    (
        df.sort_values("abs_diff_pct", ascending=False)
          .head(20)
          .to_csv(
              "claude_worst_predictions.csv",
              index=False,
          )
    )

    # ---------------------------------------------------------
    # Random sample
    # ---------------------------------------------------------

    (
        df.sample(
            30,
            random_state=42,
        )
        .to_csv(
            "claude_random_sample.csv",
            index=False,
        )
    )

    # ---------------------------------------------------------
    # Performance by make
    # ---------------------------------------------------------

    (
        df.groupby("make")
          .agg(
              count=("abs_diff_pct", "count"),
              median_ape=("abs_diff_pct", "median"),
              mean_ape=("abs_diff_pct", "mean"),
          )
          .sort_values("median_ape")
          .to_csv(
              "claude_by_make.csv",
          )
    )

    # ---------------------------------------------------------
    # Performance by year
    # ---------------------------------------------------------

    (
        df.groupby("year")
          .agg(
              count=("abs_diff_pct", "count"),
              median_ape=("abs_diff_pct", "median"),
              mean_ape=("abs_diff_pct", "mean"),
          )
          .sort_index()
          .to_csv(
              "claude_by_year.csv",
          )
    )

    print()
    print("Generated:")
    print("  claude_schema_sample.csv")
    print("  claude_best_predictions.csv")
    print("  claude_worst_predictions.csv")
    print("  claude_random_sample.csv")
    print("  claude_by_make.csv")
    print("  claude_by_year.csv")

if __name__ == "__main__":
    main()