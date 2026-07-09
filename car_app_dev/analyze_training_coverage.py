import pandas as pd

pred = pd.read_csv("experiment001/experiment001_predictions.csv")
train = pd.read_csv("experiment001/train_split.csv")

# Count training examples for each make-model-trim
counts = (
    train.groupby(
        ["make", "model", "trim"],
        dropna=False,
    )
    .size()
    .rename("train_count")
    .reset_index()
)

# Merge onto predictions
pred = pred.merge(
    counts,
    on=["make", "model", "trim"],
    how="left",
)

pred["train_count"] = (
    pred["train_count"]
    .fillna(0)
    .astype(int)
)

# Bucket the counts
def bucket(n):

    if n == 0:
        return "0"

    if n == 1:
        return "1"

    if n <= 3:
        return "2-3"

    if n <= 5:
        return "4-5"

    if n <= 10:
        return "6-10"

    return "11+"

pred["coverage_bucket"] = pred["train_count"].apply(bucket)

summary = (
    pred.groupby("coverage_bucket")
        .agg(
            rows=("abs_diff_pct", "count"),
            median_ape=("abs_diff_pct", "median"),
            mean_ape=("abs_diff_pct", "mean"),
            mae=("abs_diff", "mean"),
        )
)

print(summary)

summary.to_csv(
    "claude_training_coverage.csv"
)