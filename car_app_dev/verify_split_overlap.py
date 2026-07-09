import pandas as pd

TRAIN = "experiment001/train_split.csv"
TEST = "experiment001/test_split.csv"
MATCHES = "experiment001/exact_trim_matches.csv"

print("=" * 70)
print("LOADING FILES")
print("=" * 70)

train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)
matches = pd.read_csv(MATCHES)

print(f"Train rows : {len(train):,}")
print(f"Test rows  : {len(test):,}")
print(f"Matches    : {len(matches):,}")
print()

# -------------------------------------------------------
# Unique overlapping keys
# -------------------------------------------------------

keys = (
    matches[
        ["make", "model", "trim"]
    ]
    .drop_duplicates()
)

print(f"Unique overlapping groups : {len(keys):,}")
print()

print("=" * 70)
print("VERIFYING OVERLAPS")
print("=" * 70)

for _, key in keys.iterrows():

    make = key["make"]
    model = key["model"]
    trim = key["trim"]

    train_rows = train[
        (train["make"] == make)
        &
        (train["model"] == model)
        &
        (train["trim"] == trim)
    ]

    test_rows = test[
        (test["make"] == make)
        &
        (test["model"] == model)
        &
        (test["trim"] == trim)
    ]

    print()
    print("=" * 60)
    print(f"{make} | {model} | {trim}")
    print("=" * 60)

    print()
    print("TRAIN")
    print(
        train_rows[
            [
                "make",
                "model",
                "trim",
                "year",
            ]
        ]
    )

    print()
    print("TEST")
    print(
        test_rows[
            [
                "make",
                "model",
                "trim",
                "year",
            ]
        ]
    )

    input("\nPress ENTER for next...")