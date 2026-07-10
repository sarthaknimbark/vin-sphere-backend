import pandas as pd

TRAIN = "experiment001/train_split.csv"
TEST = "experiment001/test_split.csv"
MATCH = "experiment001/exact_trim_matches.csv"

train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)
match = pd.read_csv(MATCH)

# -----------------------------------------------------
# Change this number to inspect another row
# -----------------------------------------------------

ROW = 0

r = match.iloc[ROW]

print("=" * 80)
print("ROW FROM exact_trim_matches.csv")
print("=" * 80)

print(r)

print()

# -----------------------------------------------------
# Normalization function
# -----------------------------------------------------

def norm(x):
    return str(x).strip().upper()

# -----------------------------------------------------
# Build normalized keys
# -----------------------------------------------------

train = train.copy()
test = test.copy()

for df in (train, test):

    df["make_norm"] = df["make"].apply(norm)
    df["model_norm"] = df["model"].apply(norm)
    df["trim_norm"] = df["trim"].apply(norm)

train["key"] = (
    train["make_norm"]
    + "|||"
    + train["model_norm"]
    + "|||"
    + train["trim_norm"]
)

test["key"] = (
    test["make_norm"]
    + "|||"
    + test["model_norm"]
    + "|||"
    + test["trim_norm"]
)

key = (
    norm(r["make"])
    + "|||"
    + norm(r["model"])
    + "|||"
    + norm(r["trim"])
)

print("=" * 80)
print("NORMALIZED KEY")
print("=" * 80)

print(key)

print()

# -----------------------------------------------------
# Search train
# -----------------------------------------------------

print("=" * 80)
print("TRAIN MATCHES")
print("=" * 80)

train_hits = train[
    train["key"] == key
]

print(train_hits[
    [
        "make",
        "model",
        "trim",
        "year",
        "key",
    ]
])

print()

# -----------------------------------------------------
# Search test
# -----------------------------------------------------

print("=" * 80)
print("TEST MATCHES")
print("=" * 80)

test_hits = test[
    test["key"] == key
]

print(test_hits[
    [
        "make",
        "model",
        "trim",
        "year",
        "key",
    ]
])

print()

print("=" * 80)
print("SUMMARY")
print("=" * 80)

print("Train matches :", len(train_hits))
print("Test matches  :", len(test_hits))