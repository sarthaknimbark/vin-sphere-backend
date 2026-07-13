import pandas as pd

OLD = "experiment001_old/experiment001_predictions.csv"
NEW = "experiment001/experiment001_predictions.csv"

old = pd.read_csv(OLD)
new = pd.read_csv(NEW)

print("=" * 70)
print("ROW COUNTS")
print("=" * 70)

print("Old :", len(old))
print("New :", len(new))

print()

for name, df in [("OLD", old), ("NEW", new)]:

    ape = df["abs_diff_pct"]
    ae = df["abs_diff"]

    print("=" * 70)
    print(name)
    print("=" * 70)

    print(f"MAE        : {ae.mean():,.2f}")
    print(f"Median AE  : {ae.median():,.2f}")
    print(f"MAPE       : {ape.mean():.2f}%")
    print(f"Median APE : {ape.median():.2f}%")

    print(f"Within 5%  : {(ape <= 5).mean()*100:.2f}%")
    print(f"Within 10% : {(ape <= 10).mean()*100:.2f}%")
    print(f"Within 20% : {(ape <= 20).mean()*100:.2f}%")
    print()