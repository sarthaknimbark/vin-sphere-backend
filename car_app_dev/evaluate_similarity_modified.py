"""
evaluate_price_predictions.py
──────────────────────────────
Leave-one-out evaluation of VehicleSimilaritySearch on a price CSV.

For every row i:
  - Build reference from all rows EXCEPT row i
  - Predict price for row i
  - Record predicted_price, price_diff, price_diff_pct, provenance

Usage
-----
    python evaluate_price_predictions.py prices.csv [output.csv] [--k 10]

Output columns (appended to original CSV)
------------------------------------------
    predicted_price   — weighted-average price from k nearest neighbours
    price_diff        — predicted_price - actual price  (signed)
    price_diff_pct    — price_diff / actual price × 100  (signed %)
    abs_diff          — |price_diff|
    abs_diff_pct      — |price_diff_pct|
    n_neighbors       — how many neighbours contributed
    avg_similarity    — mean similarity score of neighbours
    provenance        — exact | fuzzy_model | make_fallback | unresolved
"""

import argparse
import sys
import time
import logging
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd


# Import from the same directory
sys.path.insert(0, str(Path(__file__).parent))
from version1_v2 import VehicleSimilaritySearch, VehicleQuery
logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")


# ──────────────────────────────────────────────────────────────────────────────
# Leave-one-out engine (thin wrapper that removes row i before each predict)
# ──────────────────────────────────────────────────────────────────────────────

class LOOPriceEvaluator:
    """
    Leave-one-out wrapper around VehicleSimilaritySearch.

    Rebuilding the full index for every row would be O(n²) — too slow.
    Instead we build the index once, then post-filter search results to
    exclude any neighbour that shares the same (make, model, year, trim, price)
    as the query row.  This is equivalent to leave-one-out for all practical
    purposes: the query row cannot contribute to its own prediction.
    """

    def __init__(self, df: pd.DataFrame, pool: pd.DataFrame, k: int = 10) -> None:
        self.df  = df.reset_index(drop=True)
        self.k   = k
        self._engine = VehicleSimilaritySearch(pool)
        self.neighbor_usage = Counter()
        self._key_cols = [c for c in ("make", "model", "year", "trim", "price")
                          if c in df.columns]

    def _row_to_query(self, row: pd.Series) -> VehicleQuery:
        return VehicleQuery(
            make  = str(row["make"])  if pd.notna(row.get("make"))  else "",
            model = str(row["model"]) if pd.notna(row.get("model")) else None,
            year  = int(row["year"])  if pd.notna(row.get("year"))  else None,
            trim  = str(row["trim"])  if pd.notna(row.get("trim"))  else None,
        )

    def _exclude_self(
        self,
        neighbors: pd.DataFrame,
        row: pd.Series,
    ) -> pd.DataFrame:
        """
        Remove any neighbour that is identical to the query row on all key
        columns.  Handles duplicate rows (same vehicle listed twice) conservatively
        — removes only ONE matching row per call.
        """
        if neighbors.empty:
            return neighbors

        mask = pd.Series([True] * len(neighbors), index=neighbors.index)
        for col in self._key_cols:
            if col not in neighbors.columns:
                continue
            query_val = row.get(col)
            if pd.isna(query_val):
                mask &= neighbors[col].isna()
            else:
                mask &= (neighbors[col] == query_val)

        # Drop at most one matching row (the query row itself)
        match_idx = neighbors[mask].index
        if len(match_idx) > 0:
            neighbors = neighbors.drop(match_idx[0])

        return neighbors

    def predict_row(self, row: pd.Series) -> dict:
        query     = self._row_to_query(row)
        neighbors = self._engine.search(query)
        if row.name == 0:
            print(neighbors.columns.tolist())
            print(neighbors.head())
        neighbors = self._exclude_self(neighbors, row)
        neighbors = neighbors.head(self.k)

        if neighbors.empty or "price" not in neighbors.columns:
            return {
                "predicted_price": None,
                "n_neighbors":     0,
                "avg_similarity": None,
                "provenance":      "unresolved",
            }

        valid = neighbors[neighbors["price"].notna()]
        if "chassisNumber" in valid.columns:
            self.neighbor_usage.update(
                valid["chassisNumber"].tolist()
            )
        if valid.empty:
            return {
                "predicted_price": None,
                "n_neighbors":     0,
                "avg_similarity": None,
                "provenance":      neighbors["_tier"].iloc[0],
            }

        prices = valid["price"].to_numpy(dtype=np.float64)

        year_diff = valid["_year_diff"].to_numpy(dtype=np.float64)

        weights = 1.0 / (year_diff + 1.0)

        predicted_price = np.median(prices)
        if row.name == 0:
            debug = valid[[
                "trim",
                "year",
                "price",
                "_year_diff",
                "_tier",
            ]].copy()

            debug["weight"] = weights

            print(debug)
            print("Prediction:", predicted_price)

        return {
            "predicted_price": round(float(predicted_price), 2),
            "n_neighbors":     len(valid),
            "avg_similarity":  None,
            "provenance":      valid["_tier"].iloc[0] if "_tier" in valid.columns else "model",
        }

    def run(self) -> pd.DataFrame:
        """
        Run leave-one-out over all rows.  Returns original df with new columns.
        """
        n       = len(self.df)
        results = []
        t0      = time.time()

        for i, row in self.df.iterrows():
            pred = self.predict_row(row)
            results.append(pred)

            if (i + 1) % 500 == 0 or (i + 1) == n:
                elapsed = time.time() - t0
                rate    = (i + 1) / elapsed
                eta     = (n - i - 1) / rate
                print(f"  {i+1:>6}/{n}  ({rate:.0f} rows/s  ETA {eta:.0f}s)", end="\r")

        print()  # newline after progress

        out = self.df.copy()
        preds_df = pd.DataFrame(results, index=self.df.index)

        out["predicted_price"] = preds_df["predicted_price"]
        out["price_diff"]      = out["predicted_price"] - out["price"]
        out["price_diff_pct"]  = (out["price_diff"] / out["price"] * 100).round(2)
        out["abs_diff"]        = out["price_diff"].abs()
        out["abs_diff_pct"]    = out["price_diff_pct"].abs()
        out["n_neighbors"]     = preds_df["n_neighbors"]
        out["avg_similarity"]  = preds_df["avg_similarity"]
        out["provenance"]      = preds_df["provenance"]
        
        usage = (
            pd.Series(self.neighbor_usage)
            .rename_axis("chassisNumber")
            .reset_index(name="times_used")
            .sort_values(
                "times_used",
                ascending=False,
            )
        )

        usage.to_csv(
            "neighbor_usage.csv",
            index=False,
        )

        print(
            f"\nSaved neighbor usage -> neighbor_usage.csv ({len(usage):,} VINs)"
        )

        return out


# ──────────────────────────────────────────────────────────────────────────────
# Summary statistics
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(out: pd.DataFrame) -> None:
    resolved = out[out["predicted_price"].notna()]
    n_total  = len(out)
    n_res    = len(resolved)

    print("\n" + "=" * 60)
    print("LEAVE-ONE-OUT EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total rows        : {n_total:,}")
    print(f"Resolved          : {n_res:,}  ({100*n_res/n_total:.1f}%)")
    print(f"Unresolved        : {n_total - n_res:,}")

    if n_res == 0:
        print("No resolved rows — check make/model column names.")
        return

    print(f"\n── Error (resolved rows only) ──────────────────────")
    print(f"  MAE             : {resolved['abs_diff'].mean():>12,.0f}")
    print(f"  Median AE       : {resolved['abs_diff'].median():>12,.0f}")
    print(f"  MAPE            : {resolved['abs_diff_pct'].mean():>11.1f}%")
    print(f"  Median APE      : {resolved['abs_diff_pct'].median():>11.1f}%")
    print(f"  Max abs error   : {resolved['abs_diff'].max():>12,.0f}")
    print(f"  Rows within 5%  : {(resolved['abs_diff_pct'] <= 5).sum():>6,}  "
          f"({100*(resolved['abs_diff_pct'] <= 5).mean():.1f}%)")
    print(f"  Rows within 10% : {(resolved['abs_diff_pct'] <= 10).sum():>6,}  "
          f"({100*(resolved['abs_diff_pct'] <= 10).mean():.1f}%)")
    print(f"  Rows within 20% : {(resolved['abs_diff_pct'] <= 20).sum():>6,}  "
          f"({100*(resolved['abs_diff_pct'] <= 20).mean():.1f}%)")

    print(f"\n── By provenance ───────────────────────────────────")
    prov_summary = (
        resolved.groupby("provenance")
        .agg(
            count=("abs_diff_pct", "count"),
            median_ape=("abs_diff_pct", "median"),
            mae=("abs_diff", "mean"),
        )
        .sort_values("count", ascending=False)
    )
    print(prov_summary.to_string())

    print(f"\n── Worst 10 predictions ────────────────────────────")
    worst_cols = [c for c in ("make", "model", "year", "trim", "price",
                               "predicted_price", "abs_diff_pct", "provenance")
                  if c in resolved.columns]
    print(resolved.nlargest(10, "abs_diff_pct")[worst_cols].to_string(index=False))
    print("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Leave-one-out price evaluation")
    parser.add_argument("input_csv",  help="CSV with make, model, year, trim, price")
    parser.add_argument("output_csv", nargs="?", help="Output path (default: input_evaluated.csv)")
    parser.add_argument("--k", type=int, default=10, help="Neighbours to use (default 10)")
    args = parser.parse_args()

    input_path  = Path(args.input_csv)
    output_path = Path(args.output_csv) if args.output_csv else \
                  input_path.parent / (input_path.stem + "_evaluated.csv")

    print(f"Loading {input_path} …")
    df = pd.read_csv(input_path)
    print(f"  {len(df):,} rows  |  columns: {list(df.columns)}")

    required = {"make", "model", "price"}
    missing  = required - set(df.columns)
    if missing:
        print(f"ERROR: CSV missing required columns: {missing}")
        sys.exit(1)

    # Clean price: strip commas, currency symbols, whitespace → float
    raw_price_sample = df["price"].dropna().iloc[0] if df["price"].notna().any() else ""
    df["price"] = (
        df["price"]
        .astype(str)
        .str.replace(r"[,\s]",  "", regex=True)   # remove commas and spaces
        .str.replace(r"[^\d.]", "", regex=True)   # strip any currency symbols
        .replace("", float("nan"))
        .astype(float)
    )
    n_null = df["price"].isna().sum()
    print(f"  price cleaned  (sample raw: {repr(raw_price_sample)!s:<14} → float) "
          f"| {n_null:,} nulls")

    # Neighbor pool = clean rows only
    if "price_flag" in df.columns:
        df_pool = df[~df["price_flag"]].copy()
        print(f"  neighbor pool  : {len(df_pool):,} rows  ({df['price_flag'].sum():,} flagged rows excluded)")
    else:
        df_pool = df.copy()
        print(f"  neighbor pool  : {len(df_pool):,} rows  (no price_flag column)")

    print(f"\nRunning leave-one-out evaluation (k={args.k}) …")
    evaluator = LOOPriceEvaluator(df, pool=df_pool, k=args.k)
    result    = evaluator.run()

    result.to_csv(output_path, index=False)
    print(f"\nSaved → {output_path}")

    print_summary(result)
    


if __name__ == "__main__":
    main()
    
