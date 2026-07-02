"""
similarity_search.py
====================
Similarity-based car price prediction module.

Given a query vehicle and a pool of reference vehicles, finds the K most
similar vehicles by a composite similarity score and predicts price as the
similarity-weighted median of their prices.

Feature space
-------------
Hard filters  : make, model  (must match; tiered fallback below)
Soft scores   : year proximity  (exponential decay)
                trim match      (exact, null-aware)
                cylinders match (exact, null-aware) — used if column present
                regionalSpec    (exact, null-aware) — used if column present

Matching tiers (tried in order until MIN_NEIGHBORS is reached)
--------------------------------------------------------------
  0 → filter on (make, model)       → provenance: "model_year"  or "model_match"
  1 → filter on (make) only         → provenance: "make_fallback"
  unresolved if still < MIN_NEIGHBORS

Provenance rules (applied to the tier-0 pool only; tier-1 always = "make_fallback")
------------------------------------------------------------------------------------
  "model_year"   avg year score of selected neighbors >= YEAR_SCORE_TIER1_THRESH
  "model_match"  avg year score < YEAR_SCORE_TIER1_THRESH

Public API
----------
  predict(query, pool, k, exclude_self) -> dict
  predict_batch(queries, pool, k)       -> pd.DataFrame
  loo_evaluate(df, k)                   -> pd.DataFrame     # leave-one-out

Usage
-----
  python similarity_search.py clean_pool.csv [--k 10] [--out results.csv]
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

import numpy as np
import pandas as pd

# ── Tuneable constants ────────────────────────────────────────────────────────
YEAR_DECAY              = 3.0   # e-folding scale: ±3 yrs → score × e⁻¹ ≈ 0.37
K_NEIGHBORS             = 10    # default neighbours
MIN_NEIGHBORS           = 3     # minimum to emit a prediction; else "unresolved"
YEAR_SCORE_TIER1_THRESH = 0.72  # avg year score ≥ this → "model_year" provenance
                                 # 0.72 ≈ exp(-1), i.e. neighbours within ~3 yrs

FALLBACK_TIERS: list[list[str]] = [
    ["make", "model"],
    ["make"],
]

# Soft feature weights — rescaled to sum to 1 dynamically for missing columns
WEIGHT_DEFAULTS: dict[str, float] = {
    "year":         0.50,
    "trim":         0.30,
    "cylinders":    0.10,
    "regionalSpec": 0.10,
}

# ── Internal helpers ──────────────────────────────────────────────────────────

def _year_score(q_year: Optional[float], pool_years: pd.Series) -> np.ndarray:
    """
    Exponential decay score based on absolute year distance.
    Returns 1.0 when years match, decays toward 0 for older/newer neighbours.
    Null years (either side) receive a neutral score of 0.5.
    """
    if pd.isna(q_year):
        return np.full(len(pool_years), 0.5)
    dist = (pool_years - q_year).abs().fillna(np.inf)
    score = np.exp(-dist.values / YEAR_DECAY)
    score[np.isinf(dist.values)] = 0.5   # null pool year → neutral
    return score


def _exact_score(q_val, pool_col: pd.Series,
                 match_score: float = 1.0,
                 mismatch_score: float = 0.0,
                 both_null_score: float = 0.5,
                 one_null_score: float = 0.3) -> np.ndarray:
    """
    Null-aware exact-match score for a categorical feature.
    """
    q_null = pd.isna(q_val) or str(q_val).strip() == ""
    p_null = pool_col.isna() | (pool_col.astype(str).str.strip() == "")

    score = np.where(
        q_null & p_null,   both_null_score,
        np.where(
            q_null | p_null,   one_null_score,
            np.where(pool_col == q_val, match_score, mismatch_score)
        )
    )
    return score.astype(float)


def _composite_score(query: pd.Series,
                     pool:  pd.DataFrame,
                     active_weights: dict[str, float]) -> np.ndarray:
    """
    Compute a [0, 1] composite similarity score for every row in pool
    relative to query, using only the features present in active_weights.
    """
    scores: dict[str, np.ndarray] = {}

    if "year" in active_weights:
        scores["year"] = _year_score(query.get("year"), pool["year"])

    for feat in ("trim", "cylinders", "regionalSpec"):
        if feat in active_weights:
            scores[feat] = _exact_score(query.get(feat), pool[feat])

    if not scores:
        return np.ones(len(pool))

    total_w = sum(active_weights[f] for f in scores)
    composite = sum(
        active_weights[f] / total_w * scores[f]
        for f in scores
    )
    return composite


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Weighted median: the value v such that cumulative weight on each side is ≥ 0.5.
    Falls back to ordinary median if all weights are zero.
    """
    if len(values) == 0:
        return np.nan
    if weights.sum() == 0 or len(values) == 1:
        return float(np.median(values))

    order    = np.argsort(values)
    vals_s   = values[order]
    wts_s    = weights[order]
    cum_w    = np.cumsum(wts_s)
    midpoint = cum_w[-1] / 2.0
    idx      = np.searchsorted(cum_w, midpoint)
    return float(vals_s[min(idx, len(vals_s) - 1)])


def _derive_provenance(tier_idx: int, avg_year_score: float) -> str:
    """
    Human-readable provenance label for a prediction.
    """
    if tier_idx == 0:
        return "model_year" if avg_year_score >= YEAR_SCORE_TIER1_THRESH else "model_match"
    if tier_idx == 1:
        return "make_fallback"
    return "unresolved"


def _active_weights(pool: pd.DataFrame) -> dict[str, float]:
    """
    Return weight dict with only features that are present in pool columns.
    """
    return {
        feat: w
        for feat, w in WEIGHT_DEFAULTS.items()
        if feat == "year" or feat in pool.columns
    }


# ── Core prediction function ──────────────────────────────────────────────────

def predict(
    query: pd.Series,
    pool: pd.DataFrame,
    k: int = K_NEIGHBORS,
    exclude_index: Optional[int] = None,
) -> dict:
    """
    Predict the price for a single query vehicle.

    Parameters
    ----------
    query         : pd.Series  — one row of vehicle data (must have make, model)
    pool          : pd.DataFrame — reference rows with a `price` column
    k             : int — number of top neighbours to use
    exclude_index : int | None — pool index to exclude (use query.name for LOO)

    Returns
    -------
    dict with keys:
        predicted_price  : float | None
        n_neighbors      : int
        avg_similarity   : float | None
        provenance       : str  ("model_year" | "model_match" | "make_fallback" | "unresolved")
        neighbor_indices : list[int]   — pool indices of the selected neighbours
    """
    weights = _active_weights(pool)

    # Exclude the query row itself when doing LOO
    if exclude_index is not None and exclude_index in pool.index:
        candidate_pool = pool.drop(index=exclude_index)
    else:
        candidate_pool = pool

    for tier_idx, key_cols in enumerate(FALLBACK_TIERS):
        # ── Hard-filter pool on this tier's key columns ──
        mask = pd.Series(True, index=candidate_pool.index)
        for col in key_cols:
            q_val = query.get(col)
            if pd.isna(q_val) or str(q_val).strip() == "":
                # Can't filter on a null query key — skip this tier
                mask &= False
                break
            mask &= (candidate_pool[col] == q_val)

        tier_pool = candidate_pool[mask].copy()

        if len(tier_pool) < MIN_NEIGHBORS:
            continue   # not enough — try next tier

        # ── Compute similarity scores ──
        sim_scores = _composite_score(query, tier_pool, weights)

        # ── Select top-K neighbours ──
        k_actual  = min(k, len(tier_pool))
        top_idx   = np.argpartition(sim_scores, -k_actual)[-k_actual:]
        top_idx   = top_idx[np.argsort(sim_scores[top_idx])[::-1]]   # sort desc

        selected_sim    = sim_scores[top_idx]
        selected_prices = tier_pool["price"].iloc[top_idx].values
        selected_index  = tier_pool.index[top_idx].tolist()

        # ── Weighted median price prediction ──
        pred_price    = _weighted_median(selected_prices, selected_sim)
        avg_sim       = float(np.mean(selected_sim))

        # Year-score component for provenance
        if "year" in weights and len(selected_sim) > 0:
            year_scores = _year_score(query.get("year"),
                                      tier_pool["year"].iloc[top_idx])
            avg_year_sc = float(np.mean(year_scores))
        else:
            avg_year_sc = 0.5

        return {
            "predicted_price":  round(pred_price),
            "n_neighbors":      k_actual,
            "avg_similarity":   round(avg_sim, 4),
            "provenance":       _derive_provenance(tier_idx, avg_year_sc),
            "neighbor_indices": selected_index,
        }

    # All tiers exhausted
    return {
        "predicted_price":  None,
        "n_neighbors":      0,
        "avg_similarity":   None,
        "provenance":       "unresolved",
        "neighbor_indices": [],
    }


# ── Batch prediction ──────────────────────────────────────────────────────────

def predict_batch(
    queries: pd.DataFrame,
    pool:    pd.DataFrame,
    k:       int = K_NEIGHBORS,
) -> pd.DataFrame:
    """
    Predict prices for a DataFrame of query vehicles.
    Queries and pool may overlap — no LOO exclusion applied here.
    Use loo_evaluate() for leave-one-out evaluation instead.

    Returns a copy of `queries` with prediction columns appended.
    """
    results = [
        predict(row, pool, k=k)
        for _, row in queries.iterrows()
    ]
    out = queries.copy()
    out["predicted_price"] = [r["predicted_price"] for r in results]
    out["n_neighbors"]     = [r["n_neighbors"]     for r in results]
    out["avg_similarity"]  = [r["avg_similarity"]  for r in results]
    out["provenance"]      = [r["provenance"]       for r in results]
    return out


# ── Leave-one-out evaluator ───────────────────────────────────────────────────

def loo_evaluate(
    df: pd.DataFrame,
    k:  int = K_NEIGHBORS,
) -> pd.DataFrame:
    """
    Leave-one-out evaluation on a single DataFrame.

    For every row i, predict price using all OTHER rows as the pool.
    Computes abs_diff and abs_diff_pct (APE) for resolved rows.

    Parameters
    ----------
    df : pd.DataFrame — must have columns: make, model, year, price
                        optional: trim, cylinders, regionalSpec
    k  : int — number of neighbours

    Returns
    -------
    df with appended columns:
        predicted_price, n_neighbors, avg_similarity, provenance,
        abs_diff, abs_diff_pct
    """
    results = []
    total   = len(df)
    for i, (idx, row) in enumerate(df.iterrows()):
        if (i + 1) % 1000 == 0 or i == 0:
            print(f"  LOO progress: {i+1:,} / {total:,}", end="\r", flush=True)
        results.append(predict(row, df, k=k, exclude_index=idx))

    print(f"  LOO progress: {total:,} / {total:,}  ✓")

    out = df.copy()
    out["predicted_price"] = [r["predicted_price"] for r in results]
    out["n_neighbors"]     = [r["n_neighbors"]     for r in results]
    out["avg_similarity"]  = [r["avg_similarity"]  for r in results]
    out["provenance"]      = [r["provenance"]       for r in results]

    resolved = out["predicted_price"].notna()
    out.loc[resolved, "abs_diff"] = (
        out.loc[resolved, "price"] - out.loc[resolved, "predicted_price"]
    ).abs()
    out.loc[resolved, "abs_diff_pct"] = (
        out.loc[resolved, "abs_diff"] / out.loc[resolved, "price"] * 100
    )

    return out


# ── Evaluation summary ────────────────────────────────────────────────────────

def print_metrics(df: pd.DataFrame, label: str = "ALL") -> None:
    """Print a metrics block for a (subset of an) evaluated DataFrame."""
    resolved = df[df["predicted_price"].notna()]
    n_total  = len(df)
    n_res    = len(resolved)
    n_unres  = n_total - n_res

    print(f"\n{'='*60}")
    print(f"{label}  (n={n_total:,}  resolved={n_res:,}  unresolved={n_unres:,})")
    print(f"{'='*60}")
    if n_res == 0:
        print("  No resolved rows — cannot compute metrics.")
        return

    print(f"  MAE          : {resolved['abs_diff'].mean():>14,.0f} AED")
    print(f"  MAPE         : {resolved['abs_diff_pct'].mean():>13.2f}%")
    print(f"  Median APE   : {resolved['abs_diff_pct'].median():>13.2f}%")
    print(f"  p75 APE      : {resolved['abs_diff_pct'].quantile(.75):>13.2f}%")
    print(f"  p90 APE      : {resolved['abs_diff_pct'].quantile(.90):>13.2f}%")
    print(f"  p95 APE      : {resolved['abs_diff_pct'].quantile(.95):>13.2f}%")
    print(f"  Avg sim      : {resolved['avg_similarity'].mean():>13.4f}")
    print(f"  Avg neighbors: {resolved['n_neighbors'].mean():>12.1f}")


def print_full_report(df: pd.DataFrame) -> None:
    """Print overall metrics + per-provenance breakdown."""
    print_metrics(df, "OVERALL")

    print("\n\n=== BY PROVENANCE TIER ===")
    for tier in ["model_year", "model_match", "make_fallback", "unresolved"]:
        subset = df[df["provenance"] == tier]
        if len(subset) > 0:
            print_metrics(subset, f"provenance = {tier}")

    resolved = df[df["predicted_price"].notna()]
    if len(resolved) > 0:
        print("\n\n=== TOP 20 WORST APE (resolved) ===")
        print(
            resolved.nlargest(20, "abs_diff_pct")[
                ["make", "model", "year", "trim", "price",
                 "predicted_price", "abs_diff_pct",
                 "n_neighbors", "avg_similarity", "provenance"]
            ].to_string()
        )


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run LOO evaluation on a cleaned vehicle price pool."
    )
    p.add_argument("pool_csv",
                   nargs="?", default="clean_pool.csv",
                   help="Path to clean pool CSV (default: clean_pool.csv)")
    p.add_argument("--k", type=int, default=K_NEIGHBORS,
                   help=f"Number of neighbours (default: {K_NEIGHBORS})")
    p.add_argument("--out", default="loo_results.csv",
                   help="Output path for evaluated CSV (default: loo_results.csv)")
    p.add_argument("--no-flagged", action="store_true",
                   help="Exclude rows where price_flag=True before evaluation")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    print(f"Loading pool: {args.pool_csv}")
    df = pd.read_csv(args.pool_csv)
    print(f"  {len(df):,} rows loaded")

    if args.no_flagged and "price_flag" in df.columns:
        before = len(df)
        df = df[~df["price_flag"]].copy()
        print(f"  {before - len(df):,} flagged rows excluded → {len(df):,} remain")

    df = df.reset_index(drop=True)

    print(f"\nRunning LOO evaluation  (k={args.k}) ...")
    evaluated = loo_evaluate(df, k=args.k)

    print_full_report(evaluated)

    evaluated.to_csv(args.out, index=False)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()