"""
similarity.py — Composite similarity search module for car price kNN.

Architecture:
  Tier 1: hard-filter candidates by exact `make` match.
  Tier 2 (fallback): if Tier 1 yields < min_neighbors, drop the make filter
          and search the full dataset, tagging provenance accordingly.
  Within each tier: composite weighted similarity over model (fuzzy),
          year (numeric decay), trim (fuzzy).
  Aggregation: similarity-weighted median price.
"""

import pandas as pd
import numpy as np
from rapidfuzz import fuzz

# ---------------------------------------------------------------------------
# Per-feature similarity functions — each returns a score in [0, 1]
# ---------------------------------------------------------------------------

def model_fuzzy_similarity(model_a: str, model_b: str) -> float:
    """
    Blend of token-set similarity (robust to trim/submodel suffixes like
    'EQS 53 AMG' vs 'EQS') and edit-distance ratio (robust to formatting
    noise like 'CLC180K' vs 'CLC 180 K').
    """
    if pd.isna(model_a) or pd.isna(model_b):
        return 0.0
    a, b = str(model_a).upper().strip(), str(model_b).upper().strip()
    if a == b:
        return 1.0

    token_sim = fuzz.token_set_ratio(a, b) / 100.0
    edit_sim = fuzz.ratio(a, b) / 100.0

    return 0.6 * token_sim + 0.4 * edit_sim


def trim_fuzzy_similarity(trim_a, trim_b) -> float:
    """Same blend as model, but nulls are treated as a weak match to each
    other (0.5), not zero — a missing trim shouldn't be penalized as hard
    as a genuinely different trim, since trim data is sparser/noisier."""
    if pd.isna(trim_a) and pd.isna(trim_b):
        return 0.5
    if pd.isna(trim_a) or pd.isna(trim_b):
        return 0.3
    a, b = str(trim_a).upper().strip(), str(trim_b).upper().strip()
    if a == b:
        return 1.0
    token_sim = fuzz.token_set_ratio(a, b) / 100.0
    edit_sim = fuzz.ratio(a, b) / 100.0
    return 0.6 * token_sim + 0.4 * edit_sim


def year_similarity(year_a, year_b, max_window: int = 15) -> float:
    """Linear decay: identical year = 1.0, decays to 0 at max_window years
    apart. max_window=15 is a placeholder — tune against eval results."""
    if pd.isna(year_a) or pd.isna(year_b):
        return 0.0
    diff = abs(float(year_a) - float(year_b))
    return max(0.0, 1.0 - diff / max_window)


# ---------------------------------------------------------------------------
# Composite similarity + neighbor search
# ---------------------------------------------------------------------------

WEIGHTS = {"model": 0.55, "year": 0.30, "trim": 0.15}  # placeholders — tune later


def composite_similarity(query: pd.Series, candidates: pd.DataFrame,
                          weights: dict = WEIGHTS) -> np.ndarray:
    """Vectorized composite similarity between a single query row and a
    DataFrame of candidates. Returns an array of scores in [0, 1]."""
    model_sims = candidates["model"].apply(lambda m: model_fuzzy_similarity(query["model"], m)).to_numpy()
    trim_sims  = candidates["trim"].apply(lambda t: trim_fuzzy_similarity(query["trim"], t)).to_numpy()
    year_sims  = candidates["year"].apply(lambda y: year_similarity(query["year"], y)).to_numpy()

    return (weights["model"] * model_sims +
            weights["year"]  * year_sims  +
            weights["trim"]  * trim_sims)


def find_neighbors(query: pd.Series, dataset: pd.DataFrame, k: int = 10,
                    min_neighbors: int = 5, exclude_index=None,
                    weights: dict = WEIGHTS):
    """
    Returns (neighbors_df, provenance) where neighbors_df has columns
    [original columns..., similarity], sorted descending by similarity,
    top-k. provenance is "exact_make" or "cross_make_fallback".

    exclude_index: pass the query's own index when doing leave-one-out
    evaluation, so a row never matches itself.
    """
    pool = dataset
    if exclude_index is not None:
        pool = pool.drop(index=exclude_index, errors="ignore")

    # Tier 1: hard filter on make
    same_make = pool[pool["make"] == query["make"]]
    provenance = "exact_make"

    candidates = same_make
    if len(candidates) < min_neighbors:
        # Tier 2: fallback, drop the make filter entirely
        candidates = pool
        provenance = "cross_make_fallback"

    if len(candidates) == 0:
        return pd.DataFrame(), "no_candidates"

    sims = composite_similarity(query, candidates, weights)
    candidates = candidates.copy()
    candidates["similarity"] = sims

    top_k = candidates.nlargest(k, "similarity")
    return top_k, provenance


def predict_price(query: pd.Series, dataset: pd.DataFrame, k: int = 10,
                   min_neighbors: int = 5, exclude_index=None,
                   weights: dict = WEIGHTS):
    """
    Returns a dict: predicted_price, n_neighbors, avg_similarity, provenance.
    Aggregation = similarity-weighted median (not mean — robust to outliers
    among the neighbor set itself).
    """
    neighbors, provenance = find_neighbors(
        query, dataset, k=k, min_neighbors=min_neighbors,
        exclude_index=exclude_index, weights=weights
    )

    if len(neighbors) == 0:
        return {"predicted_price": np.nan, "n_neighbors": 0,
                "avg_similarity": 0.0, "provenance": "unresolved"}

    prices = neighbors["price"].to_numpy()
    sims = neighbors["similarity"].to_numpy()

    # similarity-weighted median: sort by price, find the weighted 50th percentile
    order = np.argsort(prices)
    prices_sorted, sims_sorted = prices[order], sims[order]
    cum_weight = np.cumsum(sims_sorted)
    cutoff = cum_weight[-1] / 2.0
    median_idx = np.searchsorted(cum_weight, cutoff)
    weighted_median_price = prices_sorted[min(median_idx, len(prices_sorted) - 1)]

    return {
        "predicted_price": weighted_median_price,
        "n_neighbors": len(neighbors),
        "avg_similarity": float(sims.mean()),
        "provenance": provenance,
    }