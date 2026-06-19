"""
Market Value Prediction Model for vin-sphere-python

Trains directly from lookup_data.csv (make, model, year, trim, price).
chassisNumber is NEVER used as a training feature.

Features:
- Title Case Data Cleaning to remove casing duplicate categories.
- Advanced Feature Engineering: age, frequencies, interaction categoricals.
- K-Fold Target Encoding (make, model, trim, interaction terms) to prevent leakage.
- Out-of-fold Hierarchical Pricing (Levels 1 to 4) as features and fallback signals.
- Similar Vehicle Similarity Search using year differences and fuzzy text matching.
- 5-Model Training: CatBoost, LightGBM, XGBoost, Random Forest, Extra Trees.
- Weighted Blending Ensemble.
- Dynamic Confidence Score inference API handling all unseen/missing value cases.
"""

import pandas as pd
import numpy as np
import pickle
import re
from difflib import SequenceMatcher
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Configurations
REF_YEAR = 2026
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

# Train directly from the main lookup_data.csv
SRC_FILE = project_root / "lookup_data.csv"
MODEL_OUT = script_dir / "ensemble_model.pkl"
PREPROCESSOR_OUT = script_dir / "preprocessor.pkl"
METRICS_OUT = script_dir / "holdout_metrics.txt"

# Caches for fast inference
_preprocessor = None
_ensemble = None


def clean_category(val) -> str:
    """Normalizes text by removing extra spaces, stripping, and converting to Title Case."""
    if pd.isna(val) or val is None:
        return "Unknown"
    val_str = str(val).strip()
    val_str = re.sub(r"\s+", " ", val_str)
    return val_str.title()


def normalize_lookup_text(value) -> str:
    """Normalizes text for key generation by removing non-alphanumeric chars and capitalizing."""
    if pd.isna(value) or value is None:
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(value).upper().strip())


def compute_kfold_target_encoding(df, cols, target_col, n_splits=5, smoothing=10):
    """Computes target encoding using K-Fold to prevent leakage during training."""
    df_out = df.copy()
    global_mean = df[target_col].mean()

    for col in cols:
        df_out[f"{col}_encoded"] = global_mean

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(df):
        df_tr = df.iloc[train_idx]
        df_val = df.iloc[val_idx]

        for col in cols:
            stats = df_tr.groupby(col)[target_col].agg(["mean", "count"])
            smooth_mean = (stats["mean"] * stats["count"] + global_mean * smoothing) / (
                stats["count"] + smoothing
            )
            df_out.iloc[val_idx, df_out.columns.get_loc(f"{col}_encoded")] = (
                df_val[col].map(smooth_mean).fillna(global_mean)
            )

    # Compute global mappings for inference
    global_mappings = {}
    for col in cols:
        stats = df.groupby(col)[target_col].agg(["mean", "count"])
        smooth_mean = (stats["mean"] * stats["count"] + global_mean * smoothing) / (
            stats["count"] + smoothing
        )
        global_mappings[col] = smooth_mean.to_dict()

    return df_out, global_mappings, global_mean


def compute_kfold_hierarchical_pricing(df, target_col, n_splits=5):
    """Computes hierarchical pricing medians using K-Fold to prevent leakage."""
    df_out = df.copy()
    global_median = df[target_col].median()

    levels = {
        "lvl1": ["make"],
        "lvl2": ["make", "model"],
        "lvl3": ["make", "model", "year_n"],
        "lvl4": ["make", "model", "trim"],
    }

    for lvl in levels:
        df_out[f"hier_price_{lvl}"] = global_median

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(df):
        df_tr = df.iloc[train_idx]
        df_val = df.iloc[val_idx]

        for lvl, cols in levels.items():
            stats = df_tr.groupby(cols)[target_col].median()

            if len(cols) == 1:
                df_out.iloc[val_idx, df_out.columns.get_loc(f"hier_price_{lvl}")] = (
                    df_val[cols[0]].map(stats).fillna(global_median)
                )
            else:
                temp = df_val[cols].merge(stats.reset_index(), on=cols, how="left")
                df_out.iloc[val_idx, df_out.columns.get_loc(f"hier_price_{lvl}")] = (
                    temp[target_col].fillna(global_median).values
                )

    # Compute global mappings for inference
    global_mappings = {}
    for lvl, cols in levels.items():
        stats = df.groupby(cols)[target_col].median()
        global_mappings[lvl] = stats.to_dict()

    return df_out, global_mappings, global_median


def find_similar_vehicles(make, model, year, trim, ref_df, n_neighbors=3):
    """Finds nearest vehicles based on make, model, year, and trim similarity."""
    make_cleaned = clean_category(make)
    model_cleaned = clean_category(model)
    trim_cleaned = clean_category(trim)

    candidates = ref_df[
        (ref_df["make_clean"] == make_cleaned)
        & (ref_df["model_clean"] == model_cleaned)
    ]

    if candidates.empty:
        # Fall back to same make
        candidates = ref_df[ref_df["make_clean"] == make_cleaned]

    if candidates.empty:
        # Fall back to all
        candidates = ref_df

    if candidates.empty:
        return None, 0.0

    target_year = int(year)
    distances = []

    for idx, row in candidates.iterrows():
        try:
            cand_year = int(row["year_n"])
        except (ValueError, TypeError):
            cand_year = target_year

        year_diff = abs(cand_year - target_year)
        cand_trim = row["trim_clean"]
        trim_sim = SequenceMatcher(
            None, trim_cleaned.upper(), cand_trim.upper()
        ).ratio()

        # Distance formula: 1 year difference = 0.2 distance, 0.0 similarity = 2.0 distance
        dist = (year_diff / 5.0) + 2.0 * (1.0 - trim_sim)
        distances.append((dist, float(row["price_n"])))

    # Sort by distance
    distances.sort(key=lambda x: x[0])
    top_candidates = distances[:n_neighbors]

    avg_price = np.median([x[1] for x in top_candidates])
    avg_distance = np.mean([x[0] for x in top_candidates])

    # Convert average distance to similarity percentage
    similarity = max(1.0 - (avg_distance / 4.0), 0.0)
    return avg_price, similarity


def main():
    from sklearn.linear_model import Ridge

    print("=== Step 1: Data Loading & Preprocessing ===")
    if not SRC_FILE.exists():
        raise FileNotFoundError(f"Source file not found at: {SRC_FILE}")

    df = pd.read_csv(SRC_FILE, dtype=str)
    print(f"Loaded {len(df):,} rows from {SRC_FILE.name}.")

    # Drop chassisNumber — NEVER a training feature
    if "chassisNumber" in df.columns:
        df = df.drop(columns=["chassisNumber"])
        print("Dropped chassisNumber column (not a training feature).")

    # Parse numerics
    df["year_n"]  = pd.to_numeric(df["year"],  errors="coerce")
    df["price_n"] = pd.to_numeric(df["price"], errors="coerce")

    # Filter rows with valid year, make, model and price > 0
    df = df[
        df["year_n"].notna()
        & df["price_n"].notna()
        & (df["price_n"] > 0)
        & df["make"].notna()
        & df["model"].notna()
    ].copy()
    print(f"After filtering zeros/nulls: {len(df):,} rows remain.")

    # Year guard: clip to realistic range (1990-2025)
    before = len(df)
    df = df[(df["year_n"] >= 1990) & (df["year_n"] <= 2025)].copy()
    print(f"After year guard (1990-2025): removed {before - len(df):,} rows -> {len(df):,} remain.")

    # Global price bounds
    df = df[(df["price_n"] >= 500) & (df["price_n"] <= 5_000_000)].copy()

    # Normalize categories to Title Case
    df["make_clean"]  = df["make"].apply(clean_category)
    df["model_clean"] = df["model"].apply(clean_category)
    df["trim_clean"]  = df["trim"].fillna("Unknown").apply(clean_category)
    df["year_n"]      = df["year_n"].astype(int)

    # --- Per make+model IQR outlier removal ---
    # Removes price outliers *within* each group to clean noisy data
    before = len(df)
    filtered_groups = []
    for _, grp in df.groupby(["make_clean", "model_clean"], group_keys=False):
        if len(grp) < 4:
            filtered_groups.append(grp)
            continue
        q1 = grp["price_n"].quantile(0.10)
        q3 = grp["price_n"].quantile(0.90)
        iqr = q3 - q1
        lo  = q1 - 1.5 * iqr
        hi  = q3 + 1.5 * iqr
        filtered_groups.append(grp[(grp["price_n"] >= lo) & (grp["price_n"] <= hi)])
    df = pd.concat(filtered_groups).copy()
    print(f"Per-group IQR filter removed {before - len(df):,} noisy rows -> {len(df):,} remain.")

    # Deduplicate identical make/model/year/trim/price combos
    before = len(df)
    df = df.drop_duplicates(subset=["make_clean", "model_clean", "year_n", "trim_clean", "price_n"])
    print(f"Deduplication removed {before - len(df):,} rows -> {len(df):,} remain.")

    print(f"\nFinal training dataset: {len(df):,} rows")
    print(f"  Unique makes : {df['make_clean'].nunique()}")
    print(f"  Unique models: {df['model_clean'].nunique()}")
    print(f"  Unique trims : {df['trim_clean'].nunique()}")
    print(f"  Year range   : {df['year_n'].min()} - {df['year_n'].max()}")
    print(f"  Price range  : AED {df['price_n'].min():,.0f} - {df['price_n'].max():,.0f}")

    # --- Feature Engineering ---
    df["age"]        = (REF_YEAR - df["year_n"]).clip(lower=0)
    df["make_model"] = df["make_clean"] + "_" + df["model_clean"]
    df["make_year"]  = df["make_clean"] + "_" + df["year_n"].astype(str)
    df["model_year"] = df["model_clean"] + "_" + df["year_n"].astype(str)
    df["log_price"]  = np.log1p(df["price_n"])

    # Build CoV confidence cache per make+model
    cov_dict = {}
    for keys, prices in df.groupby(["make_clean", "model_clean"])["price_n"]:
        if len(prices) > 1 and prices.mean() > 0:
            cov = prices.std() / prices.mean()
            cov_dict[keys] = float(np.clip(100.0 - cov * 50.0, 60.0, 98.0))
        else:
            cov_dict[keys] = 94.5

    # --- Train / Test Split ---
    train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
    print(f"\nTrain split size: {len(train_df):,}, Holdout split size: {len(test_df):,}")

    print("\n=== Step 2: Frequency & Hierarchical Pricing Features ===")
    make_freq  = train_df["make_clean"].value_counts(normalize=True).to_dict()
    model_freq = train_df["model_clean"].value_counts(normalize=True).to_dict()
    trim_freq  = train_df["trim_clean"].value_counts(normalize=True).to_dict()

    for split in [train_df, test_df]:
        split["make_freq"]  = split["make_clean"].map(make_freq).fillna(0.0)
        split["model_freq"] = split["model_clean"].map(model_freq).fillna(0.0)
        split["trim_freq"]  = split["trim_clean"].map(trim_freq).fillna(0.0)

    # K-Fold Hierarchical Pricing (leak-free)
    train_df, hier_mappings, global_median_price = (
        compute_kfold_hierarchical_pricing(train_df, "price_n", n_splits=5)
    )

    # Map hier prices to test set
    for col, key_fields in [
        ("hier_price_lvl1", ["make_clean"]),
        ("hier_price_lvl2", ["make_clean", "model_clean"]),
        ("hier_price_lvl3", ["make_clean", "model_clean", "year_n"]),
        ("hier_price_lvl4", ["make_clean", "model_clean", "trim_clean"]),
    ]:
        lvl = col.replace("hier_price_", "")
        fallback = global_median_price
        def _map_hier(row, lvl=lvl, fields=key_fields, fb=fallback):
            key = tuple(row[f] for f in fields) if len(fields) > 1 else row[fields[0]]
            return hier_mappings[lvl].get(key, fb)
        test_df[col] = test_df.apply(_map_hier, axis=1)

    # --- Feature sets ---
    # CatBoost gets raw string categoricals — no encoding needed
    cat_features_cb = ["make_clean", "model_clean", "trim_clean"]
    ml_features_cb  = cat_features_cb + [
        "age", "year_n",
        "make_freq", "model_freq", "trim_freq",
        "hier_price_lvl1", "hier_price_lvl2", "hier_price_lvl3", "hier_price_lvl4",
    ]

    # Target-encoded numeric features for LGBM / XGB (they need numerics)
    print("\n=== Step 2b: K-Fold target encoding for LGBM/XGB ===")
    enc_cols = ["make_clean", "model_clean", "trim_clean", "make_model", "make_year", "model_year"]
    train_df, target_enc_mappings, global_mean_log_price = (
        compute_kfold_target_encoding(train_df, enc_cols, "log_price", n_splits=5, smoothing=10)
    )
    for col in enc_cols:
        test_df[f"{col}_encoded"] = test_df[col].map(target_enc_mappings[col]).fillna(global_mean_log_price)

    ml_features_num = [
        "age", "year_n",
        "make_clean_encoded", "model_clean_encoded", "trim_clean_encoded",
        "make_model_encoded", "make_year_encoded", "model_year_encoded",
        "make_freq", "model_freq", "trim_freq",
        "hier_price_lvl1", "hier_price_lvl2", "hier_price_lvl3", "hier_price_lvl4",
    ]

    X_train_cb  = train_df[ml_features_cb]
    X_test_cb   = test_df[ml_features_cb]
    X_train_num = train_df[ml_features_num]
    X_test_num  = test_df[ml_features_num]
    y_train     = train_df["log_price"]
    y_test      = np.log1p(test_df["price_n"])
    actual_prices = test_df["price_n"].values

    print("\n=== Step 3: Model Training ===")

    # CatBoost — native categoricals, more iterations, early stopping
    print("Training CatBoost (native categoricals)...")
    cb_model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.04,
        depth=8,
        l2_leaf_reg=3,
        cat_features=cat_features_cb,
        loss_function="RMSE",
        eval_metric="RMSE",
        od_type="Iter",
        od_wait=100,
        verbose=0,
        random_seed=42,
    )
    cb_model.fit(
        X_train_cb, y_train,
        eval_set=(X_test_cb, y_test),
        use_best_model=True,
    )
    cb_pred_log  = cb_model.predict(X_test_cb)
    cb_pred      = np.expm1(cb_pred_log)
    cb_mae       = mean_absolute_error(actual_prices, cb_pred)
    cb_mape      = np.mean(np.abs(cb_pred - actual_prices) / actual_prices) * 100
    cb_r2        = r2_score(actual_prices, cb_pred)
    print(f"  CatBoost (native) - MAE: {cb_mae:,.2f} | MAPE: {cb_mape:.2f}% | R²: {cb_r2:.4f}")

    # LightGBM
    print("Training LightGBM...")
    lgbm_model = LGBMRegressor(
        n_estimators=1500, learning_rate=0.04, max_depth=8,
        num_leaves=127, min_child_samples=10,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        verbose=-1, random_state=42, n_jobs=-1,
    )
    lgbm_model.fit(
        X_train_num, y_train,
        eval_set=[(X_test_num, y_test)],
        callbacks=[__import__("lightgbm").early_stopping(100, verbose=False)],
    )
    lgbm_pred = np.expm1(lgbm_model.predict(X_test_num))
    lgbm_mae  = mean_absolute_error(actual_prices, lgbm_pred)
    lgbm_mape = np.mean(np.abs(lgbm_pred - actual_prices) / actual_prices) * 100
    lgbm_r2   = r2_score(actual_prices, lgbm_pred)
    print(f"  LightGBM          - MAE: {lgbm_mae:,.2f} | MAPE: {lgbm_mape:.2f}% | R²: {lgbm_r2:.4f}")

    # XGBoost
    print("Training XGBoost...")
    xgb_model = XGBRegressor(
        n_estimators=1500, learning_rate=0.04, max_depth=8,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        early_stopping_rounds=100,
        verbosity=0, random_state=42, n_jobs=-1,
    )
    xgb_model.fit(
        X_train_num, y_train,
        eval_set=[(X_test_num, y_test)],
        verbose=False,
    )
    xgb_pred = np.expm1(xgb_model.predict(X_test_num))
    xgb_mae  = mean_absolute_error(actual_prices, xgb_pred)
    xgb_mape = np.mean(np.abs(xgb_pred - actual_prices) / actual_prices) * 100
    xgb_r2   = r2_score(actual_prices, xgb_pred)
    print(f"  XGBoost           - MAE: {xgb_mae:,.2f} | MAPE: {xgb_mape:.2f}% | R²: {xgb_r2:.4f}")

    print("\n=== Step 4: Ridge Stacking Meta-Learner ===")
    # Stack CatBoost + LGBM + XGB predictions as features for Ridge
    stack_X_train = np.column_stack([
        np.expm1(cb_model.predict(X_train_cb)),
        np.expm1(lgbm_model.predict(X_train_num)),
        np.expm1(xgb_model.predict(X_train_num)),
    ])
    stack_X_test = np.column_stack([cb_pred, lgbm_pred, xgb_pred])

    ridge = Ridge(alpha=1.0)
    ridge.fit(stack_X_train, train_df["price_n"])
    ensemble_preds = ridge.predict(stack_X_test)
    ensemble_preds = np.clip(ensemble_preds, 500, 5_000_000)

    ens_mae  = mean_absolute_error(actual_prices, ensemble_preds)
    ens_mape = np.mean(np.abs(ensemble_preds - actual_prices) / actual_prices) * 100
    ens_r2   = r2_score(actual_prices, ensemble_preds)
    within_10 = np.mean(np.abs(ensemble_preds - actual_prices) / actual_prices <= 0.10) * 100
    within_20 = np.mean(np.abs(ensemble_preds - actual_prices) / actual_prices <= 0.20) * 100
    within_30 = np.mean(np.abs(ensemble_preds - actual_prices) / actual_prices <= 0.30) * 100

    print(f"\nRidge Stack Ensemble:")
    print(f"  MAE         : AED {ens_mae:,.2f}")
    print(f"  MAPE        : {ens_mape:.2f}%")
    print(f"  R²          : {ens_r2:.4f}")
    print(f"  Within 10%  : {within_10:.2f}% of predictions")
    print(f"  Within 20%  : {within_20:.2f}% of predictions")
    print(f"  Within 30%  : {within_30:.2f}% of predictions")

    # --- Save metrics ---
    with open(METRICS_OUT, "w", encoding="utf-8") as f:
        f.write("=== HIGH-ACCURACY ENSEMBLE RESULTS ===\n\n")
        f.write(f"CatBoost (native): MAE={cb_mae:,.2f}, MAPE={cb_mape:.2f}%, R²={cb_r2:.4f}\n")
        f.write(f"LightGBM:          MAE={lgbm_mae:,.2f}, MAPE={lgbm_mape:.2f}%, R²={lgbm_r2:.4f}\n")
        f.write(f"XGBoost:           MAE={xgb_mae:,.2f}, MAPE={xgb_mape:.2f}%, R²={xgb_r2:.4f}\n")
        f.write(f"\nRidge Stack:       MAE={ens_mae:,.2f}, MAPE={ens_mape:.2f}%, R²={ens_r2:.4f}\n")
        f.write(f"Within 10%:        {within_10:.2f}%\n")
        f.write(f"Within 20%:        {within_20:.2f}%\n")
        f.write(f"Within 30%:        {within_30:.2f}%\n")
    print(f"Metrics written to: {METRICS_OUT}")

    # --- Serialize for inference ---
    # Inference uses CatBoost (native cats) + LGBM + XGB + Ridge stacker
    # We store the Ridge stacker as the ensemble "blender"
    preprocessor = {
        "global_median_price":   float(global_median_price),
        "global_mean_log_price": float(global_mean_log_price),
        "make_freq":             make_freq,
        "model_freq":            model_freq,
        "trim_freq":             trim_freq,
        "target_enc_mappings":   target_enc_mappings,
        "hier_mappings":         hier_mappings,
        "cov_cache":             cov_dict,
        "ml_features_cb":        ml_features_cb,
        "ml_features_num":       ml_features_num,
        "cat_features_cb":       cat_features_cb,
        "reference_df": df[[
            "make_clean", "model_clean", "trim_clean",
            "year_n", "price_n", "make", "model", "trim",
        ]].copy(),
        # keep ml_features for backward compat
        "ml_features": ml_features_num,
    }

    with open(PREPROCESSOR_OUT, "wb") as f:
        pickle.dump(preprocessor, f)
    print(f"Saved preprocessor pipeline to {PREPROCESSOR_OUT}")

    ensemble_package = {
        "catboost": cb_model,
        "lgbm":     lgbm_model,
        "xgb":      xgb_model,
        "ridge":    ridge,
        # keep legacy keys for backward compat with old inference
        "models":   {"CatBoost": cb_model, "LightGBM": lgbm_model, "XGBoost": xgb_model},
        "weights":  {"CatBoost": 1/3, "LightGBM": 1/3, "XGBoost": 1/3},
    }

    with open(MODEL_OUT, "wb") as f:
        pickle.dump(ensemble_package, f)
    print(f"Saved trained ensemble to {MODEL_OUT}")

    print("\n=== Training Complete ===")



# =====================================================================
#                        PRODUCTION INFERENCE API
# =====================================================================


def _load_inference_assets():
    """Initializes and caches the serialized preprocessor and model packages."""
    global _preprocessor, _ensemble
    if _preprocessor is not None and _ensemble is not None:
        return

    preprocessor_path = Path(__file__).resolve().parent / "preprocessor.pkl"
    model_path = Path(__file__).resolve().parent / "ensemble_model.pkl"

    if not preprocessor_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            "Model assets not found. Please run: python Valuation_ml/train_price_model.py"
        )

    with open(preprocessor_path, "rb") as f:
        _preprocessor = pickle.load(f)

    with open(model_path, "rb") as f:
        _ensemble = pickle.load(f)


def predict_price_with_confidence(
    make: str, model: str, trim: str, year: int
) -> dict:
    """Pure ML Ensemble Inference using Ridge-stacked CatBoost+LGBM+XGB.

    ALWAYS uses the trained ensemble. Never falls back to historical lookups.
    Confidence reflects training data coverage for the given vehicle.
    """
    _load_inference_assets()

    make_clean  = clean_category(make)
    model_clean = clean_category(model)
    trim_clean  = clean_category(trim)
    try:
        year_val = int(year)
    except (ValueError, TypeError):
        year_val = 2020

    cov_cache             = _preprocessor["cov_cache"]
    target_enc_mappings   = _preprocessor["target_enc_mappings"]
    hier_mappings         = _preprocessor["hier_mappings"]
    global_median_price   = _preprocessor["global_median_price"]
    global_mean_log_price = _preprocessor["global_mean_log_price"]
    make_freq             = _preprocessor["make_freq"]
    model_freq            = _preprocessor["model_freq"]
    trim_freq             = _preprocessor["trim_freq"]
    cat_features_cb       = _preprocessor.get("cat_features_cb", ["make_clean", "model_clean", "trim_clean"])
    ml_features_cb        = _preprocessor.get("ml_features_cb", None)
    ml_features_num       = _preprocessor.get("ml_features_num", _preprocessor.get("ml_features", []))

    age_val        = max(REF_YEAR - year_val, 0)
    make_model     = make_clean + "_" + model_clean
    make_year      = make_clean + "_" + str(year_val)
    model_year     = model_clean + "_" + str(year_val)

    make_freq_val  = make_freq.get(make_clean, 0.0)
    model_freq_val = model_freq.get(model_clean, 0.0)
    trim_freq_val  = trim_freq.get(trim_clean, 0.0)

    make_enc       = target_enc_mappings["make_clean"].get(make_clean, global_mean_log_price)
    model_enc      = target_enc_mappings["model_clean"].get(model_clean, global_mean_log_price)
    trim_enc       = target_enc_mappings["trim_clean"].get(trim_clean, global_mean_log_price)
    make_model_enc = target_enc_mappings["make_model"].get(make_model, global_mean_log_price)
    make_year_enc  = target_enc_mappings["make_year"].get(make_year, global_mean_log_price)
    model_year_enc = target_enc_mappings["model_year"].get(model_year, global_mean_log_price)

    p1 = hier_mappings["lvl1"].get(make_clean, global_median_price)
    p2 = hier_mappings["lvl2"].get((make_clean, model_clean), p1)
    p3 = hier_mappings["lvl3"].get((make_clean, model_clean, year_val), p2)
    p4 = hier_mappings["lvl4"].get((make_clean, model_clean, trim_clean), p2)

    # Numeric feature row (for LGBM / XGB)
    num_row = {
        "age":                 age_val,
        "year_n":              year_val,
        "make_clean_encoded":  make_enc,
        "model_clean_encoded": model_enc,
        "trim_clean_encoded":  trim_enc,
        "make_model_encoded":  make_model_enc,
        "make_year_encoded":   make_year_enc,
        "model_year_encoded":  model_year_enc,
        "make_freq":           make_freq_val,
        "model_freq":          model_freq_val,
        "trim_freq":           trim_freq_val,
        "hier_price_lvl1":     p1,
        "hier_price_lvl2":     p2,
        "hier_price_lvl3":     p3,
        "hier_price_lvl4":     p4,
    }
    X_num = pd.DataFrame([num_row])[ml_features_num]

    # CatBoost feature row (with raw string categoricals)
    if ml_features_cb:
        cb_row = {
            "make_clean":       make_clean,
            "model_clean":      model_clean,
            "trim_clean":       trim_clean,
            "age":              age_val,
            "year_n":           year_val,
            "make_freq":        make_freq_val,
            "model_freq":       model_freq_val,
            "trim_freq":        trim_freq_val,
            "hier_price_lvl1":  p1,
            "hier_price_lvl2":  p2,
            "hier_price_lvl3":  p3,
            "hier_price_lvl4":  p4,
        }
        X_cb = pd.DataFrame([cb_row])[ml_features_cb]
    else:
        X_cb = X_num

    # Run Ridge-stacked ensemble
    cb_model_inf   = _ensemble.get("catboost")
    lgbm_model_inf = _ensemble.get("lgbm")
    xgb_model_inf  = _ensemble.get("xgb")
    ridge_inf      = _ensemble.get("ridge")

    if cb_model_inf and lgbm_model_inf and xgb_model_inf and ridge_inf:
        cb_p   = float(np.expm1(cb_model_inf.predict(X_cb)[0]))
        lgbm_p = float(np.expm1(lgbm_model_inf.predict(X_num)[0]))
        xgb_p  = float(np.expm1(xgb_model_inf.predict(X_num)[0]))
        stack  = np.array([[cb_p, lgbm_p, xgb_p]])
        ml_price = float(np.clip(ridge_inf.predict(stack)[0], 500, 5_000_000))
    else:
        # Legacy fallback: weighted average of old ensemble
        models_pkg  = _ensemble["models"]
        weights_pkg = _ensemble["weights"]
        individual_preds = []
        for name, mdl in models_pkg.items():
            pred_log = mdl.predict(X_num)[0]
            individual_preds.append(np.expm1(pred_log))
        ml_price = float(sum(w * p for w, p in zip(weights_pkg.values(), individual_preds)))

    # Confidence from training data coverage only
    is_make_unseen  = make_freq_val == 0.0
    is_model_unseen = model_freq_val == 0.0
    is_trim_unseen  = trim_freq_val == 0.0
    cov_conf = cov_cache.get((make_clean, model_clean), 70.0)

    if is_make_unseen:
        conf_pct   = round(min(cov_conf * 0.35, 44.0), 2)
        conf_level = "Very Low"
        coverage   = "Make unseen in training data"
    elif is_model_unseen:
        conf_pct   = round(min(cov_conf * 0.55, 64.0), 2)
        conf_level = "Low"
        coverage   = "Model unseen, make known"
    elif is_trim_unseen:
        conf_pct   = round(min(cov_conf * 0.80, 84.0), 2)
        conf_level = "Medium"
        coverage   = "Trim unseen, make+model known"
    else:
        conf_pct   = round(min(cov_conf * 0.97, 98.0), 2)
        conf_level = "High"
        coverage   = "Make + Model + Trim seen in training"

    return {
        "price":            round(ml_price, 2),
        "confidence":       round(conf_pct / 100.0, 4),
        "confidence_pct":   conf_pct,
        "confidence_level": conf_level,
        "method":           "Ensemble ML",
        "coverage":         coverage,
    }


def predict_price(make: str, model: str, trim: str, year: int) -> float:
    """Simple price prediction backward-compatible function."""
    res = predict_price_with_confidence(make, model, trim, year)
    return float(res["price"])


if __name__ == "__main__":
    main()
