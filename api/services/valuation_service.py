import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from difflib import SequenceMatcher
from datetime import date, datetime

from api.config import LOOKUP_DATA_PATH, CAR_DEP_CSV_PATH
from valuation_engine import ToyotaValuationEngine

# Preloaded Data Cache
_lookup_df: Optional[pd.DataFrame] = None
_depreciation_df: Optional[pd.DataFrame] = None
_toyota_engine: Optional[ToyotaValuationEngine] = None

def load_valuation_assets():
    """Preloads CSV data files and engine instances into memory at startup."""
    global _lookup_df, _depreciation_df, _toyota_engine
    
    # 1. Load price lookup
    if LOOKUP_DATA_PATH.exists():
        try:
            df = pd.read_csv(LOOKUP_DATA_PATH)
            df["make_key"] = df["make"].apply(normalize_lookup_text)
            df["model_key"] = df["model"].apply(normalize_lookup_text)
            df["trim_key"] = df["trim"].apply(normalize_lookup_text)
            df["year_key"] = df["year"].apply(normalize_lookup_year)
            if "chassisNumber" in df.columns:
                df["vin_key"] = df["chassisNumber"].apply(normalize_lookup_text)
            else:
                df["vin_key"] = ""
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            _lookup_df = df
        except Exception as e:
            print(f"Error preloading lookup data: {e}")
            
    # 2. Load depreciation data
    if CAR_DEP_CSV_PATH.exists():
        try:
            _depreciation_df = pd.read_csv(CAR_DEP_CSV_PATH)
        except Exception as e:
            print(f"Error preloading depreciation data: {e}")
            
    # 3. Instantiate Toyota valuation engine
    try:
        _toyota_engine = ToyotaValuationEngine()
    except Exception as e:
        print(f"Error instantiating Toyota engine: {e}")

# Helper Functions
def normalize_lookup_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())

def normalize_lookup_year(value) -> str:
    year = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(year):
        return ""
    return str(int(year))

def valid_price_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["price"].notna() & (df["price"] > 0)]

def select_price(df: pd.DataFrame) -> Optional[float]:
    df = valid_price_rows(df)
    if df.empty:
        return None
    return float(df.iloc[0]["price"])

def calculate_valuation(price: float) -> Optional[Dict[str, Dict[str, int]]]:
    if price is None or pd.isna(price) or price <= 0:
        return None
    try:
        retail_avg = round(price, -2)
        retail_min = round(retail_avg * 0.9, -2)
        retail_max = round(retail_avg * 1.1, -2)

        trade_avg = round(retail_avg * 0.92, -2)
        trade_min = round(trade_avg * 0.9, -2)
        trade_max = round(trade_avg * 1.1, -2)

        return {
            "retail_price": {
                "average": int(retail_avg),
                "minimum": int(retail_min),
                "maximum": int(retail_max),
            },
            "trade_price": {
                "average": int(trade_avg),
                "minimum": int(trade_min),
                "maximum": int(trade_max),
            },
        }
    except Exception:
        return None

# --- Price Lookup Engine ---
def get_lookup_valuation(result: dict, vin: str) -> Optional[Dict[str, Any]]:
    if _lookup_df is None:
        return None

    vin_key = normalize_lookup_text(vin)
    matches = pd.DataFrame()
    if vin_key:
        vin_matches = _lookup_df[_lookup_df["vin_key"] == vin_key]
        if not vin_matches.empty:
            matches = vin_matches

    if matches.empty:
        make_key = normalize_lookup_text(result.get("make"))
        model_key = normalize_lookup_text(result.get("model"))
        year_key = normalize_lookup_year(result.get("year"))
        trim_key = normalize_lookup_text(result.get("trim"))

        # 1. Exact Match (Make + Model + Year + Trim)
        if trim_key and trim_key not in ["UNKNOWN", ""]:
            exact_matches = _lookup_df[
                (_lookup_df["make_key"] == make_key)
                & (_lookup_df["model_key"] == model_key)
                & (_lookup_df["year_key"] == year_key)
                & (_lookup_df["trim_key"] == trim_key)
            ]
            if not exact_matches.empty:
                matches = exact_matches

        # 2. Fallback to Model-Year Match (Make + Model + Year)
        if matches.empty and year_key:
            year_matches = _lookup_df[
                (_lookup_df["make_key"] == make_key)
                & (_lookup_df["model_key"] == model_key)
                & (_lookup_df["year_key"] == year_key)
            ]
            if not year_matches.empty:
                matches = year_matches

        # 3. Fallback to Model Match (Make + Model)
        if matches.empty:
            model_matches = _lookup_df[
                (_lookup_df["make_key"] == make_key)
                & (_lookup_df["model_key"] == model_key)
            ]
            if not model_matches.empty:
                matches = model_matches

    # Filter to only valid price rows
    matches = valid_price_rows(matches)
    if matches.empty:
        return None

    # Retrieve all prices to compute accurate statistics
    prices = matches["price"].dropna().tolist()
    if not prices:
        return None

    avg_price = float(np.mean(prices))
    min_price = float(np.min(prices))
    max_price = float(np.max(prices))

    # If min and max are equal, apply default 10% range bounds
    if min_price == max_price:
        min_price = avg_price * 0.9
        max_price = avg_price * 1.1

    retail_avg = round(avg_price, -2)
    retail_min = round(min_price, -2)
    retail_max = round(max_price, -2)

    trade_avg = round(retail_avg * 0.92, -2)
    trade_min = round(retail_min * 0.92, -2)
    trade_max = round(retail_max * 0.92, -2)

    return {
        "retail_price": {
            "average": int(retail_avg),
            "minimum": int(retail_min),
            "maximum": int(retail_max),
        },
        "trade_price": {
            "average": int(trade_avg),
            "minimum": int(trade_min),
            "maximum": int(trade_max),
        },
    }

# --- CatBoost ML Engine ---
def get_ml_predicted_price_info(result: dict) -> Optional[float]:
    try:
        from Valuation_ml.train_price_model import predict_price_with_confidence
        make = result.get("make", "")
        model = result.get("model", "")
        trim = result.get("trim", "")
        year = result.get("year", 2020)
        try:
            year_val = int(pd.to_numeric(year))
        except Exception:
            year_val = 2020
            
        ml_info = predict_price_with_confidence(make, model, trim, year_val)
        if ml_info and "price" in ml_info:
            return float(ml_info["price"])
    except Exception as e:
        print(f"Error in ML prediction: {e}")
    return None

# --- Toyota Engine ---
def get_toyota_engine_valuation(result: dict) -> Optional[Dict[str, Any]]:
    make = str(result.get("make", "")).strip()
    if make.upper() != "TOYOTA" or _toyota_engine is None:
        return None

    required_fields = {
        "model": result.get("model"),
        "trim": result.get("trim"),
        "year": result.get("year"),
        "body_type": result.get("body_type"),
        "regional_spec": result.get("regional_spec"),
    }
    missing = [
        field
        for field, value in required_fields.items()
        if value is None or str(value).strip().upper() in {"", "UNKNOWN", "NAN", "NONE"}
    ]
    if missing:
        return None

    try:
        payload = {
            "make": make,
            "model": str(result.get("model")).strip(),
            "trim": str(result.get("trim")).strip(),
            "model_year": int(pd.to_numeric(result.get("year"))),
            "body_type": str(result.get("body_type")).strip(),
            "regional_specs": str(result.get("regional_spec")).strip(),
        }
        _toyota_engine.calculator.current_year = date.today().year
        return _toyota_engine.value(payload)
    except Exception as exc:
        print(f"Toyota valuation failed: {exc}")
        return None

# --- Depreciation matching ---
def score_trim_match(decoded_trim: str, csv_trim: str) -> float:
    if not decoded_trim or pd.isna(decoded_trim):
        return 0.0
    if not csv_trim or pd.isna(csv_trim):
        return 0.0

    def get_clean_tokens(s):
        s_clean = re.sub(r'[^a-z0-9\s]', ' ', str(s).lower())
        tokens = s_clean.split()
        
        noise = {
            'i3', 'i4', 'i5', 'i6', 'v6', 'v8', 'v10', 'v12', 'w12',
            't', 'tc', 'tsc', 'tfsi', 'fsi', 'gdi', 'hybrid', 'phev', 'ev',
            'quattro', 'fwd', 'rwd', 'awd', '4wd', '4x4', 'specs', 'spec',
            'aed', 'nan', 'null', 'liter', 'liters', 'litre', 'litres',
            'coupe', 'sedan', 'cabriolet', 'hatchback', 'suv', 'wagon',
            'cvt', 'auto', 'manual', 'at', 'mt', 'dseg', 'cseg', 'bseg'
        }
        
        clean_toks = set()
        for t in tokens:
            if t in noise:
                continue
            if re.match(r'^\d+(\.\d+)?[tl]?$', t):
                continue
            clean_toks.add(t)
        return clean_toks

    dec_tokens = get_clean_tokens(decoded_trim)
    csv_tokens = get_clean_tokens(csv_trim)
    
    if not dec_tokens or not csv_tokens:
        dec_tokens = set(re.sub(r'[^a-z0-9\s]', ' ', str(decoded_trim).lower()).split())
        csv_tokens = set(re.sub(r'[^a-z0-9\s]', ' ', str(csv_trim).lower()).split())
        
    if not dec_tokens or not csv_tokens:
        return 0.0
        
    intersection = dec_tokens.intersection(csv_tokens)
    if not intersection:
        return 0.0
        
    score = len(intersection) / min(len(dec_tokens), len(csv_tokens))
    
    dec_clean_str = "".join(sorted(list(dec_tokens)))
    csv_clean_str = "".join(sorted(list(csv_tokens)))
    seq_ratio = SequenceMatcher(None, dec_clean_str, csv_clean_str).ratio()
    
    return score + 0.1 * seq_ratio

def find_matches_in_car_dep(df: pd.DataFrame, result: dict) -> pd.DataFrame:
    make = str(result.get("make", "")).strip().upper()
    model = str(result.get("model", "")).strip().upper()
    try:
        year_val = int(pd.to_numeric(result.get("year")))
    except Exception:
        return pd.DataFrame()

    make_matches = df[df["make"].str.upper().str.strip() == make]
    if make_matches.empty:
        make_matches = df[df["make"].str.upper().str.contains(make, na=False)]
        if make_matches.empty:
            return pd.DataFrame()

    year_matches = make_matches[make_matches["year"] == year_val]
    if year_matches.empty:
        return pd.DataFrame()

    norm_decoded_model = re.sub(r"[^A-Z0-9]", "", model)
    
    def model_matches(csv_model):
        if pd.isna(csv_model):
            return False
        norm_csv = re.sub(r"[^A-Z0-9]", "", str(csv_model).upper())
        return (norm_decoded_model == norm_csv) or (norm_decoded_model in norm_csv) or (norm_csv in norm_decoded_model)

    model_matches_df = year_matches[year_matches["model"].apply(model_matches)]
    if model_matches_df.empty:
        def slug_matches(csv_slug):
            if pd.isna(csv_slug):
                return False
            norm_slug = re.sub(r"[^A-Z0-9]", "", str(csv_slug).upper())
            return (norm_decoded_model in norm_slug)
        model_matches_df = year_matches[year_matches["model_slug"].apply(slug_matches)]
        if model_matches_df.empty:
            return pd.DataFrame()

    trim = str(result.get("trim", "")).strip().upper()
    if not trim or trim in ["UNKNOWN", ""]:
        return pd.DataFrame()

    scored_candidates = model_matches_df.copy()
    scored_candidates["trim_score"] = scored_candidates["trim_name"].apply(lambda x: score_trim_match(trim, x))
    
    valid_candidates = scored_candidates[scored_candidates["trim_score"] > 0.0]
    if not valid_candidates.empty:
        best_score = valid_candidates["trim_score"].max()
        if best_score >= 0.8:
            return valid_candidates[valid_candidates["trim_score"] == best_score]

    return pd.DataFrame()

def get_depreciated_values_by_file(result: dict) -> dict:
    if _depreciation_df is None or _depreciation_df.empty:
        return {}

    matched_df = find_matches_in_car_dep(_depreciation_df, result)
    if not matched_df.empty and "depreciated_value" in matched_df.columns:
        valid_vals = matched_df["depreciated_value"].dropna()
        if not valid_vals.empty:
            return {"DriveArabia": float(valid_vals.iloc[0])}

    return {}
