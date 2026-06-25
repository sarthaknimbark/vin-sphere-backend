import sys
import os
import re
import importlib
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st

# Configure page metadata
st.set_page_config(
    page_title="VIN Decoder",
    page_icon="",
    layout="centered",
)

# Hide Streamlit's default toolbar (Fork, GitHub, menu) and footer
st.markdown(
    """
    <style>
        /* Hide the top-right toolbar: Fork button, GitHub icon, three-dot menu */
        [data-testid="stToolbar"] { visibility: hidden !important; }
        .stDeployButton { display: none !important; }
        #MainMenu { visibility: hidden !important; }
        /* Hide Made with Streamlit footer */
        footer { visibility: hidden !important; }
        /* Hide the header bar entirely */
        header[data-testid="stHeader"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# Resolve project paths
app_dir = Path(__file__).resolve().parent
project_root = app_dir.parent
LOOKUP_DATA_PATH = project_root / "lookup_data.csv"
CAR_DEP_CSV_PATH = project_root / "Valuation" / "data" / "drivearabia_car_depreciation_valuation" / "car_dep.csv"

st.title("VIN Decoder")
st.write("Decode and get valuation for the VIN (09, 10, 11, 12, 13, 17 characters) | v2.1")

# =====================================================================
#                        DYNAMIC IMPORT ROUTER
# =====================================================================


# Short-VIN directories whose pickled vin_decoder_pipeline.pkl requires
# the folder to remain on sys.path so joblib can find 'vin_decoder' as a
# top-level module during deserialization.
_SHORT_VIN_DIRS = {
    "chat_cat(2.0)",
    "chat_cat(3.0)",
    "chat_cat_short_vin",
    "chat_cat_short_vin_09",
    "chat_cat_short_vin_11",
    "chat_cat_short_vin_12",
    "chat_cat_short_vin_13",
}


def load_predict_module(directory_name: str):
    """Loads predict.py from a sub-directory, keeping the path on sys.path for
    short-VIN modules so their pickled vin_decoder_pipeline.pkl can unpickle.

    Evicts all bare-name sibling modules (model, model_utils, feature_engineering,
    dataset, predict …) from sys.modules before importing so that any on-disk
    changes to those files are always picked up by the running Streamlit server
    without requiring a manual restart.
    """
    dir_path = project_root / directory_name
    keep_on_path = directory_name in _SHORT_VIN_DIRS

    # Bare module names that live inside chat_cat-style directories.
    # Clearing them forces Python to re-read them from disk on next import.
    _CHAT_CAT_MODULES = {
        "predict", "model", "model_utils",
        "feature_engineering", "dataset",
        "vin_decoder", "train_pipeline",
    }

    # Evict all stale bare-name modules from the module cache
    for mod_key in list(sys.modules.keys()):
        base = mod_key.split(".")[0]
        if base in _CHAT_CAT_MODULES:
            del sys.modules[mod_key]

    # Ensure the directory is at the front of sys.path so it takes priority
    if str(dir_path) in sys.path:
        sys.path.remove(str(dir_path))
    sys.path.insert(0, str(dir_path))

    try:
        predict_module = importlib.import_module("predict")
    finally:
        # Only remove path if this module doesn't need it for pickle deserialization
        if not keep_on_path and str(dir_path) in sys.path:
            sys.path.remove(str(dir_path))

    return predict_module


def decode_any_vin(vin_str: str) -> tuple:
    """Standardizes routing logic for 9, 10, 11, 12, 13, and 17 character VIN/Chassis numbers."""
    normalized = re.sub(r"[^A-Z0-9]", "", vin_str.upper())
    length = len(normalized)

    if length == 17:
        mod = load_predict_module("chat_cat(2.0)")
        res = mod.decode_vin(
            normalized, model_dir=str(project_root / "chat_cat(2.0)" / "models")
        )
        return res, 17
    elif length == 13:
        mod = load_predict_module("chat_cat_short_vin_13")
        res = mod.predict_vehicle(
            normalized,
            model_dir=str(project_root / "chat_cat_short_vin_13" / "models"),
        )
        return res, 13
    elif length == 12:
        mod = load_predict_module("chat_cat_short_vin_12")
        res = mod.predict_vehicle(
            normalized,
            model_dir=str(project_root / "chat_cat_short_vin_12" / "models"),
        )
        return res, 12
    elif length == 11:
        mod = load_predict_module("chat_cat_short_vin_11")
        res = mod.predict_vehicle(
            normalized,
            model_dir=str(project_root / "chat_cat_short_vin_11" / "models"),
        )
        return res, 11
    elif length == 10:
        mod = load_predict_module("chat_cat_short_vin")
        res = mod.predict_vehicle(
            normalized,
            model_dir=str(project_root / "chat_cat_short_vin" / "models"),
        )
        return res, 10
    elif length == 9:
        mod = load_predict_module("chat_cat_short_vin_09")
        res = mod.predict_vehicle(
            normalized,
            model_dir=str(project_root / "chat_cat_short_vin_09" / "models"),
        )
        return res, 9
    else:
        raise ValueError(
            f"Invalid length: {length} characters. Must be 9, 10, 11, 12, 13, or 17 characters."
        )


def normalize_result_keys(result: dict) -> dict:
    """Converts prediction returns into a standardized flat dictionary."""
    mapped = {}
    mapped["make"] = str(result.get("make", "UNKNOWN")).strip().upper()
    mapped["model"] = str(result.get("model", "UNKNOWN")).strip().upper()
    mapped["year"] = str(result.get("year", "UNKNOWN")).strip().upper()
    mapped["trim"] = str(result.get("trim", "UNKNOWN")).strip().upper()
    mapped["body_type"] = str(result.get("body_type", "UNKNOWN")).strip().upper()
    mapped["cylinders"] = str(result.get("cylinders", "UNKNOWN")).strip().upper()
    mapped["origin"] = str(result.get("origin", "UNKNOWN")).strip().upper()
    mapped["no_of_passengers"] = (
        str(result.get("no_of_passengers", "UNKNOWN")).strip().upper()
    )
    mapped["weight"] = str(result.get("weight", "UNKNOWN")).strip().upper()
    mapped["color"] = str(result.get("color", "UNKNOWN")).strip().upper()

    reg_spec = result.get("regional_spec", result.get("regional_specs", "UNKNOWN"))
    mapped["regional_spec"] = str(reg_spec).strip().upper()

    mapped["attribute_confidences"] = result.get("attribute_confidences", {})
    mapped["confidence"] = result.get("confidence", 0.0)
    return mapped


# =====================================================================
#                       VALUATION MODEL LOADERS
# =====================================================================


@st.cache_data
def load_price_lookup(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.copy()
    df["make_key"] = df["make"].apply(normalize_lookup_text)
    df["model_key"] = df["model"].apply(normalize_lookup_text)
    df["trim_key"] = df["trim"].apply(normalize_lookup_text)
    df["year_key"] = df["year"].apply(normalize_lookup_year)
    if "chassisNumber" in df.columns:
        df["vin_key"] = df["chassisNumber"].apply(normalize_lookup_text)
    else:
        df["vin_key"] = ""
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df


def normalize_lookup_text(value) -> str:
    if value is None:
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def normalize_lookup_year(value) -> str:
    year = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(year):
        return ""
    return str(int(year))


def valid_price_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["price"].notna() & (df["price"] > 0)]


def select_price(df: pd.DataFrame):
    df = valid_price_rows(df)
    if df.empty:
        return None
    return float(df.iloc[0]["price"])


def get_lookup_price(result: dict, vin: str):
    if not LOOKUP_DATA_PATH.exists():
        return None

    lookup_df = load_price_lookup(str(LOOKUP_DATA_PATH))

    vin_key = normalize_lookup_text(vin)
    if vin_key:
        vin_matches = lookup_df[lookup_df["vin_key"] == vin_key]
        price = select_price(vin_matches)
        if price is not None:
            return price

    # Require exact match on make, model, year, and trim
    make_key = normalize_lookup_text(result.get("make"))
    model_key = normalize_lookup_text(result.get("model"))
    year_key = normalize_lookup_year(result.get("year"))
    trim_key = normalize_lookup_text(result.get("trim"))

    if not trim_key or trim_key in ["UNKNOWN", ""]:
        return None

    exact_matches = lookup_df[
        (lookup_df["make_key"] == make_key)
        & (lookup_df["model_key"] == model_key)
        & (lookup_df["year_key"] == year_key)
        & (lookup_df["trim_key"] == trim_key)
    ]
    
    return select_price(exact_matches)


def calculate_valuation(price: float) -> dict:
    if price is None or pd.isna(price):
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


@st.cache_data
def load_car_depreciation_data():
    if not CAR_DEP_CSV_PATH.exists():
        # Fallback to alternate spelling if needed
        alt_path = project_root / "Valuation" / "data" / "drivearabia_car_depriciation_valuation" / "car_dep.csv"
        if alt_path.exists():
            return pd.read_csv(alt_path)
        return None
    try:
        return pd.read_csv(CAR_DEP_CSV_PATH)
    except Exception:
        return None


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

    # 1. Filter by make
    make_matches = df[df["make"].str.upper().str.strip() == make]
    if make_matches.empty:
        make_matches = df[df["make"].str.upper().str.contains(make, na=False)]
        if make_matches.empty:
            return pd.DataFrame()

    # 2. Filter by year
    year_matches = make_matches[make_matches["year"] == year_val]
    if year_matches.empty:
        return pd.DataFrame()

    # 3. Filter by model
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

    # 4. Filter by trim
    trim = str(result.get("trim", "")).strip().upper()
    if not trim or trim in ["UNKNOWN", ""]:
        return pd.DataFrame()

    scored_candidates = model_matches_df.copy()
    scored_candidates["trim_score"] = scored_candidates["trim_name"].apply(lambda x: score_trim_match(trim, x))
    
    valid_candidates = scored_candidates[scored_candidates["trim_score"] > 0.0]
    if not valid_candidates.empty:
        best_score = valid_candidates["trim_score"].max()
        # Require a high score (e.g. >= 0.8) to consider it a match
        if best_score >= 0.8:
            return valid_candidates[valid_candidates["trim_score"] == best_score]

    return pd.DataFrame()


def get_depreciated_values_by_file(result: dict) -> dict:
    df = load_car_depreciation_data()
    if df is None or df.empty:
        return {}

    matched_df = find_matches_in_car_dep(df, result)
    if not matched_df.empty and "depreciated_value" in matched_df.columns:
        valid_vals = matched_df["depreciated_value"].dropna()
        if not valid_vals.empty:
            return {"DriveArabia": float(valid_vals.iloc[0])}

    return {}


#VALUATION ENSEMBLE ROUTER

def get_ml_predicted_price_info(result: dict) -> dict:
    try:
        sys.path.insert(0, str(project_root))
        from Valuation_ml.train_price_model import predict_price_with_confidence

        make = result.get("make", "")
        model = result.get("model", "")
        trim = result.get("trim", "")
        year = result.get("year", 2020)
        return predict_price_with_confidence(make, model, trim, year)
    except Exception as e:
        print(f"Error in ML prediction: {e}")
        return None
    finally:
        if str(project_root) in sys.path:
            sys.path.remove(str(project_root))

#                         INPUT CONTROL PANEL

vin_input = st.text_input(
    "Enter Chassis Number to decode:",
    placeholder="Enter VIN Number here.",
    max_chars=17,
).strip().upper()

if st.button("Decode", type="primary"):
    normalized_input = re.sub(r"[^A-Z0-9]", "", vin_input)
    length = len(normalized_input)

    if length not in [9, 10, 11, 12, 13, 17]:
        st.error(
            f"Invalid chassis length ({length} characters). Input must be exactly 9, 10, 11, 12, 13, or 17 characters."
        )
    else:
        with st.spinner("Decoding VIN..."):
            try:
                # 1. Decode attributes
                res, matched_len = decode_any_vin(normalized_input)
                result = normalize_result_keys(res)

                # 2. Get Lookup Price
                price = get_lookup_price(result, normalized_input)
                price_display = (
                    "Not found" if price is None else f"{price:,.0f} AED"
                )
                result["lookup_price"] = None if price is None else float(price)

                # 3. Get ML price
                ml_info = get_ml_predicted_price_info(result)
                overall_conf = result.get("confidence", 0.0) * 100

                if ml_info is None:
                    ml_price = None
                    ml_price_display = "Not found"
                    ml_conf_display = "Not found"
                    ml_conf_level = ""
                else:
                    ml_price = ml_info["price"]
                    ml_price_display = f"{ml_price:,.0f} AED"
                    ml_conf_pct = ml_info.get("confidence_pct", ml_info.get("confidence", 0.0) * 100)
                    ml_conf_display = f"{ml_conf_pct:.1f}%"

                result["ml_predicted_price"] = (
                    None if ml_price is None else float(ml_price)
                )

                # 4. Get depreciated values
                dep_values = get_depreciated_values_by_file(result)
                result["depreciated_values"] = dep_values

                # 5. Calculate valuations
                result["lookup_price_valuation"] = calculate_valuation(price)
                result["ml_price_valuation"] = calculate_valuation(ml_price)
                result["depreciated_price_valuations"] = (
                    {
                        filename: calculate_valuation(val)
                        for filename, val in dep_values.items()
                    }
                    if dep_values
                    else {}
                )

                # Format rows for table
                attr_conf = result.get("attribute_confidences", {})

                def fmt_conf(key: str) -> str:
                    val = attr_conf.get(key)
                    if val is None:
                        return ""
                    return f"{float(val) * 100:.1f}%"

                table_data = [
                    {
                        "Attribute": "Make",
                        "Predicted Value": result.get("make"),
                        "Confidence": fmt_conf("make"),
                    },
                    {
                        "Attribute": "Model",
                        "Predicted Value": result.get("model"),
                        "Confidence": fmt_conf("model"),
                    },
                    {
                        "Attribute": "Trim",
                        "Predicted Value": result.get("trim"),
                        "Confidence": fmt_conf("trim"),
                    },
                    {
                        "Attribute": "Body Type",
                        "Predicted Value": result.get("body_type"),
                        "Confidence": fmt_conf("body_type"),
                    },
                    {
                        "Attribute": "Model Year",
                        "Predicted Value": result.get("year"),
                        "Confidence": fmt_conf("year"),
                    },
                    {
                        "Attribute": "Cylinders",
                        "Predicted Value": result.get("cylinders"),
                        "Confidence": fmt_conf("cylinders"),
                    },
                    {
                        "Attribute": "Origin",
                        "Predicted Value": result.get("origin"),
                        "Confidence": fmt_conf("origin"),
                    },
                    {
                        "Attribute": "Number of Passengers",
                        "Predicted Value": result.get("no_of_passengers"),
                        "Confidence": fmt_conf("no_of_passengers"),
                    },
                    {
                        "Attribute": "Weight (KG)",
                        "Predicted Value": result.get("weight"),
                        "Confidence": fmt_conf("weight"),
                    },
                    {
                        "Attribute": "Regional Specs",
                        "Predicted Value": result.get("regional_spec"),
                        "Confidence": fmt_conf("regional_spec"),
                    },
                    {
                        "Attribute": "Color",
                        "Predicted Value": result.get("color"),
                        "Confidence": fmt_conf("color"),
                    },
                    {
                        "Attribute": "Price (Lookup)",
                        "Predicted Value": price_display,
                        "Confidence": "Lookup table",
                    },
                    {
                        "Attribute": "Price (ML Model)",
                        "Predicted Value": ml_price_display,
                        "Confidence": ml_conf_display if ml_info is not None else "",
                    },
                ]

                # Show only matched depreciation sources (dynamic names + price).
                # If no source matched at all, show a single "Not Found" row.
                dep_rows = []
                if dep_values:
                    for filename, val in dep_values.items():
                        display_name = filename.rsplit(".", 1)[0]
                        dep_rows.append(
                            {
                                "Attribute": f"Depreciated Price ({display_name})",
                                "Predicted Value": f"{val:,.0f} AED",
                                "Confidence": "",
                            }
                        )
                else:
                    dep_rows.append(
                        {
                            "Attribute": "Depreciated Price",
                            "Predicted Value": "Not Found",
                            "Confidence": "Scraped Data",
                        }
                    )
                table_data.extend(dep_rows)

                # =====================================================
                #                   DASHBOARD RENDERING
                # =====================================================
                st.write(f"Chassis Length Detected: {matched_len} Characters")

                # Overall model confidence
                overall_conf = result.get("confidence", 0.0) * 100
                st.metric(label="Overall Model Confidence", value=f"{overall_conf:.1f}%")

                st.write("### Decoded Vehicle Attributes")
                st.table(pd.DataFrame(table_data))

                # =====================================================
                #              MARKET VALUATION DASHBOARD
                # =====================================================
                has_any_valuation = (
                    (price is not None)
                    or (ml_price is not None)
                    or bool(dep_values)
                )
                if has_any_valuation:
                    st.write("### Market Valuation Dashboard")

                    tab_titles = []
                    if price is not None:
                        tab_titles.append("Market Lookup")
                    if ml_price is not None:
                        tab_titles.append("ML Model Prediction")
                    if dep_values:
                        tab_titles.append("Depreciation Lookup")

                    tabs = st.tabs(tab_titles)
                    tab_index = 0

                    if price is not None:
                        with tabs[tab_index]:
                            val = result["lookup_price_valuation"]
                            st.write(
                                "**Retail Price vs. Trade Price (Lookup)**"
                            )
                            st.table(
                                pd.DataFrame(
                                    [
                                        {
                                            "Price Metric": "Minimum Price (-10%)",
                                            "Retail Price (AED)": f"{val['retail_price']['minimum']:,.0f}",
                                            "Trade Price (AED)": f"{val['trade_price']['minimum']:,.0f}",
                                        },
                                        {
                                            "Price Metric": "Average Price",
                                            "Retail Price (AED)": f"{val['retail_price']['average']:,.0f}",
                                            "Trade Price (AED)": f"{val['trade_price']['average']:,.0f}",
                                        },
                                        {
                                            "Price Metric": "Maximum Price (+10%)",
                                            "Retail Price (AED)": f"{val['retail_price']['maximum']:,.0f}",
                                            "Trade Price (AED)": f"{val['trade_price']['maximum']:,.0f}",
                                        },
                                    ]
                                )
                            )
                            tab_index += 1

                    if ml_price is not None:
                        with tabs[tab_index]:
                            val = result["ml_price_valuation"]
                            st.write("**Retail Price vs. Trade Price (ML Model)**")
                            st.table(
                                pd.DataFrame(
                                    [
                                        {
                                            "Price Metric": "Minimum Price (-10%)",
                                            "Retail Price (AED)": f"{val['retail_price']['minimum']:,.0f}",
                                            "Trade Price (AED)": f"{val['trade_price']['minimum']:,.0f}",
                                        },
                                        {
                                            "Price Metric": "Average Price",
                                            "Retail Price (AED)": f"{val['retail_price']['average']:,.0f}",
                                            "Trade Price (AED)": f"{val['trade_price']['average']:,.0f}",
                                        },
                                        {
                                            "Price Metric": "Maximum Price (+10%)",
                                            "Retail Price (AED)": f"{val['retail_price']['maximum']:,.0f}",
                                            "Trade Price (AED)": f"{val['trade_price']['maximum']:,.0f}",
                                        },
                                    ]
                                )
                            )
                            tab_index += 1

                    if dep_values:
                        with tabs[tab_index]:
                            st.write(
                                "**Retail Price vs. Trade Price (Depreciation)**"
                            )
                            for filename, dep_val in dep_values.items():
                                val = result["depreciated_price_valuations"][
                                    filename
                                ]
                                display_name = filename.rsplit(".", 1)[0]
                                st.markdown(f"**Source File: `{display_name}`**")
                                st.table(
                                    pd.DataFrame(
                                        [
                                            {
                                                "Price Metric": "Minimum Price (-10%)",
                                                "Retail Price (AED)": f"{val['retail_price']['minimum']:,.0f}",
                                                "Trade Price (AED)": f"{val['trade_price']['minimum']:,.0f}",
                                            },
                                            {
                                                "Price Metric": "Average Price",
                                                "Retail Price (AED)": f"{val['retail_price']['average']:,.0f}",
                                                "Trade Price (AED)": f"{val['trade_price']['average']:,.0f}",
                                            },
                                            {
                                                "Price Metric": "Maximum Price (+10%)",
                                                "Retail Price (AED)": f"{val['retail_price']['maximum']:,.0f}",
                                                "Trade Price (AED)": f"{val['trade_price']['maximum']:,.0f}",
                                            },
                                        ]
                                    )
                                )
                            tab_index += 1

                # 6. Display raw JSON response
                st.write("### Raw Model Response (JSON)")
                st.json(result)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.exception(e)
