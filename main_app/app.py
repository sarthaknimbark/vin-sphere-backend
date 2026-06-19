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

# Resolve project paths
app_dir = Path(__file__).resolve().parent
project_root = app_dir.parent
LOOKUP_DATA_PATH = project_root / "lookup_data.csv"
DEPRECIATION_DIR = project_root / "Valuation" / "data"/ "drivearabia_car_depriciation_valuation"/"depreciation"

DEPRECIATION_CSV_PATHS = {
    "audi_dep.csv": DEPRECIATION_DIR / "audi_dep.csv",
    "byd_dep.csv": DEPRECIATION_DIR / "byd_dep.csv",
    "chevelorate_dep.csv": DEPRECIATION_DIR / "chevelorate_dep.csv",
    "ford_dep.csv": DEPRECIATION_DIR / "ford_dep.csv",
    "gmc_dep.csv": DEPRECIATION_DIR / "gmc_dep.csv",
    "honda_dep.csv": DEPRECIATION_DIR / "honda_dep.csv",
    "hyundai_dep.csv": DEPRECIATION_DIR / "hyundai_dep.csv",
    "infiniti_dep.csv": DEPRECIATION_DIR / "infiniti_dep.csv",
    "jeep_dep.csv": DEPRECIATION_DIR / "jeep_dep.csv",
    "kia_dep.csv": DEPRECIATION_DIR / "kia_dep.csv",
    "lexus_dep.csv": DEPRECIATION_DIR / "lexus_dep.csv",
    "lincoln_dep.csv": DEPRECIATION_DIR / "lincoln_dep.csv",
    "mazda_dep.csv": DEPRECIATION_DIR / "mazda_dep.csv",
    "mini_dep.csv": DEPRECIATION_DIR / "mini_dep.csv",
    "mitsubishi_dep.csv": DEPRECIATION_DIR / "mitsubishi_dep.csv",
    "porsche_dep.csv": DEPRECIATION_DIR / "porsche_dep.csv",
    "rox_dep.csv": DEPRECIATION_DIR / "rox_dep.csv",
    "suzuki_dep.csv": DEPRECIATION_DIR / "suzuki_dep.csv",
    "tesla_dep.csv": DEPRECIATION_DIR / "tesla_dep.csv",
    "toyota_dep.csv": DEPRECIATION_DIR / "toyota_dep.csv",
    "volkswagen_dep.csv": DEPRECIATION_DIR / "volkswagen_dep.csv",

}

st.title("VIN Decoder")
st.write("Decode and get valuation for the VIN (09, 10, 11, 12, 17 characters)")

# =====================================================================
#                        DYNAMIC IMPORT ROUTER
# =====================================================================


# Short-VIN directories whose pickled vin_decoder_pipeline.pkl requires
# the folder to remain on sys.path so joblib can find 'vin_decoder' as a
# top-level module during deserialization.
_SHORT_VIN_DIRS = {
    "chat_cat_short_vin",
    "chat_cat_short_vin_09",
    "chat_cat_short_vin_11",
    "chat_cat_short_vin_12",
}


def load_predict_module(directory_name: str):
    """Loads predict.py from a sub-directory, keeping the path on sys.path for
    short-VIN modules so their pickled vin_decoder_pipeline.pkl can unpickle."""
    dir_path = project_root / directory_name
    keep_on_path = directory_name in _SHORT_VIN_DIRS

    # Add to sys.path if not already there
    if str(dir_path) not in sys.path:
        sys.path.insert(0, str(dir_path))

    try:
        # Clear existing 'predict' from module cache to force reload from the right dir
        for mod_key in list(sys.modules.keys()):
            if mod_key == "predict" or mod_key.startswith("predict."):
                del sys.modules[mod_key]
        predict_module = importlib.import_module("predict")
    finally:
        # Only remove path if this module doesn't need it for pickle deserialization
        if not keep_on_path and str(dir_path) in sys.path:
            sys.path.remove(str(dir_path))

    return predict_module


def decode_any_vin(vin_str: str) -> tuple:
    """Standardizes routing logic for 9, 10, 11, 12, and 17 character VIN/Chassis numbers."""
    normalized = re.sub(r"[^A-Z0-9]", "", vin_str.upper())
    length = len(normalized)

    if length == 17:
        mod = load_predict_module("chat_cat(3.0)")
        res = mod.decode_vin(
            normalized, model_dir=str(project_root / "chat_cat(3.0)" / "models")
        )
        return res, 17
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
            f"Invalid length: {length} characters. Must be 9, 10, 11, 12, or 17 characters."
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
    return float(df["price"].median())


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

    base_matches = lookup_df[
        (lookup_df["make_key"] == normalize_lookup_text(result.get("make")))
        & (lookup_df["model_key"] == normalize_lookup_text(result.get("model")))
        & (lookup_df["year_key"] == normalize_lookup_year(result.get("year")))
    ]

    if base_matches.empty:
        return None

    trim_key = normalize_lookup_text(result.get("trim"))
    if trim_key:
        exact_trim_matches = base_matches[base_matches["trim_key"] == trim_key]
        price = select_price(exact_trim_matches)
        if price is not None:
            return price

        contains_trim_matches = base_matches[
            base_matches["trim_key"].apply(
                lambda csv_trim: bool(csv_trim)
                and (trim_key in csv_trim or csv_trim in trim_key)
            )
        ]
        price = select_price(contains_trim_matches)
        if price is not None:
            return price

        fuzzy_matches = base_matches.copy()
        fuzzy_matches["trim_score"] = fuzzy_matches["trim_key"].apply(
            lambda csv_trim: SequenceMatcher(None, trim_key, csv_trim).ratio()
            if csv_trim
            else 0.0
        )
        fuzzy_matches = fuzzy_matches[fuzzy_matches["trim_score"] >= 0.82]
        if not fuzzy_matches.empty:
            best_score = fuzzy_matches["trim_score"].max()
            price = select_price(
                fuzzy_matches[fuzzy_matches["trim_score"] == best_score]
            )
            if price is not None:
                return price

    valid_base_matches = valid_price_rows(base_matches)
    if valid_base_matches["trim_key"].nunique() == 1:
        return select_price(valid_base_matches)

    return None


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
def load_all_depreciation_lookups() -> dict:
    dfs = {}
    for name, path in DEPRECIATION_CSV_PATHS.items():
        if Path(path).exists():
            try:
                dfs[name] = pd.read_csv(path)
            except Exception:
                pass
    return dfs


def find_matches_in_df(df: pd.DataFrame, result: dict) -> pd.DataFrame:
    make = str(result.get("make", "")).strip().lower()
    model = str(result.get("model", "")).strip().lower().replace(" ", "-")
    if make:
        if not model.startswith(f"{make}-"):
            model_slug = f"{make}-{model}"
        else:
            model_slug = model
    else:
        model_slug = model

    year = result.get("year")
    try:
        year_val = int(pd.to_numeric(year))
    except Exception:
        return pd.DataFrame()

    matches = df[(df["model_slug"] == model_slug) & (df["year"] == year_val)]
    if matches.empty:
        matches = df[
            df["model_slug"].str.contains(model, case=False, na=False)
            & (df["year"] == year_val)
        ]
        if matches.empty:
            return pd.DataFrame()

    trim = str(result.get("trim", "")).strip().upper()

    def normalize_trim(val):
        if pd.isna(val):
            return ""
        return re.sub(r"[^A-Z0-9]", "", str(val).upper())

    trim_key = normalize_trim(trim)

    best_match = None
    if trim_key:
        exact_matches = matches[
            matches["trim_name"].apply(normalize_trim) == trim_key
        ]
        if not exact_matches.empty:
            best_match = exact_matches
        else:
            contains_matches = matches[
                matches["trim_name"].apply(
                    lambda x: trim_key in normalize_trim(x)
                    or normalize_trim(x) in trim_key
                )
            ]
            if not contains_matches.empty:
                best_match = contains_matches
            else:
                matches_copy = matches.copy()
                matches_copy["score"] = matches_copy["trim_name"].apply(
                    lambda x: SequenceMatcher(
                        None, trim_key, normalize_trim(x)
                    ).ratio()
                    if x
                    else 0.0
                )
                fuzzy_matches = matches_copy[matches_copy["score"] >= 0.82]
                if not fuzzy_matches.empty:
                    best_score = fuzzy_matches["score"].max()
                    best_match = fuzzy_matches[
                        fuzzy_matches["score"] == best_score
                    ]

    if best_match is None or best_match.empty:
        best_match = matches

    return best_match


def get_depreciated_values_by_file(result: dict) -> dict:
    dfs_dict = load_all_depreciation_lookups()
    matched_values = {}

    for filename, df in dfs_dict.items():
        matched_df = find_matches_in_df(df, result)
        if not matched_df.empty and "depreciated_value" in matched_df.columns:
            valid_vals = matched_df["depreciated_value"].dropna()
            if not valid_vals.empty:
                matched_values[filename] = float(valid_vals.median())

    return matched_values


# =====================================================================
#                         VALUATION ENSEMBLE ROUTER
# =====================================================================


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


# =====================================================================
#                         INPUT CONTROL PANEL
# =====================================================================
vin_input = st.text_input(
    "Enter Chassis Number to decode:",
    placeholder="Enter VIN Number here.",
    max_chars=17,
).strip().upper()

if st.button("Decode", type="primary"):
    normalized_input = re.sub(r"[^A-Z0-9]", "", vin_input)
    length = len(normalized_input)

    if length not in [9, 10, 11, 12, 17]:
        st.error(
            f"Invalid chassis length ({length} characters). Input must be exactly 9, 10, 11, 12, or 17 characters."
        )
    else:
        with st.spinner("Decoding vehicle chassis metrics..."):
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
                conf_dict = result.get("attribute_confidences", {})
                table_data = [
                    {
                        "Attribute": "Make",
                        "Predicted Value": result.get("make"),
                        "Confidence": f"{conf_dict.get('make', 0.0) * 100:.2f}%",
                    },
                    {
                        "Attribute": "Model",
                        "Predicted Value": result.get("model"),
                        "Confidence": f"{conf_dict.get('model', 0.0) * 100:.2f}%",
                    },
                    {
                        "Attribute": "Trim",
                        "Predicted Value": result.get("trim"),
                        "Confidence": f"{conf_dict.get('trim', 0.0) * 100:.2f}%",
                    },
                    {
                        "Attribute": "Body Type",
                        "Predicted Value": result.get("body_type"),
                        "Confidence": f"{conf_dict.get('body_type', 0.0) * 100:.2f}%",
                    },
                    {
                        "Attribute": "Model Year",
                        "Predicted Value": result.get("year"),
                        "Confidence": f"{conf_dict.get('year', 0.0) * 100:.2f}%",
                    },
                    {
                        "Attribute": "Cylinders",
                        "Predicted Value": result.get("cylinders"),
                        "Confidence": f"{conf_dict.get('cylinders', 0.0) * 100:.2f}%",
                    },
                    {
                        "Attribute": "Origin",
                        "Predicted Value": result.get("origin"),
                        "Confidence": f"{conf_dict.get('origin', 0.0) * 100:.2f}%",
                    },
                    {
                        "Attribute": "Number of Passengers",
                        "Predicted Value": result.get("no_of_passengers"),
                        "Confidence": f"{conf_dict.get('no_of_passengers', 0.0) * 100:.2f}%",
                    },
                    {
                        "Attribute": "Weight (KG)",
                        "Predicted Value": result.get("weight"),
                        "Confidence": f"{conf_dict.get('weight', 0.0) * 100:.2f}%",
                    },
                    {
                        "Attribute": "Regional Specs",
                        "Predicted Value": result.get("regional_spec"),
                        "Confidence": f"{conf_dict.get('regional_spec', 0.0) * 100:.2f}%",
                    },
                    {
                        "Attribute": "Color",
                        "Predicted Value": result.get("color"),
                        "Confidence": f"{conf_dict.get('color', 0.0) * 100:.2f}%",
                    },
                    {
                        "Attribute": "Price (Lookup)",
                        "Predicted Value": price_display,
                        "Confidence": "Lookup Match",
                    },
                    {
                        "Attribute": "Price (ML Model)",
                        "Predicted Value": ml_price_display,
                        "Confidence": ml_conf_display,
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
                                "Confidence": "Scraped Depreciation",
                            }
                        )
                else:
                    dep_rows.append(
                        {
                            "Attribute": "Depreciated Price",
                            "Predicted Value": "Not Found",
                            "Confidence": "Scraped Depreciation",
                        }
                    )
                table_data.extend(dep_rows)

                # =====================================================
                #                   DASHBOARD RENDERING
                # =====================================================
                st.subheader(f"Chassis Length Detected: {matched_len} Characters")
                st.subheader(f"Overall Decoding Confidence: {overall_conf:.2f}%")
                st.progress(overall_conf / 100.0)

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
