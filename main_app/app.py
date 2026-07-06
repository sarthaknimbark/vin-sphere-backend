import sys
import os
import re
import time
import importlib
import warnings

# Suppress scikit-learn inconsistent version warnings during unpickling
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

# Suppress scikit-learn unpickling warnings generally
warnings.filterwarnings("ignore", message="Trying to unpickle estimator")
# Suppress user warnings (e.g. from XGBoost model loading)
warnings.filterwarnings("ignore", category=UserWarning)

from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st

# Configure page metadata
st.set_page_config(
    page_title="VIN Decoder",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)




# Resolve project paths
app_dir = Path(__file__).resolve().parent
project_root = app_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
LOOKUP_DATA_PATH = project_root / "car_app_dev"/ "dev_lookup.csv"
CAR_DEP_CSV_PATH = project_root / "Valuation" / "data" / "drivearabia_car_depreciation_valuation" / "car_dep_2.csv"

# Evict database modules from sys.modules cache to ensure Streamlit hot-reloads on-disk database changes
for mod_key in list(sys.modules.keys()):
    if mod_key == "database" or mod_key.startswith("database."):
        del sys.modules[mod_key]

from database import (
    get_db_connection,
    init_db,
    lookup_vin_in_master_dataset,
    log_prediction,
    safe_read_pending_predictions,
    save_pending_prediction,
    approve_pending_prediction,
    reject_pending_prediction,
    get_approved_predictions_count,
    update_pending_prediction
)


def format_csv_value(val):
    if pd.isna(val):
        return "UNKNOWN"
    if isinstance(val, float):
        if val.is_integer():
            return str(int(val))
        return str(val)
    return str(val)


def get_verified_vin_dict(row: dict) -> dict:
    return {
        "vin": format_csv_value(row.get("chassisNumber")),
        "year": format_csv_value(row.get("year")),
        "make": format_csv_value(row.get("make")),
        "model": format_csv_value(row.get("model")),
        "trim": format_csv_value(row.get("trim")),
        "body_type": format_csv_value(row.get("bodyType")),
        "regional_spec": format_csv_value(row.get("regionalSpec")),
        "cylinders": format_csv_value(row.get("cylinders")),
        "color": format_csv_value(row.get("color")),
        "origin": format_csv_value(row.get("origin")),
        "weight": format_csv_value(row.get("weightInKg")),
        "no_of_passengers": format_csv_value(row.get("noOfPassengers")),
        "confidence": 1.0,
        "attribute_confidences": {
            "make": 1.0, "model": 1.0, "year": 1.0, "trim": 1.0, "body_type": 1.0,
            "cylinders": 1.0, "origin": 1.0, "no_of_passengers": 1.0, "weight": 1.0,
            "regional_spec": 1.0, "color": 1.0
        }
    }


def render_admin_panel():
    # Check database connection status for Admin Panel
    db_connected, db_error = check_db_connection_cached()

    if not db_connected:
        st.error(f"Cannot access Admin Dashboard: Database is disconnected. Error: {db_error}")
        st.stop()

    # Admin PIN Authentication Gate
    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = False
        
    if not st.session_state.admin_authenticated:
        st.subheader("Admin Authentication Required")
        pin_input = st.text_input("Enter Admin PIN to Access", type="password", key="admin_pin_input")
        if st.button("Unlock Dashboard"):
            if pin_input == "123456":
                st.session_state.admin_authenticated = True
                st.success("Access Granted!")
                st.rerun()
            else:
                st.error("Incorrect PIN. Access Denied.")
        st.stop()

    st.title("Admin Verification Dashboard")
    st.write("Review, edit, approve, or reject pending decoded VIN predictions.")
    
    # Read stats
    df_pending = safe_read_pending_predictions()
    pending_count = len(df_pending) if (not df_pending.empty and "vin" in df_pending.columns) else 0
    approved_count = get_approved_predictions_count()

    # Display metric cards
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Pending Verifications", value=pending_count)
    with col2:
        st.metric(label="Chassis Added to Master Data", value=approved_count)
        
    if pending_count == 0:
        st.success("No pending VINs awaiting verification. Good job!")
        return
        
    st.subheader(f"Pending Predictions ({pending_count})")

    # Pagination Setup & Controls
    col_size, col_page, col_info = st.columns([1.5, 1.5, 5.0])
    with col_size:
        items_per_page = st.selectbox("Items per page", [10, 20, 50, 100, 200], index=0, key="admin_items_per_page")
    
    total_pages = (pending_count + items_per_page - 1) // items_per_page
    
    current_page = 1
    with col_page:
        current_page = st.number_input("Page", min_value=1, max_value=max(1, total_pages), value=1, step=1, key="admin_page_num")
        
    with col_info:
        st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
        st.caption(f"Showing page {current_page} of {total_pages} (Total: {pending_count} pending predictions)")
            
    start_idx = (current_page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, pending_count)
    df_page = df_pending.iloc[start_idx:end_idx]

    # Display the inline styling for colored buttons and Excel sheet formatting
    st.markdown(
        """
        <style>
        /* Admin table container layout */
        .admin-table-container {
            overflow-x: auto !important; /* Enable horizontal scrolling for the entire table container */
            padding-bottom: 10px !important;
            width: 100% !important;
        }

        /* Force single-line horizontal layout for all horizontal blocks inside the container */
        .admin-table-container [data-testid="stHorizontalBlock"] {
            flex-wrap: nowrap !important;
            align-items: center !important;
            display: flex !important;
            flex-direction: row !important;
            min-width: 1500px !important; /* Prevent squeezing columns below readable widths */
        }

        /* Reset stForm margins and containers inside the admin table */
        .admin-table-container div[data-testid="stForm"] {
            border: none !important;
            padding: 0px !important;
            margin-bottom: 2px !important;
            margin-top: 2px !important;
            background-color: transparent !important;
        }
        
        /* Excel style table cells and centering alignment */
        .admin-table-container [data-testid="stHorizontalBlock"] > div {
            padding: 0px 4px !important; /* Consistent horizontal padding spacing */
            margin: 0px !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
            flex-shrink: 0 !important;
        }
        
        /* Remove default margins on element containers to keep everything level */
        .admin-table-container [data-testid="stHorizontalBlock"] div[data-testid="stElementContainer"],
        .admin-table-container [data-testid="stHorizontalBlock"] div[data-testid="element-container"] {
            margin: 0px !important;
            padding: 0px !important;
            height: 38px !important;
            min-height: 38px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        
        /* Target the outer stTextInput container */
        div[data-testid="stForm"] div[data-testid="stTextInput"] {
            border: none !important;
            background-color: transparent !important;
            box-shadow: none !important;
            padding: 0px !important;
            margin: 0px !important;
            height: 38px !important;
            min-height: 38px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        
        /* Hide labels in text inputs to prevent vertical offsets */
        div[data-testid="stForm"] div[data-testid="stTextInput"] label {
            display: none !important;
        }
        
        /* Reset the Emotion cache wrapper to be transparent and borderless */
        div[data-testid="stForm"] div[data-testid="stTextInput"] div[data-testid="stTextInputRootElement"] > div {
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }
        
        /* Make the inner Styletron parent container transparent */
        div[data-testid="stForm"] div[data-testid="stTextInput"] div[data-testid="stTextInputRootElement"] > div > div > div {
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }
        
        /* Style the actual input box (grandparent of the input element) for enabled inputs */
        div[data-testid="stForm"] div[data-testid="stTextInput"]:not(:has(input:disabled)) div[data-testid="stTextInputRootElement"] > div > div {
            border: 1px solid #cccccc !important;
            border-radius: 6px !important;
            height: 38px !important;
            background-color: #ffffff !important;
            box-shadow: none !important;
        }
        
        /* Style the input box container (grandparent of the input element) for disabled inputs */
        div[data-testid="stForm"] div[data-testid="stTextInput"]:has(input:disabled) div[data-testid="stTextInputRootElement"] > div > div {
            background-color: #f0f0f0 !important;
            border: 1px solid #cccccc !important;
            border-radius: 6px !important;
            height: 38px !important;
            box-shadow: none !important;
        }
        
        /* Style the actual input tag */
        div[data-testid="stForm"] div[data-testid="stTextInput"] input {
            border: none !important;
            background-color: transparent !important;
            color: #0f172a !important;
            padding: 0px 8px !important;
            height: 36px !important;
            font-size: 0.85em !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            text-align: center !important;
        }
        
        /* Disabled input text styling */
        div[data-testid="stForm"] div[data-testid="stTextInput"] input:disabled {
            background-color: transparent !important;
            color: #888888 !important;
            font-weight: bold !important;
            -webkit-text-fill-color: #888888 !important;
        }

        /* Target Streamlit form submit button elements */
        div[data-testid="stFormSubmitButton"] {
            margin: 0px !important;
            padding: 0px !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            height: 38px !important;
            min-height: 38px !important;
            width: 100% !important;
        }
        
        /* Target Streamlit form submit buttons directly */
        div[data-testid="stFormSubmitButton"] button {
            height: 38px !important;
            min-height: 38px !important;
            width: 100% !important;
            margin: 0px !important;
            padding: 0px 4px !important;
            box-sizing: border-box !important;
        }

        /* Save button: Blue button */
        div[data-testid="stForm"] [data-testid="stHorizontalBlock"] > div:nth-child(13) button,
        div[data-testid="stForm"] [data-testid="stHorizontalBlock"] > div:nth-last-child(3) button {
            background-color: #3b82f6 !important;
            color: #ffffff !important;
            border: 1px solid #3b82f6 !important;
            border-radius: 6px !important;
            height: 38px !important;
            min-height: 38px !important;
            padding: 0px 4px !important;
            font-size: 0.85em !important;
            font-weight: 600 !important;
            width: 100% !important;
            transition: all 0.2s ease-in-out !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
        }
        div[data-testid="stForm"] [data-testid="stHorizontalBlock"] > div:nth-child(13) button:hover,
        div[data-testid="stForm"] [data-testid="stHorizontalBlock"] > div:nth-last-child(3) button:hover {
            background-color: #2563eb !important;
            border-color: #2563eb !important;
        }
        div[data-testid="stForm"] [data-testid="stHorizontalBlock"] > div:nth-child(13) button *,
        div[data-testid="stForm"] [data-testid="stHorizontalBlock"] > div:nth-child(13) button span,
        div[data-testid="stForm"] [data-testid="stHorizontalBlock"] > div:nth-child(13) button p,
        div[data-testid="stForm"] [data-testid="stHorizontalBlock"] > div:nth-last-child(3) button *,
        div[data-testid="stForm"] [data-testid="stHorizontalBlock"] > div:nth-last-child(3) button span,
        div[data-testid="stForm"] [data-testid="stHorizontalBlock"] > div:nth-last-child(3) button p {
            color: #ffffff !important;
            font-weight: 600 !important;
        }

        /* Approve button: Green button */
        div[data-testid="stForm"] [data-testid="stHorizontalBlock"] > div:nth-child(14) button,
        div[data-testid="stForm"] [data-testid="stHorizontalBlock"] > div:nth-last-child(2) button {
            background-color: #22c55e !important;
            color: #ffffff !important;
            border: 1px solid #22c55e !important;
            border-radius: 6px !important;
            height: 38px !important;
            min-height: 38px !important;
            padding: 0px 4px !important;
            font-size: 0.85em !important;
            font-weight: 600 !important;
            width: 100% !important;
            transition: all 0.2s ease-in-out !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
        }
        div[data-testid="stForm"] [data-testid="stHorizontalBlock"] > div:nth-child(14) button:hover,
        div[data-testid="stForm"] [data-testid="stHorizontalBlock"] > div:nth-last-child(2) button:hover {
            background-color: #16a34a !important;
            border-color: #16a34a !important;
        }
        div[data-testid="stForm"] [data-testid="stHorizontalBlock"] > div:nth-child(14) button *,
        div[data-testid="stForm"] [data-testid="stHorizontalBlock"] > div:nth-child(14) button span,
        div[data-testid="stForm"] [data-testid="stHorizontalBlock"] > div:nth-child(14) button p,
        div[data-testid="stForm"] [data-testid="stHorizontalBlock"] > div:nth-last-child(2) button *,
        div[data-testid="stForm"] [data-testid="stHorizontalBlock"] > div:nth-last-child(2) button span,
        div[data-testid="stForm"] [data-testid="stHorizontalBlock"] > div:nth-last-child(2) button p {
            color: #ffffff !important;
            font-weight: 600 !important;
        }

        /* Reject button: Red button */
        div[data-testid="stForm"] [data-testid="stHorizontalBlock"] > div:nth-child(15) button,
        div[data-testid="stForm"] [data-testid="stHorizontalBlock"] > div:nth-last-child(1) button {
            background-color: #ef4444 !important;
            color: #ffffff !important;
            border: 1px solid #ef4444 !important;
            border-radius: 6px !important;
            height: 38px !important;
            min-height: 38px !important;
            padding: 0px 4px !important;
            font-size: 0.85em !important;
            font-weight: 600 !important;
            width: 100% !important;
            transition: all 0.2s ease-in-out !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
        }
        div[data-testid="stForm"] [data-testid="stHorizontalBlock"] > div:nth-child(15) button:hover,
        div[data-testid="stForm"] [data-testid="stHorizontalBlock"] > div:nth-last-child(1) button:hover {
            background-color: #dc2626 !important;
            border-color: #dc2626 !important;
        }
        div[data-testid="stForm"] [data-testid="stHorizontalBlock"] > div:nth-child(15) button *,
        div[data-testid="stForm"] [data-testid="stHorizontalBlock"] > div:nth-child(15) button span,
        div[data-testid="stForm"] [data-testid="stHorizontalBlock"] > div:nth-child(15) button p,
        div[data-testid="stForm"] [data-testid="stHorizontalBlock"] > div:nth-last-child(1) button *,
        div[data-testid="stForm"] [data-testid="stHorizontalBlock"] > div:nth-last-child(1) button span,
        div[data-testid="stForm"] [data-testid="stHorizontalBlock"] > div:nth-last-child(1) button p {
            color: #ffffff !important;
            font-weight: 600 !important;
        }
        </style>

        """,
        unsafe_allow_html=True
    )
    
    # Wrap headers and row forms in the horizontally scrollable container
    st.markdown("<div class='admin-table-container'>", unsafe_allow_html=True)

    # Display inline table header for inputs
    header_cols = st.columns([1.5, 1.0, 1.0, 1.0, 0.8, 1.0, 1.0, 0.8, 1.0, 0.8, 0.8, 1.0, 1.0, 1.3, 1.0])
    headers = ["VIN", "Make", "Model", "Trim", "Year", "Color", "Body Type", "Cyl", "Origin", "Pass", "Wt", "Specs", "Save", "Approve", "Reject"]
    for col, h in zip(header_cols, headers):
        col.markdown(f"**{h}**")
        
    # Render each pending VIN in its own row (form)
    for idx, item in df_page.iterrows():
        vin = str(item["vin"]).strip().upper()
        
        # We wrap each row in a separate form with the VIN as part of the key
        with st.form(key=f"verify_form_{vin}_{idx}"):
            cols = st.columns([1.5, 1.0, 1.0, 1.0, 0.8, 1.0, 1.0, 0.8, 1.0, 0.8, 0.8, 1.0, 1.0, 1.3, 1.0])
            
            with cols[0]:
                st.text_input("VIN", value=vin, disabled=True, label_visibility="collapsed", key=f"vin_val_{vin}_{idx}")
            with cols[1]:
                make = st.text_input("Make", value=format_csv_value(item.get("make")), label_visibility="collapsed", key=f"make_val_{vin}_{idx}")
            with cols[2]:
                model = st.text_input("Model", value=format_csv_value(item.get("model")), label_visibility="collapsed", key=f"model_val_{vin}_{idx}")
            with cols[3]:
                trim = st.text_input("Trim", value=format_csv_value(item.get("trim")), label_visibility="collapsed", key=f"trim_val_{vin}_{idx}")
            with cols[4]:
                year = st.text_input("Year", value=format_csv_value(item.get("year")), label_visibility="collapsed", key=f"year_val_{vin}_{idx}")
            with cols[5]:
                color = st.text_input("Color", value=format_csv_value(item.get("color")), label_visibility="collapsed", key=f"color_val_{vin}_{idx}")
            with cols[6]:
                body_type = st.text_input("Body", value=format_csv_value(item.get("body_type")), label_visibility="collapsed", key=f"body_val_{vin}_{idx}")
            with cols[7]:
                cylinders = st.text_input("Cyl", value=format_csv_value(item.get("cylinders")), label_visibility="collapsed", key=f"cyl_val_{vin}_{idx}")
            with cols[8]:
                origin = st.text_input("Origin", value=format_csv_value(item.get("origin")), label_visibility="collapsed", key=f"origin_val_{vin}_{idx}")
            with cols[9]:
                no_of_passengers = st.text_input("Pass", value=format_csv_value(item.get("no_of_passengers")), label_visibility="collapsed", key=f"pass_val_{vin}_{idx}")
            with cols[10]:
                weight = st.text_input("Weight", value=format_csv_value(item.get("weight")), label_visibility="collapsed", key=f"weight_val_{vin}_{idx}")
            with cols[11]:
                regional_spec = st.text_input("Specs", value=format_csv_value(item.get("regional_spec")), label_visibility="collapsed", key=f"specs_val_{vin}_{idx}")
                
            with cols[12]:
                btn_save = st.form_submit_button("Save", use_container_width=True)
            with cols[13]:
                btn_approve = st.form_submit_button("Approve", use_container_width=True)
            with cols[14]:
                btn_reject = st.form_submit_button("Reject", use_container_width=True)
                
            if btn_save:
                updated_attrs = {
                    "make": make.strip().upper(),
                    "model": model.strip().upper(),
                    "trim": trim.strip().upper(),
                    "body_type": body_type.strip().upper(),
                    "year": year.strip(),
                    "cylinders": cylinders.strip(),
                    "color": color.strip().upper(),
                    "origin": origin.strip(),
                    "weight": weight.strip(),
                    "no_of_passengers": no_of_passengers.strip(),
                    "regional_spec": regional_spec.strip().upper()
                }
                if update_pending_prediction(vin, updated_attrs):
                    st.success(f"💾 **Saved:** VIN `{vin}` has been updated in the pending queue.")
                    time.sleep(1.5)
                    st.rerun()
                else:
                    st.error(f"Failed to save changes for {vin}.")
                    
            elif btn_approve:
                updated_attrs = {
                    "make": make.strip().upper(),
                    "model": model.strip().upper(),
                    "trim": trim.strip().upper(),
                    "body_type": body_type.strip().upper(),
                    "year": year.strip(),
                    "cylinders": cylinders.strip(),
                    "color": color.strip().upper(),
                    "origin": origin.strip(),
                    "weight": weight.strip(),
                    "no_of_passengers": no_of_passengers.strip(),
                    "regional_spec": regional_spec.strip().upper()
                }
                if approve_pending_prediction(vin, updated_attrs):
                    st.success(f" **Approved:** VIN `{vin}` has been verified and added to the master database.")
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error(f"Failed to approve {vin}.")
                    
            elif btn_reject:
                if reject_pending_prediction(vin):
                    st.warning(f" **Rejected:** VIN `{vin}` has been discarded and removed from the pending queue.")
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error(f"Failed to reject {vin}.")

    st.markdown("</div>", unsafe_allow_html=True)

@st.cache_resource(ttl=30)
def check_db_connection_cached() -> tuple:
    try:
        from database.connection import get_db_connection
        conn = get_db_connection()
        conn.close()
        return True, None
    except Exception as e:
        return False, str(e)

# Check database connection status
db_connected, db_error = check_db_connection_cached()

# Render Connection Status in Sidebar Header
st.sidebar.subheader("Database Connection")
if db_connected:
    st.sidebar.success("● Connected to PostgreSQL")
else:
    st.sidebar.error("● Disconnected")
    st.sidebar.caption(f"Error: {db_error}")
st.sidebar.markdown("---")

# Streamlit Navigation Sidebar
app_mode = st.sidebar.selectbox("Navigation", ["VIN Decoder", "Admin Dashboard"])

if app_mode == "Admin Dashboard":
    render_admin_panel()
    st.stop()

st.title("VIN Decoder")
st.write("Decode and get valuation for the VIN (09, 10, 11, 12, 13, 17 characters) | v3.0")

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

    Natively caches short-VIN predict modules using fully qualified package names
    so they are kept in memory (avoiding extremely slow dynamic file re-reading and
    re-unpickling on subsequent requests).
    """
    if directory_name == "chat_cat(2.0)":
        dir_path = project_root / directory_name
        
        if str(dir_path) in sys.path:
            sys.path.remove(str(dir_path))
        sys.path.insert(0, str(dir_path))
        
        # Check if predict is already in cache for this directory
        if "predict" in sys.modules:
            cached_mod = sys.modules["predict"]
            if hasattr(cached_mod, "__file__") and directory_name in str(cached_mod.__file__):
                return cached_mod
            else:
                del sys.modules["predict"]

        try:
            predict_module = importlib.import_module("predict")
        finally:
            if str(dir_path) in sys.path:
                sys.path.remove(str(dir_path))
        return predict_module
    else:
        # For short VINs, import using their fully qualified package names (which are unique and cached by Python)
        dir_path = project_root / directory_name
        if str(dir_path) not in sys.path:
            sys.path.append(str(dir_path))
            
        return importlib.import_module(f"{directory_name}.predict")


def decode_any_vin(vin_str: str) -> tuple:
    """Standardizes routing logic for 9, 10, 11, 12, 13, and 17 character VIN/Chassis numbers."""
    normalized = re.sub(r"[^A-Z0-9]", "", vin_str.upper())
    length = len(normalized)

    # 1. Existing VIN lookup in PostgreSQL master_data
    try:
        matched_row = lookup_vin_in_master_dataset(normalized)
        if matched_row:
            res = get_verified_vin_dict(matched_row)
            log_prediction(normalized, length, "master_data", res)
            return res, length
    except Exception as e:
        print(f"Error checking master dataset: {e}")

    # 2. Run decoders if not found in master dataset
    res = None
    if length == 17:
        mod = load_predict_module("chat_cat(2.0)")
        res = mod.decode_vin(
            normalized, model_dir=str(project_root / "chat_cat(2.0)" / "models")
        )
    elif length == 13:
        mod = load_predict_module("chat_cat_short_vin_13")
        res = mod.predict_vehicle(
            normalized,
            model_dir=str(project_root / "chat_cat_short_vin_13" / "models"),
        )
    elif length == 12:
        mod = load_predict_module("chat_cat_short_vin_12")
        res = mod.predict_vehicle(
            normalized,
            model_dir=str(project_root / "chat_cat_short_vin_12" / "models"),
        )
    elif length == 11:
        mod = load_predict_module("chat_cat_short_vin_11")
        res = mod.predict_vehicle(
            normalized,
            model_dir=str(project_root / "chat_cat_short_vin_11" / "models"),
        )
    elif length == 10:
        mod = load_predict_module("chat_cat_short_vin")
        res = mod.predict_vehicle(
            normalized,
            model_dir=str(project_root / "chat_cat_short_vin" / "models"),
        )
    elif length == 9:
        mod = load_predict_module("chat_cat_short_vin_09")
        res = mod.predict_vehicle(
            normalized,
            model_dir=str(project_root / "chat_cat_short_vin_09" / "models"),
        )
    else:
        raise ValueError(
            f"Invalid length: {length} characters. Must be 9, 10, 11, 12, 13, or 17 characters."
        )

    # 3. Log all predictions & save unseen predictions to pending queue
    if res is not None:
        normalized_res = normalize_result_keys(res)
        log_prediction(normalized, length, "decoder", normalized_res)
        save_pending_prediction(normalized, length, normalized_res)

    return res, length


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
        alt_path = project_root / "Valuation" / "data" / "drivearabia_car_depriciation_valuation" / "car_dep_2.csv"
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
