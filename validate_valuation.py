#!/usr/bin/env python3
"""
Toyota Valuation Validation Pipeline.
Validates the local Toyota valuation engine against ground truth prices in toyota_lookup.csv,
using decoding and valuation flows loaded directly from main_app/app.py.
"""

import os
import sys
import re
import math
import warnings
import logging
import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np

# Suppress warnings and logger noise
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

# Resolve project paths
project_root = Path(__file__).resolve().parent
main_app_path = project_root / "main_app"
input_path = project_root / "new_valuation" / "toyota_lookup.csv"
output_path = project_root / "new_valuation" / "toyota_valuation_validation_report.csv"

# Mock streamlit before importing main_app/app.py to prevent top-level execution errors
from unittest.mock import MagicMock
mock_st = MagicMock()
mock_st.text_input.return_value = ""
mock_st.button.return_value = False
mock_st.selectbox.return_value = ""
mock_st.checkbox.return_value = False
mock_st.radio.return_value = ""
mock_st.slider.return_value = 0

def dummy_decorator(*args, **kwargs):
    if len(args) == 1 and callable(args[0]):
        return args[0]
    return lambda func: func
mock_st.fragment = dummy_decorator
mock_st.cache_resource = dummy_decorator
mock_st.cache_data = dummy_decorator

sys.modules["streamlit"] = mock_st

# Mock database module to prevent database connection timeouts/errors in non-DB environment
mock_db = MagicMock()
mock_db.lookup_vin_in_master_dataset.return_value = None
mock_db.log_prediction.return_value = None
mock_db.save_pending_prediction.return_value = None
mock_db.safe_read_pending_predictions.return_value = pd.DataFrame()
mock_db.get_approved_predictions_count.return_value = 0
sys.modules["database"] = mock_db

# Load main_app/app.py dynamically to bypass namespace conflicts with root app.py
print("Loading decode and valuation flows from main_app/app.py...")
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(main_app_path))

spec = importlib.util.spec_from_file_location("main_app_app", str(main_app_path / "app.py"))
main_app = importlib.util.module_from_spec(spec)
sys.modules["main_app_app"] = main_app
spec.loader.exec_module(main_app)

decode_any_vin = main_app.decode_any_vin
normalize_result_keys = main_app.normalize_result_keys
get_toyota_engine_valuation = main_app.get_toyota_engine_valuation

def clean_price(val) -> float:
    """Safely convert price field to float, removing commas and whitespace."""
    if pd.isna(val):
        return 0.0
    val_str = str(val).replace(",", "").strip()
    try:
        return float(val_str)
    except ValueError:
        return 0.0

def run_validation():
    print(f"Loading ground truth records from: {input_path}")
    df = pd.read_csv(input_path)
    
    # Filter for valid chassisNumber values
    df_valid = df[df["chassisNumber"].notna() & (df["chassisNumber"].astype(str).str.strip() != "")]
    total_valid = len(df_valid)
    print(f"Total rows in CSV: {len(df)}")
    print(f"Rows with valid chassis number: {total_valid}")
    
    records = []
    processed = 0
    success = 0
    
    for idx, row in df_valid.iterrows():
        vin = str(row["chassisNumber"]).strip()
        ground_truth = clean_price(row["price"])
        
        if ground_truth <= 0:
            continue
            
        try:
            # 1. Decode chassis number
            res, _ = decode_any_vin(vin)
            result = normalize_result_keys(res)
            
            val_res = get_toyota_engine_valuation(result)
            
            if val_res and "final_value" in val_res and "error" not in val_res:
                pred_price = float(val_res["final_value"])
                
                # 3. Calculate errors
                diff_abs = pred_price - ground_truth
                variance_pct = (diff_abs / ground_truth) * 100
                ape = abs(variance_pct)
                
                # Determine status based on ±5% tolerance
                if ape <= 5.0:
                    status = "Within Tolerance"
                elif variance_pct > 5.0:
                    status = "Overvalued"
                else:
                    status = "Undervalued"
                    
                records.append({
                    "VIN / Chassis Number": vin,
                    "Make": result.get("make"),
                    "Model": result.get("model"),
                    "Model Year": result.get("year"),
                    "Trim": result.get("trim"),
                    "Body Type": result.get("body_type"),
                    "Regional Specs": result.get("regional_spec"),
                    "Ground Truth Price": ground_truth,
                    "Predicted Price": pred_price,
                    "Absolute Difference": diff_abs,
                    "Percentage Variance (%)": variance_pct,
                    "Absolute Percentage Error (%)": ape,
                    "Status": status
                })
                success += 1
        except Exception as e:
            if processed < 10:
                print(f"Error processing row {idx} (VIN: {vin}): {e}")
            pass
            
        processed += 1
        if processed % 10 == 0 or processed == total_valid:
            print(f"Processed {processed}/{total_valid} records (Success: {success})...", flush=True)
            
    if not records:
        print("Error: No validation records were successfully processed.")
        return
        
    # Create DataFrame and save CSV report
    report_df = pd.DataFrame(records)
    report_df.to_csv(output_path, index=False)
    print(f"\nDetailed report written to: {output_path}")
    
    # 4. Calculate Overall Metrics
    mae = report_df["Absolute Difference"].abs().mean()
    mape = report_df["Absolute Percentage Error (%)"].mean()
    rmse = math.sqrt((report_df["Absolute Difference"] ** 2).mean())
    avg_var = report_df["Percentage Variance (%)"].mean()
    
    # Calculate percentage accuracies
    def get_acc_stats(df_in, limit):
        count = (df_in["Absolute Percentage Error (%)"] <= limit).sum()
        pct = (count / len(df_in)) * 100
        return count, pct
        
    c_5, p_5 = get_acc_stats(report_df, 5.0)
    c_10, p_10 = get_acc_stats(report_df, 10.0)
    c_15, p_15 = get_acc_stats(report_df, 15.0)
    
    print("\n=============================================")
    print("        VALUATION ENGINE EVALUATION METRICS    ")
    print("=============================================")
    print(f"Total Evaluated Records:       {success}")
    print(f"Mean Absolute Error (MAE):     {mae:.2f} AED")
    print(f"Mean Absolute Pct Error (MAPE):{mape:.2f}%")
    print(f"Root Mean Squared Error (RMSE):{rmse:.2f} AED")
    print(f"Average Percentage Variance:   {avg_var:.2f}%")
    print("---------------------------------------------")
    print(f"Predictions within ±5%:        {c_5} ({p_5:.2f}%)")
    print(f"Predictions within ±10%:       {c_10} ({p_10:.2f}%)")
    print(f"Predictions within ±15%:       {c_15} ({p_15:.2f}%)")
    print("=============================================")

if __name__ == "__main__":
    run_validation()
