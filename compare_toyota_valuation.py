#!/usr/bin/env python3
"""
Toyota Valuation Comparison Engine.
This script compares ground truth prices in toyota_lookup.csv against the returned 
depreciated values from the ToyotaValuationEngine, using the local VIN decoders.
"""

import os
import sys
import re
import importlib
import warnings
import logging
from pathlib import Path
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

# Add project root to python path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from valuation_engine import ToyotaValuationEngine

def load_predict_module(directory_name: str):
    """Loads predict module dynamically from the appropriate directory."""
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
        # For short VINs, import using their fully qualified package names
        dir_path = project_root / directory_name
        if str(dir_path) not in sys.path:
            sys.path.append(str(dir_path))
            
        return importlib.import_module(f"{directory_name}.predict")

def decode_vin(vin_str: str) -> dict:
    """Decodes a chassis number/VIN of any length using the local decoder pipeline."""
    normalized = re.sub(r"[^A-Z0-9]", "", vin_str.upper())
    length = len(normalized)
    
    if length == 17:
        mod = load_predict_module("chat_cat(2.0)")
        res = mod.decode_vin(normalized, model_dir=str(project_root / "chat_cat(2.0)" / "models"))
    elif length == 13:
        mod = load_predict_module("chat_cat_short_vin_13")
        res = mod.predict_vehicle(normalized, model_dir=str(project_root / "chat_cat_short_vin_13" / "models"))
    elif length == 12:
        mod = load_predict_module("chat_cat_short_vin_12")
        res = mod.predict_vehicle(normalized, model_dir=str(project_root / "chat_cat_short_vin_12" / "models"))
    elif length == 11:
        mod = load_predict_module("chat_cat_short_vin_11")
        res = mod.predict_vehicle(normalized, model_dir=str(project_root / "chat_cat_short_vin_11" / "models"))
    elif length == 10:
        mod = load_predict_module("chat_cat_short_vin")
        res = mod.predict_vehicle(normalized, model_dir=str(project_root / "chat_cat_short_vin" / "models"))
    elif length == 9:
        mod = load_predict_module("chat_cat_short_vin_09")
        res = mod.predict_vehicle(normalized, model_dir=str(project_root / "chat_cat_short_vin_09" / "models"))
    else:
        raise ValueError(f"Unsupported VIN length: {length}")
    
    return res

def main():
    input_path = project_root / "new_valuation" / "toyota_lookup.csv"
    output_path = project_root / "new_valuation" / "toyota_comparison_results.csv"
    
    print(f"Reading input data from: {input_path}")
    df = pd.read_csv(input_path)
    
    # Filter only rows with valid chassisNumber
    df_valid = df[df["chassisNumber"].notna() & (df["chassisNumber"].astype(str).str.strip() != "")]
    total_valid = len(df_valid)
    print(f"Total rows in CSV: {len(df)}")
    print(f"Rows with chassis numbers to test: {total_valid}")
    
    # Instantiate the valuation engine
    engine = ToyotaValuationEngine()
    
    results = []
    processed_count = 0
    success_count = 0
    
    for idx, row in df_valid.iterrows():
        vin = str(row["chassisNumber"]).strip()
        price_str = str(row["price"]).replace(",", "").strip()
        try:
            ground_truth = float(price_str)
        except ValueError:
            continue
        
        try:
            # Decode the VIN to get details (including body_type and regional_spec)
            decoded = decode_vin(vin)
            
            # Map decoded values to payload schema
            payload = {
                "make": decoded.get("make"),
                "model": decoded.get("model"),
                "trim": decoded.get("trim"),
                "model_year": decoded.get("year"),
                "body_type": decoded.get("body_type"),
                "regional_specs": decoded.get("regional_spec"),
            }
            
            # Calculate valuation using engine
            val_res = engine.value(payload)
            dep_val = float(val_res.get("final_value", 0))
            
            # Calculate difference
            diff_abs = dep_val - ground_truth
            diff_pct = (diff_abs / ground_truth * 100) if ground_truth != 0 else 0.0
            
            results.append({
                "Chassis Number": vin,
                "Make": payload["make"],
                "Model": payload["model"],
                "Year": payload["model_year"],
                "Trim": payload["trim"],
                "Regional Specs": payload["regional_specs"],
                "Ground Truth Value": ground_truth,
                "Returned Depreciation Value": dep_val,
                "Difference (Absolute)": diff_abs,
                "Difference (Percentage)": diff_pct
            })
            success_count += 1
        except Exception as e:
            # Silently log errors per row to avoid console clutter
            pass
            
        processed_count += 1
        if processed_count % 500 == 0 or processed_count == total_valid:
            print(f"Processed {processed_count}/{total_valid} chassis numbers...")
            
    if not results:
        print("No comparison results generated.")
        return
        
    # Write output to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults successfully written to: {output_path}")
    
    # Print summary statistics
    print("\n--- Valuation Comparison Summary ---")
    print(f"Successfully processed: {success_count} / {total_valid} rows")
    if success_count > 0:
        mean_abs_diff = results_df["Difference (Absolute)"].abs().mean()
        mean_pct_diff = results_df["Difference (Percentage)"].abs().mean()
        print(f"Mean Absolute Difference: {mean_abs_diff:.2f} AED")
        print(f"Mean Percentage Difference: {mean_pct_diff:.2f}%")

if __name__ == "__main__":
    main()
