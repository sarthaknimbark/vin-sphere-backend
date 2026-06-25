import os
import sys
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# Add parent directory to sys.path so we can import from feature_engineering
_parent = Path(__file__).resolve().parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from feature_engineering import normalize_name, resolve_columns

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    data_path = os.path.join("car_app_dev", "17_dataset.csv")
    model_dir = os.path.join(_parent, "models")
    
    if not os.path.exists(data_path):
        # Fallback if executing from root or different folder
        data_path = os.path.join("..", "car_app_dev", "17_dataset.csv")
        model_dir = "models"
        if not os.path.exists(data_path):
            logger.error("Dataset not found at car_app_dev/17_dataset.csv")
            return
            
    logger.info(f"Loading raw dataset from {data_path}...")
    df_raw = pd.read_csv(data_path)
    logger.info(f"Loaded raw data of shape: {df_raw.shape}")
    
    # Resolve columns dynamically
    col_map = resolve_columns(df_raw)
    vin_col = col_map["VIN"]
    make_col = col_map["MAKE"]
    
    # Clean VINs & remove duplicates
    logger.info("Cleaning VINs and filtering to length 17...")
    df_raw[vin_col] = df_raw[vin_col].astype(str).str.upper().str.strip()
    df_clean = df_raw[df_raw[vin_col].str.len() == 17].copy()
    
    logger.info("Removing duplicate VIN records...")
    df_clean = df_clean.drop_duplicates(subset=[vin_col])
    logger.info(f"Data shape after cleaning and deduplication: {df_clean.shape}")
    
    # Clean and normalize Make values
    logger.info("Normalizing MAKE values...")
    normalized_makes = df_clean[make_col].apply(lambda x: normalize_name("MAKE", x))
    
    # Fit and save new LabelEncoder for MAKE
    logger.info("Fitting new LabelEncoder for MAKE...")
    le = LabelEncoder()
    le.fit(normalized_makes)
    
    make_encoder_path = os.path.join(model_dir, "make_encoder.pkl")
    os.makedirs(model_dir, exist_ok=True)
    with open(make_encoder_path, "wb") as f:
        pickle.dump(le, f)
    logger.info(f"Successfully saved make_encoder.pkl to {make_encoder_path} with {len(le.classes_)} classes.")
    
    # Rebuild MAKE fallback model using WMI + VDS hierarchy
    logger.info("Rebuilding WMI+VDS-based lookup mapping for MAKE...")
    
    train_df = pd.DataFrame({
        "WMI": df_clean[vin_col].str[0:3],
        "VDS": df_clean[vin_col].str[3:9],
        "make_clean": normalized_makes
    })
    
    # Global default calculation
    unique_classes, class_counts = np.unique(train_df["make_clean"], return_counts=True)
    best_idx = np.argmax(class_counts)
    global_default = unique_classes[best_idx]
    global_prob = float(class_counts[best_idx] / len(train_df))
    
    target_model = {}
    target_model["_default"] = (global_default, global_prob)
    
    # Define features for each level
    feature_levels = [["WMI", "VDS"], ["WMI"]]
    
    # Compile frequency dictionary for each hierarchy level
    for level_idx, features in enumerate(feature_levels):
        groupby_cols = features.copy()
        counts_df = train_df.groupby(groupby_cols + ["make_clean"]).size().reset_index(name="count")
        
        # Find class with highest count for each feature combination
        best_df = counts_df.sort_values(by=groupby_cols + ["count"], ascending=False)
        best_df = best_df.drop_duplicates(subset=groupby_cols)
        
        # Merge with total counts to compute probability/confidence
        total_counts = train_df.groupby(groupby_cols).size().reset_index(name="total_count")
        best_df = pd.merge(best_df, total_counts, on=groupby_cols)
        best_df["prob"] = best_df["count"] / best_df["total_count"]
        
        # Convert to dictionary with string class keys
        lookup_dict = {}
        for _, row in best_df.iterrows():
            key = tuple(row[f] for f in groupby_cols)
            if len(key) == 1:
                key = key[0]
            
            lookup_dict[key] = (row["make_clean"], float(row["prob"]))
            
        target_model[level_idx] = lookup_dict
    
    # Load the existing fallback model
    fallback_model_path = os.path.join(model_dir, "fallback_model.pkl")
    if os.path.exists(fallback_model_path):
        logger.info(f"Loading existing fallback model from {fallback_model_path}...")
        with open(fallback_model_path, "rb") as f:
            fallback_model = pickle.load(f)
    else:
        logger.warning(f"fallback_model.pkl not found at {fallback_model_path}. Creating a new one.")
        fallback_model = {}
        
    # Update only the MAKE target mapping
    fallback_model["MAKE"] = target_model
    
    # Save the fallback model back
    with open(fallback_model_path, "wb") as f:
        pickle.dump(fallback_model, f)
    logger.info(f"Successfully updated and saved fallback_model.pkl to {fallback_model_path}")
    print("SUCCESS")

if __name__ == "__main__":
    main()
