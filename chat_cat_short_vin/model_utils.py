import os
import logging
import joblib
import numpy as np
from typing import Any

# Configure logging
logger = logging.getLogger(__name__)

class RobustLabelEncoder:
    """
    A robust label encoder that maps unseen categories during transform or inverse_transform
    to a specified fallback value (like "UNKNOWN").
    """
    def __init__(self, fallback_value='UNKNOWN'):
        self.fallback_value = fallback_value
        self.classes_ = None
        self.class_to_idx = {}
        self.idx_to_class = {}

    def fit(self, y):
        # Flatten target list/series and find unique classes
        y_str = [str(val).upper().strip() if (val is not None and not (isinstance(val, float) and np.isnan(val))) else self.fallback_value for val in y]
        unique_labels = sorted(list(set(y_str)))
        
        # Ensure fallback_value is represented
        if self.fallback_value not in unique_labels:
            unique_labels.append(self.fallback_value)
            
        self.classes_ = np.array(unique_labels)
        self.class_to_idx = {val: idx for idx, val in enumerate(self.classes_)}
        self.idx_to_class = {idx: val for idx, val in enumerate(self.classes_)}
        return self

    def transform(self, y):
        if self.classes_ is None:
            raise ValueError("RobustLabelEncoder must be fitted before transforming.")
        y_str = [str(val).upper().strip() if (val is not None and not (isinstance(val, float) and np.isnan(val))) else self.fallback_value for val in y]
        fallback_idx = self.class_to_idx[self.fallback_value]
        return np.array([self.class_to_idx.get(val, fallback_idx) for val in y_str], dtype=int)

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self, y_idx):
        if self.classes_ is None:
            raise ValueError("RobustLabelEncoder must be fitted before inverse transforming.")
        if np.isscalar(y_idx) or (isinstance(y_idx, np.ndarray) and y_idx.ndim == 0):
            return self.idx_to_class.get(int(y_idx), self.fallback_value)
        y_idx_arr = np.array(y_idx).ravel()
        decoded = [self.idx_to_class.get(int(idx), self.fallback_value) for idx in y_idx_arr]
        return np.array(decoded)

def save_model(model: Any, filepath: str) -> None:
    """
    Saves a trained model or encoder to disk using joblib.
    """
    try:
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        logger.info(f"Saving using joblib to {filepath}...")
        joblib.dump(model, filepath)
        logger.info("Saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save to {filepath}: {e}")
        raise

def load_model(filepath: str) -> Any:
    """
    Loads a saved model or encoder from disk using joblib.
    """
    if not os.path.exists(filepath):
        logger.error(f"File not found at {filepath}")
        raise FileNotFoundError(f"File not found at: {filepath}")
        
    try:
        logger.info(f"Loading from {filepath}...")
        model = joblib.load(filepath)
        logger.info("Loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load from {filepath}: {e}")
        raise

def check_gcc_rules(chassis: str, make: str, rules: dict) -> Any:
    """
    Applies exact pattern-matching rules on the short chassis to identify GCC vs. Non-GCC specification.
    Returns 'GCC', 'NON-GCC', or None if no rule matches.
    """
    if not rules:
        return None
    chassis = str(chassis).upper().strip()
    make = str(make).upper().strip()
    
    # 0. Check manual overrides
    manual_overrides = rules.get("manual_overrides", {})
    if chassis in manual_overrides:
        return manual_overrides[chassis]
    
    # 1. Check exclusive prefixes (longest first)
    for length in [5, 4, 3]:
        pref = chassis[:length]
        if len(pref) == length:
            gcc_list = rules.get(f"gcc_exclusive_prefixes_{length}", [])
            ngcc_list = rules.get(f"non_gcc_exclusive_prefixes_{length}", [])
            if pref in ngcc_list:
                return "NON-GCC"
            if pref in gcc_list:
                return "GCC"
                
    # 2. Check make-specific position-based rules
    make_rules = rules.get("make_specific_position_rules", {}).get(make, {})
    for pos_str, char_rules in make_rules.items():
        try:
            pos_idx = int(pos_str.split("_")[1]) - 1
            if pos_idx < len(chassis):
                char_val = chassis[pos_idx]
                if char_val in char_rules.get("non_gcc_only", []):
                    return "NON-GCC"
                if char_val in char_rules.get("gcc_only", []):
                    return "GCC"
        except Exception:
            pass
            
    return None

