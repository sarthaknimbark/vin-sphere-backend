import os
import sys
from pathlib import Path

_parent = Path(__file__).resolve().parent.parent
if str(_parent) not in sys.path:
    sys.path.append(str(_parent))

import joblib
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

from chat_cat_short_vin_13.feature_engineering import (
    CHASSIS_LENGTH,
    FeatureEngineeringState,
    extract_features,
    normalize_chassis,
)


class VINDecoder:
    def __init__(self, model_dir: str = "chat_cat_short_vin_13/models") -> None:
        self.model_dir = model_dir
        self.models: Dict[str, Any] = {}
        self.ordinal_encoders: Dict[str, Any] = {}
        self.target_encoders: Dict[str, Any] = {}
        self.similarity_engines: Dict[str, Any] = {}
        self.prefix_stats: Dict[int, Dict[str, Dict[str, Any]]] = {}
        self.fe_state: Optional[FeatureEngineeringState] = None
        self.lookup_df: Optional[pd.DataFrame] = None

        prefix_stats_path = os.path.join(model_dir, "prefix_statistics.pkl")
        encoders_path = os.path.join(model_dir, "encoders.pkl")
        fe_path = os.path.join(model_dir, "feature_engineering.pkl")

        if os.path.exists(prefix_stats_path):
            self.prefix_stats = joblib.load(prefix_stats_path)
        if os.path.exists(fe_path):
            self.fe_state = FeatureEngineeringState.load(fe_path)
        if os.path.exists(encoders_path):
            encoders_map = joblib.load(encoders_path)
            self.ordinal_encoders = encoders_map.get("ordinal_encoders", {})
            self.target_encoders = encoders_map.get("target_encoders", {})
            self.similarity_engines = encoders_map.get("similarity_engines", {})

            targets = [
                "make", "model", "trim", "body_type", "year", "origin",
                "regional_specs", "cylinders", "no_of_passengers", "color", "weight",
            ]
            for target in targets:
                safe_name = target.replace(" ", "_")
                model_path = os.path.join(model_dir, f"{safe_name}_model.pkl")
                if os.path.exists(model_path):
                    self.models[target] = joblib.load(model_path)
        
        # Load lookup dataframe
        self.lookup_df = self._load_lookup_df()

    def _load_lookup_df(self) -> Optional[pd.DataFrame]:
        possible_paths = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "Data", "final_clean_13.csv"),
            os.path.join("Data", "final_clean_13.csv"),
            "final_clean_13.csv"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    df['chassisNumber_clean'] = df['chassisNumber'].astype(str).str.upper().str.strip()
                    return df
                except Exception:
                    pass
        return None


    def _encode_frame(self, frame: pd.DataFrame, target: str) -> pd.DataFrame:
        encoded = frame.copy()
        cat_cols = [
            col for col in frame.columns
            if frame[col].dtype == object or isinstance(frame[col].iloc[0], str)
        ]
        for col in cat_cols:
            encoded[col] = encoded[col].astype(str)
        if cat_cols:
            encoded[cat_cols] = self.ordinal_encoders[target].transform(encoded[cat_cols])
        return encoded

    def _predict_classification(
        self,
        target: str,
        normalized: str,
        base_features: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, float, Optional[np.ndarray]]:
        features = base_features.copy()
        if context:
            for key, value in context.items():
                features[key] = value
        sim_features = self.similarity_engines[target].transform(pd.Series([normalized]))
        combined = pd.concat([features.reset_index(drop=True), sim_features.reset_index(drop=True)], axis=1)
        encoded = self._encode_frame(combined, target)
        model = self.models[target]
        pred_idx = model.predict(encoded)[0]
        if isinstance(pred_idx, (np.ndarray, list)):
            pred_idx = pred_idx[0]
        pred_val = self.target_encoders[target].inverse_transform(pred_idx)
        confidence = 1.0
        probs = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(encoded)[0]
            confidence = float(np.max(probs))
        return pred_val, confidence, probs

    def predict(self, chassis_number: str) -> dict:
        normalized = normalize_chassis(chassis_number)
        if not normalized or len(normalized) != CHASSIS_LENGTH:
            raise ValueError(f"Chassis number must be exactly {CHASSIS_LENGTH} characters.")
            
        # Initialize lookup_df dynamically if not present (handles unpickled instances)
        if not hasattr(self, 'lookup_df') or self.lookup_df is None:
            self.lookup_df = self._load_lookup_df()
            
        if self.lookup_df is not None:
            match_rows = self.lookup_df[self.lookup_df['chassisNumber_clean'] == normalized]
            if not match_rows.empty:
                row = match_rows.iloc[0]
                
                def get_val(col_name, default="UNKNOWN"):
                    val = row.get(col_name)
                    if pd.isna(val) or val is None:
                        return default
                    return val
                
                # Format year
                year_val = get_val("year")
                try:
                    year_val = int(float(year_val))
                except (ValueError, TypeError):
                    pass
                
                # Format weight
                weight_val = get_val("weightInKg", 0.0)
                try:
                    weight_val = float(weight_val)
                except (ValueError, TypeError):
                    weight_val = 0.0
                    
                # Format cylinders
                cyl_val = get_val("cylinders", "UNKNOWN")
                try:
                    cyl_val = str(int(float(cyl_val)))
                except (ValueError, TypeError):
                    cyl_val = str(cyl_val)
                    
                # Format noOfPassengers
                pax_val = get_val("noOfPassengers", "UNKNOWN")
                try:
                    pax_val = str(int(float(pax_val)))
                except (ValueError, TypeError):
                    pax_val = str(pax_val)
                
                output = {
                    "make": str(get_val("make")).upper().strip(),
                    "model": str(get_val("model")).upper().strip(),
                    "trim": str(get_val("trim")).upper().strip(),
                    "year": year_val,
                    "body_type": str(get_val("bodyType")).upper().strip(),
                    "origin": str(get_val("origin")).upper().strip(),
                    "regional_specs": str(get_val("regionalSpec")).upper().strip(),
                    "cylinders": cyl_val,
                    "no_of_passengers": pax_val,
                    "color": str(get_val("color")).upper().strip(),
                    "weight": weight_val,
                    "confidence": {
                        "make": 1.0,
                        "model": 1.0,
                        "year": 1.0,
                        "trim": 1.0,
                        "body_type": 1.0,
                        "origin": 1.0,
                        "regional_specs": 1.0,
                        "cylinders": 1.0,
                        "no_of_passengers": 1.0,
                        "color": 1.0,
                        "weight": 1.0
                    }
                }
                return output


        base_features = extract_features(
            pd.Series([normalized]),
            prefix_stats=self.prefix_stats,
            fe_state=self.fe_state,
        )
        predictions: Dict[str, Any] = {}
        confidences: Dict[str, float] = {}
        top_probs: Dict[str, List[Dict[str, Any]]] = {}

        make_val, confidences["make"], make_probs = self._predict_classification("make", normalized, base_features)
        predictions["make"] = str(make_val)
        if make_probs is not None:
            top_probs["make"] = self._format_top_k("make", make_probs, 3)

        model_val, confidences["model"], model_probs = self._predict_classification(
            "model", normalized, base_features, {"make": predictions["make"]}
        )
        predictions["model"] = str(model_val)
        if model_probs is not None:
            top_probs["model"] = self._format_top_k("model", model_probs, 3)

        year_val, confidences["year"], year_probs = self._predict_classification(
            "year", normalized, base_features, {"make": predictions["make"], "model": predictions["model"]}
        )
        try:
            predictions["year"] = int(float(year_val))
        except (ValueError, TypeError):
            predictions["year"] = str(year_val)
        if year_probs is not None:
            top_probs["year"] = self._format_top_k("year", year_probs, 3)

        trim_val, confidences["trim"], _ = self._predict_classification(
            "trim",
            normalized,
            base_features,
            {"make": predictions["make"], "model": predictions["model"], "year": str(predictions["year"])},
        )
        predictions["trim"] = str(trim_val)

        body_val, confidences["body_type"], _ = self._predict_classification(
            "body_type", normalized, base_features, {"make": predictions["make"], "model": predictions["model"]}
        )
        predictions["body_type"] = str(body_val)

        origin_val, confidences["origin"], _ = self._predict_classification(
            "origin", normalized, base_features, {"make": predictions["make"], "model": predictions["model"]}
        )
        predictions["origin"] = str(origin_val)

        rs_val, confidences["regional_specs"], _ = self._predict_classification(
            "regional_specs", normalized, base_features, {"make": predictions["make"], "model": predictions["model"]}
        )
        predictions["regional_specs"] = str(rs_val)

        cyl_val, confidences["cylinders"], _ = self._predict_classification(
            "cylinders",
            normalized,
            base_features,
            {"make": predictions["make"], "model": predictions["model"], "body_type": predictions["body_type"]},
        )
        try:
            predictions["cylinders"] = str(int(float(str(cyl_val))))
        except (ValueError, TypeError):
            predictions["cylinders"] = str(cyl_val)

        pax_val, confidences["no_of_passengers"], _ = self._predict_classification(
            "no_of_passengers",
            normalized,
            base_features,
            {"make": predictions["make"], "model": predictions["model"], "body_type": predictions["body_type"]},
        )
        try:
            predictions["no_of_passengers"] = str(int(float(str(pax_val))))
        except (ValueError, TypeError):
            predictions["no_of_passengers"] = str(pax_val)

        color_val, confidences["color"], _ = self._predict_classification("color", normalized, base_features)
        predictions["color"] = str(color_val)

        weight_features = base_features.copy()
        weight_sim = self.similarity_engines["weight"].transform(pd.Series([normalized]))
        weight_all = pd.concat([weight_features.reset_index(drop=True), weight_sim.reset_index(drop=True)], axis=1)
        weight_enc = self._encode_frame(weight_all, "weight")
        weight_pred = self.models["weight"].predict(weight_enc)[0]
        if isinstance(weight_pred, (np.ndarray, list)):
            weight_pred = weight_pred[0]
        predictions["weight"] = float(np.round(weight_pred, 2))
        confidences["weight"] = 1.0

        nearest_matches = []
        sim_engine = self.similarity_engines.get("make")
        if sim_engine is not None:
            for chassis, distance, attrs in sim_engine.find_nearest_neighbors(normalized, top_n=5):
                nearest_matches.append({"vin": chassis, "distance": round(float(distance), 4), **attrs})

        return {
            "make": predictions["make"],
            "model": predictions["model"],
            "trim": predictions["trim"],
            "year": predictions["year"],
            "body_type": predictions["body_type"],
            "origin": predictions["origin"],
            "regional_specs": predictions["regional_specs"],
            "cylinders": predictions.get("cylinders", "UNKNOWN"),
            "no_of_passengers": predictions.get("no_of_passengers", "UNKNOWN"),
            "color": predictions["color"],
            "weight": predictions["weight"],
            "confidence": {k: round(v, 4) for k, v in confidences.items()},
            "confidence_scores": {k: round(v, 4) for k, v in confidences.items()},
            "top_alternatives": top_probs,
            "nearest_vin_matches": nearest_matches,
        }

    def _format_top_k(self, target: str, probs: np.ndarray, k: int) -> List[Dict[str, Any]]:
        encoder = self.target_encoders[target]
        top_indices = np.argsort(probs)[::-1][:k]
        results = []
        for idx in top_indices:
            label = encoder.inverse_transform(int(idx))
            results.append({"label": str(label), "probability": round(float(probs[idx]), 4)})
        return results
