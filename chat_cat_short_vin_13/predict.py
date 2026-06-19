import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

_parent = Path(__file__).resolve().parent.parent
if str(_parent) not in sys.path:
    sys.path.append(str(_parent))

from chat_cat_short_vin_13.model_utils import load_model

logger = logging.getLogger(__name__)
_MODEL_CACHE: Dict[str, Any] = {}


def predict_vehicle(chassis_number: str, model_dir: str = "chat_cat_short_vin_13/models") -> Dict[str, Any]:
    """Decode a 13-character short chassis number using the unified VINDecoder pipeline."""
    pipeline_path = os.path.join(model_dir, "vin_decoder_pipeline.pkl")
    if not os.path.exists(pipeline_path):
        pipeline_path = os.path.join(_parent, "chat_cat_short_vin_13", "vin_decoder_pipeline.pkl")

    if os.path.exists(pipeline_path):
        try:
            decoder = load_model(pipeline_path)
            if not hasattr(decoder, "fe_state") or getattr(decoder, "model_dir", "").endswith("_12"):
                raise ValueError("Stale pipeline artifact detected.")
            res = decoder.predict(chassis_number)
        except Exception as exc:
            logger.warning("Failed to load pipeline: %s. Falling back to manual load.", exc)
            from chat_cat_short_vin_13.vin_decoder import VINDecoder
            decoder = VINDecoder(model_dir=model_dir)
            res = decoder.predict(chassis_number)
    else:
        from chat_cat_short_vin_13.vin_decoder import VINDecoder
        decoder = VINDecoder(model_dir=model_dir)
        res = decoder.predict(chassis_number)

    res_conf = res.get("confidence_scores", res.get("confidence", {}))

    def to_str_val(val: Any) -> str:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "UNKNOWN"
        val_str = str(val).strip()
        if val_str.upper() in ["NAN", "NONE", "UNKNOWN", "0", "0.0", ""]:
            return "UNKNOWN"
        try:
            float_val = float(val_str)
            if float_val.is_integer():
                return str(int(float_val))
            return str(round(float_val, 2))
        except ValueError:
            return val_str

    output = {
        "vin": str(chassis_number).upper().strip(),
        "year": to_str_val(res.get("year")),
        "make": to_str_val(res.get("make")),
        "model": to_str_val(res.get("model")),
        "trim": to_str_val(res.get("trim")),
        "body_type": to_str_val(res.get("body_type")),
        "regional_spec": to_str_val(res.get("regional_specs")),
        "cylinders": to_str_val(res.get("cylinders")),
        "origin": to_str_val(res.get("origin")),
        "no_of_passengers": to_str_val(res.get("no_of_passengers")),
        "weight": to_str_val(res.get("weight")),
        "color": to_str_val(res.get("color")),
        "confidence": 0.0,
        "confidence_scores": {
            "make": round(float(res_conf.get("make", 0.0)), 4),
            "model": round(float(res_conf.get("model", 0.0)), 4),
            "year": round(float(res_conf.get("year", 0.0)), 4),
            "trim": round(float(res_conf.get("trim", 0.0)), 4),
            "body_type": round(float(res_conf.get("body_type", 0.0)), 4),
            "origin": round(float(res_conf.get("origin", 0.0)), 4),
            "regional_spec": round(float(res_conf.get("regional_specs", 0.0)), 4),
            "cylinders": round(float(res_conf.get("cylinders", 0.0)), 4),
            "no_of_passengers": round(float(res_conf.get("no_of_passengers", 0.0)), 4),
            "weight": round(float(res_conf.get("weight", 0.0)), 4),
            "color": round(float(res_conf.get("color", 0.0)), 4),
        },
        "attribute_confidences": {
            "make": round(float(res_conf.get("make", 0.0)), 4),
            "model": round(float(res_conf.get("model", 0.0)), 4),
            "year": round(float(res_conf.get("year", 0.0)), 4),
            "trim": round(float(res_conf.get("trim", 0.0)), 4),
            "body_type": round(float(res_conf.get("body_type", 0.0)), 4),
            "origin": round(float(res_conf.get("origin", 0.0)), 4),
            "regional_spec": round(float(res_conf.get("regional_specs", 0.0)), 4),
            "cylinders": round(float(res_conf.get("cylinders", 0.0)), 4),
            "no_of_passengers": round(float(res_conf.get("no_of_passengers", 0.0)), 4),
            "weight": round(float(res_conf.get("weight", 0.0)), 4),
            "color": round(float(res_conf.get("color", 0.0)), 4),
        },
        "top_alternatives": res.get("top_alternatives", {}),
        "nearest_vin_matches": res.get("nearest_vin_matches", []),
    }
    confs = list(output["confidence_scores"].values())
    output["confidence"] = round(float(np.mean(confs)), 4)
    return output


def explain_prediction(chassis_number: str, target: str = "make", model_dir: str = "chat_cat_short_vin_13/models") -> Dict[str, Any]:
    from difflib import SequenceMatcher
    from chat_cat_short_vin_13.feature_engineering import extract_features, normalize_chassis
    from chat_cat_short_vin_13.vin_decoder import VINDecoder

    normalized = normalize_chassis(chassis_number)
    decoder = VINDecoder(model_dir=model_dir)
    sim_engine = decoder.similarity_engines.get("make")
    closest_prefixes = []
    if sim_engine and hasattr(sim_engine, "train_chassis") and sim_engine.train_chassis:
        train_prefixes = sorted({c[:5] for c in sim_engine.train_chassis})
        input_prefix = normalized[:5].upper()
        matches = []
        for train_pref in train_prefixes:
            score = SequenceMatcher(None, input_prefix, train_pref).ratio()
            matches.append({"prefix": train_pref, "similarity": round(score, 4)})
        closest_prefixes = sorted(matches, key=lambda x: x["similarity"], reverse=True)[:5]

    res = decoder.predict(normalized)
    return {
        "closest_prefixes": closest_prefixes,
        "top_alternatives": res.get("top_alternatives", {}).get(target, []),
        "nearest_vin_matches": res.get("nearest_vin_matches", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="13-Character Short Chassis Model Inference Tool")
    parser.add_argument("--chassis", type=str, required=True, help="13-character short chassis number.")
    parser.add_argument("--model_dir", type=str, default="chat_cat_short_vin_13/models")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    try:
        print(json.dumps(predict_vehicle(args.chassis, args.model_dir), indent=4))
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, indent=4))


if __name__ == "__main__":
    main()
