import os
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

project_root = Path(__file__).resolve().parent.parent
module_dir = project_root / "chat_cat_short_vin_13"
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from chat_cat_short_vin_13.predict import predict_vehicle, explain_prediction

st.set_page_config(page_title="13-Char Short VIN Decoder", layout="centered")
st.title("Short VIN Decoder (13 Characters)")
st.write("Production hierarchical decoder trained with prefix-disjoint validation.")

chassis_input = st.text_input(
    "Enter a 13-Character Short Chassis Number:",
    placeholder="RZN1470016178",
).strip().upper()

normalized = re.sub(r"[^A-Z0-9]", "", chassis_input)
length = len(normalized)
model_dir = str(project_root / "chat_cat_short_vin_13" / "models")

if length > 0 and length != 13:
    st.error(f"Invalid length: {length} characters. Must be exactly 13.")
elif length == 13:
    st.info("Detected: 13-Character Chassis Model")

if st.button("Decode Chassis", use_container_width=True):
    if length != 13:
        st.error("Chassis must be exactly 13 characters.")
    elif not os.path.exists(model_dir) or not os.listdir(model_dir):
        st.warning("13-Character models not found. Run: python chat_cat_short_vin_13/train.py")
    else:
        with st.spinner("Decoding vehicle attributes..."):
            try:
                result = predict_vehicle(normalized, model_dir=model_dir)
                st.session_state["last_result"] = result
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.exception(exc)

if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    st.subheader("Predicted Vehicle Attributes")
    display_attrs = [
        ("Make", "make"),
        ("Model", "model"),
        ("Model Year", "year"),
        ("Trim", "trim"),
        ("Body Type", "body_type"),
        ("Cylinders", "cylinders"),
        ("Origin", "origin"),
        ("No of Passengers", "no_of_passengers"),
        ("Regional Spec", "regional_spec"),
        ("Color", "color"),
        ("Weight", "weight"),
    ]
    conf_dict = result.get("confidence_scores", result.get("attribute_confidences", {}))
    rows = []
    for label, key in display_attrs:
        rows.append({
            "Attribute": label,
            "Predicted Value": str(result.get(key, "UNKNOWN")),
            "Confidence": f"{conf_dict.get(key, 0.0) * 100:.2f}%",
        })
    st.table(pd.DataFrame(rows))

    top_alts = result.get("top_alternatives", {})
    if top_alts:
        st.subheader("Top Alternative Predictions")
        for target, options in top_alts.items():
            if options:
                st.markdown(f"**{target.upper()}**")
                st.write(", ".join([f"{item['label']} ({item['probability']:.1%})" for item in options]))

    nearest = result.get("nearest_vin_matches", [])
    if nearest:
        st.subheader("Nearest VIN Matches")
        st.table(pd.DataFrame(nearest))

    with st.expander("Raw JSON Response"):
        st.json(result)
