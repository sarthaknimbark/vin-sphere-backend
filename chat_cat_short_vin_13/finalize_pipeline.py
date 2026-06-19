"""Rebuild encoders.pkl, prefix_statistics.pkl, and vin_decoder_pipeline.pkl from trained target models."""

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

_parent = Path(__file__).resolve().parent.parent
if str(_parent) not in sys.path:
    sys.path.append(str(_parent))

from chat_cat_short_vin_13.feature_engineering import CHASSIS_LENGTH, FeatureEngineeringState, load_and_preprocess_data
from chat_cat_short_vin_13.model_utils import load_model, save_model
from chat_cat_short_vin_13.prefix_similarity import PrefixSimilarityEngine
from chat_cat_short_vin_13.train import COLUMN_MAPPING, HIERARCHICAL_ORDER
from chat_cat_short_vin_13.vin_decoder import VINDecoder


def main() -> None:
    data_path = _parent / "Data" / "final_clean_13.csv"
    model_dir = Path(__file__).resolve().parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_preprocess_data(str(data_path))
    df["prefix5"] = df["chassisNumber"].str[:5]
    train_prefixes, _ = train_test_split(list(df["prefix5"].unique()), test_size=0.2, random_state=42)
    train_prefixes = set(train_prefixes)
    train_df = df[df["prefix5"].isin(train_prefixes)].copy()

    fe_state = FeatureEngineeringState().fit(train_df["chassisNumber"])
    save_model(fe_state, model_dir / "feature_engineering.pkl")

    base_engine = PrefixSimilarityEngine()
    base_engine.fit(train_df)
    similarity_engines = {target: base_engine for target in HIERARCHICAL_ORDER}

    prefix_stats_unified = {length: base_engine.prefix_stats[length] for length in range(3, 8)}
    save_model(prefix_stats_unified, model_dir / "prefix_statistics.pkl")

    ordinal_encoders = {}
    target_encoders = {}
    for target in HIERARCHICAL_ORDER:
        fe_path = model_dir / f"{target}_fe_encoder.pkl"
        if fe_path.exists():
            ordinal_encoders[target] = load_model(str(fe_path))
        label_path = model_dir / f"{target}_label_encoder.pkl"
        if label_path.exists():
            target_encoders[target] = load_model(str(label_path))

    save_model(
        {
            "ordinal_encoders": ordinal_encoders,
            "target_encoders": target_encoders,
            "similarity_engines": similarity_engines,
        },
        model_dir / "encoders.pkl",
    )

    save_model(
        {
            "chassis_length": CHASSIS_LENGTH,
            "targets": HIERARCHICAL_ORDER,
            "column_mapping": COLUMN_MAPPING,
            "train_prefixes": list(train_prefixes),
            "record_count": len(df),
        },
        model_dir / "vin_metadata.pkl",
    )

    pipeline = VINDecoder(model_dir=str(model_dir))
    save_model(pipeline, model_dir / "vin_decoder_pipeline.pkl")
    print(f"Rebuilt pipeline artifacts in {model_dir}")


if __name__ == "__main__":
    main()
