import hashlib
import logging
import os
import joblib
import re
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

CHASSIS_LENGTH = 13
_parent = Path(__file__).resolve().parent
DEFAULT_STATS_PATH = _parent / "models" / "prefix_statistics.pkl"


def normalize_chassis(chassis: str) -> str:
    if not isinstance(chassis, str):
        logger.warning("Expected string for chassis normalization, got %s", type(chassis))
        return ""
    return chassis.upper().strip().replace(" ", "").replace("-", "")


def load_prefix_stats(path: str | Path = DEFAULT_STATS_PATH) -> Dict[int, Dict[str, Dict[str, Any]]]:
    path = Path(path)
    if path.exists():
        try:
            return joblib.load(path)
        except Exception as exc:
            logger.warning("Could not load prefix statistics from %s: %s", path, exc)
    return {}


class FeatureEngineeringState:
    """Holds fitted TF-IDF and frequency encoders learned on the training split."""

    def __init__(self) -> None:
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.char_frequency_maps: Dict[str, Dict[str, int]] = {}
        self.chassis_length = CHASSIS_LENGTH

    def fit(self, chassis_series: pd.Series) -> "FeatureEngineeringState":
        normalized = chassis_series.astype(str).apply(normalize_chassis)
        self.tfidf_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=64)
        self.tfidf_vectorizer.fit(normalized.tolist())

        for pos in range(1, CHASSIS_LENGTH + 1):
            col = f"char_{pos}"
            counts = normalized.apply(lambda x, i=pos - 1: x[i] if len(x) > i else "?").value_counts()
            self.char_frequency_maps[col] = counts.to_dict()
        return self

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureEngineeringState":
        state = joblib.load(path)
        if not isinstance(state, cls):
            raise TypeError(f"Expected FeatureEngineeringState at {path}")
        return state


def _hash_feature(value: str, bucket: int = 32) -> int:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return int(digest, 16) % bucket


def extract_features(
    chassis_series: pd.Series,
    prefix_stats: Optional[dict] = None,
    fe_state: Optional[FeatureEngineeringState] = None,
) -> pd.DataFrame:
    """Extract 13-character VIN features for training and inference."""
    logger.info("Extracting features from 13-character chassis numbers...")
    normalized_series = chassis_series.astype(str).apply(normalize_chassis)

    if prefix_stats is None:
        prefix_stats = load_prefix_stats()
    if fe_state is None and (_parent / "models" / "feature_engineering.pkl").exists():
        try:
            fe_state = FeatureEngineeringState.load(_parent / "models" / "feature_engineering.pkl")
        except Exception:
            fe_state = None

    features: Dict[str, pd.Series] = {}

    for i in range(CHASSIS_LENGTH):
        features[f"char_{i + 1}"] = normalized_series.apply(lambda x, idx=i: x[idx] if len(x) > idx else "?")

    for i in range(2, 9):
        features[f"prefix_{i}"] = normalized_series.apply(lambda x, n=i: x[:n] if len(x) >= n else x.ljust(n, "?"))

    for i in range(2, 6):
        features[f"suffix_{i}"] = normalized_series.apply(lambda x, n=i: x[-n:] if len(x) >= n else x.rjust(n, "?"))

    features["pattern"] = normalized_series.apply(
        lambda x: "".join("L" if c.isalpha() else "D" if c.isdigit() else "?" for c in x)
    )
    features["pattern_length"] = normalized_series.apply(len)
    features["digit_count"] = normalized_series.apply(lambda x: sum(c.isdigit() for c in x))
    features["letter_count"] = normalized_series.apply(lambda x: sum(c.isalpha() for c in x))
    features["num_letters"] = features["letter_count"]
    features["num_digits"] = features["digit_count"]

    for i in range(CHASSIS_LENGTH):
        features[f"ascii_char_{i + 1}"] = normalized_series.apply(lambda x, idx=i: ord(x[idx]) if len(x) > idx else 0)

    for i in range(CHASSIS_LENGTH - 1):
        features[f"bigram_{i + 1}"] = normalized_series.apply(lambda x, idx=i: x[idx : idx + 2] if len(x) >= idx + 2 else "??")
    for i in range(CHASSIS_LENGTH - 2):
        features[f"trigram_{i + 1}"] = normalized_series.apply(lambda x, idx=i: x[idx : idx + 3] if len(x) >= idx + 3 else "???")
    for i in range(CHASSIS_LENGTH - 3):
        features[f"fourgram_{i + 1}"] = normalized_series.apply(lambda x, idx=i: x[idx : idx + 4] if len(x) >= idx + 4 else "????")

    features["letter_positions"] = normalized_series.apply(
        lambda x: ",".join(str(idx) for idx, char in enumerate(x) if char.isalpha())
    )
    features["digit_positions"] = normalized_series.apply(
        lambda x: ",".join(str(idx) for idx, char in enumerate(x) if char.isdigit())
    )

    def extract_serial_int(val: str) -> int:
        match = re.search(r"(\d+)\D*$", val)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return 0
        return 0

    features["serial_number"] = normalized_series.apply(extract_serial_int)
    features["letters_only"] = normalized_series.apply(lambda x: "".join(re.findall(r"[A-Z]", x)))

    def get_letter_prefix(val: str) -> str:
        match = re.match(r"^[A-Z]+", val)
        if match:
            return match.group(0)
        match_any = re.search(r"[A-Z]+", val)
        return match_any.group(0) if match_any else "?"

    features["letter_prefix"] = normalized_series.apply(get_letter_prefix)

    def get_first_digit_idx(val: str) -> int:
        for idx, char in enumerate(val):
            if char.isdigit():
                return idx
        return -1

    features["first_digit_idx"] = normalized_series.apply(get_first_digit_idx)

    def get_last_letter_idx(val: str) -> int:
        last_idx = -1
        for idx, char in enumerate(val):
            if char.isalpha():
                last_idx = idx
        return last_idx

    features["last_letter_idx"] = normalized_series.apply(get_last_letter_idx)

    for length in [3, 4, 5, 6, 7]:
        def get_pref_purity(val: str, L: int, target: str | None = None) -> float:
            pref = val[:L]
            if prefix_stats and L in prefix_stats and pref in prefix_stats[L]:
                if target:
                    return float(prefix_stats[L][pref].get(f"{target}_confidence", 0.0))
                return float(prefix_stats[L][pref].get("confidence_score", 0.0))
            return 0.0

        features[f"prefix_{length}_purity"] = normalized_series.apply(lambda x, L=length: get_pref_purity(x, L))

        for target in ["make", "model", "trim", "body_type", "origin", "regional_specs"]:
            def get_pref_target_enc(val: str, L: int, tgt: str) -> str:
                pref = val[:L]
                if prefix_stats and L in prefix_stats and pref in prefix_stats[L]:
                    return str(prefix_stats[L][pref].get(f"most_common_{tgt}", "UNKNOWN"))
                return "UNKNOWN"

            features[f"prefix_{length}_target_enc_{target}"] = normalized_series.apply(
                lambda x, L=length, tgt=target: get_pref_target_enc(x, L, tgt)
            )
            features[f"prefix_{length}_purity_{target}"] = normalized_series.apply(
                lambda x, L=length, tgt=target: get_pref_purity(x, L, tgt)
            )

    for i in range(1, 13):
        features[f"hash_bigram_{i}"] = normalized_series.apply(
            lambda x, idx=i - 1: _hash_feature(x[idx : idx + 2] if len(x) >= idx + 2 else "??")
        )
    for i in range(1, 12):
        features[f"hash_trigram_{i}"] = normalized_series.apply(
            lambda x, idx=i - 1: _hash_feature(x[idx : idx + 3] if len(x) >= idx + 3 else "???", bucket=64)
        )

    if fe_state is not None and fe_state.char_frequency_maps:
        for pos in range(1, CHASSIS_LENGTH + 1):
            col = f"char_{pos}"
            freq_map = fe_state.char_frequency_maps.get(col, {})
            features[f"freq_{col}"] = normalized_series.apply(
                lambda x, idx=pos - 1, fm=freq_map: float(fm.get(x[idx] if len(x) > idx else "?", 0))
            )

    df_features = pd.DataFrame(features)

    if fe_state is not None and fe_state.tfidf_vectorizer is not None:
        tfidf_matrix = fe_state.tfidf_vectorizer.transform(normalized_series.tolist()).toarray()
        tfidf_cols = [f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
        df_tfidf = pd.DataFrame(tfidf_matrix, columns=tfidf_cols, index=df_features.index)
        df_features = pd.concat([df_features, df_tfidf], axis=1)

    logger.info("Feature extraction complete. Shape: %s", df_features.shape)
    return df_features


def load_and_preprocess_data(data_path: str) -> pd.DataFrame:
    if not data_path:
        raise ValueError("Data path must be provided.")

    logger.info("Loading dataset from: %s", data_path)
    df = pd.read_csv(data_path)

    rename_dict = {}
    for col in df.columns:
        if col.lower() == "chassisnumber":
            rename_dict[col] = "chassisNumber"
    if rename_dict:
        df = df.rename(columns=rename_dict)

    if "chassisNumber" not in df.columns:
        raise KeyError("Required input column 'chassisNumber' not found in dataset.")

    df["chassisNumber"] = df["chassisNumber"].astype(str).apply(normalize_chassis)
    invalid = df["chassisNumber"].str.len() != CHASSIS_LENGTH
    if invalid.any():
        logger.warning("Dropping %d rows with chassis length != %d", invalid.sum(), CHASSIS_LENGTH)
        df = df[~invalid].copy()
    return df.reset_index(drop=True)
