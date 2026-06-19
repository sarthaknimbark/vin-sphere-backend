import logging
from typing import Any, Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from rapidfuzz.distance import JaroWinkler
except ImportError:
    JaroWinkler = None


def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def hamming_distance(s1: str, s2: str) -> int:
    min_len = min(len(s1), len(s2))
    dist = sum(c1 != c2 for c1, c2 in zip(s1[:min_len], s2[:min_len]))
    dist += abs(len(s1) - len(s2))
    return dist


def prefix_similarity(s1: str, s2: str) -> int:
    common_len = 0
    for c1, c2 in zip(s1, s2):
        if c1 == c2:
            common_len += 1
        else:
            break
    return common_len


def jaro_winkler_similarity(s1: str, s2: str) -> float:
    if JaroWinkler is not None:
        return float(JaroWinkler.similarity(s1, s2))
    max_len = max(len(s1), len(s2), 1)
    return 1.0 - (levenshtein_distance(s1, s2) / max_len)


class PrefixSimilarityEngine:
    def __init__(self) -> None:
        self.train_chassis: List[str] = []
        self.train_df: pd.DataFrame | None = None
        self.prefix_stats: Dict[int, Dict[str, Dict[str, Any]]] = {}
        self.targets = ["make", "model", "trim", "body_type", "origin", "regional_specs"]
        self.csv_cols = {
            "make": "make",
            "model": "model",
            "trim": "trim",
            "body_type": "bodyType",
            "origin": "origin",
            "regional_specs": "regionalSpec",
        }

    def fit(self, df: pd.DataFrame) -> "PrefixSimilarityEngine":
        self.train_df = df.copy()
        self.train_df["chassisNumber"] = self.train_df["chassisNumber"].astype(str).str.upper().str.strip()
        for _, col in self.csv_cols.items():
            if col in self.train_df.columns:
                self.train_df[col] = self.train_df[col].astype(str).str.upper().str.strip()

        self.train_chassis = self.train_df["chassisNumber"].tolist()
        self.train_records: List[Dict[str, Any]] = []
        self.prefix_records: Dict[int, Dict[str, List[Dict[str, Any]]]] = {}

        for _, row in self.train_df.iterrows():
            rec = {"chassisNumber": row["chassisNumber"]}
            for target, col in self.csv_cols.items():
                rec[target] = row[col] if (col in row and pd.notna(row[col])) else "UNKNOWN"
            try:
                rec["year"] = int(float(row["year"])) if pd.notna(row.get("year")) else 0
            except ValueError:
                rec["year"] = 0
            try:
                rec["weight"] = float(row["weightInKg"]) if pd.notna(row.get("weightInKg")) else 0.0
            except ValueError:
                rec["weight"] = 0.0
            self.train_records.append(rec)

            chassis_num = rec["chassisNumber"]
            for length in [3, 2]:
                pref = chassis_num[:length]
                self.prefix_records.setdefault(length, {}).setdefault(pref, []).append(rec)

        self.prefix_stats = {}
        for length in range(3, 8):
            self.prefix_stats[length] = {}
            prefix_col = f"prefix_{length}"
            temp_df = self.train_df.copy()
            temp_df[prefix_col] = temp_df["chassisNumber"].str[:length]

            for prefix_val, group in temp_df.groupby(prefix_col):
                prefix_entry: Dict[str, Any] = {}
                for target, col in self.csv_cols.items():
                    if col in group.columns:
                        valid_group = group[group[col].notna() & (~group[col].astype(str).isin(["NAN", "NONE", "", "UNKNOWN"]))]
                        if len(valid_group) > 0:
                            counts = valid_group[col].value_counts()
                            prefix_entry[f"most_common_{target}"] = counts.index[0]
                            prefix_entry[f"{target}_confidence"] = float(counts.iloc[0] / len(group))
                            prefix_entry[target] = counts.index[0]
                        else:
                            prefix_entry[f"most_common_{target}"] = "UNKNOWN"
                            prefix_entry[f"{target}_confidence"] = 0.0
                            prefix_entry[target] = "UNKNOWN"
                    else:
                        prefix_entry[f"most_common_{target}"] = "UNKNOWN"
                        prefix_entry[f"{target}_confidence"] = 0.0
                        prefix_entry[target] = "UNKNOWN"
                prefix_entry["confidence_score"] = (
                    prefix_entry.get("make_confidence", 0.0) + prefix_entry.get("model_confidence", 0.0)
                ) / 2.0
                self.prefix_stats[length][prefix_val] = prefix_entry
        return self

    def find_nearest_neighbors(self, chassis: str, top_n: int = 5, max_candidates: int = 120) -> List[Tuple[str, float, Dict[str, Any]]]:
        chassis = chassis.upper().strip()
        candidates: List[Dict[str, Any]] = []
        pref_3 = chassis[:3]
        if 3 in self.prefix_records and pref_3 in self.prefix_records[3]:
            candidates = self.prefix_records[3][pref_3]
        if not candidates:
            pref_2 = chassis[:2]
            if 2 in self.prefix_records and pref_2 in self.prefix_records[2]:
                candidates = self.prefix_records[2][pref_2]
        if not candidates:
            candidates = self.train_records
        if len(candidates) > max_candidates:
            candidates = candidates[:max_candidates]

        results: List[Tuple[str, float, Dict[str, Any]]] = []
        for rec in candidates:
            other_chassis = rec["chassisNumber"]
            lev = levenshtein_distance(chassis, other_chassis)
            ham = hamming_distance(chassis, other_chassis)
            pref = prefix_similarity(chassis, other_chassis)
            jw = jaro_winkler_similarity(chassis, other_chassis)
            distance = lev + 0.5 * ham - 0.2 * pref - 0.5 * jw
            attributes = {"year": rec["year"], "weight": rec["weight"]}
            for target in self.targets:
                attributes[target] = rec[target]
            results.append((other_chassis, distance, attributes))

        return sorted(results, key=lambda x: x[1])[:top_n]

    def get_prefix_intel(self, chassis: str) -> Dict[str, Any]:
        chassis = chassis.upper().strip()
        for length in range(7, 2, -1):
            pref_val = chassis[:length]
            if length in self.prefix_stats and pref_val in self.prefix_stats[length]:
                return self.prefix_stats[length][pref_val]
        return {target: "UNKNOWN" for target in self.targets}

    def transform(self, chassis_series: pd.Series) -> pd.DataFrame:
        logger.info("Generating similarity search and prefix intelligence features...")
        output_features = []
        for chassis in chassis_series:
            chassis_clean = str(chassis).upper().strip()
            row_feat: Dict[str, Any] = {}
            intel = self.get_prefix_intel(chassis_clean)
            for target in self.targets:
                row_feat[f"pref_intel_{target}"] = intel[target]

            neighbors = self.find_nearest_neighbors(chassis_clean, top_n=3)
            for n_idx in range(3):
                if n_idx < len(neighbors):
                    nb_chassis, dist, nb_attrs = neighbors[n_idx]
                    row_feat[f"sim_score_{n_idx + 1}"] = float(dist)
                    row_feat[f"sim_chassis_{n_idx + 1}"] = nb_chassis
                    row_feat[f"sim_jaro_{n_idx + 1}"] = float(jaro_winkler_similarity(chassis_clean, nb_chassis))
                    for target in self.targets:
                        row_feat[f"sim_{target}_{n_idx + 1}"] = nb_attrs[target]
                    row_feat[f"sim_year_{n_idx + 1}"] = str(nb_attrs["year"])
                    row_feat[f"sim_weight_{n_idx + 1}"] = float(nb_attrs["weight"])
                else:
                    row_feat[f"sim_score_{n_idx + 1}"] = 999.0
                    row_feat[f"sim_chassis_{n_idx + 1}"] = "UNKNOWN"
                    row_feat[f"sim_jaro_{n_idx + 1}"] = 0.0
                    for target in self.targets:
                        row_feat[f"sim_{target}_{n_idx + 1}"] = "UNKNOWN"
                    row_feat[f"sim_year_{n_idx + 1}"] = "UNKNOWN"
                    row_feat[f"sim_weight_{n_idx + 1}"] = 0.0
            output_features.append(row_feat)
        return pd.DataFrame(output_features, index=chassis_series.index)
