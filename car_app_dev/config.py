"""
Evaluation Framework Configuration.

Every module imports configuration from here.
"""

# =============================================================================
# Required Columns
# =============================================================================

REQUIRED_PREDICTION_COLUMNS = [
    "chassisNumber",
    "make",
    "model",
    "trim",
    "year",
    "price",
    "predicted_price",
]

REQUIRED_SPLIT_COLUMNS = [
    "chassisNumber",
    "difficulty",
    "difficulty_name",
    "known_make",
    "known_model",
    "known_trim",
    "train_model_count",
    "train_trim_count",
]

# =============================================================================
# Metrics
# =============================================================================

ERROR_THRESHOLDS = [
    5,
    10,
    20,
]

# =============================================================================
# Output
# =============================================================================

OUTPUT_ANALYSIS_FILE = "prediction_analysis.csv"

OUTPUT_METRICS_FILE = "metrics_summary.txt"