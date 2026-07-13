# run_analysis.py

from evaluation.core.loader import DataLoader
from evaluation.core.validator import DataValidator

loader = DataLoader()

pred_df = loader.load_predictions(
    "experiment001/experiment001_predictions.csv"
)

split_df = loader.load_split_analysis(
    "experiment001/split_analysis.csv"
)

DataValidator.validate_predictions(pred_df)
DataValidator.validate_split(split_df)

print("Validation passed.")