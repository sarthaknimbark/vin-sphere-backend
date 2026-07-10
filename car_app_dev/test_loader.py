from evaluation.core.loader import DataLoader
from evaluation.core.validator import DataValidator

loader = DataLoader()

predictions = loader.load_predictions(
    "experiment001/experiment001_predictions.csv"
)

split = loader.load_split_analysis(
    "experiment001/split_analysis.csv"
)

DataValidator.validate_predictions(predictions)
DataValidator.validate_split(split)

print("Everything passed.")