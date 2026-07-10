from pathlib import Path

import pandas as pd


class DataLoader:
    """
    Responsible only for loading CSV files.
    """

    @staticmethod
    def load_predictions(path: str | Path) -> pd.DataFrame:

        return pd.read_csv(path)

    @staticmethod
    def load_split_analysis(path: str | Path) -> pd.DataFrame:

        return pd.read_csv(path)