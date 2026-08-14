import os
from pathlib import Path

# Resolve project paths
api_dir = Path(__file__).resolve().parent
PROJECT_ROOT = api_dir.parent

# Database Connection Status Helper
def get_db_connection_safe():
    try:
        from database.connection import get_db_connection
        conn = get_db_connection()
        return conn, None
    except Exception as e:
        return None, str(e)

# File Paths
LOOKUP_DATA_PATH = PROJECT_ROOT / "car_app_dev" / "dev_lookup.csv"
CAR_DEP_CSV_PATH = PROJECT_ROOT / "Valuation" / "data" / "car_dep.csv"

# Model Directories
MODEL_DIRS = {
    17: PROJECT_ROOT / "chat_cat(2.0)" / "models",
    13: PROJECT_ROOT / "chat_cat_short_vin_13" / "models",
    12: PROJECT_ROOT / "chat_cat_short_vin_12" / "models",
    11: PROJECT_ROOT / "chat_cat_short_vin_11" / "models",
    10: PROJECT_ROOT / "chat_cat_short_vin" / "models",
    9: PROJECT_ROOT / "chat_cat_short_vin_09" / "models",
}

# API Version/Title
API_TITLE = "VIN Sphere Decoding & Valuation API"
API_VERSION = "v1"
