
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Raw dataset path
DATASET_PATH = (
    PROJECT_ROOT / "data" / "raw" / "crop_recommendation.csv"
)

# Target column to predict
TARGET_COLUMN = "CROPS"

