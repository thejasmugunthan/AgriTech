
"""
Save the selected trained crop recommendation pipeline.
"""

import logging

import joblib

from src.crop_recommendation.config import PROJECT_ROOT


MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"

SOURCE_MODEL = MODEL_DIR / "best_crop_model.joblib"
FINAL_MODEL = MODEL_DIR / "crop_recommendation_pipeline.joblib"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "save_model.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def save_pipeline() -> None:
    """Load and save the complete fitted pipeline."""

    if not SOURCE_MODEL.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {SOURCE_MODEL}. "
            "Run train.py first."
        )

    # Load the previously trained complete pipeline.
    pipeline = joblib.load(SOURCE_MODEL)

    # Confirm the object supports prediction.
    if not hasattr(pipeline, "predict"):
        raise TypeError(
            "The saved object does not support prediction."
        )

    # Save the complete pipeline under its final filename.
    joblib.dump(
        pipeline,
        FINAL_MODEL,
        compress=3,
    )

    logger.info(
        "Complete trained pipeline saved to %s",
        FINAL_MODEL,
    )

    print("Pipeline saved successfully.")
    print(f"Location: {FINAL_MODEL}")


if __name__ == "__main__":
    save_pipeline()
