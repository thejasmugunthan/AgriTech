import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from src.crop_recommendation.config import (
    PROJECT_ROOT,
    TARGET_COLUMN,
)

# File paths
MODEL_PATH = PROJECT_ROOT / "models" / "best_crop_model.joblib"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "predictions"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def load_model():
    """Load the complete trained preprocessing and prediction pipeline."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}. "
            "Run the training module first."
        )

    model = joblib.load(MODEL_PATH)

    logger.info("Trained pipeline loaded successfully.")

    return model


def load_test_data():
    """Load test data and separate features from actual crop labels."""

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_DATA_PATH}. "
            "Run the data-splitting module first."
        )

    dataset = pd.read_csv(TEST_DATA_PATH)

    if TARGET_COLUMN not in dataset.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing from the dataset."
        )

    X_test = dataset.drop(columns=[TARGET_COLUMN])
    y_test = dataset[TARGET_COLUMN]

    logger.info("Test dataset loaded: %s records", len(dataset))

    return X_test, y_test


def test_predictions(model, X_test, y_test):
    """Predict crops and compare predictions with actual labels."""

    predicted_crops = model.predict(X_test)

    accuracy = accuracy_score(y_test, predicted_crops)

    results = pd.DataFrame({
        "Actual_Crop": y_test.to_numpy(),
        "Predicted_Crop": predicted_crops,
    })

    results["Correct_Prediction"] = (
        results["Actual_Crop"] == results["Predicted_Crop"]
    )

    output_path = OUTPUT_DIR / "test_predictions.csv"
    results.to_csv(output_path, index=False)

    print("\n========== TEST PREDICTION RESULTS ==========")
    print(f"Total test records : {len(y_test)}")
    print(f"Correct predictions: {results['Correct_Prediction'].sum()}")
    print(f"Incorrect predictions: {(~results['Correct_Prediction']).sum()}")
    print(f"Test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    print("\nFirst 10 predictions:")
    print(results.head(10).to_string(index=False))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predicted_crops,
            zero_division=0,
        )
    )

    print(f"\nPrediction results saved to: {output_path}")

    logger.info("Test predictions completed.")
    logger.info("Test accuracy: %.4f", accuracy)

    return results, accuracy


def main():
    model = load_model()
    X_test, y_test = load_test_data()

    test_predictions(model, X_test, y_test)


if __name__ == "__main__":
    main()

