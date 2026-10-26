
"""
Select the best crop recommendation model.

Selection is based primarily on cross-validation accuracy,
with macro F1-score used as a tie-breaker when available.

The selected fitted pipeline is saved as the final candidate model.
"""

import logging
import shutil

import joblib
import pandas as pd

from src.crop_recommendation.config import PROJECT_ROOT


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_DIR = PROJECT_ROOT / "models"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "training"
LOG_DIR = PROJECT_ROOT / "logs"

COMPARISON_PATH = ARTIFACT_DIR / "model_comparison.csv"

# Models saved by train.py.
MODEL_FILES = {
    "Decision Tree": MODEL_DIR / "decision_tree.joblib",
    "Random Forest": MODEL_DIR / "random_forest.joblib",
    "Gradient Boosting": MODEL_DIR / "gradient_boosting.joblib",
}

# Existing training code saves the selected fitted pipeline here.
BEST_CANDIDATE_PATH = MODEL_DIR / "best_crop_model.joblib"

FINAL_MODEL_PATH = MODEL_DIR / "final_crop_model.joblib"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOGGING
# ============================================================

logger = logging.getLogger("crop_best_model_selection")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        LOG_DIR / "model_selection.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


# ============================================================
# 3. LOAD MODEL COMPARISON REPORT
# ============================================================

def load_comparison() -> pd.DataFrame:
    """Load and validate the model comparison report."""

    if not COMPARISON_PATH.is_file():
        raise FileNotFoundError(
            f"Model comparison report not found: "
            f"{COMPARISON_PATH}. Run train.py first."
        )

    comparison = pd.read_csv(COMPARISON_PATH)

    required_columns = {
        "model",
        "cv_accuracy_mean",
    }

    missing_columns = required_columns - set(comparison.columns)

    if missing_columns:
        raise ValueError(
            "Model comparison report is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    comparison = comparison.dropna(
        subset=["cv_accuracy_mean"]
    )

    if comparison.empty:
        raise ValueError(
            "No valid cross-validation scores are available. "
            "Review the training results before selecting a model."
        )

    return comparison


# ============================================================
# 4. SELECT BEST CANDIDATE
# ============================================================

def select_best_candidate(
    comparison: pd.DataFrame,
) -> str:
    """
    Select the model with the highest cross-validation accuracy.

    Macro F1-score is used as a tie-breaker if available.
    """

    sort_columns = ["cv_accuracy_mean"]
    ascending = [False]

    if "f1_macro" in comparison.columns:
        sort_columns.append("f1_macro")
        ascending.append(False)

    ranked = comparison.sort_values(
        by=sort_columns,
        ascending=ascending,
    )

    best_model_name = ranked.iloc[0]["model"]

    logger.info(
        "Best candidate selected using: %s",
        sort_columns,
    )

    logger.info(
        "Selected model: %s",
        best_model_name,
    )

    return best_model_name


# ============================================================
# 5. SAVE SELECTED MODEL
# ============================================================

def save_selected_model(best_model_name: str) -> None:
    """
    Save the selected fitted pipeline under a standard filename.

    The current train.py implementation saves the best fitted
    pipeline as best_crop_model.joblib.

    If you later save each model separately, update this function
    to load the corresponding file from MODEL_FILES.
    """

    if not BEST_CANDIDATE_PATH.is_file():
        raise FileNotFoundError(
            f"Saved candidate model not found: "
            f"{BEST_CANDIDATE_PATH}. Run train.py first."
        )

    # Verify that the saved model can be loaded.
    model = joblib.load(BEST_CANDIDATE_PATH)

    # Save the same fitted pipeline under the final model name.
    joblib.dump(
        model,
        FINAL_MODEL_PATH,
    )

    logger.info(
        "Selected model '%s' saved to: %s",
        best_model_name,
        FINAL_MODEL_PATH,
    )


# ============================================================
# 6. MAIN EXECUTION
# ============================================================

def main() -> None:
    """Execute best-candidate selection."""

    logger.info("=" * 60)
    logger.info("BEST MODEL SELECTION STARTED")

    comparison = load_comparison()

    print("\nModel comparison:")
    print(
        comparison.sort_values(
            "cv_accuracy_mean",
            ascending=False,
        ).to_string(index=False)
    )

    best_model_name = select_best_candidate(
        comparison
    )

    save_selected_model(
        best_model_name
    )

    print("\n" + "=" * 60)
    print("BEST CANDIDATE MODEL SELECTED")
    print("=" * 60)
    print(f"Selected model: {best_model_name}")
    print(f"Saved model: {FINAL_MODEL_PATH}")

    logger.info("BEST MODEL SELECTION COMPLETED")


if __name__ == "__main__":
    main()

