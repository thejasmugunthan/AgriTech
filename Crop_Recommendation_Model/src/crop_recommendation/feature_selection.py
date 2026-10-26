
"""
Feature selection for the AgriTech Crop Recommendation Model.

Responsibilities:
1. Load and validate the dataset.
2. Separate input features (X) and target (y).
3. Identify numerical and categorical features.
4. Report missing values and constant features.
5. Save feature-selection reports.
"""

import json
import logging
from pathlib import Path

import pandas as pd

from src.crop_recommendation.config import (
    PROJECT_ROOT,
    TARGET_COLUMN,
)
from src.crop_recommendation.data_validation import (
    load_dataset,
    validate_dataset,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "feature_selection"
LOG_DIR = PROJECT_ROOT / "logs"

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "feature_selection.log"


# ============================================================
# 2. LOGGING
# ============================================================

logger = logging.getLogger("crop_feature_selection")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


# ============================================================
# 3. FEATURE SELECTION
# ============================================================

def select_features(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Separate features and target, then generate a feature report.

    Args:
        dataset: Validated crop recommendation dataset.

    Returns:
        X: Input features.
        y: Target crop labels.
        feature_report: Information about each input feature.
    """

    logger.info("Starting feature selection.")

    # --------------------------------------------------------
    # Step 1: Validate basic dataset structure
    # --------------------------------------------------------

    if dataset.empty:
        raise ValueError("Cannot select features from an empty dataset.")

    if not dataset.columns.is_unique:
        raise ValueError("Dataset contains duplicate column names.")

    if TARGET_COLUMN not in dataset.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' does not exist."
        )

    # --------------------------------------------------------
    # Step 2: Separate X and y
    # --------------------------------------------------------

    X = dataset.drop(columns=[TARGET_COLUMN]).copy()
    y = dataset[TARGET_COLUMN].copy()

    if X.shape[1] == 0:
        raise ValueError(
            "No input features remain after separating the target."
        )

    logger.info("Input feature count: %s", X.shape[1])
    logger.info("Target column: %s", TARGET_COLUMN)
    logger.info("Target classes: %s", y.nunique())

    # --------------------------------------------------------
    # Step 3: Identify feature types
    # --------------------------------------------------------

    numerical_features = X.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    other_features = [
        column
        for column in X.columns
        if column not in numerical_features
        and column not in categorical_features
    ]

    if other_features:
        logger.warning(
            "Features with unrecognized or special data types: %s",
            other_features,
        )

    logger.info("Numerical features: %s", numerical_features)
    logger.info("Categorical features: %s", categorical_features)

    # --------------------------------------------------------
    # Step 4: Inspect missing values
    # --------------------------------------------------------

    missing_counts = X.isna().sum()

    # --------------------------------------------------------
    # Step 5: Identify constant features
    # --------------------------------------------------------

    constant_features = [
        column
        for column in X.columns
        if X[column].nunique(dropna=False) <= 1
    ]

    if constant_features:
        logger.warning(
            "Constant features detected: %s",
            constant_features,
        )
        logger.warning(
            "Review these columns before training. "
            "They may be removed if they provide no useful information."
        )
    else:
        logger.info("No constant features detected.")

    # --------------------------------------------------------
    # Step 6: Generate feature-selection report
    # --------------------------------------------------------

    records = []

    for column in X.columns:
        records.append(
            {
                "feature": column,
                "data_type": str(X[column].dtype),
                "feature_type": (
                    "numerical"
                    if column in numerical_features
                    else "categorical"
                    if column in categorical_features
                    else "other"
                ),
                "missing_values": int(missing_counts[column]),
                "unique_values": int(
                    X[column].nunique(dropna=True)
                ),
                "is_constant": column in constant_features,
                "selected_initially": (
                    column not in constant_features
                ),
            }
        )

    feature_report = pd.DataFrame(records)

    report_path = ARTIFACT_DIR / "feature_report.csv"

    feature_report.to_csv(
        report_path,
        index=False,
    )

    # --------------------------------------------------------
    # Step 7: Save selected feature names
    # --------------------------------------------------------

    # Keep all non-constant features in the initial candidate set.
    # Do not remove correlated features automatically.
    selected_features = [
        column
        for column in X.columns
        if column not in constant_features
    ]

    selection_path = ARTIFACT_DIR / "selected_features.json"

    with selection_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "target_column": TARGET_COLUMN,
                "selected_features": selected_features,
                "numerical_features": [
                    column
                    for column in selected_features
                    if column in numerical_features
                ],
                "categorical_features": [
                    column
                    for column in selected_features
                    if column in categorical_features
                ],
            },
            file,
            indent=4,
        )

    # Return only the initial candidate feature set.
    X_selected = X[selected_features].copy()

    logger.info(
        "Initial candidate feature count: %s",
        X_selected.shape[1],
    )
    logger.info("Feature report saved to: %s", report_path)
    logger.info("Selected feature list saved to: %s", selection_path)

    return X_selected, y, feature_report


# ============================================================
# 4. MAIN EXECUTION
# ============================================================

def main() -> None:
    """Execute the feature-selection workflow."""

    logger.info("=" * 60)
    logger.info("CROP RECOMMENDATION FEATURE SELECTION STARTED")

    # Step 1: Load dataset using the existing function.
    dataset = load_dataset()

    # Step 2: Validate dataset.
    validate_dataset(dataset)

    # Step 3: Select and inspect features.
    X, y, feature_report = select_features(dataset)

    # Step 4: Display results.
    print("\n" + "=" * 60)
    print("FEATURE SELECTION SUMMARY")
    print("=" * 60)

    print(f"Total samples: {len(X)}")
    print(f"Selected candidate features: {X.shape[1]}")
    print(f"Target column: {TARGET_COLUMN}")
    print(f"Number of crop classes: {y.nunique()}")

    print("\nSelected feature names:")
    for feature in X.columns:
        print(f"  - {feature}")

    print("\nFeature report:")
    print(feature_report.to_string(index=False))

    print(f"\nReports saved to: {ARTIFACT_DIR}")
    print(f"Log file: {LOG_FILE}")

    logger.info("Feature selection completed successfully.")


if __name__ == "__main__":
    main()

