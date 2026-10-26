
"""
Dataset loading, inspection, and validation for the
AgriTech Crop Recommendation Model.

Responsibilities:
1. Load the dataset.
2. Inspect dataset structure and quality.
3. Validate dataset requirements.
4. Generate inspection reports.
5. Record execution details in a log file.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.crop_recommendation.config import DATASET_PATH, TARGET_COLUMN


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# Project root:
# Crop_Recommendation_Model/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOG_DIR = PROJECT_ROOT / "logs"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

LOG_FILE = LOG_DIR / "data_validation.log"

# Create required directories if they do not exist.
LOG_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOGGING CONFIGURATION
# ============================================================

def configure_logger():
    """Configure logging to write to a file and the terminal."""

    logger = logging.getLogger("crop_data_validation")
    logger.setLevel(logging.INFO)

    # Prevent duplicate messages when this module is imported.
    logger.propagate = False

    # Avoid adding duplicate handlers if configured previously.
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    # Write logs to a file.
    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Display logs in the terminal.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = configure_logger()


# ============================================================
# 3. LOAD DATASET
# ============================================================

def load_dataset() -> pd.DataFrame:
    """
    Load the crop recommendation dataset from a CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If the dataset does not exist.
        ValueError: If the dataset is empty or has no columns.
    """

    logger.info("=" * 60)
    logger.info("STARTING DATASET LOADING")
    logger.info("Dataset path: %s", DATASET_PATH)

    # Check whether the dataset exists.
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            "Place the CSV file inside data/raw/ and update "
            "DATASET_PATH in config.py."
        )

    # Load the CSV file.
    try:
        dataset = pd.read_csv(DATASET_PATH)
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        raise ValueError(
            f"Unable to parse the CSV dataset: {error}"
        ) from error

    # Check whether the dataset contains records.
    if dataset.empty:
        raise ValueError("The dataset contains no rows.")

    # Check whether the dataset contains columns.
    if dataset.shape[1] == 0:
        raise ValueError("The dataset contains no columns.")

    logger.info("Dataset loaded successfully.")
    logger.info("Rows: %s", dataset.shape[0])
    logger.info("Columns: %s", dataset.shape[1])

    return dataset


# ============================================================
# 4. INSPECT DATASET
# ============================================================

def inspect_dataset(dataset: pd.DataFrame) -> dict:
    """
    Inspect the dataset structure and quality.

    Generates:
        artifacts/numerical_statistics.csv
        artifacts/crop_distribution.csv

    Returns:
        dict: Dataset inspection summary.
    """

    logger.info("=" * 60)
    logger.info("STARTING DATASET INSPECTION")

    # --------------------------------------------------------
    # Basic structure
    # --------------------------------------------------------

    row_count, column_count = dataset.shape

    logger.info("Dataset shape: %s", dataset.shape)
    logger.info("Column names: %s", dataset.columns.tolist())

    logger.info("Column data types:\n%s", dataset.dtypes.to_string())

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing_values = dataset.isna().sum()
    missing_columns = missing_values[missing_values > 0]

    if missing_columns.empty:
        logger.info("Missing values: None")
    else:
        logger.warning(
            "Columns containing missing values:\n%s",
            missing_columns.to_string(),
        )

    # --------------------------------------------------------
    # Duplicate rows
    # --------------------------------------------------------

    duplicate_count = int(dataset.duplicated().sum())

    logger.info("Duplicate rows: %s", duplicate_count)

    # --------------------------------------------------------
    # Numerical statistics
    # --------------------------------------------------------

    numerical_columns = dataset.select_dtypes(
        include=np.number
    ).columns

    if len(numerical_columns) > 0:
        numerical_statistics = dataset[
            numerical_columns
        ].describe().T

        statistics_path = (
            ARTIFACT_DIR / "numerical_statistics.csv"
        )

        numerical_statistics.to_csv(
            statistics_path,
            index=True,
        )

        logger.info(
            "Numerical statistics saved to: %s",
            statistics_path,
        )
    else:
        logger.warning(
            "No numerical columns found for statistical analysis."
        )

    # --------------------------------------------------------
    # Target distribution
    # --------------------------------------------------------

    if TARGET_COLUMN in dataset.columns:
        crop_distribution = (
            dataset[TARGET_COLUMN]
            .value_counts(dropna=False)
            .rename_axis(TARGET_COLUMN)
            .reset_index(name="count")
        )

        distribution_path = (
            ARTIFACT_DIR / "crop_distribution.csv"
        )

        crop_distribution.to_csv(
            distribution_path,
            index=False,
        )

        logger.info(
            "Target distribution:\n%s",
            crop_distribution.to_string(index=False),
        )

        logger.info(
            "Target distribution saved to: %s",
            distribution_path,
        )
    else:
        logger.warning(
            "Target column '%s' is missing. "
            "Target distribution cannot be generated.",
            TARGET_COLUMN,
        )

    # --------------------------------------------------------
    # Inspection summary
    # --------------------------------------------------------

    summary = {
        "rows": int(row_count),
        "columns": int(column_count),
        "missing_values": int(dataset.isna().sum().sum()),
        "duplicate_rows": duplicate_count,
        "numerical_columns": len(numerical_columns),
        "categorical_columns": len(
            dataset.select_dtypes(
                include=["object", "category", "string"]
            ).columns
        ),
    }

    logger.info("Inspection summary: %s", summary)
    logger.info("DATASET INSPECTION COMPLETED")

    return summary


# ============================================================
# 5. VALIDATE DATASET
# ============================================================

def validate_dataset(dataset: pd.DataFrame) -> bool:
    """
    Validate dataset requirements.

    Raises:
        ValueError: If a critical validation check fails.

    Returns:
        bool: True if validation succeeds.
    """

    logger.info("=" * 60)
    logger.info("STARTING DATASET VALIDATION")

    # --------------------------------------------------------
    # Check 1: Dataset must not be empty
    # --------------------------------------------------------

    if dataset.empty:
        raise ValueError("Validation failed: dataset is empty.")

    # --------------------------------------------------------
    # Check 2: Column names must be valid and unique
    # --------------------------------------------------------

    if not dataset.columns.is_unique:
        duplicate_columns = dataset.columns[
            dataset.columns.duplicated()
        ].tolist()

        raise ValueError(
            f"Duplicate column names found: {duplicate_columns}"
        )

    invalid_columns = [
        column
        for column in dataset.columns
        if not isinstance(column, str) or not column.strip()
    ]

    if invalid_columns:
        raise ValueError(
            f"Invalid or blank column names: {invalid_columns}"
        )

    # --------------------------------------------------------
    # Check 3: Target column must exist
    # --------------------------------------------------------

    if TARGET_COLUMN not in dataset.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not found. "
            f"Available columns: {dataset.columns.tolist()}"
        )

    # --------------------------------------------------------
    # Check 4: Target must not contain missing values
    # --------------------------------------------------------

    target = dataset[TARGET_COLUMN]

    if target.isna().any():
        missing_target_count = int(target.isna().sum())

        raise ValueError(
            f"Target column '{TARGET_COLUMN}' contains "
            f"{missing_target_count} missing values."
        )

    # --------------------------------------------------------
    # Check 5: Target labels must not be blank
    # --------------------------------------------------------

    if (
        pd.api.types.is_object_dtype(target.dtype)
        or isinstance(target.dtype, pd.StringDtype)
    ):
        blank_labels = target.map(
            lambda value: isinstance(value, str)
            and not value.strip()
        )

        if blank_labels.any():
            raise ValueError(
                f"Target column '{TARGET_COLUMN}' "
                "contains blank labels."
            )

    # --------------------------------------------------------
    # Check 6: Target must contain at least two classes
    # --------------------------------------------------------

    if target.nunique(dropna=True) < 2:
        raise ValueError(
            "The target column must contain at least "
            "two distinct crop classes."
        )

    # --------------------------------------------------------
    # Check 7: Detect infinite numerical values
    # --------------------------------------------------------

    numerical_data = dataset.select_dtypes(
        include=np.number
    )

    if not numerical_data.empty:
        infinite_counts = np.isinf(
            numerical_data.to_numpy(dtype=float)
        ).sum(axis=0)

        infinite_columns = {
            numerical_data.columns[index]: int(count)
            for index, count in enumerate(infinite_counts)
            if count > 0
        }

        if infinite_columns:
            raise ValueError(
                "Infinite numerical values found: "
                f"{infinite_columns}"
            )

    # --------------------------------------------------------
    # Check 8: Report feature missing values
    # --------------------------------------------------------

    feature_columns = dataset.drop(
        columns=[TARGET_COLUMN]
    )

    missing_features = feature_columns.isna().sum()
    missing_features = missing_features[
        missing_features > 0
    ]

    if not missing_features.empty:
        logger.warning(
            "Feature columns contain missing values:\n%s",
            missing_features.to_string(),
        )
        logger.warning(
            "Missing feature values must be handled "
            "during preprocessing."
        )

    # --------------------------------------------------------
    # Check 9: Report duplicate records
    # --------------------------------------------------------

    duplicate_count = int(dataset.duplicated().sum())

    if duplicate_count > 0:
        logger.warning(
            "Found %s duplicate rows. Review them before "
            "training; they have not been removed.",
            duplicate_count,
        )

    # --------------------------------------------------------
    # Validation completed
    # --------------------------------------------------------

    logger.info(
        "Dataset validation completed successfully."
    )

    logger.info(
        "Number of crop classes: %s",
        target.nunique(),
    )

    logger.info(
        "Crop classes: %s",
        sorted(target.astype(str).unique().tolist()),
    )

    return True


# ============================================================
# 6. DISPLAY DATASET SUMMARY
# ============================================================

def display_summary(
    dataset: pd.DataFrame,
    summary: dict,
) -> None:
    """Display a readable summary in the terminal."""

    print("\n" + "=" * 60)
    print("CROP RECOMMENDATION DATASET SUMMARY")
    print("=" * 60)

    print(f"Dataset path       : {DATASET_PATH}")
    print(f"Number of rows     : {summary['rows']}")
    print(f"Number of columns  : {summary['columns']}")
    print(f"Missing values     : {summary['missing_values']}")
    print(f"Duplicate rows     : {summary['duplicate_rows']}")
    print(f"Numerical columns  : {summary['numerical_columns']}")
    print(f"Categorical columns: {summary['categorical_columns']}")

    print("\nFirst five rows:")
    print(dataset.head())

    print("\nDataset information:")
    dataset.info()

    print("\nDataset validation: PASSED")

    print(f"\nLog file: {LOG_FILE}")
    print(f"Reports directory: {ARTIFACT_DIR}")


# ============================================================
# 7. MAIN EXECUTION
# ============================================================

def main() -> None:
    """Execute loading, inspection, and validation."""

    logger.info("CROP RECOMMENDATION DATA VALIDATION STARTED")

    try:
        # Step 1: Load the dataset.
        dataset = load_dataset()

        # Step 2: Inspect the dataset.
        summary = inspect_dataset(dataset)

        # Step 3: Validate the dataset.
        validate_dataset(dataset)

        # Step 4: Display the summary.
        display_summary(dataset, summary)

        logger.info(
            "DATA LOADING, INSPECTION, AND VALIDATION "
            "COMPLETED SUCCESSFULLY"
        )

    except Exception:
        logger.exception(
            "Data loading, inspection, or validation failed."
        )
        raise


if __name__ == "__main__":
    main()

