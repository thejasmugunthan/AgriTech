"""
Split the crop recommendation dataset into training and testing sets.

Training data: 80%
Testing data: 20%

Stratified splitting preserves crop-class proportions in both sets.
"""

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

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

TEST_SIZE = 0.20
RANDOM_STATE = 42

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
LOG_DIR = PROJECT_ROOT / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "split_data.log"


# ============================================================
# 2. LOGGING
# ============================================================

logger = logging.getLogger("crop_data_split")
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
# 3. STRATIFIED DATASET SPLIT
# ============================================================

def split_dataset(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a dataset into 80% training and 20% testing data.

    Stratification ensures that each crop class is represented
    in both sets when the class counts permit it.

    Returns:
        train_data: Training DataFrame.
        test_data: Testing DataFrame.
    """

    logger.info("Starting stratified dataset splitting.")

    # --------------------------------------------------------
    # Step 1: Validate basic requirements
    # --------------------------------------------------------

    if dataset.empty:
        raise ValueError("Cannot split an empty dataset.")

    if TARGET_COLUMN not in dataset.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' does not exist."
        )

    if dataset[TARGET_COLUMN].isna().any():
        raise ValueError(
            "Target contains missing values. Resolve them "
            "before performing a stratified split."
        )

    # Stratification needs at least two classes.
    class_counts = dataset[TARGET_COLUMN].value_counts()

    if len(class_counts) < 2:
        raise ValueError(
            "Stratified splitting requires at least two crop classes."
        )

    # Every class needs at least two examples to appear in both sets.
    insufficient_classes = class_counts[class_counts < 2]

    if not insufficient_classes.empty:
        raise ValueError(
            "Every crop needs at least two records for this "
            "stratified split. Insufficient crop counts:\n"
            f"{insufficient_classes.to_string()}"
        )

    # --------------------------------------------------------
    # Step 2: Separate features and target
    # --------------------------------------------------------

    X = dataset.drop(columns=[TARGET_COLUMN])
    y = dataset[TARGET_COLUMN]

    # --------------------------------------------------------
    # Step 3: Perform stratified 80:20 split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # --------------------------------------------------------
    # Step 4: Reconstruct the training and testing DataFrames
    # --------------------------------------------------------

    train_data = X_train.copy()
    train_data[TARGET_COLUMN] = y_train

    test_data = X_test.copy()
    test_data[TARGET_COLUMN] = y_test

    # Preserve the original column order.
    train_data = train_data[dataset.columns]
    test_data = test_data[dataset.columns]

    # --------------------------------------------------------
    # Step 5: Verify split sizes and class representation
    # --------------------------------------------------------

    if len(train_data) + len(test_data) != len(dataset):
        raise RuntimeError(
            "Split verification failed: total row counts do not match."
        )

    original_classes = set(y.unique())
    train_classes = set(train_data[TARGET_COLUMN].unique())
    test_classes = set(test_data[TARGET_COLUMN].unique())

    missing_train_classes = original_classes - train_classes
    missing_test_classes = original_classes - test_classes

    if missing_train_classes:
        raise RuntimeError(
            f"Training set is missing crop classes: "
            f"{missing_train_classes}"
        )

    if missing_test_classes:
        raise RuntimeError(
            f"Testing set is missing crop classes: "
            f"{missing_test_classes}"
        )

    logger.info(
        "Training set: %s rows (%.2f%%)",
        len(train_data),
        len(train_data) / len(dataset) * 100,
    )

    logger.info(
        "Testing set: %s rows (%.2f%%)",
        len(test_data),
        len(test_data) / len(dataset) * 100,
    )

    logger.info(
        "All %s crop classes are present in both sets.",
        len(original_classes),
    )

    return train_data, test_data


# ============================================================
# 4. SAVE SPLIT DATASETS
# ============================================================

def save_datasets(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> None:
    """Save training and testing datasets as CSV files."""

    train_path = OUTPUT_DIR / "train.csv"
    test_path = OUTPUT_DIR / "test.csv"

    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    logger.info("Training dataset saved to: %s", train_path)
    logger.info("Testing dataset saved to: %s", test_path)


# ============================================================
# 5. DISPLAY SPLIT SUMMARY
# ============================================================

def display_split_summary(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> None:
    """Display dataset sizes and crop counts."""

    total_rows = len(train_data) + len(test_data)

    print("\n" + "=" * 60)
    print("DATASET SPLIT SUMMARY")
    print("=" * 60)

    print(f"Total records : {total_rows}")
    print(
        f"Training set  : {len(train_data)} "
        f"({len(train_data) / total_rows * 100:.2f}%)"
    )
    print(
        f"Testing set   : {len(test_data)} "
        f"({len(test_data) / total_rows * 100:.2f}%)"
    )

    print(
        f"\nCrop classes in training set: "
        f"{train_data[TARGET_COLUMN].nunique()}"
    )
    print(
        f"Crop classes in testing set : "
        f"{test_data[TARGET_COLUMN].nunique()}"
    )

    print("\nCrop distribution comparison:")

    comparison = pd.DataFrame(
        {
            "total": (
                pd.concat(
                    [
                        train_data[TARGET_COLUMN],
                        test_data[TARGET_COLUMN],
                    ]
                )
                .value_counts()
            ),
            "train": train_data[TARGET_COLUMN].value_counts(),
            "test": test_data[TARGET_COLUMN].value_counts(),
        }
    ).fillna(0).astype(int)

    comparison["train_percentage"] = (
        comparison["train"] / len(train_data) * 100
    ).round(2)

    comparison["test_percentage"] = (
        comparison["test"] / len(test_data) * 100
    ).round(2)

    print(comparison.sort_index().to_string())

    print("\nSaved files:")
    print(OUTPUT_DIR / "train.csv")
    print(OUTPUT_DIR / "test.csv")


# ============================================================
# 6. MAIN EXECUTION
# ============================================================

def main() -> None:
    """Load, validate, split, and save the dataset."""

    logger.info("=" * 60)
    logger.info("CROP RECOMMENDATION DATA SPLITTING STARTED")

    # Step 1: Load the original dataset.
    dataset = load_dataset()

    # Step 2: Validate the dataset.
    validate_dataset(dataset)

    # Step 3: Split the dataset.
    train_data, test_data = split_dataset(dataset)

    # Step 4: Save the resulting datasets.
    save_datasets(train_data, test_data)

    # Step 5: Display the results.
    display_split_summary(train_data, test_data)

    logger.info("Dataset splitting completed successfully.")


if __name__ == "__main__":
    main()
