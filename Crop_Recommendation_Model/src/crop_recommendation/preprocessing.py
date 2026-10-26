"""
Preprocessing pipeline for the AgriTech Crop Recommendation Model.

Responsibilities:
1. Load the training and testing datasets.
2. Separate features (X) and target (y).
3. Identify numerical and categorical features.
4. Handle missing feature values.
5. Encode categorical features.
6. Optionally scale numerical features.
7. Fit preprocessing on training data only.
8. Transform training and testing data.
9. Save the fitted preprocessing pipeline.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.crop_recommendation.config import (
    PROJECT_ROOT,
    TARGET_COLUMN,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "preprocessing"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"

# Set True when using algorithms that benefit from scaling,
# such as Logistic Regression, KNN, or SVM.
#
# Set False for an initial Decision Tree / Random Forest model.
SCALE_NUMERICAL_FEATURES = False

RANDOM_STATE = 42

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOGGING
# ============================================================

logger = logging.getLogger("crop_preprocessing")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        LOG_DIR / "preprocessing.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


# ============================================================
# 3. LOAD TRAINING AND TESTING DATA
# ============================================================

def load_split_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the previously created training and testing CSV files."""

    if not TRAIN_PATH.is_file():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAIN_PATH}. "
            "Run split_data.py first."
        )

    if not TEST_PATH.is_file():
        raise FileNotFoundError(
            f"Testing dataset not found: {TEST_PATH}. "
            "Run split_data.py first."
        )

    train_data = pd.read_csv(TRAIN_PATH)
    test_data = pd.read_csv(TEST_PATH)

    if train_data.empty or test_data.empty:
        raise ValueError(
            "Training and testing datasets must not be empty."
        )

    if TARGET_COLUMN not in train_data.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing "
            "from the training dataset."
        )

    if TARGET_COLUMN not in test_data.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing "
            "from the testing dataset."
        )

    if not train_data.columns.is_unique:
        raise ValueError(
            "Training dataset contains duplicate column names."
        )

    if not test_data.columns.is_unique:
        raise ValueError(
            "Testing dataset contains duplicate column names."
        )

    train_features = [
        column
        for column in train_data.columns
        if column != TARGET_COLUMN
    ]

    test_features = [
        column
        for column in test_data.columns
        if column != TARGET_COLUMN
    ]

    if set(train_features) != set(test_features):
        raise ValueError(
            "Training and testing feature columns do not match."
        )

    # Use the training column order consistently.
    test_data = test_data[
        train_features + [TARGET_COLUMN]
    ]

    logger.info(
        "Training dataset loaded: %s rows, %s columns.",
        train_data.shape[0],
        train_data.shape[1],
    )

    logger.info(
        "Testing dataset loaded: %s rows, %s columns.",
        test_data.shape[0],
        test_data.shape[1],
    )

    return train_data, test_data


# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================

def separate_features_target(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
]:
    """Separate input features from crop labels."""

    X_train = train_data.drop(
        columns=[TARGET_COLUMN]
    ).copy()

    y_train = train_data[TARGET_COLUMN].copy()

    X_test = test_data.drop(
        columns=[TARGET_COLUMN]
    ).copy()

    y_test = test_data[TARGET_COLUMN].copy()

    if X_train.shape[1] == 0:
        raise ValueError(
            "No input features are available for preprocessing."
        )

    if y_train.isna().any() or y_test.isna().any():
        raise ValueError(
            "Target labels contain missing values. "
            "Resolve target issues before preprocessing."
        )

    # The training and test sets must use the same features.
    X_test = X_test[X_train.columns]

    # Check for columns containing only missing values in training.
    all_missing_columns = X_train.columns[
        X_train.isna().all()
    ].tolist()

    if all_missing_columns:
        raise ValueError(
            "The following features contain only missing values "
            "in the training data: "
            f"{all_missing_columns}. Review or remove them."
        )

    # Prevent accidental model training on infinite values.
    numeric_train = X_train.select_dtypes(
        include=np.number
    )

    numeric_test = X_test.select_dtypes(
        include=np.number
    )

    if not numeric_train.empty:
        if np.isinf(numeric_train.to_numpy(dtype=float)).any():
            raise ValueError(
                "Infinite values found in numerical training features."
            )

    if not numeric_test.empty:
        if np.isinf(numeric_test.to_numpy(dtype=float)).any():
            raise ValueError(
                "Infinite values found in numerical testing features."
            )

    logger.info(
        "Number of training features: %s",
        X_train.shape[1],
    )

    logger.info(
        "Number of crop classes: %s",
        y_train.nunique(),
    )

    return X_train, y_train, X_test, y_test


# ============================================================
# 5. BUILD PREPROCESSING PIPELINE
# ============================================================

def build_preprocessor(
    X_train: pd.DataFrame,
) -> ColumnTransformer:
    """
    Build transformations based on training feature data types.

    Numerical features:
        Median imputation and optional scaling.

    Categorical features:
        Most-frequent imputation and one-hot encoding.
    """

    numerical_features = X_train.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    unsupported_features = [
        column
        for column in X_train.columns
        if column not in numerical_features
        and column not in categorical_features
    ]

    if unsupported_features:
        raise TypeError(
            "Unsupported feature data types found: "
            f"{unsupported_features}. "
            "Convert them to numeric or categorical values "
            "before preprocessing."
        )

    transformers = []

    # --------------------------------------------------------
    # Numerical features
    # --------------------------------------------------------

    if numerical_features:
        numerical_steps = [
            (
                "imputer",
                SimpleImputer(strategy="median"),
            )
        ]

        if SCALE_NUMERICAL_FEATURES:
            numerical_steps.append(
                ("scaler", StandardScaler())
            )

        numerical_pipeline = Pipeline(
            steps=numerical_steps
        )

        transformers.append(
            (
                "numerical",
                numerical_pipeline,
                numerical_features,
            )
        )

    # --------------------------------------------------------
    # Categorical features
    # --------------------------------------------------------

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )

    if not transformers:
        raise ValueError(
            "No supported numerical or categorical features found."
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    logger.info(
        "Numerical features: %s",
        numerical_features,
    )

    logger.info(
        "Categorical features: %s",
        categorical_features,
    )

    logger.info(
        "Numerical scaling enabled: %s",
        SCALE_NUMERICAL_FEATURES,
    )

    return preprocessor


# ============================================================
# 6. FIT AND TRANSFORM DATA
# ============================================================

def preprocess_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    """
    Fit the preprocessor on training data and transform both sets.

    The test data is never used to fit preprocessing parameters.
    """

    logger.info("Building preprocessing pipeline.")

    preprocessor = build_preprocessor(X_train)

    # Learn imputation, encoding, and scaling parameters
    # from the training data only.
    X_train_processed_array = preprocessor.fit_transform(
        X_train
    )

    # Reuse the learned parameters on the test data.
    X_test_processed_array = preprocessor.transform(
        X_test
    )

    # Recover transformed feature names.
    feature_names = preprocessor.get_feature_names_out()

    # Return DataFrames for easier inspection and debugging.
    X_train_processed = pd.DataFrame(
        X_train_processed_array,
        columns=feature_names,
        index=X_train.index,
    )

    X_test_processed = pd.DataFrame(
        X_test_processed_array,
        columns=feature_names,
        index=X_test.index,
    )

    if list(X_train_processed.columns) != list(
        X_test_processed.columns
    ):
        raise RuntimeError(
            "Training and testing feature columns do not match "
            "after preprocessing."
        )

    if X_train_processed.isna().any().any():
        raise ValueError(
            "Missing values remain in the processed training data."
        )

    if X_test_processed.isna().any().any():
        raise ValueError(
            "Missing values remain in the processed testing data."
        )

    logger.info(
        "Processed training shape: %s",
        X_train_processed.shape,
    )

    logger.info(
        "Processed testing shape: %s",
        X_test_processed.shape,
    )

    return (
        X_train_processed,
        X_test_processed,
        preprocessor,
    )


# ============================================================
# 7. SAVE PREPROCESSOR AND OPTIONAL PROCESSED DATA
# ============================================================

def save_preprocessor(
    preprocessor: ColumnTransformer,
) -> None:
    """Save the fitted preprocessing pipeline."""

    joblib.dump(
        preprocessor,
        PREPROCESSOR_PATH,
    )

    logger.info(
        "Fitted preprocessor saved to: %s",
        PREPROCESSOR_PATH,
    )


def save_processed_data(
    X_train_processed: pd.DataFrame,
    y_train: pd.Series,
    X_test_processed: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Save processed feature sets and their labels for inspection."""

    X_train_processed.to_csv(
        ARTIFACT_DIR / "X_train_processed.csv",
        index=False,
    )

    X_test_processed.to_csv(
        ARTIFACT_DIR / "X_test_processed.csv",
        index=False,
    )

    # Keep the target labels separate from the transformed features.
    y_train.to_frame(name=TARGET_COLUMN).to_csv(
        ARTIFACT_DIR / "y_train.csv",
        index=False,
    )

    y_test.to_frame(name=TARGET_COLUMN).to_csv(
        ARTIFACT_DIR / "y_test.csv",
        index=False,
    )

    logger.info(
        "Processed datasets and labels saved to: %s",
        ARTIFACT_DIR,
    )


# ============================================================
# 8. MAIN EXECUTION
# ============================================================

def main() -> None:
    """Execute the preprocessing workflow."""

    logger.info("=" * 60)
    logger.info("CROP RECOMMENDATION PREPROCESSING STARTED")

    # Step 1: Load train/test datasets.
    train_data, test_data = load_split_datasets()

    # Step 2: Separate features and labels.
    X_train, y_train, X_test, y_test = (
        separate_features_target(
            train_data,
            test_data,
        )
    )

    # Step 3: Fit and apply preprocessing.
    (
        X_train_processed,
        X_test_processed,
        preprocessor,
    ) = preprocess_data(
        X_train,
        X_test,
    )

    # Step 4: Save the fitted preprocessor.
    save_preprocessor(preprocessor)

    # Step 5: Save processed data for inspection.
    save_processed_data(
        X_train_processed,
        y_train,
        X_test_processed,
        y_test,
    )

    # Step 6: Display results.
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"Training samples: {len(X_train_processed)}")
    print(f"Testing samples : {len(X_test_processed)}")

    print(
        f"Features before preprocessing: {X_train.shape[1]}"
    )

    print(
        "Features after preprocessing: "
        f"{X_train_processed.shape[1]}"
    )

    print("\nProcessed training preview:")
    print(X_train_processed.head())

    print(f"\nPreprocessor saved to: {PREPROCESSOR_PATH}")
    print(f"Processed data saved to: {ARTIFACT_DIR}")

    logger.info(
        "CROP RECOMMENDATION PREPROCESSING COMPLETED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()
