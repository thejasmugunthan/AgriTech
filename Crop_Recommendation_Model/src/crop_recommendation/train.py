
"""
Train and compare machine learning models for the
AgriTech Crop Recommendation System.

Models:
1. Decision Tree
2. Random Forest
3. Gradient Boosting

Outputs:
- Model comparison report
- Best candidate model
- Training log
"""

import logging
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from src.crop_recommendation.config import (
    PROJECT_ROOT,
    TARGET_COLUMN,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "training"
LOG_DIR = PROJECT_ROOT / "logs"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

BEST_MODEL_PATH = MODEL_DIR / "best_crop_model.joblib"
COMPARISON_PATH = ARTIFACT_DIR / "model_comparison.csv"
REPORT_PATH = ARTIFACT_DIR / "best_model_classification_report.csv"

RANDOM_STATE = 42
CV_FOLDS = 3

MODEL_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOGGING
# ============================================================

logger = logging.getLogger("crop_model_training")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        LOG_DIR / "training.log",
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

def load_data():
    """Load the previously split training and testing datasets."""

    if not TRAIN_PATH.is_file():
        raise FileNotFoundError(
            f"Training file not found: {TRAIN_PATH}. "
            "Run split_data.py first."
        )

    if not TEST_PATH.is_file():
        raise FileNotFoundError(
            f"Testing file not found: {TEST_PATH}. "
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
            "from training data."
        )

    if TARGET_COLUMN not in test_data.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing "
            "from testing data."
        )

    if train_data[TARGET_COLUMN].isna().any():
        raise ValueError("Training targets contain missing values.")

    if test_data[TARGET_COLUMN].isna().any():
        raise ValueError("Testing targets contain missing values.")

    train_features = [
        column for column in train_data.columns
        if column != TARGET_COLUMN
    ]

    test_features = [
        column for column in test_data.columns
        if column != TARGET_COLUMN
    ]

    if set(train_features) != set(test_features):
        raise ValueError(
            "Training and testing feature columns do not match."
        )

    # Keep a consistent feature order.
    test_data = test_data[
        train_features + [TARGET_COLUMN]
    ]

    X_train = train_data[train_features].copy()
    y_train = train_data[TARGET_COLUMN].copy()

    X_test = test_data[train_features].copy()
    y_test = test_data[TARGET_COLUMN].copy()

    if y_train.nunique() < 2:
        raise ValueError(
            "Training data must contain at least two crop classes."
        )

    if set(y_test.unique()) - set(y_train.unique()):
        raise ValueError(
            "The test set contains crop classes absent from training."
        )

    logger.info("Training data shape: %s", X_train.shape)
    logger.info("Testing data shape: %s", X_test.shape)
    logger.info("Number of crop classes: %s", y_train.nunique())

    return X_train, X_test, y_train, y_test


# ============================================================
# 4. BUILD PREPROCESSING PIPELINE
# ============================================================

def build_preprocessor(X_train):
    """
    Build preprocessing transformations using training columns.

    Numerical columns:
        Median imputation.

    Categorical columns:
        Most-frequent imputation and one-hot encoding.

    Scaling is not needed for the tree-based models used here.
    """

    numerical_features = X_train.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    unsupported_features = [
        column for column in X_train.columns
        if column not in numerical_features
        and column not in categorical_features
    ]

    if unsupported_features:
        raise TypeError(
            "Unsupported feature data types: "
            f"{unsupported_features}"
        )

    transformers = []

    if numerical_features:
        numerical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                )
            ]
        )

        transformers.append(
            (
                "numerical",
                numerical_pipeline,
                numerical_features,
            )
        )

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore"),
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
        raise ValueError("No usable input features were found.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


# ============================================================
# 5. DEFINE MODELS
# ============================================================

def get_models():
    """Return resource-conscious candidate classifiers."""

    return {
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=2,
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=50,
            random_state=RANDOM_STATE,
        ),
    }

# ============================================================
# 6. EVALUATE A MODEL
# ============================================================

def evaluate_model(
    model_name,
    pipeline,
    X_test,
    y_test,
):
    """Calculate evaluation metrics for a fitted pipeline."""

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    logger.info(
        "%s | Accuracy: %.4f | Precision: %.4f | "
        "Recall: %.4f | F1: %.4f",
        model_name,
        accuracy,
        precision,
        recall,
        f1,
    )

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
    }


# ============================================================
# 7. TRAIN AND COMPARE MODELS
# ============================================================

def train_models(X_train, y_train, X_test, y_test):
    """Train, cross-validate, and compare candidate models."""

    models = get_models()
    results = []
    fitted_models = {}

    minimum_class_count = int(y_train.value_counts().min())
    number_of_folds = min(CV_FOLDS, minimum_class_count)

    if number_of_folds >= 2:
        cross_validator = StratifiedKFold(
            n_splits=number_of_folds,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
    else:
        cross_validator = None
        logger.warning(
            "Not enough records per class for cross-validation."
        )

    total_start = time.perf_counter()

    logger.info("=" * 65)
    logger.info("TRAINING STARTED")
    logger.info("Training records: %d", len(X_train))
    logger.info("Testing records: %d", len(X_test))
    logger.info("Features: %d", X_train.shape[1])
    logger.info("Crop classes: %d", y_train.nunique())
    logger.info("Cross-validation folds: %d", number_of_folds)
    logger.info("Candidate models: %s", list(models.keys()))
    logger.info("=" * 65)

    for model_index, (model_name, classifier) in enumerate(
        models.items(), start=1
    ):
        model_start = time.perf_counter()

        logger.info(
            "[MODEL %d/%d] Starting %s",
            model_index,
            len(models),
            model_name,
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("classifier", classifier),
            ]
        )

        fold_scores = []

        # Run cross-validation with visible fold-level progress.
        if cross_validator is not None:
            from sklearn.base import clone

            for fold_number, (train_indices, val_indices) in enumerate(
                cross_validator.split(X_train, y_train),
                start=1,
            ):
                fold_start = time.perf_counter()

                logger.info(
                    "[%s] Fold %d/%d started",
                    model_name,
                    fold_number,
                    number_of_folds,
                )

                fold_model = clone(pipeline)

                X_fold_train = X_train.iloc[train_indices]
                y_fold_train = y_train.iloc[train_indices]

                X_fold_val = X_train.iloc[val_indices]
                y_fold_val = y_train.iloc[val_indices]

                fold_model.fit(X_fold_train, y_fold_train)

                fold_predictions = fold_model.predict(X_fold_val)

                fold_accuracy = accuracy_score(
                    y_fold_val,
                    fold_predictions,
                )

                fold_scores.append(fold_accuracy)

                logger.info(
                    "[%s] Fold %d/%d completed | "
                    "Accuracy: %.4f | Time: %.1f seconds",
                    model_name,
                    fold_number,
                    number_of_folds,
                    fold_accuracy,
                    time.perf_counter() - fold_start,
                )

            cv_mean = float(np.mean(fold_scores))

            logger.info(
                "[%s] Cross-validation completed | "
                "Mean accuracy: %.4f",
                model_name,
                cv_mean,
            )
        else:
            cv_mean = np.nan

        # Fit the final pipeline on the complete training set.
        logger.info(
            "[%s] Fitting final model on %d records",
            model_name,
            len(X_train),
        )

        fit_start = time.perf_counter()
        pipeline.fit(X_train, y_train)

        logger.info(
            "[%s] Final fitting completed in %.1f seconds",
            model_name,
            time.perf_counter() - fit_start,
        )

        # Evaluate the model on the held-out test data.
        logger.info("[%s] Evaluating test data", model_name)

        evaluation_start = time.perf_counter()

        metrics = evaluate_model(
            model_name,
            pipeline,
            X_test,
            y_test,
        )

        logger.info(
            "[%s] Test evaluation completed in %.1f seconds",
            model_name,
            time.perf_counter() - evaluation_start,
        )

        metrics["cv_accuracy_mean"] = cv_mean

        results.append(metrics)
        fitted_models[model_name] = pipeline

        logger.info(
            "[MODEL %d/%d] %s finished in %.1f seconds",
            model_index,
            len(models),
            model_name,
            time.perf_counter() - model_start,
        )

    comparison = pd.DataFrame(results)

    # Select the model using cross-validation, not test accuracy.
    if comparison["cv_accuracy_mean"].notna().any():
        comparison = comparison.sort_values(
            by="cv_accuracy_mean",
            ascending=False,
        )
        selection_metric = "cv_accuracy_mean"
    else:
        raise RuntimeError(
            "No valid cross-validation scores were produced. "
            "Cannot reliably select the best model."
        )

    best_model_name = comparison.iloc[0]["model"]
    best_model = fitted_models[best_model_name]

    logger.info(
        "Best candidate selected using %s: %s",
        selection_metric,
        best_model_name,
    )

    logger.info(
        "All model training and comparison completed in %.1f seconds",
        time.perf_counter() - total_start,
    )

    return comparison, best_model_name, best_model



# ============================================================
# 8. SAVE RESULTS
# ============================================================

def save_results(
    comparison,
    best_model_name,
    best_model,
    X_test,
    y_test,
):
    """Save the comparison report, model, and class report."""

    comparison.to_csv(
        COMPARISON_PATH,
        index=False,
    )

    joblib.dump(
        best_model,
        BEST_MODEL_PATH,
    )

    predictions = best_model.predict(X_test)

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).T.to_csv(
        REPORT_PATH,
        index=True,
    )

    logger.info(
        "Model comparison saved to: %s",
        COMPARISON_PATH,
    )

    logger.info(
        "Best candidate model saved to: %s",
        BEST_MODEL_PATH,
    )

    logger.info(
        "Classification report saved to: %s",
        REPORT_PATH,
    )


# ============================================================
# 9. MAIN EXECUTION
# ============================================================

def main():
    """Execute the complete model training workflow."""

    logger.info("CROP RECOMMENDATION MODEL TRAINING STARTED")

    X_train, X_test, y_train, y_test = load_data()

    comparison, best_model_name, best_model = train_models(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    save_results(
        comparison,
        best_model_name,
        best_model,
        X_test,
        y_test,
    )

    print("\n" + "=" * 70)
    print("MODEL TRAINING AND COMPARISON RESULTS")
    print("=" * 70)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(f"\nBest candidate model: {best_model_name}")
    print(f"Saved model: {BEST_MODEL_PATH}")
    print(f"Comparison report: {COMPARISON_PATH}")
    print(f"Classification report: {REPORT_PATH}")

    logger.info("MODEL TRAINING COMPLETED SUCCESSFULLY")


def save_trained_pipeline(
    pipeline,
    model_name: str,
) -> None:
    """
    Save the complete fitted preprocessing and model pipeline.
    """

    import joblib

    from src.crop_recommendation.config import PROJECT_ROOT

    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "best_crop_model.joblib"

    # Save the complete fitted pipeline.
    joblib.dump(pipeline, model_path)

    logger.info(
        "Saved trained pipeline: %s",
        model_path,
    )

    print(f"\nSelected model: {model_name}")
    print(f"Complete pipeline saved to: {model_path}")




if __name__ == "__main__":
    main()
