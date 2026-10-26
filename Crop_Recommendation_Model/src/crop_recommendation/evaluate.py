
"""
Evaluate and compare crop recommendation models.

Loads the saved model and held-out test dataset.

Generates:
1. Accuracy, precision, recall, F1-score and macro F1-score.
2. Per-crop classification report.
3. Confusion matrix.
4. Model comparison charts and reports.
5. Evaluation log.
"""

import logging

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.crop_recommendation.config import (
    PROJECT_ROOT,
    TARGET_COLUMN,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
TRAINING_DIR = PROJECT_ROOT / "artifacts" / "training"
EVALUATION_DIR = PROJECT_ROOT / "artifacts" / "evaluation"
LOG_DIR = PROJECT_ROOT / "logs"

TEST_PATH = DATA_DIR / "test.csv"
MODEL_PATH = MODEL_DIR / "best_crop_model.joblib"
COMPARISON_PATH = TRAINING_DIR / "model_comparison.csv"

EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOGGING
# ============================================================

logger = logging.getLogger("crop_model_evaluation")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        LOG_DIR / "evaluation.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


# ============================================================
# 3. LOAD MODEL AND TEST DATA
# ============================================================

def load_evaluation_data():
    """Load the saved pipeline and held-out test data."""

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Saved model not found: {MODEL_PATH}. "
            "Run train.py first."
        )

    if not TEST_PATH.is_file():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_PATH}. "
            "Run split_data.py first."
        )

    if not COMPARISON_PATH.is_file():
        raise FileNotFoundError(
            f"Model comparison report not found: "
            f"{COMPARISON_PATH}. Run train.py first."
        )

    # Only load trusted model files created by your own project.
    model = joblib.load(MODEL_PATH)

    test_data = pd.read_csv(TEST_PATH)

    if test_data.empty:
        raise ValueError("The test dataset is empty.")

    if TARGET_COLUMN not in test_data.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in test data."
        )

    if test_data[TARGET_COLUMN].isna().any():
        raise ValueError(
            "Test dataset contains missing target labels."
        )

    X_test = test_data.drop(columns=[TARGET_COLUMN])
    y_test = test_data[TARGET_COLUMN]

    expected_features = getattr(
        model,
        "feature_names_in_",
        None,
    )

    # Pipelines commonly expose the input feature names through
    # feature_names_in_ after fitting on a DataFrame.
    if expected_features is not None:
        expected_features = list(expected_features)

        missing_features = [
            column
            for column in expected_features
            if column not in X_test.columns
        ]

        extra_features = [
            column
            for column in X_test.columns
            if column not in expected_features
        ]

        if missing_features or extra_features:
            raise ValueError(
                "Test feature columns do not match the saved model. "
                f"Missing: {missing_features}; "
                f"unexpected: {extra_features}"
            )

        X_test = X_test[expected_features]

    logger.info("Model loaded from: %s", MODEL_PATH)
    logger.info("Test dataset shape: %s", test_data.shape)

    return model, X_test, y_test


# ============================================================
# 4. CALCULATE METRICS
# ============================================================

def calculate_metrics(model, X_test, y_test):
    """Calculate overall evaluation metrics."""

    logger.info("Generating test predictions.")

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision_weighted": precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "recall_weighted": recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "f1_weighted": f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        ),
    }

    logger.info("Evaluation metrics: %s", metrics)

    return predictions, metrics


# ============================================================
# 5. GENERATE CLASSIFICATION REPORT
# ============================================================

def save_classification_report(y_test, predictions):
    """Save precision, recall and F1-score for every crop."""

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).T

    report_path = (
        EVALUATION_DIR / "classification_report.csv"
    )

    report_df.to_csv(report_path, index=True)

    logger.info(
        "Classification report saved to: %s",
        report_path,
    )

    return report_df


# ============================================================
# 6. GENERATE CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(y_test, predictions):
    """Save the confusion matrix as a CSV and PNG chart."""

    labels = sorted(
        set(y_test.unique()) | set(predictions),
        key=str,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    )

    matrix_df.index.name = "Actual crop"
    matrix_df.columns.name = "Predicted crop"

    matrix_df.to_csv(
        EVALUATION_DIR / "confusion_matrix.csv"
    )

    # Annotate only smaller matrices to avoid unreadable text.
    annotate = len(labels) <= 20

    fig, ax = plt.subplots(
        figsize=(
            max(10, len(labels) * 0.65),
            max(8, len(labels) * 0.55),
        )
    )

    sns.heatmap(
        matrix_df,
        annot=annotate,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )

    ax.set_title("Crop Recommendation Confusion Matrix")
    ax.set_xlabel("Predicted crop")
    ax.set_ylabel("Actual crop")

    ax.tick_params(axis="x", rotation=90)
    ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()

    fig.savefig(
        EVALUATION_DIR / "confusion_matrix.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    logger.info("Confusion matrix saved.")


# ============================================================
# 7. COMPARE MODELS
# ============================================================

def compare_models(current_metrics):
    """
    Display existing training comparison results.

    Test metrics are kept separate from cross-validation results.
    """

    comparison = pd.read_csv(COMPARISON_PATH)

    required_columns = {
        "model",
        "accuracy",
        "f1_weighted",
        "cv_accuracy_mean",
    }

    missing_columns = required_columns - set(comparison.columns)

    if missing_columns:
        raise ValueError(
            "Model comparison report is missing columns: "
            f"{sorted(missing_columns)}"
        )

    # The training report contains test metrics generated during
    # model training. These are not independent of model selection
    # if the test scores were used to choose a model.
    comparison = comparison.rename(
        columns={
            "accuracy": "training_stage_test_accuracy",
            "f1_weighted": "training_stage_test_f1_weighted",
            "cv_accuracy_mean": "cv_accuracy_mean",
        }
    )

    comparison["final_evaluation_accuracy"] = (
        current_metrics["accuracy"]
    )

    comparison["final_evaluation_f1_macro"] = (
        current_metrics["f1_macro"]
    )

    output_path = (
        EVALUATION_DIR / "model_comparison.csv"
    )

    comparison.to_csv(output_path, index=False)

    logger.info(
        "Model comparison saved to: %s",
        output_path,
    )

    # Plot cross-validation results when available.
    valid_comparison = comparison.dropna(
        subset=["cv_accuracy_mean"]
    )

    if not valid_comparison.empty:
        plot_data = valid_comparison.sort_values(
            "cv_accuracy_mean",
            ascending=False,
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.barplot(
            data=plot_data,
            x="model",
            y="cv_accuracy_mean",
            ax=ax,
        )

        ax.set_title("Model Comparison: Cross-Validation Accuracy")
        ax.set_xlabel("Model")
        ax.set_ylabel("Mean Cross-Validation Accuracy")
        ax.set_ylim(0, 1)

        fig.tight_layout()

        fig.savefig(
            EVALUATION_DIR / "model_comparison.png",
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(fig)

    return comparison


# ============================================================
# 8. MAIN EXECUTION
# ============================================================

def main():
    """Run model evaluation and comparison."""

    logger.info("=" * 60)
    logger.info("CROP MODEL EVALUATION STARTED")

    model, X_test, y_test = load_evaluation_data()

    predictions, metrics = calculate_metrics(
        model,
        X_test,
        y_test,
    )

    classification_report_df = save_classification_report(
        y_test,
        predictions,
    )

    save_confusion_matrix(
        y_test,
        predictions,
    )

    comparison = compare_models(metrics)

    metrics_df = pd.DataFrame(
        [metrics]
    )

    metrics_df.to_csv(
        EVALUATION_DIR / "evaluation_metrics.csv",
        index=False,
    )

    print("\n" + "=" * 60)
    print("FINAL MODEL EVALUATION")
    print("=" * 60)

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name:25s}: {metric_value:.4f}")

    print("\nPer-crop classification report:")
    print(classification_report_df.to_string())

    print("\nModel comparison:")
    print(comparison.to_string(index=False))

    print(f"\nEvaluation reports: {EVALUATION_DIR}")

    logger.info("CROP MODEL EVALUATION COMPLETED")


if __name__ == "__main__":
    main()
