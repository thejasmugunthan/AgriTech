
import argparse
import logging
import subprocess
import sys
from pathlib import Path


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

LOG_DIR = PROJECT_ROOT / "logs"
MODEL_PATH = PROJECT_ROOT / "models" / "best_crop_model.joblib"

LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "main.log"


# ============================================================
# 2. LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("agritech_main")


# ============================================================
# 3. WORKFLOW STEPS
# ============================================================

WORKFLOW = [
    (
        "Dataset Validation",
        "src.crop_recommendation.data_validation",
    ),
    (
        "Exploratory Data Analysis",
        "src.crop_recommendation.eda",
    ),
    (
        "Feature Selection",
        "src.crop_recommendation.feature_selection",
    ),
    (
        "Train/Test Data Splitting",
        "src.crop_recommendation.split_data",
    ),
    (
        "Model Training",
        "src.crop_recommendation.train",
    ),
    (
        "Model Evaluation",
        "src.crop_recommendation.evaluate",
    ),
    (
        "Test Predictions",
        "src.crop_recommendation.predict",
    ),
]


# ============================================================
# 4. EXECUTE A WORKFLOW STEP
# ============================================================

def run_step(step_name: str, module_name: str) -> None:
    """Execute one Python module and stop if it fails."""

    logger.info("=" * 65)
    logger.info("STARTING: %s", step_name)
    logger.info("=" * 65)

    command = [
        sys.executable,
        "-m",
        module_name,
    ]

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Step failed: {step_name}. "
            f"Exit code: {result.returncode}. "
            f"Check the terminal output and {LOG_FILE}."
        )

    logger.info("COMPLETED: %s", step_name)


# ============================================================
# 5. RUN THE ML WORKFLOW
# ============================================================

def run_ml_pipeline() -> None:
    """Run validation, analysis, training, and prediction."""

    logger.info("AG RITECH CROP RECOMMENDATION WORKFLOW STARTED")

    for step_name, module_name in WORKFLOW:
        run_step(step_name, module_name)

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Expected trained model was not created: {MODEL_PATH}"
        )

    logger.info("=" * 65)
    logger.info("ALL ML WORKFLOW STEPS COMPLETED SUCCESSFULLY")
    logger.info("Saved model: %s", MODEL_PATH)
    logger.info("Log file: %s", LOG_FILE)
    logger.info("=" * 65)


# ============================================================
# 6. START THE SWAGGER API
# ============================================================

def start_api() -> None:
    """Start the FastAPI server with Swagger documentation."""

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Trained model not found. Run the ML workflow first."
        )

    logger.info("Starting AgriTech Crop Recommendation API.")
    logger.info("Swagger UI will be available at http://127.0.0.1:8000/docs")

    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.crop_recommendation.api:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--reload",
    ]

    # The server runs until you stop it with Ctrl+C.
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
    )


# ============================================================
# 7. MAIN ENTRY POINT
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AgriTech Crop Recommendation Workflow"
    )

    parser.add_argument(
        "--api-only",
        action="store_true",
        help="Start Swagger API without retraining the model.",
    )

    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Run the ML workflow without starting the API.",
    )

    args = parser.parse_args()

    try:
        if args.api_only:
            start_api()
            return

        run_ml_pipeline()

        if not args.no_api:
            start_api()

    except (RuntimeError, FileNotFoundError) as error:
        logger.error("%s", error)
        logger.error("Workflow stopped. Fix the error before continuing.")
        sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Workflow interrupted by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()






    #          main.py
    #             |
    #             v
    #    1. Validate Dataset
    #             |
    #             v
    #    2. Exploratory Analysis
    #             |
    #             v
    #    3. Feature Selection
    #             |
    #             v
    #    4. Split Train/Test Data
    #             |
    #             v
    #    5. Train Candidate Models
    #             |
    #             v
    #    6. Evaluate Best Model
    #             |
    #             v
    #    7. Test Predictions
    #             |
    #             v
    #    8. Start FastAPI Server
    #             |
    #             v
    #       Swagger UI