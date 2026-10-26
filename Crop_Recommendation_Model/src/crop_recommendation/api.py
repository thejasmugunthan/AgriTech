
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# 1. Locate the trained model
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "best_crop_model.joblib"

model = None


# 2. Define and validate input parameters

class CropInput(BaseModel):
    TYPE_OF_CROP: str
    SOIL: str
    SEASON: str
    SOWN: float = Field(ge=0)
    HARVESTED: float = Field(ge=0)
    WATER_SOURCE: str

    SOIL_PH: float = Field(ge=0, le=14)
    SOIL_PH_HIGH: float = Field(ge=0, le=14)

    CROPDURATION: float = Field(ge=0)
    CROPDURATION_MAX: float = Field(ge=0)

    TEMP: float
    MAX_TEMP: float

    WATERREQUIRED: float = Field(ge=0)
    WATERREQUIRED_MAX: float = Field(ge=0)

    RELATIVE_HUMIDITY: float = Field(ge=0, le=100)
    RELATIVE_HUMIDITY_MAX: float = Field(ge=0, le=100)

    N: float = Field(ge=0)
    N_MAX: float = Field(ge=0)
    P: float = Field(ge=0)
    P_MAX: float = Field(ge=0)
    K: float = Field(ge=0)
    K_MAX: float = Field(ge=0)



# 3. Load the trained model when the API starts
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found: {MODEL_PATH}. "
            "Run the training module first."
        )

    model = joblib.load(MODEL_PATH)

    print("Crop recommendation model loaded successfully.")

    yield

    model = None


# 4. Create the FastAPI application
app = FastAPI(
    title="AgriTech Crop Recommendation API",
    description="Predict suitable crops using soil and weather parameters.",
    version="1.0.0",
    lifespan=lifespan,
)


# 5. Home endpoint
@app.get("/")
def home():
    return {
        "message": "AgriTech Crop Recommendation API is running"
    }


# 6. Health-check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# 7. Crop prediction endpoint
@app.post("/predict")
def predict_crop(data: CropInput):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="The prediction model is not loaded."
        )

    # Convert the request into a DataFrame
    input_data = pd.DataFrame([data.model_dump()])

    try:
        # Predict using the complete trained pipeline
        prediction = model.predict(input_data)

        return {
            "success": True,
            "recommended_crop": str(prediction[0]),
            "message": "Crop prediction completed successfully"
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the model's expected input features."
        ) from error

