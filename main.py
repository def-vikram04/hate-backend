from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import logging
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend URL for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class TextInput(BaseModel):
    text: str

# Load trained model
try:
    model = joblib.load("hate_speech_model.pkl")
    logger.info("✅ Model loaded successfully.")
except Exception as e:
    logger.error("❌ Failed to load model: %s", str(e))
    model = None

# Root route
@app.get("/")
def read_root():
    return {"message": "Welcome to the Hate Speech Classifier API"}

# Prediction route
@app.post("/predict")
async def predict(input: TextInput):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    if not input.text.strip():
        raise HTTPException(status_code=422, detail="Input text cannot be empty.")

    try:
        # Predict using pipeline
        prediction = model.predict([input.text])[0]

        # Optional mapping
        label_map = {
            0: "Hate Speech",
            1: "Offensive Language",
            2: "Neither"
        }

        return {
            "prediction": int(prediction),
            "label": label_map.get(int(prediction), "Unknown")
        }

    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        logger.error("Prediction failed:\n%s", traceback_str)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
