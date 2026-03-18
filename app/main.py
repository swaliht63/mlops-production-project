from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

from app.model import load_model
from app.predict import predict
from app.logger import logger

app = FastAPI()
model = load_model()

# 👇 Define input with validation
class InputData(BaseModel):
    features: List[float] = Field(..., min_items=4, max_items=4)

# 👇 Map numeric output → human-readable
flower_map = {
    0: "Setosa 🌸",
    1: "Versicolor 🌼",
    2: "Virginica 🌺"
}

@app.get("/")
def home():
    return {
        "status": "success",
        "message": "MLOps Pipeline Running 🚀"
    }

@app.post("/predict")
def make_prediction(data: InputData):
    try:
        features = data.features

        logger.info(f"Received input: {features}")

        result = predict(model, features)[0]

        logger.info(f"Prediction result: {result}")

        return {
            "status": "success",
            "input": features,
            "prediction": {
                "class_id": result,
                "label": flower_map.get(result)
            }
        }

    except Exception as e:
        logger.error(f"Error: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )