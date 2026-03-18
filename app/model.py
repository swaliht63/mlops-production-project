import joblib
import os

def load_model():
    if not os.path.exists("app/model.joblib"):
        raise Exception("Model not found! Run training first.")
    return joblib.load("app/model.joblib")