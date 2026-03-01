from pathlib import Path
import pandas as pd
from fastapi import FastAPI
from joblib import load

from .schemas import InsuranceInput

app = FastAPI(title="Insurance Cost Prediction API")

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"

model = load(MODEL_PATH)

@app.get("/")
def home():
    return {"message": "Insurance Cost Prediction API is running 🚀"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}

@app.post("/predict")
def predict(data: InsuranceInput):
    row = {
        "age": data.age,
        "sex": data.sex,
        "bmi": data.bmi,
        "children": data.children,
        "smoker": data.smoker,
        "region": data.region,
    }

    X = pd.DataFrame([row])
    pred = model.predict(X)[0]

    return {"predicted_charges": round(float(pred), 2)}