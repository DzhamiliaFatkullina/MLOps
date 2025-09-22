import os
import json
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np

class PredictRequest(BaseModel):
    features: list[float]

app = FastAPI(title="Simple Model API")

# Allow JS/Streamlit to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.joblib")
METADATA_PATH = os.environ.get("METADATA_PATH", "models/metadata.json")

model = None
metadata = {}

@app.on_event("startup")
def load_model():
    global model, metadata
    # MODEL_PATH is relative to container/workdir or repo root when running locally
    if not Path(MODEL_PATH).exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    if Path(METADATA_PATH).exists():
        metadata = json.load(open(METADATA_PATH))
    else:
        metadata = {"feature_names": [], "target_names": []}

@app.get("/")
def root():
    return {"status": "ok", "n_features": len(metadata.get("feature_names", []))}

@app.post("/predict")
def predict(req: PredictRequest):
    features = np.array(req.features).reshape(1, -1)
    if model is None:
        return {"error": "model not loaded"}
    pred = model.predict(features)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = [float(x) for x in model.predict_proba(features)[0]]
    return {
        "prediction": int(pred),
        "probabilities": proba,
        "target_names": metadata.get("target_names", [])
    }
