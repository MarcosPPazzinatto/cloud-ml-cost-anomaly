import os
from pathlib import Path
from typing import List


import joblib
from fastapi import FastAPI
from pydantic import BaseModel


MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/model_isoforest.pkl"))


if not MODEL_PATH.exists():
raise RuntimeError("Model file not found. Train the model first.")


model = joblib.load(MODEL_PATH)
app = FastAPI(title="Cost Anomaly API")


FEATURE_ORDER = ["cost_ma7", "cost_ma30", "cost_lag1", "dow", "month", "deploy_velocity"]


class Item(BaseModel):
cost_ma7: float
cost_ma30: float
cost_lag1: float
dow: int
month: int
deploy_velocity: float


class BatchInput(BaseModel):
items: List[Item]


@app.get("/health")
async def health():
return {"status": "ok"}


@app.post("/predict")
async def predict(batch: BatchInput):
import numpy as np


X = np.array([[getattr(it, f) for f in FEATURE_ORDER] for it in batch.items])
preds = (model.predict(X) == -1).astype(int)
return {"predictions": preds.tolist()}
