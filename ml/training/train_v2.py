import os
from pathlib import Path


import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score


FEAT_PATH = Path("ml/features/features.parquet")
ART_DIR = Path("artifacts")
ART_DIR.mkdir(exist_ok=True)
MODEL_PATH = ART_DIR / "model_isoforest.pkl"


assert FEAT_PATH.exists(), "Run ml/features/build_features.py first."


df = pd.read_parquet(FEAT_PATH)


feature_cols = ["cost_ma7", "cost_ma30", "cost_lag1", "dow", "month", "deploy_velocity"]
X = df[feature_cols].values
labels = df["is_anomaly"].values.astype(int)


# Time-based split
cutoff = int(len(df) * 0.8)
X_train, X_test = X[:cutoff], X[cutoff:]
y_train, y_test = labels[:cutoff], labels[cutoff:]


with mlflow.start_run() as run:
model = IsolationForest(
n_estimators=300,
contamination=max(0.01, labels.mean()),
random_state=7,
n_jobs=-1,
)
model.fit(X_train)


pred_test = (model.predict(X_test) == -1).astype(int)
precision = precision_score(y_test, pred_test, zero_division=0)
recall = recall_score(y_test, pred_test, zero_division=0)
f1 = f1_score(y_test, pred_test, zero_division=0)


mlflow.log_metric("precision", float(precision))
mlflow.log_metric("recall", float(recall))
mlflow.log_metric("f1", float(f1))
mlflow.log_param("n_estimators", 300)


joblib.dump(model, MODEL_PATH)
mlflow.log_artifact(str(MODEL_PATH))


print(f"Model saved to {MODEL_PATH}")
