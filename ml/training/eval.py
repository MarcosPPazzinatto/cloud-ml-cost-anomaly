from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import classification_report


FEAT_PATH = Path("ml/features/features.parquet")
MODEL_PATH = Path("artifacts/model_isoforest.pkl")


assert FEAT_PATH.exists(), "Missing features.parquet."
assert MODEL_PATH.exists(), "Missing trained model."


df = pd.read_parquet(FEAT_PATH)
feature_cols = ["cost_ma7", "cost_ma30", "cost_lag1", "dow", "month", "deploy_velocity"]
X = df[feature_cols].values
labels = df["is_anomaly"].astype(int).values


model = joblib.load(MODEL_PATH)
preds = (model.predict(X) == -1).astype(int)


print(classification_report(labels, preds, digits=4))
