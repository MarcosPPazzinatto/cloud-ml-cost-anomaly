import mlflow, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score


# load features (replace with Feast/BigQuery)
df = pd.read_parquet("data/daily_cost.parquet")
X = df[["cost_ma7", "cost_ma30", "dow", "month", "deploy_velocity"]]
labels = df["is_anomaly"]


with mlflow.start_run():
model = IsolationForest(n_estimators=200, contamination=0.02, random_state=7)
preds = model.fit_predict(X)
f1 = f1_score(labels, (preds == -1).astype(int))
mlflow.log_metric("f1", f1)
mlflow.sklearn.log_model(model, "model")
