import pandas as pd
from pathlib import Path


RAW_PATH = Path("ml/data/daily_cost.parquet")
OUT_DIR = Path("ml/features")
OUT_DIR.mkdir(parents=True, exist_ok=True)


assert RAW_PATH.exists(), "Run ml/data/generate_synthetic_costs.py first."


df = pd.read_parquet(RAW_PATH)


# Feature engineering per project/service
df["date"] = pd.to_datetime(df["date"])


df = df.sort_values(["project", "service", "date"]).copy()
df["dow"] = df["date"].dt.dayofweek
f = []
for (p, s), g in df.groupby(["project", "service"], sort=False):
g = g.copy()
g["cost_ma7"] = g["cost"].rolling(7, min_periods=3).mean()
g["cost_ma30"] = g["cost"].rolling(30, min_periods=10).mean()
g["cost_lag1"] = g["cost"].shift(1)
# Simple synthetic deploy velocity proxy: absolute day-to-day change
g["deploy_velocity"] = (g["cost"].pct_change().abs().fillna(0) * 100).clip(0, 100)
f.append(g)


feat = pd.concat(f).dropna().copy()
feat["month"] = feat["date"].dt.month


OUT_PATH = OUT_DIR / "features.parquet"
feat.to_parquet(OUT_PATH, index=False)
print(f"Features saved to {OUT_PATH} with shape {feat.shape}")
