import numpy as np
import pandas as pd
from pathlib import Path


RNG_SEED = 7
np.random.seed(RNG_SEED)


OUT_DIR = Path("ml/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Parameters
n_days = 365
projects = ["core", "analytics", "edge"]
services = ["api", "worker", "storage"]


# Generate base time series with weekly seasonality and trend
start_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=n_days)
idx = pd.date_range(start_date, periods=n_days, freq="D")


def gen_series(base_level: float) -> np.ndarray:
trend = np.linspace(0, base_level * 0.2, n_days)
weekly = 1 + 0.15 * np.sin(2 * np.pi * np.arange(n_days) / 7)
noise = np.random.normal(0, base_level * 0.05, n_days)
return base_level * weekly + trend + noise


rows = []
for prj in projects:
for svc in services:
base = np.random.uniform(50, 200)
y = gen_series(base)
# Insert anomalies on random days
is_anom = np.zeros(n_days, dtype=int)
spike_days = np.random.choice(np.arange(n_days), size=max(3, n_days // 40), replace=False)
for d in spike_days:
y[d] *= np.random.uniform(1.8, 3.2)
is_anom[d] = 1
for i, day in enumerate(idx):
rows.append({
"date": day,
"project": prj,
"service": svc,
"cost": max(y[i], 0.0),
"is_anomaly": int(is_anom[i])
})


_df = pd.DataFrame(rows)
_df.sort_values(["date", "project", "service"], inplace=True)


# Save parquet
out_path = OUT_DIR / "daily_cost.parquet"
_df.to_parquet(out_path, index=False)
print(f"Wrote {_df.shape[0]} rows to {out_path}")
