# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 02:24:43 2026

@author: cpcch
"""

import pandas as pd
import os
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from catboost import CatBoostRegressor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(BASE_DIR, "data", "f1_dataset_features.csv")
df = pd.read_csv(data_path)

features = [
    "GridPosition",
    "GridAdvantage",
    "PitStops",
    "AvgLapTime",
    "AirTemp",
    "TrackTemp",
    "Rain",
    "DriverAvgFinish",
    "DriverTop10Rate",
    "TeamAvgFinish",
    "TeamTop10Rate",
    "TeammateDiff",
    "BeatsTeammate",
    "DriverVsTeammatePace",
    "DriverRainAvgFinish",
    "DriverRecentForm"
]

X = df[features].copy()
X = X.fillna(X.mean())

y = df["FinalPosition"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = CatBoostRegressor(
    random_state=42,
    verbose=0
)
model.fit(X_train, y_train)

pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("Modelo de posición final entrenado.")
print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R2:   {r2:.3f}")

model_path = os.path.join(BASE_DIR, "models", "model_position.joblib")
joblib.dump(model, model_path)

print(f"Modelo guardado en: {model_path}")