# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 13:52:02 2026

@author: cpcch
"""

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

from catboost import CatBoostClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_csv(
    os.path.join(BASE_DIR, "data", "f1_dataset_features.csv")
)

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

# Convertimos posiciones 1-20 en clases 0-19 para el clasificador
y = df["FinalPosition"].astype(int)
y_class = y - 1

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_class,
    test_size=0.2,
    random_state=42
)

model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function="MultiClass",
    verbose=0,
    random_state=42
)

model.fit(X_train, y_train)

pred_class = model.predict(X_test).flatten()

# Volvemos de clases 0-19 a posiciones 1-20
pred_position = pred_class + 1
real_position = y_test + 1

accuracy = accuracy_score(real_position, pred_position)
mae = mean_absolute_error(real_position, pred_position)
rmse = np.sqrt(mean_squared_error(real_position, pred_position))

print("Modelo multiclase entrenado.")
print(f"Accuracy exacta: {accuracy:.4f}")
print(f"MAE posiciones: {mae:.3f}")
print(f"RMSE posiciones: {rmse:.3f}")

comparacion = pd.DataFrame({
    "Real": real_position.values,
    "Predicho": pred_position,
    "ErrorAbsoluto": abs(real_position.values - pred_position)
})

print("\nPrimeras predicciones:")
print(comparacion.head(20))

print("\nDistribución de errores:")
print(comparacion["ErrorAbsoluto"].value_counts().sort_index())