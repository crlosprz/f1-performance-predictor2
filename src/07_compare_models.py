# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 13:21:05 2026

@author: cpcch
"""

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor
)

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

df = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "data",
        "f1_dataset_features.csv"
    )
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

X = df[features]

# Sustituir NaN por la media de cada columna
X = X.fillna(X.mean())

y = df["FinalPosition"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

models = {
    "Linear Regression":
        LinearRegression(),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=300,
            random_state=42
        ),
        
    "Hist Gradient Boosting":
    HistGradientBoostingRegressor(
        random_state=42
        ),

    "Extra Trees":
        ExtraTreesRegressor(
            n_estimators=300,
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            random_state=42
        ),
    "LightGBM":
        LGBMRegressor(
            random_state=42,
            verbose=-1
        ),

    "CatBoost":
        CatBoostRegressor(
            random_state=42,
            verbose=0
        ),

    "XGBoost":
        XGBRegressor(
            n_estimators=400,
            learning_rate=0.04,
            max_depth=4,
            random_state=42,
            objective="reg:squarederror"
        )
}

results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            pred
        )
    )

    r2 = r2_score(
        y_test,
        pred
    )

    results.append([
        name,
        mae,
        rmse,
        r2
    ])

results_df = pd.DataFrame(
    results,
    columns=[
        "Modelo",
        "MAE",
        "RMSE",
        "R2"
    ]
)

print(
    results_df.sort_values(
        "MAE"
    )
)

results_path = os.path.join(
    BASE_DIR,
    "models",
    "model_comparison_results.csv"
)

results_df.to_csv(results_path, index=False)

print(f"\nResultados guardados en: {results_path}")