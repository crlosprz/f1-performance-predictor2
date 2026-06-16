    # -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 14:30:17 2026

@author: cpcch
"""

# src/09_catboost_tuning.py

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from catboost import CatBoostRegressor

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

X = df[features].fillna(df[features].mean())
y = df["FinalPosition"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

results = []

for iterations in [300, 500, 1000]:
    for depth in [4, 6, 8]:

        print(
            f"\nProbando iterations={iterations} depth={depth}"
        )

        model = CatBoostRegressor(
            iterations=iterations,
            depth=depth,
            learning_rate=0.05,
            loss_function="RMSE",
            verbose=0,
            random_state=42
        )

        model.fit(
            X_train,
            y_train
        )

        pred = model.predict(X_test)

        mae = mean_absolute_error(
            y_test,
            pred
        )

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
            iterations,
            depth,
            mae,
            rmse,
            r2
        ])

results_df = pd.DataFrame(
    results,
    columns=[
        "Iterations",
        "Depth",
        "MAE",
        "RMSE",
        "R2"
    ]
)

results_df = results_df.sort_values(
    "MAE"
)

print("\n======================")
print("MEJORES RESULTADOS")
print("======================\n")

print(results_df)

results_df.to_csv(
    os.path.join(
        BASE_DIR,
        "models",
        "catboost_tuning_results.csv"
    ),
    index=False
)