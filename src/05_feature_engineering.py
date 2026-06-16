# -*- coding: utf-8 -*-
"""
Created on Sun May 24 14:44:37 2026

@author: cpcch
"""

import pandas as pd
import os

# ======================================
# RUTA BASE
# ======================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ======================================
# CARGAR DATASET LIMPIO
# ======================================

data_path = os.path.join(
    BASE_DIR,
    "data",
    "f1_dataset_clean.csv"
)

df = pd.read_csv(data_path)

print("Dataset cargado.")
print(df.head())

# ======================================
# DRIVER PERFORMANCE
# ======================================

driver_avg_finish = (
    df.groupby("Driver")["FinalPosition"]
    .mean()
)

driver_top10_rate = (
    df.groupby("Driver")["Top10"]
    .mean()
)

# ======================================
# TEAM PERFORMANCE
# ======================================

team_avg_finish = (
    df.groupby("Team")["FinalPosition"]
    .mean()
)

team_top10_rate = (
    df.groupby("Team")["Top10"]
    .mean()
)

# ======================================
# AÑADIR VARIABLES
# ======================================

df["DriverAvgFinish"] = df["Driver"].map(driver_avg_finish)

df["DriverTop10Rate"] = df["Driver"].map(driver_top10_rate)

df["TeamAvgFinish"] = df["Team"].map(team_avg_finish)

df["TeamTop10Rate"] = df["Team"].map(team_top10_rate)

# ======================================
# VENTAJA DE POSICIÓN DE SALIDA
# ======================================

df["GridAdvantage"] = 21 - df["GridPosition"]

# ======================================
# TEAMMATE COMPARISON
# ======================================

df["TeammateDiff"] = 0
df["BeatsTeammate"] = 0

# ======================================
# DRIVER VS TEAMMATE PACE
# ======================================

df["DriverVsTeammatePace"] = 0.0

for idx, row in df.iterrows():

    same_race = df[
        (df["Season"] == row["Season"]) &
        (df["GrandPrix"] == row["GrandPrix"]) &
        (df["Team"] == row["Team"])
    ]

    # Debe haber exactamente dos pilotos
    # Debe haber exactamente dos pilotos
    if len(same_race) == 2:

        teammate = same_race[same_race["Driver"] != row["Driver"]]

        if not teammate.empty:

            teammate_position = teammate.iloc[0]["FinalPosition"]
            
            diff = teammate_position - row["FinalPosition"]

            df.at[idx, "TeammateDiff"] = diff

        # ======================================
        # COMPARACIÓN DE RITMO VS COMPAÑERO
        # ======================================

            if (
                pd.notna(row["AvgLapTime"]) and
                pd.notna(teammate.iloc[0]["AvgLapTime"])
            ):

                teammate_lap = teammate.iloc[0]["AvgLapTime"]

                driver_lap = row["AvgLapTime"]

            # Negativo = más rápido que compañero
                lap_diff = driver_lap - teammate_lap

                df.at[idx, "DriverVsTeammatePace"] = lap_diff

            if row["FinalPosition"] < teammate_position:
                df.at[idx, "BeatsTeammate"] = 1

# ======================================
# RENDIMIENTO EN LLUVIA POR PILOTO
# ======================================

rain_performance = (
    df[df["Rain"] == 1]
    .groupby("Driver")["FinalPosition"]
    .mean()
)

df["DriverRainAvgFinish"] = df["Driver"].map(rain_performance)

df["DriverRainAvgFinish"] = df["DriverRainAvgFinish"].fillna(
    df["DriverAvgFinish"]
)

# ======================================
# RENDIMIENTO DEL EQUIPO POR TEMPORADA
# ======================================

df["TeamSeasonAvgFinish"] = (
    df.groupby(["Team", "Season"])["FinalPosition"]
    .transform("mean")
)

df["TeamSeasonTop10Rate"] = (
    df.groupby(["Team", "Season"])["Top10"]
    .transform("mean")
)

# ======================================
# RENDIMIENTO DEL PILOTO POR TEMPORADA
# ======================================

df["DriverSeasonAvgFinish"] = (
    df.groupby(["Driver", "Season"])["FinalPosition"]
    .transform("mean")
)

df["DriverSeasonTop10Rate"] = (
    df.groupby(["Driver", "Season"])["Top10"]
    .transform("mean")
)

# ======================================
# FORMA RECIENTE DEL PILOTO
# Media de posición en las últimas 5 carreras
# ======================================

df = df.sort_values(["Driver", "Season", "GrandPrix"])

df["DriverRecentForm"] = (
    df.groupby("Driver")["FinalPosition"]
    .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
)

# ======================================
# DIFERENCIA RESPECTO A LA MEDIA DE CARRERA
# ======================================

df["RaceAvgFinish"] = (
    df.groupby(["Season", "GrandPrix"])["FinalPosition"]
    .transform("mean")
)

df["FinishVsRaceAverage"] = df["FinalPosition"] - df["RaceAvgFinish"]



# ======================================
# GUARDAR DATASET FINAL
# ======================================

output_path = os.path.join(
    BASE_DIR,
    "data",
    "f1_dataset_features.csv"
)

df.to_csv(output_path, index=False)

print("\nFeature engineering completado.")

print("\nNuevas columnas:")
print(df[[
    "Driver",
    "Team",
    "DriverAvgFinish",
    "TeamAvgFinish",
    "TeammateDiff",
    "BeatsTeammate"
]].head())