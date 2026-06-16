# -*- coding: utf-8 -*-
"""
Created on Sat May 23 19:40:52 2026

@author: cpcch
"""

import fastf1
import pandas as pd
import time
import os

# Activamos caché
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "cache")

fastf1.Cache.enable_cache(CACHE_DIR)

# Lista donde guardaremos todas las filas
all_race_data = []

# Temporadas a descargar
seasons = [2024]

for season in seasons:

    print(f"\n========== TEMPORADA {season} ==========")

    # Calendario de carreras
    schedule = fastf1.get_event_schedule(season)

    for _, event in schedule.iterrows():

        gp_name = event["EventName"]

        try:
            print(f"\nDescargando: {gp_name}")

            # Cargar sesión de carrera
            session = fastf1.get_session(season, gp_name, "R")
            session.load()

            # Resultados de carrera
            results = session.results

            # Clima medio de la sesión
            weather = session.weather_data

            avg_air_temp = weather["AirTemp"].mean()
            avg_track_temp = weather["TrackTemp"].mean()
            rain = weather["Rainfall"].any()

            # Recorrer pilotos
            for _, row in results.iterrows():

                driver = row["Abbreviation"]
                team = row["TeamName"]

                grid_position = row["GridPosition"]
                final_position = row["Position"]

                points = row["Points"]

                # Vueltas del piloto
                driver_laps = session.laps.pick_drivers(driver)

                # Neumático más usado
                if not driver_laps.empty:
                    main_compound = (
                        driver_laps["Compound"]
                        .mode()
                        .iloc[0]
                    )
                else:
                    main_compound = None

                # Número de paradas aproximado
                pit_stops = driver_laps["Stint"].nunique()

                # Tiempo medio de vuelta
                avg_lap_time = (
                    driver_laps["LapTime"]
                    .dt.total_seconds()
                    .mean()
                )

                # Guardamos fila
                all_race_data.append({
                    "Season": season,
                    "GrandPrix": gp_name,
                    "Driver": driver,
                    "Team": team,
                    "GridPosition": grid_position,
                    "FinalPosition": final_position,
                    "Points": points,
                    "MainCompound": main_compound,
                    "PitStops": pit_stops,
                    "AvgLapTime": avg_lap_time,
                    "AirTemp": avg_air_temp,
                    "TrackTemp": avg_track_temp,
                    "Rain": rain
                })

            # Pausa pequeña para evitar saturar
            time.sleep(1)

        except Exception as e:
            print(f"Error en {gp_name}: {e}")

# Convertimos a DataFrame
df = pd.DataFrame(all_race_data)

# Creamos variables objetivo
df["Top10"] = (df["FinalPosition"] <= 10).astype(int)
df["Top5"] = (df["FinalPosition"] <= 5).astype(int)
df["Top3"] = (df["FinalPosition"] <= 3).astype(int)
df["Victory"] = (df["FinalPosition"] == 1).astype(int)

# Guardamos CSV
output_path = os.path.join(BASE_DIR, "data", f"f1_dataset_{seasons[0]}.csv")
df.to_csv(output_path, index=False)

print("\nDataset guardado correctamente.")
print(df.head())