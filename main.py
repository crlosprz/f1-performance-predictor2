# -*- coding: utf-8 -*-
"""
Created on Sun May 24 14:25:37 2026

@author: cpcch
"""

import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "f1_dataset_features.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

df = pd.read_csv(DATA_PATH)


model_position = joblib.load(os.path.join(MODELS_DIR, "model_position.joblib"))

st.set_page_config(
    page_title="F1 Performance Predictor",
    page_icon="🏎️",
    layout="wide"
)

st.title("🏎️ F1 Performance Predictor")
st.markdown(
    "Aplicación interactiva para analizar el rendimiento de pilotos de Fórmula 1 "
    "y estimar probabilidades de Top 10, Top 5, podio y victoria."
)

st.sidebar.header("⚙️ Simulador de carrera")

season = st.sidebar.selectbox(
    "Temporada",
    sorted(df["Season"].unique())
)

season_df = df[df["Season"] == season]

drivers = sorted(df["Driver"].dropna().unique())
teams = sorted(season_df["Team"].dropna().unique())
circuits = sorted(season_df["GrandPrix"].dropna().unique())

driver = st.sidebar.selectbox("Piloto", drivers)
team = st.sidebar.selectbox("Equipo", teams)
circuit = st.sidebar.selectbox("Gran Premio", circuits)

grid_position = st.sidebar.slider("Posición de salida", 1, 20, 5)
pit_stops = st.sidebar.slider("Número de paradas", 1, 5, 2)
# =========================================
# ESTIMACIÓN JERÁRQUICA DEL TIEMPO DE VUELTA
# =========================================

driver_team_circuit = df[
    (df["Driver"] == driver) &
    (df["Team"] == team) &
    (df["GrandPrix"] == circuit) &
    (df["Season"] == season)
]

team_circuit = df[
    (df["Team"] == team) &
    (df["GrandPrix"] == circuit) &
    (df["Season"] == season)
]

driver_circuit = df[
    (df["Driver"] == driver) &
    (df["GrandPrix"] == circuit)
]

circuit_season = df[
    (df["GrandPrix"] == circuit) &
    (df["Season"] == season)
]

# =========================================
# MEDIAS
# =========================================

driver_time = driver_team_circuit["AvgLapTime"].mean()
team_time = team_circuit["AvgLapTime"].mean()
circuit_time = circuit_season["AvgLapTime"].mean()

# =========================================
# FALLBACKS + PONDERACIONES
# =========================================

if pd.notna(driver_time) and pd.notna(team_time):

    avg_lap_time = (
        0.35 * driver_time +
        0.45 * team_time +
        0.20 * circuit_time
    )

    lap_source = "Piloto + Equipo + Circuito + Temporada"

elif pd.notna(team_time):

    avg_lap_time = (
        0.75 * team_time +
        0.25 * circuit_time
    )

    lap_source = "Equipo + Circuito + Temporada"

elif pd.notna(driver_circuit["AvgLapTime"].mean()):

    avg_lap_time = (
        0.75 * driver_circuit["AvgLapTime"].mean() +
        0.25 * circuit_time
    )

    lap_source = "Piloto + Circuito"

elif pd.notna(circuit_time):

    avg_lap_time = circuit_time

    lap_source = "Circuito + Temporada"

else:

    avg_lap_time = df["AvgLapTime"].mean()

    lap_source = "Media general del dataset"

# =========================================
# MOSTRAR INFO
# =========================================

st.sidebar.info(
    f"Tiempo medio estimado: {avg_lap_time:.2f} s\n\n"
    f"Fuente: {lap_source}"
)
air_temp = st.sidebar.slider("Temperatura del aire", 5.0, 40.0, 25.0)
track_temp = st.sidebar.slider("Temperatura de pista", 10.0, 65.0, 35.0)
rain = st.sidebar.checkbox("¿Lluvia?", value=False)

selected_data = df[
    (df["Driver"] == driver) &
    (df["Team"] == team) &
    (df["Season"] == season)
]

if not selected_data.empty:
    driver_avg_finish = selected_data["DriverAvgFinish"].mean()
    driver_top10_rate = selected_data["DriverTop10Rate"].mean()
    team_avg_finish = selected_data["TeamAvgFinish"].mean()
    team_top10_rate = selected_data["TeamTop10Rate"].mean()
    teammate_diff = selected_data["TeammateDiff"].mean()
    beats_teammate = selected_data["BeatsTeammate"].mean()
    driver_vs_teammate_pace = selected_data["DriverVsTeammatePace"].mean()
else:
    driver_avg_finish = df[df["Driver"] == driver]["DriverAvgFinish"].mean()
    driver_top10_rate = df[df["Driver"] == driver]["DriverTop10Rate"].mean()
    team_avg_finish = df[df["Team"] == team]["TeamAvgFinish"].mean()
    team_top10_rate = df[df["Team"] == team]["TeamTop10Rate"].mean()
    teammate_diff = df[df["Driver"] == driver]["TeammateDiff"].mean()
    beats_teammate = df[df["Driver"] == driver]["BeatsTeammate"].mean()
    driver_vs_teammate_pace = df[
    df["Driver"] == driver
]["DriverVsTeammatePace"].mean()

# Valores de seguridad si algún dato queda vacío
driver_avg_finish = driver_avg_finish if pd.notna(driver_avg_finish) else df["DriverAvgFinish"].mean()
driver_top10_rate = driver_top10_rate if pd.notna(driver_top10_rate) else df["DriverTop10Rate"].mean()
team_avg_finish = team_avg_finish if pd.notna(team_avg_finish) else df["TeamAvgFinish"].mean()
team_top10_rate = team_top10_rate if pd.notna(team_top10_rate) else df["TeamTop10Rate"].mean()
teammate_diff = teammate_diff if pd.notna(teammate_diff) else df["TeammateDiff"].mean()
beats_teammate = beats_teammate if pd.notna(beats_teammate) else df["BeatsTeammate"].mean()
driver_vs_teammate_pace = (
    driver_vs_teammate_pace
    if pd.notna(driver_vs_teammate_pace)
    else 0
)

# =========================================
# GRID ADVANTAGE
# =========================================

grid_advantage = 21 - grid_position

# =========================================
# NUEVAS FEATURES XGBOOST
# =========================================

driver_rain_avg_finish = df[
    df["Driver"] == driver
]["DriverRainAvgFinish"].mean()

driver_recent_form = df[
    df["Driver"] == driver
]["DriverRecentForm"].mean()

driver_rain_avg_finish = (
    driver_rain_avg_finish
    if pd.notna(driver_rain_avg_finish)
    else df["DriverRainAvgFinish"].mean()
)

driver_recent_form = (
    driver_recent_form
    if pd.notna(driver_recent_form)
    else df["DriverRecentForm"].mean()
)

input_data = pd.DataFrame([{
    "GridPosition": grid_position,
    "GridAdvantage": grid_advantage,
    "PitStops": pit_stops,
    "AvgLapTime": avg_lap_time,
    "AirTemp": air_temp,
    "TrackTemp": track_temp,
    "Rain": int(rain),
    "DriverAvgFinish": driver_avg_finish,
    "DriverTop10Rate": driver_top10_rate,
    "TeamAvgFinish": team_avg_finish,
    "TeamTop10Rate": team_top10_rate,
    "TeammateDiff": teammate_diff,
    "BeatsTeammate": beats_teammate,
    "DriverVsTeammatePace": driver_vs_teammate_pace,
    "DriverRainAvgFinish": driver_rain_avg_finish,
    "DriverRecentForm": driver_recent_form
}])

# Predicción principal: posición final estimada
predicted_position = model_position.predict(input_data)[0]

# Limitamos la posición estimada a un rango razonable
predicted_position = max(1, min(20, predicted_position))

predicted_position_rounded = int(round(predicted_position))
predicted_position_rounded = max(1, min(20, predicted_position_rounded))

# Función logística para convertir posición estimada en probabilidad
def position_to_probability(position, threshold, scale=1.2):
    return 1 / (1 + np.exp((position - threshold) / scale))

prob_top10 = position_to_probability(predicted_position, 10)
prob_top5 = position_to_probability(predicted_position, 5)
prob_top3 = position_to_probability(predicted_position, 3)
prob_victory = position_to_probability(predicted_position, 1, scale=0.8)

st.subheader("🎯 Predicción del escenario")

st.metric(
    "Posición final estimada",
    f"P{predicted_position_rounded}"
)

st.caption(
    f"Valor continuo estimado por el modelo: {predicted_position:.2f}"
)
col1, col2, col3, col4 = st.columns(4)

col1.metric("Top 10", f"{prob_top10*100:.1f}%")
col2.metric("Top 5", f"{prob_top5*100:.1f}%")
col3.metric("Podio", f"{prob_top3*100:.1f}%")
col4.metric("Victoria", f"{prob_victory*100:.1f}%")



st.markdown("---")

st.subheader("📊 Análisis histórico")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Piloto",
    "Equipos",
    "Circuitos",
    "Lluvia",
    "Vs compañero",
    "Dataset",
    "Modelos"
])

with tab1:
    st.markdown(f"### Rendimiento histórico de {driver}")

    driver_df = df[df["Driver"] == driver]

    col_a, col_b, col_c, col_d = st.columns(4)

    col_a.metric("Carreras", len(driver_df))
    col_b.metric("Victorias", int(driver_df["Victory"].sum()))
    col_c.metric("Podios", int(driver_df["Top3"].sum()))
    col_d.metric("Top 10", int(driver_df["Top10"].sum()))

    fig = px.bar(
        driver_df.groupby("Season")[["Victory", "Top3", "Top5", "Top10"]].sum().reset_index(),
        x="Season",
        y=["Victory", "Top3", "Top5", "Top10"],
        title=f"Evolución de resultados de {driver}"
    )

    st.plotly_chart(fig, use_container_width=True, key="grafico_piloto")

with tab2:
    st.markdown("### Resultados por equipo")

    team_summary = (
        df.groupby("Team")[["Victory", "Top3", "Top5", "Top10"]]
        .sum()
        .sort_values("Victory", ascending=False)
        .reset_index()
    )

    fig = px.bar(
        team_summary,
        x="Team",
        y="Victory",
        title="Victorias por equipo"
    )

    st.plotly_chart(fig, use_container_width=True, key="grafico_equipos")
    st.dataframe(df, use_container_width=True)



with tab3:
    st.markdown(f"### Datos históricos del {circuit}")

    selected_season_circuit = st.selectbox(
        "Selecciona temporada",
        sorted(df["Season"].unique()),
        key="circuit_season"
    )

    circuit_df = df[
        (df["GrandPrix"] == circuit) &
        (df["Season"] == selected_season_circuit)
    ]

    fig = px.scatter(
        circuit_df,
        x="GridPosition",
        y="FinalPosition",
        color="Driver",
        size="Points",
        hover_data=["Season", "Team", "Points"],
        title=f"Relación entre posición de salida y posición final - {selected_season_circuit}",
        color_discrete_sequence=px.colors.qualitative.Dark24
    )

    fig.update_traces(
        marker=dict(
            size=12,
            opacity=0.9,
            line=dict(width=1, color="black")
        )
    )

    fig.update_layout(
        height=600,
        legend_title_text="Piloto",
        xaxis_title="Posición de salida",
        yaxis_title="Posición final"
    )

    fig.update_yaxes(autorange="reversed")

    st.plotly_chart(fig, use_container_width=True, key="grafico_circuitos")

    st.dataframe(circuit_df, use_container_width=True)
    
    

with tab4:
    st.markdown("### 🌧️ Mejores pilotos en condiciones de lluvia")

    rain_df = df[df["Rain"] == 1]

    if rain_df.empty:
        st.warning("No hay suficientes carreras con lluvia en el dataset.")
    else:
        rain_summary = (
            rain_df.groupby("Driver")
            .agg(
                CarrerasLluvia=("Driver", "count"),
                PosicionMedia=("FinalPosition", "mean"),
                Top10Rate=("Top10", "mean"),
                Podios=("Top3", "sum"),
                Victorias=("Victory", "sum")
            )
            .reset_index()
        )

        rain_summary = rain_summary[rain_summary["CarrerasLluvia"] >= 2]
        rain_summary = rain_summary.sort_values("PosicionMedia")

        fig = px.bar(
            rain_summary,
            x="Driver",
            y="PosicionMedia",
            title="Posición media en carreras con lluvia",
            hover_data=["CarrerasLluvia", "Top10Rate", "Podios", "Victorias"]
        )

        st.plotly_chart(fig, use_container_width=True, key="grafico_lluvia")
        st.dataframe(rain_summary, use_container_width=True)

with tab5:
    st.markdown("### 🧑‍🤝‍🧑 Rendimiento frente al compañero de equipo")

    selected_season_vs = st.selectbox(
        "Selecciona temporada",
        sorted(df["Season"].unique()),
        key="vs_season"
    )

    vs_df = df[df["Season"] == selected_season_vs]

    teammate_summary = (
        vs_df.groupby("Driver")
        .agg(
            DiferenciaMediaPosicion=("TeammateDiff", "mean"),
            GanaCompaneroRate=("BeatsTeammate", "mean"),
            RitmoVsCompanero=("DriverVsTeammatePace", "mean"),
            Carreras=("Driver", "count"),
            EquipoPrincipal=("Team", lambda x: x.mode().iloc[0])
        )
        .reset_index()
    )

    teammate_summary = teammate_summary[teammate_summary["Carreras"] >= 5]
    teammate_summary = teammate_summary.sort_values("RitmoVsCompanero")

    fig = px.bar(
        teammate_summary,
        x="Driver",
        y="RitmoVsCompanero",
        color="EquipoPrincipal",
        title=f"Ritmo medio frente al compañero de equipo - {selected_season_vs}",
        hover_data=[
            "EquipoPrincipal",
            "Carreras",
            "DiferenciaMediaPosicion",
            "GanaCompaneroRate"
        ]
    )

    st.plotly_chart(fig, use_container_width=True, key="grafico_companero")

    st.dataframe(teammate_summary, use_container_width=True)

    st.info(
        "Valores negativos indican que el piloto fue más rápido que su compañero. "
        "Valores positivos indican que fue más lento. "
        "El análisis está filtrado por temporada para evitar mezclar compañeros de distintos años."
    )
with tab6:
    st.markdown("### Dataset utilizado")

    st.dataframe(df, use_container_width=True)

st.markdown("---")
st.info(
    "Nota: el modelo estima probabilidades a partir de datos históricos de 2021-2024. "
    "No simula accidentes, safety cars, sanciones ni abandonos mecánicos."
)

with tab7:
    st.markdown("### Comparación de modelos")

    image_path = os.path.join(
        BASE_DIR,
        "images",
        "model_comparison_mae.png"
    )

    

    from PIL import Image

    img = Image.open(image_path)

    st.image(
    img,
    caption="Gráfica generada en R con ggplot2",
    width=900
    )   

    comparison_path = os.path.join(
        BASE_DIR,
        "models",
        "model_comparison_results.csv"
    )

    comparison_df = pd.read_csv(comparison_path)

    st.dataframe(
        comparison_df.sort_values("MAE"),
        use_container_width=True
    )

    st.info(
        "Se seleccionó CatBoost Regressor porque obtuvo el menor MAE, es decir, el menor error medio en posiciones finales."
    )