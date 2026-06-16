# -*- coding: utf-8 -*-
"""
Created on Sun May 24 14:02:51 2026

@author: cpcch
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ruta base
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Leer dataset final
data_path = os.path.join(BASE_DIR, "data", "f1_dataset_final.csv")

df = pd.read_csv(data_path)

print(df.head())

# =========================
# TOP 10 PILOTOS POR VICTORIAS
# =========================

wins = (
    df[df["Victory"] == 1]
    .groupby("Driver")
    .size()
    .sort_values(ascending=False)
)

print("\nVictorias por piloto:")
print(wins)

plt.figure(figsize=(10,6))

wins.plot(kind="bar")

plt.title("Victorias por piloto (2021-2024)")
plt.xlabel("Piloto")
plt.ylabel("Victorias")

plt.tight_layout()
plt.show()

# =========================
# PODIOS POR PILOTO
# =========================

podiums = (
    df[df["Top3"] == 1]
    .groupby("Driver")
    .size()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10,6))

podiums.plot(kind="bar", color="orange")

plt.title("Podios por piloto")
plt.xlabel("Piloto")
plt.ylabel("Número de podios")

plt.tight_layout()
plt.show()

# =========================
# RELACIÓN SALIDA vs FINAL
# =========================

plt.figure(figsize=(8,6))

sns.scatterplot(
    data=df,
    x="GridPosition",
    y="FinalPosition",
    hue="Rain"
)

plt.title("Posición de salida vs posición final")

plt.xlabel("Posición de salida")
plt.ylabel("Posición final")

plt.tight_layout()
plt.show()

# =========================
# NEUMÁTICOS MÁS UTILIZADOS
# =========================

plt.figure(figsize=(8,5))

sns.countplot(
    data=df,
    x="MainCompound",
    order=df["MainCompound"].value_counts().index
)

plt.title("Compuestos más utilizados")

plt.xlabel("Neumático")
plt.ylabel("Frecuencia")

plt.tight_layout()
plt.show()

# =========================
# TEMPERATURA DE PISTA
# =========================

plt.figure(figsize=(8,5))

sns.histplot(df["TrackTemp"], bins=20)

plt.title("Distribución de temperatura de pista")

plt.xlabel("Temperatura pista")
plt.ylabel("Frecuencia")

plt.tight_layout()
plt.show()