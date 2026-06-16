# -*- coding: utf-8 -*-
"""
Created on Sun May 24 14:12:37 2026

@author: cpcch
"""

import pandas as pd
import os

# Ruta base
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Leer dataset final
data_path = os.path.join(BASE_DIR, "data", "f1_dataset_final.csv")

df = pd.read_csv(data_path)

print("Filas originales:")
print(len(df))

# ======================================
# LIMPIEZA GRID POSITION
# ======================================

df = df[df["GridPosition"] >= 1]
df = df[df["GridPosition"] <= 20]

# ======================================
# ELIMINAR TESTS Y SESIONES NO OFICIALES
# ======================================

df = df[
    ~df["GrandPrix"].str.contains("Test", case=False, na=False)
]

# ======================================
# ELIMINAR FILAS SIN POSICIÓN FINAL
# ======================================

df = df.dropna(subset=["FinalPosition"])

# ======================================
# ELIMINAR POSICIONES INVÁLIDAS
# ======================================

df = df[df["FinalPosition"] > 0]

# ======================================
# ELIMINAR VALORES NULOS IMPORTANTES
# ======================================

important_cols = [
    "GridPosition",
    "FinalPosition",
    "Points",
    "MainCompound",
    "TrackTemp"
]

df = df.dropna(subset=important_cols)

# ======================================
# CONVERTIR VARIABLES
# ======================================

df["Rain"] = df["Rain"].astype(int)

# ======================================
# RESETEAR ÍNDICES
# ======================================

df.reset_index(drop=True, inplace=True)

# ======================================
# GUARDAR DATASET LIMPIO
# ======================================

output_path = os.path.join(
    BASE_DIR,
    "data",
    "f1_dataset_clean.csv"
)

df.to_csv(output_path, index=False)

print("\nFilas finales:")
print(len(df))

print("\nDataset limpio guardado correctamente.")

print("\nPrimeras filas:")
print(df.head())