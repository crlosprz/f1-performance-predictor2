# -*- coding: utf-8 -*-
"""
Created on Sun May 24 13:56:28 2026

@author: cpcch
"""

import pandas as pd
import os

# Ruta base del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Carpeta data
DATA_DIR = os.path.join(BASE_DIR, "data")

# CSVs a unir
files = [
    "f1_dataset_2021.csv",
    "f1_dataset_2022.csv",
    "f1_dataset_2023.csv",
    "f1_dataset_2024.csv"
]

# Lista donde guardaremos los dataframes
dfs = []

for file in files:

    path = os.path.join(DATA_DIR, file)

    print(f"Cargando {file}...")

    df = pd.read_csv(path)

    dfs.append(df)

# Unimos todo
final_df = pd.concat(dfs, ignore_index=True)

# Eliminamos duplicados
final_df.drop_duplicates(inplace=True)

# Reiniciamos índices
final_df.reset_index(drop=True, inplace=True)

# Guardamos dataset final
output_path = os.path.join(DATA_DIR, "f1_dataset_final.csv")

final_df.to_csv(output_path, index=False)

print("\nDataset final creado correctamente.")
print(final_df.head())

print("\nNúmero total de filas:")
print(len(final_df))