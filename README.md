
# 🏎️ F1 Performance Predictor

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-ScikitLearn-orange.svg)](https://scikit-learn.org/)

👉 **Repositorio GitHub:** https://github.com/crlosprz/f1-position-predictor

👉 **Aplicación desplegada:** https://f1-position-predictor-kbdnmvwuiagd8kvuwnefkm.streamlit.app/

Aplicación interactiva desarrollada con Streamlit para analizar y predecir el rendimiento de pilotos de Fórmula 1 utilizando datos históricos reales de las temporadas 2021-2024.

La aplicación permite simular distintos escenarios de carrera, comparar pilotos y equipos, analizar rendimiento en lluvia y estudiar diferencias frente a compañeros de equipo mediante técnicas de Machine Learning y análisis de datos.

---

## 📈 Resultados del modelo


El proyecto utiliza CatBoost Regressor para estimar la posición final de un piloto de Fórmula 1 a partir de variables históricas de rendimiento, posición de salida, condiciones meteorológicas y rendimiento del equipo. 
A partir de la posición estimada se calculan probabilidades de Top10, Top5, Podio y Victoria.
Los resultados son:
MAE = 1.759
RMSE = 2.454
R² = 0.817

# Objetivo del proyecto

El objetivo principal del proyecto es desarrollar una solución de análisis predictivo aplicada al mundo de la Fórmula 1 utilizando datos históricos reales.

La aplicación permite:

- Predecir la posición final y de ahí precedir:
  - Top 10
  - Top 5
  - Podio
  - Victoria

- Comparar:
  - pilotos
  - equipos
  - rendimiento frente al compañero
  - rendimiento en lluvia

- Simular escenarios hipotéticos:
  - pilotos en otros equipos
  - temporadas distintas
  - diferentes circuitos y condiciones

---

# Tecnologías utilizadas

- Python
- Streamlit
- Pandas
- Scikit-learn
- Plotly
- FastF1
- Joblib
- Matplotlib

---

# Dataset

Los datos han sido obtenidos utilizando la librería FastF1 a partir de datos históricos oficiales de Fórmula 1 entre las temporadas 2021 y 2024.

Se incluyen variables como:

- posición de salida
- posición final
- puntos
- neumáticos
- temperatura
- lluvia
- paradas en boxes
- tiempo medio por vuelta
- rendimiento frente al compañero
- métricas históricas de piloto y equipo

---

# Machine Learning

Durante el desarrollo se compararon distintos algoritmos de Machine Learning:

- Linear Regression
- Random Forest Regressor
- Extra Trees Regressor
- Gradient Boosting Regressor
- Hist Gradient Boosting Regressor
- XGBoost Regressor
- LightGBM Regressor
- CatBoost Regressor

CatBoost Regressor obtuvo el mejor rendimiento, alcanzando el menor error absoluto medio (MAE) y fue seleccionado como modelo final.

# 🔄 Reproducibilidad

El proyecto incluye:
- datasets procesados,
- modelos entrenados,
- scripts de descarga y limpieza,
- y la aplicación Streamlit completa.

Los datos pueden regenerarse ejecutando secuencialmente:

```bash
python src/01_download_data.py
python src/02_merge_datasets.py
python src/03_clean_dataset.py
python src/05_feature_engineering.py
python src/06_train_position_model.py

# Estructura del proyecto

```text
f1-performance-predictor/
│
├── main.py
├── README.md
├── requirements.txt
│
├── data/
│   ├── f1_dataset.csv
│   ├── f1_dataset_clean.csv
│   └── f1_dataset_features.csv
│
├── models/
│   └── model_position.joblib
│
├── src/
│   ├── 01_download_data.py
│   ├── 02_merge_datasets.py
│   ├── 03_clean_dataset.py
│   ├── 05_feature_engineering.py
│   ├── 06_train_position_model.py
│   ├── 07_compare_models.py
│   ├── 08_multiclass_position_model.py
│   └── 09_catboost_tuning.py
│
├── models/
│   ├── model_position.joblib
│   ├── model_comparison_results.csv
│   └── catboost_tuning_results.csv
│
├── images/
│   └── model_comparison_mae.png
│
└── notebooks/
    └── 01_exploratory_analysis.py
