# Predictor de victorias en la MLB - Deep Learning

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Estado](https://img.shields.io/badge/Estado-Listo_%7C_Expandiendo-purple)

![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)

![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-000000?logo=python&logoColor=white)

Este proyecto implementa una **Red Neuronal** (NN) capaz de predecir el ganador de partidos de la MLB (Major League Baseball) basándose en estadísticas históricas de los equipos.

El modelo analiza métricas ofensivas y defensivas para calcular la probabilidad de victoria.

## Arquitectura del Modelo
El núcleo del proyecto es una red neuronal densa (Feed-Forward) construida con librerías **TensorFlow/Keras**:
- **Entrada:** Estadísticas históricas de ambos equipos (Promedio de bateo, ERA, victorias recientes, etc...).
- **Capas Ocultas:** 6 capas densas con activación `swish` y `BatchNormalization`.
- **Regularización:** Capas de `Dropout` para prevenir el sobreajuste (overfitting).
- **Salida:** Probabilidad binaria (`Sigmoid`) indicando si el equipo local gana o no.

## Estructura del Repositorio

```text
├── datos/              # Dataset (teamstats.csv) gestionado con Git LFS
├── fuente/             # Código fuente modular
│   ├── modelo.py       # Definición de la arquitectura de la red
│   ├── preprocesamiento.py # Limpieza y transformación de datos
│   └── particion.py    # División de Train/Test/Validation
├── modelos/            # Archivo final entrenado (.keras)
├── notebooks/          # Análisis exploratorio de datos (EDA)
├── app.py              # Interfaz web (Streamlit) para interactuar con el modelo
├── entrenamiento.py    # Script maestro de entrenamiento
└── requerimientos.txt  # Dependencias del proyecto