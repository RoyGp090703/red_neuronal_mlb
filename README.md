# Predictor de victorias en la MLB - Deep Learning

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Status](https://img.shields.io/badge/Status-Completed-green)

Este proyecto implementa una **Red Neuronal** (NN) capaz de predecir el ganador de partidos de la MLB (Major League Baseball) basándose en estadísticas históricas de los equipos.

El modelo analiza métricas ofensivas y defensivas para calcular la probabilidad de victoria.

## Arquitectura del Modelo
El núcleo del proyecto es una red neuronal densa (Feed-Forward) construida con librerías **TensorFlow/Keras**:
- **Input:** Estadísticas históricas de ambos equipos (Promedio de bateo, ERA, victorias recientes, etc...).
- **Capas Ocultas:** 6 capas densas con activación `swish` y `BatchNormalization`.
- **Regularización:** Capas de `Dropout` para prevenir el sobreajuste (overfitting).
- **Output:** Probabilidad binaria (Sigmoid) indicando si el equipo local gana o no.

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