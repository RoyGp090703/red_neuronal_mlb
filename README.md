# Predictor de victorias en la MLB - Deep Learning
**Desarrollado por: Rodrigo García Peláez**

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Estado](https://img.shields.io/badge/Estado-Funcionando_%7C_Expandiendo...-purple)

![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)

![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-000000?logo=python&logoColor=white)

Este proyecto implementa una **Red Neuronal** (NN) capaz de predecir el ganador de partidos de la MLB (Major League Baseball) basándose en estadísticas históricas de los equipos.

El modelo analiza métricas ofensivas y defensivas para calcular la probabilidad de victoria.

## Arquitectura del Modelo
El núcleo del proyecto es una red neuronal densa, construida con librerías **TensorFlow/Keras**:
- **Entrada:** Estadísticas históricas de ambos equipos (Promedio de bateo, ERA, victorias recientes, etc...).
- **Capas Ocultas:** 6 capas densas con activación `swish` y `BatchNormalization`.
- **Regularización:** Capas de `Dropout` para prevenir el sobreajuste (overfitting).
- **Salida:** Probabilidad binaria (`Sigmoid`) indicando si el equipo local gana o no.

## Estructura del Repositorio

```text
├── datos/                      # Carpeta de archivos 
│   └── teamstats.csv           # Base de datos
├── fuente/                     # Carpeta de códigos fuente
│   ├── modelo.py               # Definición de la arquitectura de la red
│   ├── preprocesamiento.py     # Limpieza y transformación de datos
│   └── particion.py            # División de entrenamiento, validación y prueba
├── modelos/                    # Carpeta de modelos entrenados
│   └── predictor.keras         # Primer modelo de red
├── notebooks/                  # Carpeta de códigos para graficar resultados
│   └── resultados.ipynb        # Grafiación de resultados
├── app.py                      # PROXIMAMENTE - Interfaz web para interactuar con el modelo
├── entrenamiento.py            # Código de entrenamiento de la red
└── requerimientos.txt          # Dependencias del proyecto
```
## Instalación
A continuación se muestran los pasos para poder descargar los archivos y librerías necesarias.

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/RoyGp090703/red_neuronal_mlb.git](https://github.com/RoyGp090703/red_neuronal_mlb.git)
   cd red_neuronal_mlb
   ```
2. **Instalar dependencias:**
   ```bash
   pip install -r requerimientos.txt
   ```
## Entrenamiento de la red (opcional)
   Si desea re-entrenar la red neuronal con nuevos datos o si has modificado la arquitectura de la misma, ejecuta los siguientes comandos