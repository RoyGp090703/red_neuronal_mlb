import pandas as pd                                                                                                 #Importación de librería "pandas" para manipulación de datos
import numpy as np                                                                                                  #Importación de librería "numpy" para manipulación numérica
from sklearn.preprocessing import StandardScaler                                                                    #Importación de la herramienta para escalar los datos numéricos, con media 0 y desviación estándar 1

def preparacion(df_modelo):                                                                                         #Definición de la función para preparación de datos para el modelo   

    target_col = 'target'                                                                                           #Nombre de la columna objetivo

    caracteristicas_cols = [c for c in df_modelo.columns if c not in ['gid_home', 'año_temp', target_col]]          #Lista con nombre de las columnas de datos

    x_completo = df_modelo[caracteristicas_cols]                                                                    #Selección de columnas de datos
    y_completo = df_modelo[target_col]                                                                              #Selección de columna objetivo

    total_filas = len(df_modelo)                                                                                    #Conteo total de registros
    indice_70 = int(total_filas * 0.70)                                                                             #Cálculo del 70% de los datos
    indice_85 = int(total_filas * 0.85)                                                                             #Cálculo del 85% de los datos

    x_entreno_1 = x_completo.iloc[:indice_70]                                                                       #Separación de datos para entrenamiento al 70%
    y_entreno = y_completo.iloc[:indice_70]                                                                         #Separación de columna objetivo para entrenamiento al 70%

    x_valid_1 = x_completo.iloc[indice_70:indice_85]                                                                #Separación de datos para validación al 15%
    y_valid = y_completo.iloc[indice_70:indice_85]                                                                  #Separación de columna objetivo para validación al 15%

    x_prueba_1 = x_completo.iloc[indice_85:]                                                                        #Separación de datos para prueba al 15%
    y_prueba = y_completo.iloc[indice_85:]                                                                          #Separación de columna objetivo para prueba al 15%

    escalador = StandardScaler()                                                                                    #Creación del escalador de datos
    escalador.fit(x_entreno_1)                                                                                      #Cálculo del escalador con respecto al conjunto de entrenamiento

    x_entreno = escalador.transform(x_entreno_1)                                                                    #Escalamiento del conjunto de entrenamiento
    x_valid   = escalador.transform(x_valid_1)                                                                      #Escalamiento del conjunto de validación
    x_prueba  = escalador.transform(x_prueba_1)                                                                     #Escalamiento del conjunto de prueba

    return x_entreno, y_entreno, x_valid, y_valid, x_prueba, y_prueba                                               #Retorno de los conjuntos preparados para el modelo