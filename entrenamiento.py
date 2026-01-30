import tensorflow as tf                                                                                         #Importación de librería "tensorflow" para crear modelos de aprendizaje
from tensorflow.keras import models, layers, callbacks, optimizers                                              #Importación de herramientas de "keras" para la creación y ajuste de la red
from tensorflow.keras.callbacks import CSVLogger                                                                #Importación de la herramienta para guardar el log del entrenamiento en un archivo CSV
import os                                                                                                       #Importación de librería "os" para manejo de rutas

from fuente.preprocesamiento import carga_limpia, caracterizacion                                               #Importación de funciones de preprocesamiento
from fuente.particion import preparacion                                                                        #Importación de la función para particionar los datos
from fuente.modelo import crear_modelo                                                                          #Importación de la función para crear el modelo de la red

df_crudo = carga_limpia('datos/teamstats.csv')                                                                  #Carga y limpieza del conjunto de datos  

df_listo = caracterizacion(df_crudo)                                                                            #Caracterización del conjunto de datos   

x_entreno, y_entreno, x_valid, y_valid, x_prueba, y_prueba = preparacion(df_listo)                              #Partición de los datos para entrenamiento, validación y prueba

input_shape = x_entreno.shape[1]                                                                                #Definición de la forma de entrada para el modelo

model = crear_modelo(entrada_shape=input_shape)                                                                   #Creación del modelo de red neuronal 

parada = callbacks.EarlyStopping(                                                                               #Declaración del callback para frenado temprano
              monitor='val_loss',                                                                               #Cuidado del parámetro de pérdida durante la validación
              patience=15,                                                                                      #Definición de 15 épocas para detener el proceso si el error no baja
              restore_best_weights=True)                                                                        #Obtención de la mejor época si el proceso termina

reducir_velocidad = callbacks.ReduceLROnPlateau(                                                                #Declaración del callback para el ajuste del parámetro de aprendizaje
              monitor='val_loss',                                                                               #Cuidado del parámetro de pérdida durante la validación
              factor=0.5,                                                                                       #Reducción de la velocidad a la mitad
              patience=5,                                                                                       #Definición de 5 épocas para el frenado
              min_lr=0.00001,                                                                                   #Declaración del límite mínimo
              verbose=1)                                                                                        #Aviso de modificación en el parámetro de aprendizaje

guardar_log = CSVLogger('entreno_log.csv', separator=',', append=False)                                         #Callback para guardar el log del entrenamiento en un archivo CSV 

historia = model.fit(                                                                                           #Inicio del entrenamiento del modelo creado
              x_entreno,                                                                                        #Definición de los datos de entrenamiento
              y_entreno,                                                                                        #Definición de los datos de predicción
              epochs=50,                                                                                        #Declaración del número de épocas que la red hará (50)
              batch_size=256,                                                                                   #Declaración del tamaño del "batch" con el que la red entrenará (256)
              validation_data=(x_valid, y_valid),                                                               #Validación con los datos de validacion declarados
              callbacks=[parada, reducir_velocidad, guardar_log],                                               #Declaración de los callbacks para regulación
              verbose=1)                                                                                        #Mensaje de progreso durante el entrenamiento

if not os.path.exists('modelos'):                                                                               #Creación de la carpeta para guardar el modelo si no existe  
    os.makedirs('modelos')                                                                                      #Creación de la carpeta para guardar el modelo si no existe 

ruta_modelo = 'modelos/predictor.keras'                                                                            #Definición de la ruta para guardar el modelo

model.save(ruta_modelo)                                                                                         #Guardado del modelo entrenado