import tensorflow as tf                                                                                         #Importación de librería "tensorflow" para crear modelos de aprendizaje
from tensorflow.keras import models, layers, callbacks, optimizers                                              #Importación de herramientas de "keras" para la creación y ajuste de la red

def crear_modelo(entrada_shape, learning_rate=0.001):                                                           #Función para crear el modelo de red neuronal, con el tamaño de la entrada y la tasa de aprendizaje como parámetros

    model = models.Sequential()                                                                                 #Creación del modelo de red

    model.add(layers.Dense(128, input_dim=entrada_shape))                                                       #Definición de la capa de entrada según el tamaño de "x_entreno" (114)
    model.add(layers.BatchNormalization())                                                                      #Normalización de los datos
    model.add(layers.Activation('swish'))                                                                       #Función de activación "swish"
    model.add(layers.Dropout(0.4))                                                                              #Dropout para evitar overfitting

    model.add(layers.Dense(64))                                                                                 #Definición de la primera capa oculta con 64 neuronas
    model.add(layers.BatchNormalization())                                                                      #Normalización de los datos
    model.add(layers.Activation('swish'))                                                                       #Función de activación "swish"
    model.add(layers.Dropout(0.3))                                                                              #Dropout para evitar overfitting

    model.add(layers.Dense(32))                                                                                 #Definición de la segunda capa oculta con 32 neuronas
    model.add(layers.BatchNormalization())                                                                      #Normalización de los datos
    model.add(layers.Activation('swish'))                                                                       #Función de activación "swish"
    model.add(layers.Dropout(0.2))                                                                              #Dropout para evitar overfitting

    model.add(layers.Dense(16))                                                                                 #Definición de la tercera capa oculta con 16 neuronas
    model.add(layers.BatchNormalization())                                                                      #Normalización de los datos
    model.add(layers.Activation('swish'))                                                                       #Función de activación "swish"
    model.add(layers.Dropout(0.1))                                                                              #Dropout para evitar overfitting

    model.add(layers.Dense(8))                                                                                  #Definición de la cuarta capa oculta con 8 neuronas
    model.add(layers.BatchNormalization())                                                                      #Normalización de los datos
    model.add(layers.Activation('swish'))                                                                       #Función de activación "swish"

    model.add(layers.Dense(4))                                                                                  #Definición de la quinta capa oculta con 4 neuronas
    model.add(layers.BatchNormalization())                                                                      #Normalización de los datos
    model.add(layers.Activation('swish'))                                                                       #Función de activación "swish"

    model.add(layers.Dense(2))                                                                                  #Definición de la sexta capa oculta con 2 neuronas
    model.add(layers.BatchNormalization())                                                                      #Normalización de los datos
    model.add(layers.Activation('swish'))                                                                       #Función de activación "swish"

    model.add(layers.Dense(1, activation='sigmoid'))                                                            #Definición de la capa de salida con 1 neurona y función de activación sigmoide para clasificación binaria

    optimizador = optimizers.Adam(learning_rate=learning_rate)                                                  #Definición del parámetro de aprendizaje inicial y de la función de optimización "adam"

    model.compile(                                                                                              #Declaración de funciones para la red
                optimizer=optimizador,                                                                          #Elección de la función de optimización
                loss='binary_crossentropy',                                                                     #Elección de la función de pérdida "binary_crossentropy", porque es la correcta para el resultado binario (0 o 1)
                metrics=['accuracy','precision','auc'])                                                         #Elección de la métrica "accuracy"
    
    return model                                                                                                #Retorno del modelo creado