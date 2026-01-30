import pandas as pd                                                                                             #Importación de librería "pandas" para manipulación de datos
import numpy as np                                                                                              #Importación de librería "numpy" para manipulación numérica

def carga_limpia(file_path):                                                                                    #Definición de la función para cargar y limpiar los datos                                  

    df = pd.read_csv(file_path, low_memory=False)                                                               #Carga del dataframe desde el archivo CSV

    columnas_relevantes = [                                                                                     #Declaración de una variable con los nombres de las columnas útiles
        'gid',                                                                                                  #Columna del identificador del juego
        'team',                                                                                                 #Columna del identificador del equipo
        'date',                                                                                                 #Columna de la fecha del juego
        'vishome',                                                                                              #Columna para indicar localía
        'gametype',                                                                                             #Columna para indicar estancia del juego (pretemporada, temporada regular, postemporada, serie mundial)
        'stattype',                                                                                             #Columna para indicar validez oficial de las estadísticas registradas
        'pbp',                                                                                                  #Columna para indicar si el juego se registró jugada a jugada
        'win',                                                                                                  #Columna para indicar si el equipo ganó el juego
        'lob',                                                                                                  #Columna para número de corredores dejados en base
        'b_pa',                                                                                                 #Columna para número de apariciones en el plato del equipo
        'b_ab',                                                                                                 #Columna para número de turnos al bat del equipo
        'b_r',                                                                                                  #Columna para número de carreras anotadas del equipo
        'b_h',                                                                                                  #Columna para número de hits del equipo
        'b_d',                                                                                                  #Columna para número de dobles del equipo
        'b_t',                                                                                                  #Columna para número de triples del equipo
        'b_hr',                                                                                                 #Columna para número de home runs del equipo
        'b_rbi',                                                                                                #Columna para número de carreras impulsadas del equipo
        'b_sh',                                                                                                 #Columna para número de hits de sacrificio del equipo
        'b_sf',                                                                                                 #Columna para número de elevados de sacrificio del equipo
        'b_hbp',                                                                                                #Columna para número de bases por golpe que recibió el equipo
        'b_w',                                                                                                  #Columna para número de bases por bola que recibió el equipo
        'b_iw',                                                                                                 #Columna para número de bases intencionales que recibió el equipo
        'b_k',                                                                                                  #Columna para número de ponches que recibió el equipo
        'b_sb',                                                                                                 #Columna para número de bases robadas por el equipo
        'b_cs',                                                                                                 #Columna para número de carreras robadas por el equipo
        'b_gdp',                                                                                                #Columna para número de grounded into double plays
        'b_xi',                                                                                                 #Columna para número de interferencias del catcher
        'b_roe',                                                                                                #Columna para número de embasados por un error
        'p_ipouts',                                                                                             #Columna para número de outs por los pitchers con mínimo 3 entradas lanzadas
        'p_noout',                                                                                              #Columna para número de bateadores enfrentados sin out registrado
        'p_bfp',                                                                                                #Columna para número de bateadores enfrentados
        'p_h',                                                                                                  #Columna para número de hits permitidos
        'p_d',                                                                                                  #Columna para número de dobles permitidos
        'p_t',                                                                                                  #Columna para número de triples permitidos
        'p_hr',                                                                                                 #Columna para número de home runs permitidos
        'p_r',                                                                                                  #Columna para número de carreras permitidas
        'p_er',                                                                                                 #Columna para número carreras limpias permitidas
        'p_w',                                                                                                  #Columna para número de bases por bola permitidas
        'p_iw',                                                                                                 #Columna para número de bases intencionales permitidas
        'p_k',                                                                                                  #Columna para número de ponches hechos por los pitchers
        'p_hbp',                                                                                                #Columna para número de bases por golpe permitidas
        'p_wp',                                                                                                 #Columna para número de lanzamientos descontrolados
        'p_bk',                                                                                                 #Columna para número de balks
        'p_sh',                                                                                                 #Columna para número de hits de sacrificio permitidos
        'p_sf',                                                                                                 #Columna para número de elevados de sacrificio permitidos
        'p_sb',                                                                                                 #Columna para número de bases robadas permitidas
        'p_cs',                                                                                                 #Columna para número de atrapados robando
        'p_pb',                                                                                                 #Columna para número de lanzamientos no atrapados por el catcher permitidos
        'd_po',                                                                                                 #Columna para número de putouts
        'd_a',                                                                                                  #Columna para número de asistencias
        'd_e',                                                                                                  #Columna para número de errores hechos
        'd_dp',                                                                                                 #Columna para número de double plays hechos
        'd_tp',                                                                                                 #Columna para número de triple plays hechos
        'd_sb',                                                                                                 #Columna para número de bases robadas permitidas
        'd_cs',                                                                                                 #Columna para número de atrapados robando permitidos
        'inn1',                                                                                                 #Columna para número de carreras anotadas en la primera entrada
        'inn2',                                                                                                 #Columna para número de carreras anotadas en la segunda entrada
        'inn3',                                                                                                 #Columna para número de carreras anotadas en la tercera entrada
        'inn4',                                                                                                 #Columna para número de carreras anotadas en la cuarta entrada
        'inn5',                                                                                                 #Columna para número de carreras anotadas en la quinta entrada
        'inn6',                                                                                                 #Columna para número de carreras anotadas en la sexta entrada
        'inn7',                                                                                                 #Columna para número de carreras anotadas en la séptima entrada
        'inn8',                                                                                                 #Columna para número de carreras anotadas en la octava entrada
        'inn9',                                                                                                 #Columna para número de carreras anotadas en la novena entrada
    ]

    df = df[columnas_relevantes].copy()                                                                         #Selección de columnas relevantes
    df = df[df['stattype'] == 'value']                                                                          #Filtro de juegos con registros oficiales
    df = df[df['pbp'] == 'y']                                                                                   #Filtro de juegos con registros jugada por jugada
    df = df[df['gametype'] == 'regular']                                                                        #Filtro de juegos solo de temporada regular
    df = df.drop(columns=['stattype', 'pbp', 'gametype'])                                                       #Eliminación de columnas que no interesan

    cols_problematicas = [
        'b_w', 'b_k', 'p_w', 'p_k',
        'inn1', 'inn2', 'inn3', 'inn4', 'inn5', 'inn6', 'inn7', 'inn8', 'inn9'
    ]                                                                                                           #Selección de columnas que contienen datos no numéricos

    for col in cols_problematicas:                                                                              #Inicio del bucle sobre todas las columnas a cambiar
        df[col] = pd.to_numeric(df[col], errors='coerce')                                                       #Conversión de la columna a tipo numérico (errors='coerce' transforma cualquier texto raro en NaN)

    df = df.fillna(0)                                                                                           #Cambio de NaN´s a 0 (numérico)
    df['season'] = df['date'] // 10000                                                                          #Creación de la columna "season" a partir de la columna "date"
    df = df.sort_values(by=['date', 'gid'])                                                                     #Ordenamiento del dataframe por fecha e ID de juego

    return df                                                                                                   #Retorno del dataframe limpio

def caracterizacion(df):                                                                                        #Definición de la función para la creación de características     

    grupos = df.groupby(['team', 'season'])                                                                     #Agrupamiento de datos por equipo y por temporada para el cálculo de estadísticas

    columnas_identificadores = ['gid', 'team', 'date', 'vishome', 'season', 'gametype', 'stattype', 'pbp']      #Lista con nombres de las columnas no numéricas

    columnas_para_promedio = [c for c in df.columns if c not in columnas_identificadores]                       #Lista con nombres de las columnas numéricas

    df_promedios = grupos[columnas_para_promedio].cumsum()                                                      #Suma de valores acumulados en cada columna numérica

    df_corrido = df_promedios.groupby([df['team'], df['season']]).shift(1)                                      #Agrupamiento de la suma acumulada por equipo y temporada y corrimiento de datos

    jugados = grupos.cumcount()                                                                                 #Conteo de juegos jugados por equipo y temporada

    juegos_divisibles = jugados.replace(0, 1)                                                                   #Cambio de 0 por 1 para evitar errores durante el cálculo de promedios

    car_finales = []                                                                                            #Creación de lista vacía para las columnas de promedios

    for col in columnas_para_promedio:                                                                          #Inicio del bucle sobre todas las columnas en las se hará promedio
        nombre_promedio = f'prom_{col}'                                                                         #Nombramiento de cada columna de promedio

        if col == 'win':                                                                                        #Condición para encontrar la columna de victorias
            nombre_promedio = 'win_pct'                                                                         #Nombramiento especial a la columna de porcentaje de victorias

        df[nombre_promedio] = df_corrido[col] / juegos_divisibles                                               #Calculo del promedio sobre la columna
        df[nombre_promedio] = df[nombre_promedio].fillna(0)                                                     #Cambio de NaN´s a 0 (numérico)

        car_finales.append(nombre_promedio)                                                                     #Guardado de la nueva columna creada en la lista vacía

    columnas_modelo = ['gid', 'vishome', 'win'] + car_finales                                                   #Lista con columnas necesarias más las columnas de promedios

    df_base = df[columnas_modelo].copy()                                                                        #Creación del nuevo dataframe
    df_base = df_base[jugados > 0]                                                                              #Eliminación del primer juego de cada equipo en cada temporada por falta de datos

    df_local = df_base[df_base['vishome'] == 'h'].add_suffix('_home')                                           #Identificación de datos como local

    df_vis  = df_base[df_base['vishome'] == 'v'].add_suffix('_vis')                                             #Identificación de datos como visitante

    df_modelo = pd.merge(df_local, df_vis, left_on='gid_home', right_on='gid_vis')                              #Unión de los datos de un mismo partido (uno por equipo) en una misma fila
    df_modelo = df_modelo.rename(columns={'win_home': 'target'})                                                #Renombramiento de la columna de victorias como local

    cols_borrar = ['gid_vis', 'win_vis', 'vishome_home', 'vishome_vis']                                         #Lista con nombres de columnas redundantes

    df_modelo = df_modelo.drop(columns=cols_borrar)                                                             #Eliminación de columnas redundantes a causa de la unión
    df_modelo['año_temp'] = df_modelo['gid_home'].astype(str).str[3:7].astype(int)                              #Selección de caracteres que representan el año de la temporada
    df_modelo = df_modelo.sort_values(by=['año_temp', 'gid_home'])                                              #Ordenamiento cronológico de los partidos

    return df_modelo                                                                                            #Retorno del dataframe listo para el modelado