"""
Script de las funciones básicas 
"""

import numpy as np
import pandas as pd

# Constantes de selección y error
PEARSON = 1
COSENO = 2
EUCLIDEA = 3
SIMPLE = 1
MEDIA = 2
NONE = 0
ERROR = -1

CORR_COL_0 = "u"
CORR_COL_1 = "v"
CORR_COL_2 = "corr"

# Lee y procesa el fichero de input
def process_input_file(file_name): 
    with open(file_name, 'r') as f:
        min = float(f.readline())
        # print(min)
        max = float(f.readline())
        # print(max)

    input = np.genfromtxt(file_name, dtype='f', delimiter=' ', skip_header=2, missing_values="-")
    input_data = [min, max, input]
    return input_data

# Recoge y normaliza las opciones
def process_options(metrics, neighbors, prediction): 
    norm_metrics = ERROR
    if metrics == 'pearson': 
        norm_metrics = PEARSON
    elif metrics == 'coseno':
        norm_metrics = COSENO
    elif metrics == 'euclidea':
        norm_metrics = EUCLIDEA
    else:
        norm_metrics = NONE
        
    if neighbors <= 0:
        norm_neighbors = ERROR
    else:
        norm_neighbors = neighbors
        
    norm_prediction = ERROR
    if prediction == 'simple':
        norm_prediction = SIMPLE
    elif prediction == 'media':
        norm_prediction = MEDIA
    else:
        norm_prediction = NONE
    
    options = [norm_metrics, norm_neighbors, norm_prediction]
    return options
          
        
# Normaliza los valores entre 0 y 1
def normalizar(val: float, min: float, max: float):
    return (val-min)/(max-min)

# Desnormaliza
def desnormalizar(val: float, min: float, max: float):
    sol = val * (max - min) + min
    return sol 

# Devuelve una matriz con las posiciones de NaN
def find_nan_positions(df): 
    nan_positions = np.argwhere(df.isnull().values)
    return nan_positions

# Crea un data_frame a partir de la matriz de numpy
def create_utility_df(matrix):
    df = pd.DataFrame(matrix)
    # Renombrar las columans automáticamente
    new_column_names = [f'Item{i}' for i in range(len(df.columns))]
    df.columns = new_column_names
    
    # Renombrar las filas automáticamente
    new_index_names = [f'User{j}' for j in range(len(df.index))]
    df.index = new_index_names
    return df

def create_sim_df(matrix):
    df = pd.DataFrame(matrix)
    # Generar los nombres de las filas automáticamente basados en los valores de las columnas 1 y 2
    new_index_names = [f'Sim{int(col[0])}{int(col[1])}' for _, col in df.iterrows()]
    
    # Renombrar las filas
    df.index = new_index_names
    
    # Renombrar las columnas automáticamente
    new_column_names = [CORR_COL_0, CORR_COL_1, CORR_COL_2]
    df.columns = new_column_names
    
    return df

def create_sol_df():
    sol_df = pd.DataFrame(columns=['Pos_NaN', 'Predic_Value'])
    return sol_df

        