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

# Devuelve una matriz con las posiciones de NaN
def find_nan_positions(df): 
    nan_positions = np.argwhere(df.isnull().values)
    return nan_positions

# Crea un data_frame a partir de la matriz de numpy
def create_df(matrix):
    df = pd.DataFrame(matrix)
    return df
        