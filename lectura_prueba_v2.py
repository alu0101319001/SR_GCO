import argparse
import sys
import numpy as np
import pandas as pd
import functions as func
import metrica_func as mf

parser = argparse.ArgumentParser()

parser.add_argument("-r", "--read", type=str, nargs='?', default='utility_matrix_A.txt')
parser.add_argument("-m", "--metrics", type=str, nargs='?', default='pearson')
parser.add_argument("-n", "--neighbors", type=int, nargs='?', default=2)
parser.add_argument("-p", "--prediction", type=str, nargs='?', default='simple')
args = parser.parse_args()

# print(args.read, args.metrics, args.neighbors, args.prediction)

file_name: str = args.read
metrics: str = args.metrics
neighbors: int = args.neighbors
prediction: str = args.prediction

options = func.process_options(metrics, neighbors, prediction)

print("Métrica seleccionada:", metrics, "-", options[0])
print("Número de vecinos:", neighbors)
print("Tipo de predicción:", prediction, "-", options[2])

print("Nombre del fichero:", file_name)

input_data = func.process_input_file(file_name)

print("Valor Mínimo:", input_data[0])
print("Valor Máximo:", input_data[1])
print("Matriz de Utilidad:\n", input_data[2])

matrix_normalizada = func.normalizar(input_data[2], input_data[0], input_data[1])

df = func.create_df(matrix_normalizada)

print("Matriz de Utilidad Normalizada (DF): \n", df)

nan_positions = func.find_nan_positions(df)

print("Posiciones de NaN en el DataFrame: \n", nan_positions)

all_corr = []
for i in range(len(nan_positions)):
    data_corr = mf.pearson(df, nan_positions[i,0])
    all_corr.append(data_corr)
    
df2 = func.create_df(all_corr)

print("Correlación de Pearson de usuarios con NaN: \n", df2)

"""
PRUEBA
"""
