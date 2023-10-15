import argparse
import sys
import numpy as np
import pandas as pd
import functions as func
import metrica_func as mf
import predicción_func as pf

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

utility_df = func.create_utility_df(matrix_normalizada)

print("Matriz de Utilidad Normalizada (DF): \n", utility_df)

nan_positions = func.find_nan_positions(utility_df)

print("\nPosiciones de NaN en el DataFrame: \n", nan_positions)

# Aquí faltaría un proceso de selección del NaN a calcular

# Se elige un NaN y empieza un ciclo (ahora forzado)
nan_selected = nan_positions[0]
print("\nPosición NaN seleccionada:", nan_selected)

# Obtiene el df con la similaridad entre usuarios 
df_sim = mf.calculate_similarity(utility_df, nan_selected, options[0])
print("\nCalculo de similitudes:")    
print(df_sim)      

# Proceso de selección de vecinos
# Devuelve df_sim, con solo las similitudes seleccionadas
neighbors_selected = pf.select_neighbors(df_sim, neighbors)
print("\nVecinos seleccionados:")
print(neighbors_selected)

# Cálculo de predicciones
sol_val = pf.calculate_prediction(utility_df, nan_selected, neighbors_selected, options[2])
print("\nValor de la predicción: ", sol_val)
print("Desnormalizado:", func.desnormalizar(sol_val, input_data[0], input_data[1]))

# Añadir los valores calculados a un data-frame de soluciones
sol_df = func.create_sol_df()

    
# df2 = func.create_df(all_corr)

# print("Correlación de Pearson de usuarios con NaN: \n", df2)

"""
PRUEBA
"""
