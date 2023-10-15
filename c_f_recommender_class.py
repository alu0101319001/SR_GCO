# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 18:59:24 2023

@author: eduar

Intento de aunar todo en forma de clase
"""

import argparse
import sys
import numpy as np
import pandas as pd

# Visualización del dataframe
pd.options.display.max_columns = 50
pd.options.display.max_columns = None
pd.options.display.max_rows = 50
pd.options.display.max_rows = None

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
ROUND_VALUE = 2
MIN_NEIGHBORS = 2

# La clase que recoje el proceso de un recomendador por el método de filtro colaborativo
class C_F_Recommender:
    
    # Inicializador de la clase - parámetros de entrada
    def __init__(self, file_name, metrics, neighbors, prediction):
        # Atributos de la clase
        # Se dan como parámetros de creación
        self.file_name = file_name
        self.input_metrics = metrics
        self.neighbors = neighbors
        self.input_prediction  = prediction   
        # Se asignaran en el proceso de inicializacion
        self.norm_metrics = None
        self.norm_prediction = None
        self.min_value = None
        self.max_value = None
        self.__utility_matrix = None  
        self.num_col = None
        self.num_rows = None
        self.utility_df = None
        self.nan_positions = None
        self.invalid_items = None
        # Cambian cada ciclo
        self.copy_nan_positions = None
        self.nan_selected = None
        self.sim_df = None
        self.neighbors_selected = None
        self.sol_val = None
        # Se inicializa vacía y se van añadiendo sol cada ciclo
        self.sol_df = None

    def start(self):
        # Procesa toda la información de entrada y crea la estructura inicial de datos
        self.process_input()
        print(self.utility_df)
        print(self.nan_positions)
        print(self.invalid_items)
        print(self.copy_nan_positions)
        # Empieza el ciclo
        while self.copy_nan_positions.size > 0:
            # Selecciona el NaN a calcular
            self.select_nan_to_calculate()
            print(self.nan_selected)
            # Calcula la similaridad
            self.calculate_similarity()
            print(self.sim_df)
            # Selecciona vecinos
            self.select_neighbors()
            print(self.neighbors_selected)
            # Calcula la predicción
            self.calcualte_prediction()
            print(self.sol_val, self.desnormalizar(self.sol_val))
            # Añade la solución
            
            # Decisión: usar valores de NaN calculados o no?
            
        print('finish')
            
        

    def process_input(self):
        self.process_input_file()
        self.process_options()
        self.create_utility_df()
        self.check_min_item_rating()
        self.find_nan_positions() 

        
    def process_input_file(self):
        with open(self.file_name, 'r') as f:
            self.min_value = float(f.readline())
            # print(min)
            self.max_value = float(f.readline())
            # print(max)

        self.__utility_matrix = np.genfromtxt(self.file_name, dtype='f', delimiter=' ', skip_header=2, missing_values="-")
        
    def process_options(self):
        if self.input_metrics == 'pearson': 
            self.norm_metrics = PEARSON
        elif self.input_metrics == 'coseno':
            self.norm_metrics = COSENO
        elif self.input_metrics == 'euclidea':
            self.norm_metrics = EUCLIDEA
        else:
            self.norm_metrics = None
            
        if self.input_prediction == 'simple':
            self.norm_prediction = SIMPLE
        elif self.input_prediction == 'media':
            self.norm_prediction = MEDIA
        else:
            self.norm_prediction = None
            
        
    def find_nan_positions(self):
        self.nan_positions = np.argwhere(self.utility_df.isnull().values)
        self.copy_nan_positions = np.array([pos for pos in self.nan_positions if pos[1] not in self.invalid_items])
        
    def select_nan_to_calculate(self):
        # Calcula el recuente de repeticiones de las filas en la lista
        row_counter = {}
        for pos in self.copy_nan_positions:
            row = pos[0]
            if row in row_counter:
                row_counter[row] += 1
            else:
                row_counter[row] = 1
        
        # Encontrar el mínimo recuento de repeticiones de filas
        min_row_counted = min(row_counter.values())
        
        # Filtrar las filas con el mínimo recuente de repeticiones de filas
        preselect = [pos for pos in self.copy_nan_positions if row_counter[pos[0]] == min_row_counted]
        
        if len(preselect) == 1:
            self.nan_selected = preselect[0]
            self.copy_nan_positions = self.copy_nan_positions[~np.all(self.copy_nan_positions == self.nan_selected, axis=1)]
            return
        # Calcular el recuento de repeticiones de columnas en la lista completa
        col_counter = {}
        for pos in self.copy_nan_positions:
            col = pos[1]
            if col in col_counter:
                col_counter[col] += 1
            else:
                col_counter[col] = 1
      
        # Encontrar el mínimo recuento de repeticiones de columnas en la lista completa
        min_col_counted = min(col_counter.values())
      
        # Filtrar las posiciones con el mínimo recuento de repeticiones de columnas en la preselección
        selection = [pos for pos in preselect if col_counter[pos[1]] == min_col_counted]
      
        if selection:
            # Si hay resultados en la selección, elige el primero
            self.nan_selected = np.array(selection[0])
        else:
            # Si no hay resultados, elige el primero de la preselección
            self.nan_selected = preselect[0]
      
        # Eliminamos la posición seleccionada de la lista
        self.copy_nan_positions = self.copy_nan_positions[~np.all(self.copy_nan_positions == self.nan_selected, axis=1)]
       
    def calculate_similarity(self):
        if self.norm_metrics == PEARSON:
            data_corr = self.pearson()
        elif self.norm_metrics == COSENO:
            data_corr = []
        elif self.norm_metrics == EUCLIDEA:
            data_corr = []
        else:
            # Error
            return -1
        self.create_sim_df(data_corr)
        
    def select_neighbors(self): 
        eliminate_sim = []
        for sim in self.sim_df.index:
            user_label = "User" + str(self.sim_df.at[sim, CORR_COL_1])
            if pd.isna(self.utility_df.at[user_label, self.utility_df.columns[self.nan_selected[1]]]):
                eliminate_sim.append(sim)
        sim_df_copy = self.sim_df.drop(index = eliminate_sim)
        self.neighbors_selected = sim_df_copy.nlargest(self.neighbors, CORR_COL_2)
        
    def calcualte_prediction(self):
        if self.norm_prediction == SIMPLE:
            self.simple_prediction()
        elif self.norm_prediction == MEDIA:
            self.media_prediction()
        else:
            # Error
            return None
        
    def check_min_item_rating(self):
        self.invalid_items = []
        for item in self.utility_df.columns:
            non_nan_values = self.utility_df[item].dropna()
            if len(non_nan_values) < MIN_NEIGHBORS:
                self.invalid_items.append(int(item[4:]))
                

### MÉTRICAS DE SIMILITUD ###
    def pearson(self):
        data_corr = []
        user_selected_label = self.utility_df.index[self.nan_selected[0]]
        calif_user_selected = self.utility_df.loc[user_selected_label].dropna()
        for user in self.utility_df.index:
            if user != user_selected_label:
                calif_user_current = self.utility_df.loc[user].dropna()
                comun_calif = calif_user_selected.index.intersection(calif_user_current.index)
                
                # Pearson
                if len(comun_calif) > 1:
                    corr = np.corrcoef(calif_user_selected[comun_calif], calif_user_current[comun_calif])[0, 1]
                    data_corr.append([user_selected_label[4:], user[4:], round(corr, ROUND_VALUE + 1)])
        return(data_corr)
        
### CALCULO DE PREDICCIONES ###
    def simple_prediction(self):
        top_summation = 0
        bot_summation = 0
        for i in range(len(self.neighbors_selected)):
            v = int(self.neighbors_selected.iloc[i][CORR_COL_1])
            sim_u_v = self.neighbors_selected.iloc[i][CORR_COL_2]
            user_label = "User" + str(v)
            r_v_i = self.utility_df.at[user_label, self.utility_df.columns[self.nan_selected[1]]]
            top_summation += sim_u_v * r_v_i
            bot_summation += abs(sim_u_v)
        self.sol_val = round((top_summation / bot_summation), ROUND_VALUE)
        
    def media_prediction(self):
        top_summation = 0
        bot_summation = 0
        user_u_label = "User" + str(int(self.nan_selected[0]))
        u_mean = round(self.utility_df.loc[user_u_label].mean(skipna = True), ROUND_VALUE)
        for i in range(len(self.neighbors_selected)):
            v = int(self.neighbors_selected.iloc[i][CORR_COL_1])
            sim_u_v = self.neighbors_selected.iloc[i][CORR_COL_2]
            user_v_label = "User" + str(v)
            r_v_i = self.utility_df.at[user_v_label, self.utility_df.columns[self.nan_selected[1]]]
            v_mean = round(self.utility_df.loc[user_v_label].mean(skipna = True), ROUND_VALUE)
            top_summation = top_summation + (sim_u_v * (r_v_i - v_mean))
            bot_summation = abs(bot_summation + sim_u_v)
        sol = u_mean + (top_summation / bot_summation)
        self.sol_val = round(sol, ROUND_VALUE)
        
        
### CREACIONES DE DATAFRAMES ####
    def create_utility_df(self):
        normalized_matrix = self.normalizar(self.__utility_matrix)
        
        self.utility_df = pd.DataFrame(normalized_matrix)
        self.num_col = len(self.utility_df.columns)
        self.num_rows = len(self.utility_df.index) 
        
        new_columns_names = [f'Item{i}' for i in range(self.num_col)]
        self.utility_df.columns = new_columns_names
        
        new_index_name = [f'User{j}' for j in range(self.num_rows)]
        self.utility_df.index = new_index_name
        
    def create_sim_df(self, data_corr):
        self.sim_df = pd.DataFrame(data_corr)
        # Generar los nombres de las filas automáticamente basados en los valores de las columnas 1 y 2
        new_index_names = [f'Sim{int(col[0])}{int(col[1])}' for _, col in self.sim_df.iterrows()]
        self.sim_df.index = new_index_names
        
        # Renombrar las columnas automáticamente
        new_column_names = [CORR_COL_0, CORR_COL_1, CORR_COL_2]
        self.sim_df.columns = new_column_names
    
### NORMALIZACION ###
    # Normaliza los valores entre 0 y 1
    def normalizar(self, val: float):
        return (val-self.min_value)/(self.max_value-self.min_value)
    
    # Desnormaliza
    def desnormalizar(self, val: float):
        sol = val * (self.max_value - self.min_value) + self.min_value
        return sol 
                
        
        
        
        

