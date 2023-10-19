# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 18:59:24 2023

@author: eduar

Intento de aunar todo en forma de clase
"""

import argparse
import sys
import os
import logging
import numpy as np
import pandas as pd

# Configura la configuración del registro
logging.basicConfig(filename='tracking.log', level=logging.INFO, format='%(levelname)s - %(message)s')

# Visualización del dataframe
pd.options.display.max_columns = 50
pd.options.display.max_columns = None
pd.options.display.max_rows = 50
pd.options.display.max_rows = None

# Constantes de selección y error
PEARSON = 1
COSINE = 2
EUCLIDEAN = 3
SIMPLE = 1
MEDIA = 2
NONE = 0
ERROR = -1
CORR_COL_0 = "u"
CORR_COL_1 = "v"
CORR_COL_2 = "sim"
ROUND_VALUE = 2
MIN_NEIGHBORS = 2
SOL_COL_0 = 'NaN_Pos'
SOL_COL_1 = 'Sol_Val'
SOL_COL_2 = 'Desn_Sol_Val'

# La clase que recoje el proceso de un recomendador por el método de filtro colaborativo
class C_F_Recommender:
    
    # Inicializador de la clase - parámetros de entrada
    def __init__(self, file_name, metrics, neighbors, prediction, 
                 output_mode="console", use_calculated_nan=True):
        # Atributos de la clase
        # Se dan como parámetros de creación
        self.file_name = file_name
        self.input_metrics = metrics
        self.neighbors = neighbors
        self.input_prediction  = prediction   
        self.output_mode = output_mode
        self.use_calculated_nan = use_calculated_nan
        # Se asginan en esta inicialización
        self.output_file = file_name.replace(".txt", "_output.txt")
        # Se asignaran en la función start
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
        try:
            self.restore_output_file()
            # Procesa toda la información de entrada y crea la estructura inicial de datos
            self.process_input()
            self.create_sol_df()
    
            self.log(self.utility_df)
            self.log(self.nan_positions)
            self.log(self.invalid_items)
            self.log(self.copy_nan_positions)
            # Empieza el ciclo
            while self.copy_nan_positions.size > 0:
                # Selecciona el NaN a calcular
                self.select_nan_to_calculate()
                self.log(self.nan_selected)
                # Calcula la similaridad
                self.calculate_similarity()
                self.log(self.sim_df)
                # Selecciona vecinos
                self.select_neighbors()
                self.log(self.neighbors_selected)
                # Calcula la predicción
                self.calcualte_prediction()
                sol = "Valor: " + str(self.sol_val)
                sol += " | Valor Desnormalizado: " + str(self.desnormalizar(self.sol_val))
                self.log(sol)
                # Añade la solución
                self.add_solution()
                # Decisión: usar valores de NaN calculados o no?
                if self.use_calculated_nan:
                    self.add_calculated_NaN()
                    self.log("NUEVA UTILITY DF")
                    self.log(self.utility_df)
                    
                
            self.log(self.sol_df)
            self.log('finish')
            
        except Exception:
            return ERROR
        except:
            print("Something else went wrong")
          
        
    def process_input(self):
        try:
            self.process_input_file()
            self.check_utility_matrix_values()
            self.process_options()
            self.create_utility_df()
            self.check_utility_df_values()
            self.check_min_item_rating()
            self.find_nan_positions() 
            
        except Exception:
            raise

        
    def process_input_file(self):
        try:
            logging.info(f'Archivo a procesar: {self.file_name}')
            with open(self.file_name, 'r') as f:
                self.min_value = float(f.readline())
                self.max_value = float(f.readline())
                if self.min_value >= self.max_value:  
                    raise ValueError(f'The minimum value cannot exceed or equal the maximum value: Min({self.min_value}) - Max({self.max_value})')
            self.__utility_matrix = np.genfromtxt(self.file_name, dtype='f', delimiter=' ', skip_header=2, missing_values="-")
            logging.info(f'Procesamiento:\nValor mínimo:{self.min_value}\nValor Máximo:{self.max_value}\nUtility Matrix:\n{self.__utility_matrix}')

            
        except FileNotFoundError as fnf_error:
            self.log(fnf_error)
            self.log(f"Explanation: Cannot load file {self.file_name}")
            raise
        
    def process_options(self):
        try:
            if self.input_metrics == 'pearson': 
                self.norm_metrics = PEARSON
            elif self.input_metrics == 'cosine':
                self.norm_metrics = COSINE
            elif self.input_metrics == 'euclidean':
                self.norm_metrics = EUCLIDEAN
            else:
                raise ValueError(f'The value of input_metrics {self.input_metrics} is not recognized')
                
            logging.info(f'Metrica escogida: {self.input_metrics} - {self.norm_metrics}')
            
            if self.input_prediction == 'simple':
                self.norm_prediction = SIMPLE
            elif self.input_prediction == 'media':
                self.norm_prediction = MEDIA
            else:
                raise ValueError(f'The value of input_prediction {self.input_prediction} is not recognized')
                
            logging.info(f'Predicción escogida: {self.input_prediction} - {self.norm_prediction}')

        except ValueError as ve:
            self.log(ve)
            self.log("Explanation: Review the values given by parameters")
            raise
            
        
    def find_nan_positions(self):
        try:
            self.nan_positions = np.argwhere(self.utility_df.isnull().values)
            self.copy_nan_positions = np.array([pos for pos in self.nan_positions if pos[1] not in self.invalid_items])
        except Exception as error:
            self.log(f'Exception in find_nan_positions: {error}')
            raise
        
    def select_nan_to_calculate(self):
        try:
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
            
            # Filtrar las filas con el mínimo recuento de repeticiones de filas
            preselect = [pos for pos in self.copy_nan_positions if row_counter[pos[0]] == min_row_counted]
            
            if len(preselect) == 0:
                raise Exception(f'No unknow value to choose is viable. Preselect: {preselect}')
                
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
            
        except Exception as error:
            self.log(f'Exception in select_nan_to_calculate: {error}')
            raise
       
    def calculate_similarity(self):
        try:
            if self.norm_metrics == PEARSON:
                data_corr = self.pearson()
            elif self.norm_metrics == COSINE:
                data_corr = self.cosine()
            elif self.norm_metrics == EUCLIDEAN:
                data_corr = self.euclidean()
            else:
                raise ValueError(f'The value of metrics {self.norm_metrics} is not recognized')
                
            self.create_sim_df(data_corr)
            
        except ValueError as ve:
            self.log(f'Exception in calculate_similarity: {ve}')
            self.log('Explanation: Something happened in the normalization of metrics')
            raise
        except Exception:
            raise
        
    def select_neighbors(self): 
        eliminate_sim = []
        for sim in self.sim_df.index:
            user_label = "User" + str(self.sim_df.at[sim, CORR_COL_1])
            # Elimina aquellos usuarios que no han valorado el item en cuestión
            if pd.isna(self.utility_df.at[user_label, self.utility_df.columns[self.nan_selected[1]]]):
                eliminate_sim.append(sim)
            # Elimina aquellas similitudes con valor negativo o 0
            elif self.sim_df.at[sim, CORR_COL_2] <= 0:
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
        try:
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
                    else:
                        logging.warning(f'There are no common qualifications with which to calculate the correlation value: u:{user_selected_label} - v:{user} - Comun:{comun_calif}')
            return(data_corr)
        
        except Exception as error:
            self.log(f'Exception in pearson: {error}')
            raise
    
    def cosine(self):
        data_corr = []
        user_selected_label = self.utility_df.index[self.nan_selected[0]]
        calif_user_selected = self.utility_df.loc[user_selected_label].dropna()
        for user in self.utility_df.index:
            if user != user_selected_label:
                calif_user_current = self.utility_df.loc[user].dropna()
                comun_calif = calif_user_selected.index.intersection(calif_user_current.index)
                
                # Cosine
                # Prdocuto escalar entre los vectores de calificaciones
                dot_product = np.dot(calif_user_selected[comun_calif], calif_user_current[comun_calif])
                # Calcula las normas de los vectores
                norm_user_selected = np.linalg.norm(calif_user_selected[comun_calif])
                norm_user_current = np.linalg.norm(calif_user_current[comun_calif])
                # Evita divisiones por cero
                if norm_user_selected == 0 or norm_user_current == 0:
                    return 0.0
                # Calcula la similitud del coseno
                similarity = dot_product / (norm_user_selected * norm_user_current)
                data_corr.append([user_selected_label[4:], user[4:], round(similarity, ROUND_VALUE + 1)])
        return(data_corr)
    
    def euclidean(self):
        data_corr = []
        user_selected_label = self.utility_df.index[self.nan_selected[0]]
        calif_user_selected = self.utility_df.loc[user_selected_label].dropna()
        for user in self.utility_df.index:
            if user != user_selected_label:
                calif_user_current = self.utility_df.loc[user].dropna()
                comun_calif = calif_user_selected.index.intersection(calif_user_current.index)
                
                # Euclidean
                euclidean_dist = np.sqrt(np.nansum((calif_user_selected[comun_calif] - calif_user_current[comun_calif])**2))
                self.log(euclidean_dist)
                similarity = 1 / (1 + euclidean_dist)
                data_corr.append([user_selected_label[4:], user[4:], round(similarity, ROUND_VALUE + 1)])
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
        
        
### CREACIONES y MODIFICACIONES DE DATAFRAMES ####
    def create_utility_df(self):
        normalized_matrix = self.normalizar(self.__utility_matrix)
        
        self.utility_df = pd.DataFrame(normalized_matrix)
        self.num_col = len(self.utility_df.columns)
        self.num_rows = len(self.utility_df.index) 
        
        new_columns_names = [f'Item{i}' for i in range(self.num_col)]
        self.utility_df.columns = new_columns_names
        
        new_index_name = [f'User{j}' for j in range(self.num_rows)]
        self.utility_df.index = new_index_name
        
        logging.info(f'Utility_df creada {self.num_col}x{self.num_rows}:\n{self.utility_df}')
        
    def create_sim_df(self, data_corr):
        self.sim_df = pd.DataFrame(data_corr)
        # Generar los nombres de las filas automáticamente basados en los valores de las columnas 1 y 2
        new_index_names = [f'Sim{int(col[0])}{int(col[1])}' for _, col in self.sim_df.iterrows()]
        self.sim_df.index = new_index_names
        
        # Renombrar las columnas automáticamente
        new_column_names = [CORR_COL_0, CORR_COL_1, CORR_COL_2]
        self.sim_df.columns = new_column_names
        
    def create_sol_df(self):
        col = [SOL_COL_0, SOL_COL_1, SOL_COL_2]
        self.sol_df = pd.DataFrame(columns=col)
        
                        
    def add_solution(self):
        temp_df = pd.DataFrame([[self.nan_selected, self.sol_val, self.desnormalizar(self.sol_val)]],
                               columns=[SOL_COL_0, SOL_COL_1, SOL_COL_2])
        self.sol_df = pd.concat([self.sol_df, temp_df], ignore_index=True)
        
    def add_calculated_NaN(self):
        # Actualiza el valor en utility_df
        user_index = self.utility_df.index[self.nan_selected[0]]
        item_column = self.utility_df.columns[self.nan_selected[1]]
        self.utility_df.at[user_index, item_column] = self.sol_val
    
### NORMALIZACION ###
    # Normaliza los valores entre 0 y 1
    def normalizar(self, val: float):
        return (val-self.min_value)/(self.max_value-self.min_value)
    
    # Desnormaliza
    def desnormalizar(self, val: float):
        sol = val * (self.max_value - self.min_value) + self.min_value
        return round(sol, ROUND_VALUE) 
    
### VISUALIZACION ###
    def log(self, msg):
        if self.output_mode == 'console':
            print(msg)
        elif self.output_mode == 'file':
            with open(self.output_file, "a") as f:
                f.write(str(msg) + "\n")
    
    def restore_output_file(self):
            if os.path.exists(self.output_file):
                os.remove(self.output_file)
                
### CONTROL DE ERRORES
    def check_utility_matrix_values(self):
        try:
            check = np.all((np.isnan(self.__utility_matrix) | 
                             ((self.__utility_matrix >= self.min_value) 
                              & (self.__utility_matrix <= self.max_value))))
            if not check:
                 raise ValueError(f'Not all values ​​in the utility matrix are within the set range:\n{self.__utility_matrix}')
        except ValueError as ve:
            self.log(f'Exception in check_utility_matrix_values: {ve}')
            self.log('Explanation: Check the file input data.')
            raise
    
    def check_utility_df_values(self):
        try:
            check = np.all((np.isnan(self.utility_df) | 
                             ((self.utility_df >= 0.0) 
                              & (self.utility_df <= 1.0))))
            if not check:
                 raise ValueError(f'Not all values ​​in the utility dataframe are normalized:\n{self.utility_df}')
        except ValueError as ve:
            self.log(f'Exception in check_utility_df_values: {ve}')
            self.log('Explanation: Something went wrong in normalization process.')
            raise
         
        
        
        
        

