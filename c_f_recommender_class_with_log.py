# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 18:59:24 2023

@author: eduar

Clase para gestionar todo el proceso del método de filtrado colaborativo
"""

import argparse
import sys
import os
import logging
import numpy as np
import pandas as pd
import openpyxl as oxl

# Configura la configuración del registro
logging.basicConfig(filename='tracking.log', level=logging.INFO, filemode='w',format='%(message)s')

# Visualización del dataframe
pd.options.display.max_columns = None
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
MIN_NEIGHBORS = 1
SOL_COL_0 = 'NaN_Pos'
SOL_COL_1 = 'User'
SOL_COL_2 = 'Sol_Val'
SOL_COL_3 = 'Desn_Val'
SOL_COL_4 = 'Neighbors_Selected'
RESULT_XLSX = 'results.xlsx'


# La clase que recoje el proceso de un recomendador por el método de filtro colaborativo
class C_F_Recommender:
    
    # Inicializador de la clase - parámetros de entrada
    def __init__(self, file_name, metrics, neighbors, prediction, use_calculated_nan=True):
        # Atributos de la clase
        # Se dan como parámetros de creación
        self.file_name = file_name
        self.input_metrics = metrics
        self.neighbors = neighbors
        self.input_prediction  = prediction   
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
        # Para el recalculate
        self.new_nan_positions = None
        # Usados para la visualización final
        self.complete_sim_df = None
        # Control de errores
        self.not_valid = False


    def start(self):
        try:
            # Procesa toda la información de entrada y crea la estructura inicial de datos
            if not self.process_input():
                return
            self.create_sol_df()
            self.create_complete_sim_df()
    
            # Empieza el ciclo
            count = 0
            logging.info('### START ###')
            while self.copy_nan_positions.size > 0:
                logging.info(f'---> Ciclo {count}')
                count = count + 1
                # Restablece not_valid
                self.not_valid = False
                # Selecciona el NaN a calcular
                self.select_nan_to_calculate()
                # Calcula la similaridad
                self.calculate_similarity()
                # Añade al df con la similaridad completa
                self.add_sims()
                # Selecciona vecinos
                self.select_neighbors()
                # Si no existen vecinos suficientes, no es posible resolver el valor
                if len(self.neighbors_selected) < self.neighbors:
                    
                    if len(self.neighbors_selected) < MIN_NEIGHBORS:
                        logging.warning(f'There are not enough viable neighbors to predict: Min: {MIN_NEIGHBORS} - Valid: {len(self.neighbors_selected)}')
                        self.incalculable_nan_value()
                        continue
                    else:
                        logging.warning(f'There are not enough neighbors as those considered, but the minimum is met: Considered: {self.neighbors} - Valid: {len(self.neighbors_selected)}')
                        logging.warning('The value will be calculated with the available neighbors, but will not be considered a valid value to add to the utility dataframe.')
                        self.not_valid = True
                        
                # Calcula la predicción
                self.calcualte_prediction()
                self.add_solution()
                
                if self.not_valid == False:
                    # Decisión: usar valores de NaN calculados o no?
                    if self.use_calculated_nan:
                        logging.info('Sol_Val added to utility_df for calculate the new nan_pos')
                        self.add_calculated_NaN()
                    else:
                        logging.info('Sol_Val will not be added to utility_df for calculate new nan_pos')
                    
            self.incalculable_nan_list()
            logging.info('#### FINISH START ####\n')
            logging.info(f'-- Complete Sim DF --\n{self.complete_sim_df}')
            logging.info(f'-- Solution DF --\n{self.sol_df}')
            logging.info(f'-- Utility DF --\n{self.utility_df}\n')
            logging.info(f'-- NaN Positions --\n{self.nan_positions}')
            logging.info(f'-- Incalculable NaN List --\n{self.new_nan_positions}')
            if not self.check_same_nan_positions(self.nan_positions, self.new_nan_positions):
                logging.info('### GO RECALCULATE ###')
                self.recalculate()
            
            logging.info('### FINAL RESULT ###')
            logging.info(f'-- Complete Sim DF --\n{self.complete_sim_df}')
            logging.info(f'-- Solution DF --\n{self.sol_df}')
            logging.info(f'-- Utility DF --\n{self.utility_df}\n')
            logging.info('#### FINISH ALL ####\n')
            self.final_console_output()
            self.final_xlsx_output()
            return [self.utility_df, self.complete_sim_df, self.sol_df]
            
        except Exception:
            return ERROR
        except:
            print("Something else went wrong")
          
    def recalculate(self):
        try:
            self.check_min_item_rating()
            if not self.find_nan_positions():
                self.incalculable_nan_list()
                logging.info('### FINISH RECALCULATE - Not nan_pos ###')
                return # END OF RECALCULATE
            count = 0
            while self.copy_nan_positions.size > 0:
                logging.info(f'---> Ciclo {count}')
                count = count + 1
                self.not_valid = False
                self.select_nan_to_calculate()
                self.calculate_similarity()
                self.add_sims()
                self.select_neighbors()
                if len(self.neighbors_selected) < self.neighbors:
                    if len(self.neighbors_selected) < MIN_NEIGHBORS:
                        logging.warning(f'There are not enough viable neighbors to predict: Min: {MIN_NEIGHBORS} - Valid: {len(self.neighbors_selected)}')
                        self.incalculable_nan_value()
                        continue
                    else:
                        logging.warning(f'There are not enough neighbors as those considered, but the minimum is met: Considered: {self.neighbors} - Valid: {len(self.neighbors_selected)}')
                        logging.warning('The value will be calculated with the available neighbors, but will not be considered a valid value to add to the utility dataframe.')
                        self.not_valid = True
                self.calcualte_prediction()
                self.add_solution()
                if self.not_valid == False:
                    # Decisión: usar valores de NaN calculados o no?
                    if self.use_calculated_nan:
                        logging.info('Sol_Val added to utility_df for calculate the new nan_pos')
                        self.add_calculated_NaN()
                    else:
                        logging.info('Sol_Val will not be added to utility_df for calculate new nan_pos')
                self.incalculable_nan_list()
            logging.info('#### FINISH RECALCULATE ####\n')
            logging.info(f'-- Complete Sim DF --\n{self.complete_sim_df}')
            logging.info(f'-- Solution DF --\n{self.sol_df}')
            logging.info(f'-- Utility DF --\n{self.utility_df}\n')
            logging.info(f'-- NaN Positions --\n{self.nan_positions}')
            logging.info(f'-- Incalculable NaN List --\n{self.new_nan_positions}')
            if not self.check_same_nan_positions(self.nan_positions, self.new_nan_positions):
                logging.info('### GO RECALCULATE ###')
                self.recalculate()
            
            self.incalculable_nan_list()
            logging.info('### FINISH RECALCULATE - ending ###')
            return # END OF RECALCULATE
        
        except Exception:
            raise
        
    def process_input(self):
        try:
            self.process_input_file()
            self.check_utility_matrix_values()
            self.process_options()
            self.create_utility_df()
            self.check_utility_df_values()
            self.check_min_item_rating()
            if not self.find_nan_positions():
                return False
            return True
            
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
            logging.info(f'#### PROCESAMIENTO ####\n\tValor mínimo:{self.min_value}\n\tValor Máximo:{self.max_value}\n\t-- Utility Matrix --\n{self.__utility_matrix}')

            
        except FileNotFoundError as fnf_error:
            print(fnf_error)
            print(f"Explanation: Cannot load file {self.file_name}")
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
            print(ve)
            print("Explanation: Review the values given by parameters")
            raise
            
        
    def find_nan_positions(self):
        try:
            self.nan_positions = np.argwhere(self.utility_df.isnull().values)
            if len(self.nan_positions) == 0:
                return False
            
            logging.info(f'- NaN Positions -\n{self.nan_positions}')
            
            self.copy_nan_positions = np.array([pos for pos in self.nan_positions if pos[1] not in self.invalid_items])
            if len(self.copy_nan_positions) == 0:
                return False
            
            logging.info(f'- Copy NaN Positions (1º Filter) -\n{self.copy_nan_positions}\n')            
            
            return True
        
        except Exception as error:
            print(f'Exception in find_nan_positions: {error}')
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
                logging.info(f'NaN selected --> {self.nan_selected}')
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
            logging.info(f'NaN selected --> {self.nan_selected}')
            
        except Exception as error:
            print(f'Exception in select_nan_to_calculate: {error}')
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
            logging.info(f' -- Similarity Calculated --\n{self.sim_df}')
            
        except ValueError as ve:
            print(f'Exception in calculate_similarity: {ve}')
            print('Explanation: Something happened in the normalization of metrics')
            raise
        except Exception:
            raise
        
    def select_neighbors(self): 
        try:
            logging.info(f'## Select Neighbors Process ##')
            self.neighbors_selected = []
            eliminate_sim = []
            for sim in self.sim_df.index:
                user_label = "User" + str(self.sim_df.at[sim, CORR_COL_1])
                # Elimina aquellos usuarios que no han valorado el item en cuestión
                if pd.isna(self.utility_df.at[user_label, self.utility_df.columns[self.nan_selected[1]]]):
                    eliminate_sim.append(sim)
                # Elimina aquellas similitudes con valor negativo, 0 o NaN
                elif ((self.sim_df.at[sim, CORR_COL_2] <= 0) | (np.isnan(self.sim_df.at[sim, CORR_COL_2]))):
                    eliminate_sim.append(sim)
            logging.info(f'Sim eliminated: {eliminate_sim}')
            sim_df_copy = self.sim_df.drop(index = eliminate_sim)
            self.neighbors_selected = sim_df_copy.nlargest(self.neighbors, CORR_COL_2)
            logging.info(f' -- Neighbors Selected --\n{self.neighbors_selected}')
        
        except Exception as error:
            print(f'Exception in select_neighbors: {error}')
            raise
        
    def calcualte_prediction(self):
        try:
            if self.norm_prediction == SIMPLE:
                self.simple_prediction()
            elif self.norm_prediction == MEDIA:
                self.media_prediction()
            else:
                raise ValueError(f'The value of prediction {self.norm_prediction} is not recognized')
                
        except ValueError as ve:
            print(f'Exception in calculate_prediction: {ve}')
            print('Explanation: Something happened in the normalization of prediction')
            raise
        except Exception:
            raise
        
        
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
                        corr = np.nan
                        data_corr.append([user_selected_label[4:], user[4:], corr])
            return(data_corr)
        
        except Exception as error:
            print(f'Exception in pearson: {error}')
            raise
    
    def cosine(self):
        try:
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
                        similarity = np.nan
                        data_corr.append([user_selected_label[4:], user[4:], similarity])
                        return(data_corr)
                    # Calcula la similitud del coseno
                    similarity = dot_product / (norm_user_selected * norm_user_current)
                    data_corr.append([user_selected_label[4:], user[4:], round(similarity, ROUND_VALUE + 1)])
            return(data_corr)
        
        except Exception as error:
            print(f'Exception in cosine: {error}')
            raise
    
    def euclidean(self):
        try:
            data_corr = []
            user_selected_label = self.utility_df.index[self.nan_selected[0]]
            calif_user_selected = self.utility_df.loc[user_selected_label].dropna()
            for user in self.utility_df.index:
                if user != user_selected_label:
                    calif_user_current = self.utility_df.loc[user].dropna()
                    comun_calif = calif_user_selected.index.intersection(calif_user_current.index)
                    
                    # Euclidean
                    euclidean_dist = np.sqrt(np.nansum((calif_user_selected[comun_calif] - calif_user_current[comun_calif])**2))
                    similarity = 1 / (1 + euclidean_dist)
                    data_corr.append([user_selected_label[4:], user[4:], round(similarity, ROUND_VALUE + 1)])
            return(data_corr)
        
        except Exception as error:
            print(f'Exception in euclidean: {error}')
            raise
    
### CALCULO DE PREDICCIONES ###
    def simple_prediction(self):
        try:
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
            self.limit_sol_val()

            logging.info(f': : Sol_Val calculated by simple prediction: {self.sol_val} - {self.desnormalizar(self.sol_val)}')
            
        except Exception as error:
            print(f'Exception in simple_prediction: {error}')
            raise
        
        
    def media_prediction(self):
        try:
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
            self.limit_sol_val()

            logging.info(f': : Sol_Val calculated by media prediction: {self.sol_val} - {self.desnormalizar(self.sol_val)}')

        
        except Exception as error:
            print(f'Exception in media_prediction: {error}')
            raise
        
        
### CREACIONES y MODIFICACIONES DE DATAFRAMES ####
    def create_utility_df(self):
        try:
            normalized_matrix = self.normalizar(self.__utility_matrix)
            
            self.utility_df = pd.DataFrame(normalized_matrix)
            self.num_col = len(self.utility_df.columns)
            self.num_rows = len(self.utility_df.index) 
            
            new_columns_names = [f'Item{i}' for i in range(self.num_col)]
            self.utility_df.columns = new_columns_names
            
            new_index_name = [f'User{j}' for j in range(self.num_rows)]
            self.utility_df.index = new_index_name
            
            logging.info(f'-- Utility_df creada {self.num_col}x{self.num_rows} --\n{self.utility_df}')
        
        except Exception as error:
            print(f'Exception in create_utility_df: {error}')
            raise
        
    def add_calculated_NaN(self):
        try:
            # Actualiza el valor en utility_df
            user_index = self.utility_df.index[self.nan_selected[0]]
            item_column = self.utility_df.columns[self.nan_selected[1]]
            self.utility_df.at[user_index, item_column] = np.float32(self.sol_val)

        except Exception as error:
            print(f'Exception in add_calculated_NaN: {error}')
            raise
    
    def add_solution_values_to_utility_df(self):
        try:
            for index, row in self.sol_df.iterrows():
                if((row[SOL_COL_2] != np.nan) & (len(row[SOL_COL_4]) >= self.neighbors)):
                    nan_pos = row[SOL_COL_0]
                    user = "User" + str(nan_pos[0])
                    item = "Item" + str(nan_pos[1])
                    value = np.float32(row[SOL_COL_2])
                    self.utility_df.at[user, item] = value
            
        except Exception as error:
            print(f'Exception in add_solution_values: {error}')
            raise
                
    def create_sim_df(self, data_corr):
        try:
            self.sim_df = pd.DataFrame(data_corr)
            # Generar los nombres de las filas automáticamente basados en los valores de las columnas 1 y 2
            new_index_names = [f'Sim{int(col[0])}{int(col[1])}' for _, col in self.sim_df.iterrows()]
            self.sim_df.index = new_index_names
            
            # Renombrar las columnas automáticamente
            new_column_names = [CORR_COL_0, CORR_COL_1, CORR_COL_2]
            self.sim_df.columns = new_column_names

        except Exception as error:
            print(f'Exception in create_sim_df: {error}')
            raise
        
    def create_sol_df(self):
        try:
            col = [SOL_COL_0, SOL_COL_1, SOL_COL_2, SOL_COL_3, SOL_COL_4]
            self.sol_df = pd.DataFrame(columns=col)
        except Exception as error:
            print(f'Exception in create_sol_df: {error}')
            raise
       
                       
    def add_solution(self):
        try:
            if self.not_valid:
                value = str(self.sol_val) + '*'
            else:
                value = np.float32(self.sol_val)
            
            for index, row in self.sol_df.iterrows():
                if np.array_equal(row[SOL_COL_0], self.nan_selected):
                    self.sol_df.at[index, SOL_COL_2] = value
                    self.sol_df.at[index, SOL_COL_3] = self.desnormalizar(self.sol_val)
                    self.sol_df.at[index, SOL_COL_4] = self.list_neighbors_selected()
                    return
                
            list_neighbors_selected = self.list_neighbors_selected()
            user_selected = 'User' + str(self.nan_selected[0])
            
            temp_df = pd.DataFrame([[self.nan_selected, user_selected, value, self.desnormalizar(self.sol_val), list_neighbors_selected]],
                                   columns=[SOL_COL_0, SOL_COL_1, SOL_COL_2, SOL_COL_3, SOL_COL_4])
            
            if self.sol_df.empty:
                self.sol_df = temp_df
            else:
                self.sol_df = pd.concat([self.sol_df, temp_df], ignore_index=True)
                
        except Exception as error:
            print(f'Exception in add_solution: {error}')
            raise
        
    def list_neighbors_selected(self):
        try:
            list_neighbors_selected = []
            for index, sim in self.neighbors_selected.iterrows():
                list_neighbors_selected.append(sim[CORR_COL_1])
            return list_neighbors_selected
        
        except Exception as error:
            print(f'Exception in list_neighbors_selected: {error}')
            raise
    
    def create_complete_sim_df(self):
        try:
            name_col_row = [f'User{j}' for j in range(self.num_rows)]
            self.complete_sim_df = pd.DataFrame(index=name_col_row, columns=name_col_row)
            
            for i in range(self.num_rows):
                self.complete_sim_df.at[f'User{i}', f'User{i}'] = '#'
                            
        except Exception as error:
            print(f'Exception in create_complete_sim_df: {error}')
            raise
    
    def add_sims(self):
        try:
            user_u = self.sim_df.iloc[0,0]
            user_u_label = 'User' + user_u
            for index, row in self.sim_df.iterrows():
                user_v = row[CORR_COL_1]
                user_v_label = 'User' + user_v
                self.complete_sim_df.at[user_u_label, user_v_label] = row[CORR_COL_2]
                
        except Exception as error:
            print(f'Exception in add_sims: {error}')
            raise
    
### NORMALIZACION ###
    # Normaliza los valores entre 0 y 1
    def normalizar(self, val: float):
        try:
            return (val-self.min_value)/(self.max_value-self.min_value)

        except Exception as error:
            print(f'Exception in normalizar: {error}')
            raise
    
    # Desnormaliza
    def desnormalizar(self, val: float):
        try:
            sol = val * (self.max_value - self.min_value) + self.min_value
            return round(sol, ROUND_VALUE) 

        except Exception as error:
            print(f'Exception in desnormalziar: {error}')
                    
### CONTROL DE ERRORES
    def check_utility_matrix_values(self):
        try:
            check = np.all((np.isnan(self.__utility_matrix) | 
                             ((self.__utility_matrix >= self.min_value) 
                              & (self.__utility_matrix <= self.max_value))))
            if not check:
                 raise ValueError(f'Not all values ​​in the utility matrix are within the set range:\n{self.__utility_matrix}')
        except ValueError as ve:
            print(f'Exception in check_utility_matrix_values: {ve}')
            print('Explanation: Check the file input data.')
            raise
    
    def check_utility_df_values(self):
        try:
            check = np.all((np.isnan(self.utility_df) | 
                             ((self.utility_df >= 0.0) 
                              & (self.utility_df <= 1.0))))
            if not check:
                 raise ValueError(f'Not all values ​​in the utility dataframe are normalized:\n{self.utility_df}')
        except ValueError as ve:
            print(f'Exception in check_utility_df_values: {ve}')
            print('Explanation: Something went wrong in normalization process.')
            raise
            
    def check_same_nan_positions(self, list1, list2):
        try:
            if len(list1) != len(list2):
                return False
            
            for arr1, arr2 in zip(list1, list2):
                if not np.array_equal(arr1, arr2):
                    return False
    
            return True            
        except Exception as error:
            print(f'Exception in check_same_nan_positions: {error}')
            raise
            
            
    def check_min_item_rating(self):
        try:
            self.invalid_items = []
            for item in self.utility_df.columns:
                non_nan_values = self.utility_df[item].dropna()
                if len(non_nan_values) < MIN_NEIGHBORS:
                    self.invalid_items.append(int(item[4:]))
            
            logging.info(f'Invalid items detected: {self.invalid_items}')
        
        except Exception as error:
            print(f'Exception in check_min_item_rating: {error}')
                
                
    def limit_sol_val(self):
        try:
            if self.sol_val < 0.0:
                self.sol_val = 0.0
            elif self.sol_val > 1.0:
                self.sol_val = 1.0
            return self.sol_val
        
        except Exception as error:
            print(f'Exception in limit_sol_val: {error}')
            raise
    
    def incalculable_nan_value(self):
        try:
            self.sol_val = np.nan
            self.add_solution()
            logging.info(': : Sol_Val stay NaN')
            
        except Exception as error:
            print(f'Exception in incalculable_nan_value: {error}')
            raise
            
    def incalculable_nan_list(self):
        try:
            if not self.use_calculated_nan:
                self.add_solution_values_to_utility_df()
            
            self.new_nan_positions = np.argwhere(self.utility_df.isnull().values)
            return self.new_nan_positions
                               
        except Exception as error:
            print(f'Exception en incalculable_nan_list: {error}')
            raise

### VISUALIZACIÓN    
    def final_console_output(self):
        try:
            input_data = "### INPUT ###\n"
            input_data += f"File : {self.file_name}\n"
            input_data += f"Metric: {self.input_metrics}\n"
            input_data += f"Neighbors: {self.neighbors}\n"
            input_data += f"Prediction: {self.input_prediction}\n"
            input_data += f"Use calculated NaN values: {self.use_calculated_nan}\n\n"
            print(input_data)

            output_data = "### OUTPUT ###\n"
            pd.options.display.max_columns = 7
            pd.options.display.max_rows = 20
            output_data += f"-- Utility Matrix --\n{self.utility_df}\n\n"
            output_data += f"-- Similarity --\n{self.complete_sim_df}\n\n"
            output_data += f"-- Solution Table --\n{self.sol_df}\n\n"
            pd.options.display.max_columns = None
            pd.options.display.max_rows = None
            output_data += f"-- Incalculable --\n{self.new_nan_positions}\n"
            print(output_data)

            print("Complete tracking in the generated tracking.log file")
            print(f"The results have been exported to excel: {RESULT_XLSX}")

        
        except Exception as error:
            print(f"Exception in final_console_output: {error}")
            raise

    def final_xlsx_output(self):
        try:
            with pd.ExcelWriter(RESULT_XLSX, engine='openpyxl', mode='w') as writer:
                self.utility_df.to_excel(writer, sheet_name="Utility DataFrame")
                self.complete_sim_df.to_excel(writer, sheet_name="Complete Similarity DataFrame")
                self.sol_df.to_excel(writer, sheet_name="Solution DataFrame")

        except Exception as error:
            print(f'Exception in final_csv_output: {error}')
            raise

            

         
        
        
        
        

