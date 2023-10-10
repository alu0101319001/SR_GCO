"""
Created on Sat Oct  7 16:59:31 2023

@author: eduar

Script del cálculo de la similitud entre usuarios
"""

import pandas as pd
import matplotlib.pyplot as plt
import functions as func

ROUND_VALUE = 2

def calculate_similarity(df, nan_positions, metrics):
    # Falta un switch para el selector de métrica
    data_corr = pearson(df, nan_positions[0])
    
    df_sim = func.create_sim_df(data_corr)
       
    return df_sim

def pearson(df, user_select):
    # Faltaría seleccionar los items que se van a utilizar para la correlación
    # antes de usar directamente pearson. Para tener control sobre ellos
    # usar NaN previamentes calculados o no? 
    
    # print("User select ", user_select, "\n",df.iloc[user_select])
    data_corr = []
    for i in range(len(df)):
        if user_select != i:     
            # print("User ", i, "\n",df.iloc[i])
            corr = df.iloc[user_select].corr(df.iloc[i], method='pearson')
            list_corr = [user_select, i, round(corr, ROUND_VALUE)]
            data_corr.append(list_corr)
            
    return data_corr
            