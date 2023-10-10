# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 20:23:06 2023

@author: eduar
"""

import pandas as pd
import functions as func

ROUND_VALUE = 2

def select_neighbors(df_corr, neighbors): 
    selected = df_corr.nlargest(neighbors, func.CORR_COL_2)
    return selected

def calculate_prediction(utility_df, nan_selected, neighbors_selected, prediction):
    # Switch selector de tipo de predicción
    sol_val = -1
    if prediction == func.SIMPLE:
        sol_val = simple_prediction(utility_df, nan_selected, neighbors_selected)
    elif prediction == func.MEDIA:
        sol_val = media_prediction(utility_df, nan_selected, neighbors_selected)
    else:
        # error
        sol_val = -99
    return sol_val

def simple_prediction(utility_df, nan_selected, neighbors_selected):
    top_summation = 0
    bot_summation = 0
    for i in range(len(neighbors_selected)):
        v = int(neighbors_selected.iloc[i]['v'])
        #print(v)
        sim_u_v = neighbors_selected.iloc[i]['corr']
        #print(sim_u_v)
        user_label = "User" + str(v)
        r_v_i = utility_df.at[user_label, utility_df.columns[nan_selected[1]]]
        #print(r_v_i)
        
        top_summation = top_summation + (sim_u_v * r_v_i)
        bot_summation = abs(bot_summation + sim_u_v)
        
    #print(top_summation, bot_summation)
    solution = round((top_summation / bot_summation), ROUND_VALUE)
    return solution
    
def media_prediction(utility_df, nan_selected, neighbors_selected):
    top_summation = 0
    bot_summation = 0
    user_u_label = "User" + str(int(nan_selected[0]))
    u_mean = round(utility_df.loc[user_u_label].mean(skipna = True), ROUND_VALUE)
    for i in range(len(neighbors_selected)):
        v = int(neighbors_selected.iloc[i]['v'])
        #print(v)
        sim_u_v = neighbors_selected.iloc[i]['corr']
        #print(sim_u_v)
        user_v_label = "User" + str(v)
        r_v_i = utility_df.at[user_v_label, utility_df.columns[nan_selected[1]]]
        #print(r_v_i)
        v_mean = round(utility_df.loc[user_v_label].mean(skipna = True), ROUND_VALUE)
        
        top_summation = top_summation + (sim_u_v * (r_v_i - v_mean))
        bot_summation = abs(bot_summation + sim_u_v)
        
    #print(top_summation, bot_summation)
    sol = u_mean + (top_summation / bot_summation)
    solution = round(sol, ROUND_VALUE)
    return solution
        
    

