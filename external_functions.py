# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 19:38:19 2023

@author: eduar
"""
import csv

ROUND_VALUE = 2
def export_df(df, file_name):
    df.to_csv(file_name, sep=" ",
              quoting=csv.QUOTE_NONE, escapechar= " ")
    
def pearson(df, user_select):
    # Faltaría seleccionar los items que se van a utilizar para la correlación
    # antes de usar directamente pearson. Para tener control sobre ellos
    # usar NaN previamentes calculados o no? --> Esto sería mas bien en el bucle
    
    # print("User select ", user_select, "\n",df.iloc[user_select])
    data_corr = []
    for i in range(len(df)):
        if user_select != i:     
            # print("User ", i, "\n",df.iloc[i])
            corr = df.iloc[user_select].corr(df.iloc[i], method='pearson')
            list_corr = [user_select, i, round(corr, ROUND_VALUE)]
            data_corr.append(list_corr)
            
    return data_corr
