"""
Created on Sat Oct  7 16:59:31 2023

@author: eduar

Script del cálculo de la similitud entre usuarios
"""

import pandas as pd
import matplotlib.pyplot as plt
ROUND_VALUE = 2

def pearson(df, user_select):
    # print("User select ", user_select, "\n",df.iloc[user_select])
    data_corr = []
    for i in range(len(df)):
        if user_select != i:     
            # print("User ", i, "\n",df.iloc[i])
            corr = df.iloc[user_select].corr(df.iloc[i], method='pearson')
            list_corr = [user_select, i, round(corr, ROUND_VALUE)]
            data_corr.append(list_corr)
            
    return data_corr
            