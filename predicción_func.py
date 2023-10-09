# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 20:23:06 2023

@author: eduar
"""

import pandas as pd
import functions as func

def select_neighbors(df_corr, neighbors): 
    selected = df_corr.nlargest(neighbors, func.CORR_COL_2)
    return selected

"""
def simple_prediction(utility_df, neighbors_selected):
    summation = 0
    for i in range(len(neighbors_selected)):
        #v = 
        summation = summation + (neighbors_selected[i]['corr'] * utility_df[]
    """

