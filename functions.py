import numpy as np
import pandas as pd


def normalizar(val: float, min: float, max: float):
    return (val-min)/(max-min)

def find_nan(df, index): 
    result = df[index].isnull().to_numpy().nonzero()
    print(result)

def create_df(matrix):
    df = pd.DataFrame(matrix)
    print(df)
    return df
        

    