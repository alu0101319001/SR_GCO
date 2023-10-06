import argparse
import sys
import numpy as np
import pandas as pd
import functions as func

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--read", type=str, nargs='?', default='utility_matrix_A.txt')
args = parser.parse_args()
print(args.read)
file_name: str = args.read

with open(file_name, 'r') as f:
    min = float(f.readline())
    print(min)
    max = float(f.readline())
    print(max)

input = np.genfromtxt(file_name, dtype='f', delimiter=' ', skip_header=2, missing_values="-")
print(input)
matrix_normalizada = func.normalizar(input, min, max)
print(matrix_normalizada)

df = func.create_df(matrix_normalizada)

for i in range(df.shape[0]):
    func.find_nan(df, i)
    

"""
"""