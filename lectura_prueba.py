import argparse
import sys
import numpy as np
import normalizar as nor

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--read", type=str, nargs='?', default='utility_matrix_B.txt')
args = parser.parse_args()
print(args.read)
file_name: str = args.read

with open(file_name, 'r') as f:
    min: float = f.readline()
    max: float = f.readline()

input = np.genfromtxt(file_name, dtype='f', delimiter=' ', skip_header=2, missing_values="-")
print(input)

print(nor.normalizar(input[1], min, max))

"""
"""
