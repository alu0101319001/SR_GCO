# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 19:46:07 2023

@author: eduar
"""

import argparse
import numpy as np
import pandas as pd
import time
import c_f_recommender_class_with_log as cfrc_lg

parser = argparse.ArgumentParser()

parser.add_argument("-r", "--read", type=str, nargs='?', default='utility_matrix_A.txt')
parser.add_argument("-m", "--metrics", type=str, nargs='?', default='pearson')
parser.add_argument("-n", "--neighbors", type=int, nargs='?', default=2)
parser.add_argument("-p", "--prediction", type=str, nargs='?', default='simple')
parser.add_argument("-u", "--use_nan", action='store_true')
args = parser.parse_args()

# print(args.read, args.metrics, args.neighbors, args.prediction)

file_name: str = args.read
metrics: str = args.metrics
neighbors: int = args.neighbors
prediction: str = args.prediction
use_nan = args.use_nan

start_time = time.time()

test = cfrc_lg.C_F_Recommender(file_name, metrics, neighbors, prediction, use_nan)
result = test.start()

end_time = time.time()
print(f"Execution time: {end_time - start_time}")

