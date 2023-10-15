# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 19:46:07 2023

@author: eduar
"""

import argparse
import sys
import numpy as np
import pandas as pd
import c_f_recommender_class as cfrc

parser = argparse.ArgumentParser()

parser.add_argument("-r", "--read", type=str, nargs='?', default='utility_matrix_A.txt')
parser.add_argument("-m", "--metrics", type=str, nargs='?', default='pearson')
parser.add_argument("-n", "--neighbors", type=int, nargs='?', default=2)
parser.add_argument("-p", "--prediction", type=str, nargs='?', default='simple')
args = parser.parse_args()

# print(args.read, args.metrics, args.neighbors, args.prediction)

file_name: str = args.read
metrics: str = args.metrics
neighbors: int = args.neighbors
prediction: str = args.prediction

test = cfrc.C_F_Recommender(file_name, metrics, neighbors, prediction)
test.start()