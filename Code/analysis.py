import argparse
import time
import c_f_recommender_class_with_log as cflg

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
REP = 10
execution_times = []
test = cflg.C_F_Recommender(file_name, metrics, neighbors, prediction, use_nan)

for _ in range(REP):
  inicio = time.time()
  test.start()
  fin = time.time()
  execution_times.append((fin - inicio))

promedio = sum(execution_times) / REP
print('######### FINAL ANALISIS #########')
print(f'Tiempo medio de ejecución: {promedio:.4f} segundos')