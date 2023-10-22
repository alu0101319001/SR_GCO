# Ejecutar con args: 
## CORRECTOS

runfile('tracking_main.py', args='--read utility_matrix_A.txt --metrics pearson --neighbors 2 --prediction media')
runfile('recommender_main.py', args='--read utility-matrix-5-10-1.txt --metrics pearson --neighbors 2 --prediction media --output file --use_nan')
runfile('recommender_main.py', args='--read utility-matrix-5-10-1.txt --metrics pearson --neighbors 2 --prediction media --output console --use_nan')
runfile('recommender_main.py', args='--read utility-matrix-5-10-1.txt --metrics cosine --neighbors 2 --prediction media --output console --use_nan')
runfile('recommender_main.py', args='--read utility-matrix-5-10-1.txt --metrics euclidean --neighbors 2 --prediction media --output console --use_nan')
runfile('recommender_main.py', args='--read utility_matrix_B.txt --metrics cosine --neighbors 2 --prediction media --output console --use_nan')
runfile('recommender_main.py', args='--read utility_matrix_B.txt --metrics pearson --neighbors 2 --prediction media --output console --use_nan')



## ERRORES
runfile('recommender_main.py', args='--read utility_matrix_ERROR_1.txt --metrics cosine --neighbors 2 --prediction media --output console --use_nan')
runfile('recommender_main.py', args='--read utility-matrix-5-10-10.txt --metrics cosine --neighbors 2 --prediction media --output console --use_nan')
runfile('recommender_main.py', args='--read utility-matrix-5-10-9.txt --metrics cosine --neighbors 2 --prediction media --output console --use_nan')

## CONSOLA NORMAL
python3 recommender_main.py --read utility-matrix-5-10-10.txt --metrics pearson --neighbors 2 --prediction media --output console --use_nan


