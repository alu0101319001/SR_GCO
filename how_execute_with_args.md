# Ejecutar con args: 

runfile('tracking_main.py', args='--read utility_matrix_A.txt --metrics pearson --neighbors 2 --prediction media')
runfile('recommender_main.py', args='--read utility-matrix-5-10-1.txt --metrics pearson --neighbors 2 --prediction media --output file --use_nan')
runfile('recommender_main.py', args='--read utility-matrix-5-10-1.txt --metrics pearson --neighbors 2 --prediction media --output console --use_nan')
runfile('recommender_main.py', args='--read utility-matrix-5-10-1.txt --metrics cosine --neighbors 2 --prediction media --output console --use_nan')