# UM 10-25
## PEARSON
### SIMPLE

### MEDIA
- python3 analysis.py --read examples-utility-matrices/utility-matrix-10-25-1.txt --metrics pearson --neighbors 4 --prediction media --use_nan
## COSINE
### SIMPLE

### MEDIA
- python3 analysis.py --read examples-utility-matrices/utility-matrix-10-25-1.txt --metrics cosine --neighbors 4 --prediction media --use_nan
## EUCLIDEAN
### SIMPLE

### MEDIA
- python3 analysis.py --read examples-utility-matrices/utility-matrix-10-25-1.txt --metrics euclidean --neighbors 4 --prediction media --use_nan
# UM 100-1000
## PEARSON
### SIMPLE

### MEDIA
- python3 recommender_main.py --read examples-utility-matrices/utility-matrix-100-1000-1.txt --metrics pearson --neighbors 10 --prediction media --use_nan

## COSINE
### SIMPLE

### MEDIA
python3 recommender_main.py --read examples-utility-matrices/utility-matrix-100-1000-1.txt --metrics cosine --neighbors 10 --prediction media --use_nan

## EUCLIDEAN
### SIMPLE

### MEDIA
python3 recommender_main.py --read examples-utility-matrices/utility-matrix-100-1000-1.txt --metrics euclidean --neighbors 10 --prediction media --use_nan

# UM 25-100
## PEARSON
### SIMPLE

### MEDIA
- python3 recommender_main.py --read examples-utility-matrices/utility-matrix-25-100-1.txt --metrics pearson --neighbors 8 --prediction media --use_nan

## COSINE
### SIMPLE

### MEDIA
- python3 recommender_main.py --read examples-utility-matrices/utility-matrix-25-100-1.txt --metrics cosine --neighbors 8 --prediction media --use_nan

## EUCLIDEAN
### SIMPLE

### MEDIA
- python3 recommender_main.py --read examples-utility-matrices/utility-matrix-25-100-1.txt --metrics euclidean --neighbors 8 --prediction media --use_nan

# UM 5-10
## PEARSON
### SIMPLE
- python3 recommender_main.py --read examples-utility-matrices/utility-matrix-5-10-10.txt --metrics pearson --neighbors 2 --prediction simple --use_nan
- python3 recommender_main.py --read examples-utility-matrices/utility-matrix-5-10-4.txt --metrics pearson --neighbors 2 --prediction simple --use_nan

### MEDIA
- python3 recommender_main.py --read examples-utility-matrices/utility-matrix-5-10-10.txt --metrics pearson --neighbors 2 --prediction media --use_nan
- python3 recommender_main.py --read examples-utility-matrices/utility-matrix-5-10-10.txt --metrics pearson --neighbors 2 --prediction media 
- python3 recommender_main.py --read examples-utility-matrices/utility-matrix-5-10-4.txt --metrics pearson --neighbors 2 --prediction media --use_nan
- python3 recommender_main.py --read examples-utility-matrices/utility-matrix-5-10-4.txt --metrics pearson --neighbors 2 --prediction media

## COSINE
### SIMPLE
- python3 recommender_main.py --read examples-utility-matrices/utility-matrix-5-10-4.txt --metrics cosine --neighbors 2 --prediction simple --use_nan

### MEDIA
- python3 recommender_main.py --read examples-utility-matrices/utility-matrix-5-10-4.txt --metrics cosine --neighbors 2 --prediction media --use_nan

## EUCLIDEAN
### SIMPLE
- python3 recommender_main.py --read examples-utility-matrices/utility-matrix-5-10-4.txt --metrics euclidean --neighbors 2 --prediction simple --use_nan

### MEDIA
- python3 recommender_main.py --read examples-utility-matrices/utility-matrix-5-10-4.txt --metrics euclidean --neighbors 2 --prediction media --use_nan

# UM 50-250
## PEARSON
### SIMPLE

### MEDIA
- python3 recommender_main.py --read examples-utility-matrices/utility-matrix-50-250-1.txt --metrics pearson --neighbors 10 --prediction media --use_nan

## COSINE
### SIMPLE

### MEDIA
- python3 recommender_main.py --read examples-utility-matrices/utility-matrix-50-250-1.txt --metrics cosine --neighbors 10 --prediction media --use_nan
## EUCLIDEAN
### SIMPLE

### MEDIA
- python3 recommender_main.py --read examples-utility-matrices/utility-matrix-50-250-1.txt --metrics euclidean --neighbors 10 --prediction media --use_nan


