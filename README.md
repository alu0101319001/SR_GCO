# **Sistema de recomendación**
## **Métodos de filtrado colaborativo**
> Participantes:
   - Eduardo González Pérez
   - Jonathan Martínez Pérez

##### El codigo fuente de la practica se encuentra dividido en 2 partes
#####    Ficheros con extensión .py
#####    Ficheros .txt con los datos de entrada

## **Modo de uso**:
#####   **Descargar: Mediante un git clone https://github.com/alu0101319001/SR_GCO.git**
#####   **Instalar Python3 en caso de no tenerlo:**
#####   **sudo apt update**
#####   **sudo apt install python3**  
#####   **Instalar librerías en caso necesario: pip3 install numpy, pip3 install pandas**
#####   **Ejecutar: python3 recommender_main.py --read utility-matrix-5-10-1.txt --metrics pearson --neighbors 2 --prediction media --output file --use_nan** 
    - recommender_main.py es el ejecutable.
    - --read utility-matrix-5-10-1.txt es la matriz de entrada.
    - --metrics pearson/cosine/euclidean son las métricas.
    - --neighbors 2: número de vecinos (mínimo 2).
    - --prediction media/simple son las predicciones.

###### El ejecutable se encontrara en la carpeta bin
###### Una vez ejecutado el programa este pedira al usuario:
    - Fichero de entrada.
    - Métrica: Pearson, distancia coseno o euclídea.
    - Número de vecinos para seleccionar.
    - Tipo de predicción: Predicción simple o diferencia con la media.

## **Descripcion del código desarrollado**:
