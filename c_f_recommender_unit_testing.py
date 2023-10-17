# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 22:30:39 2023

@author: eduar

FICHERO DE PRUEBAS UNITARIAS
"""

import unittest
import c_f_recommender_class as cfrc

### DATOS
file_1 = "utility_matrix_A.txt"
file_2 = "utility_matrix_B.txt"
file_3 = "utility-matrix-5-10-1.txt"

metric = ['pearson', 'cosine', 'euclidean']
prediction = ['simple', 'media']
output = ['console', 'file']


class CF_Recommender_Test(unittest.TestCase):
    
    def test_simple(self):
        s = 'SI'
        self.assertEqual(s, 'SI')

if __name__ == '__main__':
    unittest.main()