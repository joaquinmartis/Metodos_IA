# -*- coding: utf-8 -*-
"""
Created on Wed May  6 17:56:16 2020

@author: Daniel Albornoz

Implementación similar a genfis de Matlab.
Sugeno type FIS. Generado a partir de clustering substractivo.

"""
__author__ = 'Daniel Albornoz'

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
import time
from substractive_clustering import substractive_clustering
from fis import *
from reglas_fis import *
from inputs_fis import *

def main():

    data_x = np.arange(-10,10,0.1)
    data_y = -0.5*data_x**3-0.6*data_x**2+10*data_x+1 #my_exponential(9, 0.5,1, data_x)

    plt.plot(data_x, data_y)
    # plt.ylim(-20,20)
    plt.xlim(-7,7)

    data = np.vstack((data_x, data_y)).T

    fis2 = FIS()
    fis2.genFIS(data, 1.1)
    fis2.viewInputs()
    r = fis2.evalFIS(np.vstack(data_x))

    plt.figure()
    plt.plot(data_x,data_y)
    plt.plot(data_x,r,linestyle='--')
    plt.show()
    fis2.solutions

    # r1 = data_x*-2.29539539+ -41.21850973
    # r2 = data_x*-15.47376916 -79.82911266
    # r3 = data_x*-15.47376916 -79.82911266
    # plt.plot(data_x,r1)
    # plt.plot(data_x,r2)
    # plt.plot(data_x,r3)

if __name__=="__main__":
    main()