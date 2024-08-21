#Python con IA: Clustering
# Imports
from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris





def kmeans():
    pass

def setup_kmeans():
    centro_1=np.array([1,1],dtype="int")
    centro_2=np.array([8,1],dtype="int")
    centro_3=np.array([2,6],dtype="int")

    nube_1=np.random.randn(200,2)
    nube_2=np.random.randn(200,2)
    nube_3=np.random.randn(200,2)

    nube_1+=centro_1
    nube_2+=centro_2
    nube_3+=centro_3

    datos=np.concatenate((nube_1,nube_2,nube_3),axis=0)
    plt.scatter(datos[:,0],datos[:,1],s=7)
    plt.show()


def main():
    iris = load_iris()
    setup_kmeans()
    
if __name__ == "__main__":

    main()  