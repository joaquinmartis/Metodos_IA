#Python con IA: Clustering
# Imports
from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris


def kmeans(k,datos):
    n=datos.shape[0]#N número de datos
    d=datos.shape[1]#D número de parametros

    media=np.mean(datos,axis=0)
    std=np.std(datos,axis=0)
    centros_clusters=np.random.randn(k,d)*std+media

    plt.scatter(datos[:,0],datos[:,1],s=7)
    plt.scatter(centros_clusters[:,0], centros_clusters[:,1], marker='*', c='g', s=150)
    #plt.show()

    mat_distancia_ccluster=np.zeros((n,k))
    nuevos_centros=deepcopy(centros_clusters)
    viejos_centros=np.zeros((k,d))
    error=1
    while error!=0:
        viejos_centros=nuevos_centros
        #Calcular distancia a centros
        for centro in range(k):
            #                                               Resta en la matriz datos fila por fila por el vector centro(el actual)
            mat_distancia_ccluster[:,centro]=np.linalg.norm(datos-viejos_centros[centro],axis=1) #axis=1 significa que va por filas para hacer la norma
        
        #Buscamos a que cluster pertenece cada punto
        vec_pertenencia=np.zeros(n)
        for i in range(n): #Recorre cada punto
            min_centro=0
            for j_centro in range(k): #Recorre cada centro cluster
                if (mat_distancia_ccluster[i,j_centro]<mat_distancia_ccluster[i,min_centro]):
                    min_centro=j_centro
            vec_pertenencia[i]=min_centro

        #Recalculamos los centros de cluster
        acumulador_puntos=np.zeros((k,d))
        contador_puntos=np.zeros(k)
        for i_dato in range(n):
            for j_centro in range(k):
                if vec_pertenencia[i_dato]==j_centro:
                    acumulador_puntos[j_centro]+=datos[i]
                    contador_puntos[j_centro]+=1

        #Recalculo cluster
        
        for i in range(k):
            if contador_puntos[i]!=0:
                nuevos_centros=acumulador_puntos[i]/contador_puntos[i]

        error = np.linalg.norm(nuevos_centros - viejos_centros)    

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
    #plt.scatter(datos[:,0],datos[:,1],s=7)
    
    k=3
    kmeans(k,datos)




def main():
    iris = load_iris()
    setup_kmeans()
    
if __name__ == "__main__":

    main()  