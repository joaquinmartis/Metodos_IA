"""Subtractive Clustering Algorithm
"""
__author__ = 'Daniel Albornoz'


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance_matrix

def subclust2(data, Ra, Rb=0, AcceptRatio=0.3, RejectRatio=0.1):
    if Rb==0:
        Rb = Ra*1.15

    scaler = MinMaxScaler()
    scaler.fit(data)
    normalized_data = scaler.transform(data)

    matriz_distancias = distance_matrix(normalized_data,normalized_data)
    alpha=(Ra/2)**2
    matriz_potenciales = np.sum(np.exp(-matriz_distancias**2/alpha),axis=0)

    centers = []
    i=np.argmax(matriz_potenciales)
    C = normalized_data[i]
    p=matriz_potenciales[i]
    centers = [C]

    continuar=True
    restarP = True
    while continuar:
        pAnt = p
        if restarP:
            matriz_potenciales=matriz_potenciales-p*np.array([np.exp(-np.linalg.norm(v-C)**2/(Rb/2)**2) for v in normalized_data])
        restarP = True
        i=np.argmax(matriz_potenciales)
        C = normalized_data[i]
        p=matriz_potenciales[i]
        if p>AcceptRatio*pAnt:
            centers = np.vstack((centers,C))
        elif p<RejectRatio*pAnt:
            continuar=False
        else:
            dr = np.min([np.linalg.norm(v-C) for v in centers])
            if dr/Ra+p/pAnt>=1:
                centers = np.vstack((centers,C))
            else:
                matriz_potenciales[i]=0
                restarP = False
        if not any(v>0 for v in matriz_potenciales):
            continuar = False
    distancias = [[np.linalg.norm(p-c) for p in normalized_data] for c in centers]
    labels = np.argmin(distancias, axis=0)
    centers = scaler.inverse_transform(centers)
    return labels, centers

# c1 = np.random.rand(15,2)+[1,1]
# c2 = np.random.rand(10,2)+[10,1.5]
# c3 = np.random.rand(5,2)+[4.9,5.8]
# m = np.append(c1,c2, axis=0)
# m = np.append(m,c3, axis=0)

# r,c = subclust2(m,2)

# plt.figure()
# plt.scatter(m[:,0],m[:,1])
# plt.scatter(c[:,0],c[:,1], marker='X')
# print(c)
if __name__=="__main__":
    c1 = np.random.rand(150,2)+[1,1]
    c2 = np.random.rand(100,2)+[10,1.5]
    c3 = np.random.rand(50,2)+[4.9,5.8]
    m = np.append(c1,c2, axis=0)
    m = np.append(m,c3, axis=0)

    r,c = subclust2(m,1)

    plt.figure()
    plt.scatter(m[:,0],m[:,1], c=r)
    plt.scatter(c[:,0],c[:,1], marker='X')
    plt.show()