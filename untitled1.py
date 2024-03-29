# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 21:47:42 2024

@author: yan
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import io

zeroTolerance = 1e-9 # values below this number are zero

#%%
saveDirRoot = 'experiments'
# dataFolder = os.listdir(saveDirRoot)
# for i in range(len(dataFolder)):
#     saveDir = os.path.join(saveDirRoot, dataFolder[i])
        
# the following is for temperature use
folderName = "TimeSync-050-20240322083752"
saveDir = os.path.join(saveDirRoot, folderName)
saveDir = os.path.join(saveDir,'savedTanh')

#%%
def ReLU(x):
    return x * (x > 0)

#%%

nAgents = 50
duration = 10. # simulation duration 
updateTime = 0.01 # clock update time
tSamples = np.int64(duration/updateTime)
savingInstant = np.array([1, 101, 201, 301, 401, 501, 601, 701, 801, 901, 1001])
preSavingInstant = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])

modelStructure = np.array(['2-64-2', '2-64-64-2', '2-64-64-2'])
beforeActivationValue = np.array(['1-100-beforeAct', '101-200-beforeAct', '201-300-beforeAct', '301-400-beforeAct', '401-500-beforeAct', \
                            '501-600-beforeAct', '601-700-beforeAct', '701-800-beforeAct', '801-900-beforeAct', '901-1000-beforeAct'])
afterActivationValue = np.array(['1-100-afterAct', '101-200-afterAct', '201-300-afterAct', '301-400-afterAct', '401-500-afterAct', \
                            '501-600-afterAct', '601-700-afterAct', '701-800-afterAct', '801-900-afterAct', '901-1000-afterAct'])
    
# for element in modelStructure:
savedTanhDir = os.path.join(saveDir, '2-64-64-2')    
graphFile = np.load(os.path.join(savedTanhDir, 'graph.npz'), allow_pickle=True)    
graph = graphFile['graph']

assert len(beforeActivationValue) == len(afterActivationValue)
    
laplacianMatrix = np.zeros((graph.shape[0], tSamples, nAgents, nAgents), dtype = np.float64)
eigenValues = np.zeros((graph.shape[0], tSamples, nAgents), dtype = np.float64)
eigenVectors = np.zeros((graph.shape[0], tSamples, nAgents, nAgents), dtype = np.float64)
beforeActValues = np.zeros((graph.shape[0], tSamples, 64, nAgents), dtype = np.float64) # values before the activation function
afterActValues = np.zeros((graph.shape[0], tSamples, 64, nAgents), dtype = np.float64) # values after the activation function

index = 0
for t in range(1, 900):
    
    if t == savingInstant[index]:
        savedBeforeTanhDir = os.path.join(savedTanhDir, beforeActivationValue[0])
        savedAfterTanhDir = os.path.join(savedTanhDir, afterActivationValue[0])
        
        if 'beforeActivation' in globals():
            del beforeActivation
        if 'afterActivation' in globals():
            del afterActivation                
        
        with open(savedBeforeTanhDir, "rb") as fp:   # Unpickling
            beforeActivation = pickle.load(fp)    
        
        with open(savedAfterTanhDir, "rb") as fp:   # Unpickling
            afterActivation = pickle.load(fp)
        
        index = index + 1
    
    thisBeforeAct = beforeActivation[(t-1) - preSavingInstant[index-1]][1] # values before the activation function
    thisAfterAct = afterActivation[(t-1) - preSavingInstant[index-1]][1] # values after the activation function    
        
    thisGraph = graph[:,t,:,:]
    thisGraph[thisGraph > zeroTolerance] = 1. # reset the normalised adjacency matrix to the normal graph matrix
    thisAdjacencyMatrix = thisGraph
    thisDegreeMatrix = np.sum(thisAdjacencyMatrix, axis=2)

    for i in range(thisDegreeMatrix.shape[0]): # in the experiment instant dimension                    
        laplacianMatrix[i, t, :, :] = np.diag(thisDegreeMatrix[i, :]) - thisAdjacencyMatrix[i, :, :]  # Non-Normalized laplacian matrix
        # The column eigenvectors[:, i] is the normalized eigenvector corresponding to the eigenvalue eigenvalues[i]
        eigenValues[i, t, :], eigenVectors[i, t, :, :] = np.linalg.eigh(laplacianMatrix[i, t, :, :])

        assert thisBeforeAct.shape[1] == thisAfterAct.shape[1]
        for j in range(thisBeforeAct.shape[1]): # in the feature dimension                    
            beforeActValues[i, t, j, :] = np.matmul(np.transpose(eigenVectors[i, t, :, :]), np.float64(thisBeforeAct[i, j, :])) # values before the activation function
            afterActValues[i, t, j, :] = np.matmul(np.transpose(eigenVectors[i, t, :, :]), np.float64(thisAfterAct[i, j, :])) # values after the activation function
            # afterActValues[i, t, j, :] = np.matmul(np.transpose(eigenVectors[i, t, :, :]), np.float64(ReLU(thisBeforeAct[i, j, :]))) # values after the activation function

mdic = {"eigenValues": eigenValues, \
        "eigenVectors": eigenVectors, \
        "toTanhFunc": beforeActValues, \
        "fromTanhFunc": afterActValues}            
# io.savemat(os.path.join(os.path.join(saveDir, '2-64-64-2'), 'first-layer')+'.mat', mdic)
io.savemat(os.path.join(os.path.join(saveDir, '2-64-64-2'), 'second-layer')+'.mat', mdic)

#%%
t = 1 # 1, 10, 50, 100, 200, 500, 800
cnt = 10
for i in range(cnt+0, thisBeforeAct.shape[1]):
    plt.figure()
    plt.rcParams["figure.figsize"] = (6.4,4.8)
    # for t in range(0, tSamples-1):
    plt.vlines(eigenValues[0, t, :], 0, beforeActValues[0, t, i, :], color='#D3D3D3', alpha=0.4)
    plt.scatter(eigenValues[0, t, :], beforeActValues[0, t, i, :], color='#D3D3D3', alpha=0.4)
    
    plt.vlines(eigenValues[0, t, :], 0, afterActValues[0, t, i, :], color='#7BC8F6', alpha=0.4)
    plt.scatter(eigenValues[0, t, :], afterActValues[0, t, i, :], color='#7BC8F6', alpha=0.4)
    
    if i == cnt+9:
        break
        