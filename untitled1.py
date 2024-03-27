# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 21:47:42 2024

@author: yan
"""
import os
import pickle
import numpy as np

# with open("test", "rb") as fp:   # Unpickling
# ...   b = pickle.load(fp)

zeroTolerance = 1e-9 # values below this number are zero

#%%
saveDirRoot = 'experiments'
dataFolder = os.listdir(saveDirRoot)
for i in range(len(dataFolder)):
    saveDir = os.path.join(saveDirRoot, dataFolder[i])
        
# the following is for temperature use
folderName = "TimeSync-050-20240320204410"
saveDir = os.path.join(saveDirRoot, folderName)
savedTanhDir = os.path.join(saveDir,'savedTanh')
savedTanhDir = os.path.join(savedTanhDir,'1-100afterAct')
#%%

nAgents = 50
duration = 10. # simulation duration 
updateTime = 0.01 # clock update time

tSamples = np.int64(duration/updateTime)
readTime = np.array([1, 101, 201, 301, 401, 501, 601, 701, 801, 901])

graphFile = np.load(saveFile + '-' + str(model.name) + '-nDAggers-' + str(nDAggers) + '.npz', allow_pickle=True) # the data file loaded from the example folder

graph = graphFile['graph']

laplacianMatrix = np.zeros((graph.shape[0], tSamples, nAgents, nAgents), dtype = np.float64)
eigenValues = np.zeros((graph.shape[0], tSamples, nAgents), dtype = np.float64)
eigenVectors = np.zeros((graph.shape[0], tSamples, nAgents, nAgents), dtype = np.float64)
beforeActValues = np.zeros((graph.shape[0], tSamples, beforeAct[0][0].shape[1], nAgents), dtype = np.float64) # values before the activation function
afterActValues = np.zeros((graph.shape[0], tSamples, afterAct[0][0].shape[1], nAgents), dtype = np.float64) # values after the activation function

i = 0

for t in range(1, tSamples):
        
    if t == readTime[i]:
        del beforeActivation, afterActivation 
        
        with open(savedTanhDir, "rb") as fp:   # Unpickling
            afterActivation = pickle.load(fp)
            
        with open(savedTanhDir, "rb") as fp:   # Unpickling
            beforeActivation = pickle.load(fp)    
            
        i = i + 1    

    for t in range(1, tSamples-1):
        thisBeforeAct = beforeActivation[t][0] # values before the activation function
        thisAfterAct = afterActivation[t][0] # values after the activation function    
        
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

    for i in range(0, thisBeforeAct.shape[1]):
        plt.figure()
        plt.rcParams["figure.figsize"] = (6.4,4.8)
        for t in range(0, tSamples-1):
            plt.vlines(eigenValues[0, t, :], 0, beforeActValues[0, t, i, :], color='#D3D3D3', alpha=0.4)
            plt.scatter(eigenValues[0, t, :], beforeActValues[0, t, i, :], color='#D3D3D3', alpha=0.4)
            
            plt.vlines(eigenValues[0, t, :], 0, afterActValues[0, t, i, :], color='#7BC8F6', alpha=0.4)
            plt.scatter(eigenValues[0, t, :], afterActValues[0, t, i, :], color='#7BC8F6', alpha=0.4)