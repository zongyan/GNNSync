# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 17:34:14 2023

@author: yan
"""

import os
import numpy as np
import copy

import matplotlib.pyplot as plt

nAgents = 50  # number of UAVs during training 
updateTime = 0.01 # clock update time
addedLayerNum = np.array([9, 4, 4, 4, 9, 4, 4, 4]) + 1
addedLayerName = np.array(['0 layers', '1 layers', '2 layers', '3 layers', '4 layers', '5 layers', '6 layers', '7 layers', '8 layers', '9 layers'])
gnnName = np.array(['GNNOne', 'GNNTwo', 'GNNThree', 'GNNFour', 'GNNFive', 'GNNSix', 'GNNSeven', 'GNNEight'])
numSavedFiles = 6

saveDirRoot = 'experiments' 
dataFolder = os.listdir(saveDirRoot)

for j in range(len(dataFolder)-numSavedFiles-1):
    
    saveDir = os.path.join(saveDirRoot, os.listdir(saveDirRoot)[j+numSavedFiles]) 
    if not os.path.exists(saveDir):
        raise Exception("error in finding data folder!")    
    saveDir = os.path.join(saveDir, "savedData")
    savedEndToEndData = os.path.join(saveDir, os.listdir(saveDir)[0])
    savedLayerWiseData = os.path.join(saveDir, os.listdir(saveDir)[1])

    #%%  
    print("Computing the cost of each best %s model in the layer-wise training..." %(gnnName[j]))    
    for i in range(addedLayerNum[j]):
        
        testingData = np.load(os.path.join(savedLayerWiseData, os.listdir(savedLayerWiseData)[i]), allow_pickle=True)
        
        offsetTest = copy.deepcopy(testingData['offsetTestBest'])
        skewTest = copy.deepcopy(testingData['skewTestBest'])
        adjTest = copy.deepcopy(testingData['adjTestBest'])
        stateTest = copy.deepcopy(testingData['stateTestBest'])
        commGraphTest = copy.deepcopy(testingData['commGraphTestBest'])
        bestL = copy.deepcopy(testingData['bestL'])
        bestIteration = copy.deepcopy(testingData['bestIteration'])
        bestEpoch = copy.deepcopy(testingData['bestEpoch'])
        bestBatch = copy.deepcopy(testingData['bestBatch'])
        lossTrain = copy.deepcopy(testingData['lossTrain'])
        accValid = copy.deepcopy(testingData['accValid'])
        
        offset = offsetTest[:, :, :, :]
        skew = skewTest[:, :, :, :]
        avgOffset = np.mean(offset, axis = 3) # nSamples x tSamples x 1
        avgSkew = np.mean(skew/10, axis= 3) # nSamples x tSamples x 1, change unit from 10ppm to 100ppm               
        
        diffOffset = offset - np.tile(np.expand_dims(avgOffset, 3), (1, 1, 1, nAgents)) # nSamples x tSamples x 1 x nAgents
        diffSkew = skew/10 - np.tile(np.expand_dims(avgSkew, 3), (1, 1, 1, nAgents)) # nSamples x tSamples x 1 x nAgents
        
        diffOffset = np.sum(diffOffset**2, 2) # nSamples x tSamples x nAgents
        diffSkew = np.sum(diffSkew**2, 2) # nSamples x tSamples x nAgents
        
        diffOffsetAvg = np.mean(diffOffset, axis = 2) # nSamples x tSamples
        diffSkewAvg = np.mean(diffSkew, axis = 2) # nSamples x tSamples
        
        costPerSample = np.sum(diffOffsetAvg, axis = 1) + np.sum(diffSkewAvg, axis = 1)*updateTime # nSamples
        
        cost = np.mean(costPerSample) # scalar
        print("\tThe cost of time sync for %s (added) best model: %.4f" %(addedLayerName[i], cost), flush = True)
        
    trainLoss = lossTrain.reshape((lossTrain.shape[0], lossTrain.shape[-1] * lossTrain.shape[-2]))    
    validAcc = accValid.reshape((accValid.shape[0], accValid.shape[-1] * accValid.shape[-2]))    
        
    plt.figure()
    plt.rcParams["figure.figsize"] = (6.4,4.8)            
    for i in range(lossTrain.shape[0]):
        plt.plot(trainLoss[i,:])               
    plt.legend(['0 layer', '1 layer', '2 layers', '3 layers', '4 layers', '5 layers', '6 layers', '7 layers', '8 layers', '9 layers'])
    
    plt.figure()
    plt.rcParams["figure.figsize"] = (6.4,4.8)            
    for i in range(lossTrain.shape[0]):
        plt.plot(validAcc[i,:])               
    plt.legend(['0 layer', '1 layer', '2 layers', '3 layers', '4 layers', '5 layers', '6 layers', '7 layers', '8 layers', '9 layers'])

    #%%
    print("Computing the cost of each best %s model in the end-to-end training..." %(gnnName[j]))
    for i in range(addedLayerNum[j]):
        
        testingData = np.load(os.path.join(savedEndToEndData, os.listdir(savedEndToEndData)[i]), allow_pickle=True)
        
        offsetTest = copy.deepcopy(testingData['offsetTestBest'])
        skewTest = copy.deepcopy(testingData['skewTestBest'])
        adjTest = copy.deepcopy(testingData['adjTestBest'])
        stateTest = copy.deepcopy(testingData['stateTestBest'])
        commGraphTest = copy.deepcopy(testingData['commGraphTestBest'])
        bestL = copy.deepcopy(testingData['bestL'])
        bestIteration = copy.deepcopy(testingData['bestIteration'])
        bestEpoch = copy.deepcopy(testingData['bestEpoch'])
        bestBatch = copy.deepcopy(testingData['bestBatch'])
        lossTrain = copy.deepcopy(testingData['lossTrain'])
        accValid = copy.deepcopy(testingData['accValid'])
        
        offset = offsetTest[:, :, :, :]
        skew = skewTest[:, :, :, :]
        avgOffset = np.mean(offset, axis = 3) # nSamples x tSamples x 1
        avgSkew = np.mean(skew/10, axis= 3) # nSamples x tSamples x 1, change unit from 10ppm to 100ppm               
        
        diffOffset = offset - np.tile(np.expand_dims(avgOffset, 3), (1, 1, 1, nAgents)) # nSamples x tSamples x 1 x nAgents
        diffSkew = skew/10 - np.tile(np.expand_dims(avgSkew, 3), (1, 1, 1, nAgents)) # nSamples x tSamples x 1 x nAgents
        
        diffOffset = np.sum(diffOffset**2, 2) # nSamples x tSamples x nAgents
        diffSkew = np.sum(diffSkew**2, 2) # nSamples x tSamples x nAgents
        
        diffOffsetAvg = np.mean(diffOffset, axis = 2) # nSamples x tSamples
        diffSkewAvg = np.mean(diffSkew, axis = 2) # nSamples x tSamples
        
        costPerSample = np.sum(diffOffsetAvg, axis = 1) + np.sum(diffSkewAvg, axis = 1)*updateTime # nSamples   
     
        cost = np.mean(costPerSample) # scalar
        print("\tThe cost of time sync for %s (added) best model: %.4f" %(addedLayerName[i], cost), flush = True)        
        
    trainLoss = lossTrain.reshape((lossTrain.shape[0], lossTrain.shape[-1] * lossTrain.shape[-2]))    
    validAcc = accValid.reshape((accValid.shape[0], accValid.shape[-1] * accValid.shape[-2]))    
        
    plt.figure()
    plt.rcParams["figure.figsize"] = (6.4,4.8)            
    for i in range(lossTrain.shape[0]):
        plt.plot(trainLoss[i,:])               
    plt.legend(['0 layer', '1 layer', '2 layers', '3 layers', '4 layers', '5 layers', '6 layers', '7 layers', '8 layers', '9 layers'])
    
    plt.figure()
    plt.rcParams["figure.figsize"] = (6.4,4.8)            
    for i in range(lossTrain.shape[0]):
        plt.plot(validAcc[i,:])               
    plt.legend(['0 layer', '1 layer', '2 layers', '3 layers', '4 layers', '5 layers', '6 layers', '7 layers', '8 layers', '9 layers'])
    
    print("\n")