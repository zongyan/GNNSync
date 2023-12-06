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

saveDirRoot = 'experiments' 
saveDir = os.path.join(saveDirRoot, os.listdir(saveDirRoot)[0]) 
if not os.path.exists(saveDir):
    raise Exception("error in finding data folder!")    
saveDir = os.path.join(saveDir, "savedData")
savedEndToEndData = os.path.join(saveDir, os.listdir(saveDir)[0])
savedLayerWiseData = os.path.join(saveDir, os.listdir(saveDir)[1])

print("Computing the cost of each best model in the end-to-end training...")
for i in range(10):
    
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
    print("\tThe cost of time sync for each best model: %.4f" %(cost), flush = True)
    
trainLoss = lossTrain.reshape((lossTrain.shape[0], lossTrain.shape[-1] * lossTrain.shape[-2]))    
validAcc = accValid.reshape((accValid.shape[0], accValid.shape[-1] * accValid.shape[-2]))    

print("Plotting the training loss and validation accuract in the end-to-end training...")

plt.figure()
plt.rcParams["figure.figsize"] = (6.4,4.8)            
for i in range(lossTrain.shape[0]):
    plt.plot(trainLoss[i,:])               
plt.legend(['1', '2', '3', '4', '5', '6', '7', '8', '9'])

plt.figure()
plt.rcParams["figure.figsize"] = (6.4,4.8)            
for i in range(lossTrain.shape[0]):
    plt.plot(validAcc[i,:])               
plt.legend(['1', '2', '3', '4', '5', '6', '7', '8', '9'])

print("\n")
#%%  
print("Computing the cost of each best model in the layer-wise training...")    
for i in range(10):
    
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
    print("\tThe cost of time sync for each best model: %.4f" %(cost), flush = True)
    
trainLoss = lossTrain.reshape((lossTrain.shape[0], lossTrain.shape[-1] * lossTrain.shape[-2]))    
validAcc = accValid.reshape((accValid.shape[0], accValid.shape[-1] * accValid.shape[-2]))    

print("Plotting the training loss and validation accuract in the layer-wise training...")

plt.figure()
plt.rcParams["figure.figsize"] = (6.4,4.8)            
for i in range(lossTrain.shape[0]):
    plt.plot(trainLoss[i,:])               
plt.legend(['1', '2', '3', '4', '5', '6', '7', '8', '9'])

plt.figure()
plt.rcParams["figure.figsize"] = (6.4,4.8)            
for i in range(lossTrain.shape[0]):
    plt.plot(validAcc[i,:])               
plt.legend(['1', '2', '3', '4', '5', '6', '7', '8', '9'])