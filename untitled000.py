# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 17:34:14 2023

@author: yan
"""

import os
import numpy as np
import copy

import matplotlib.pyplot as plt

thisFilename = 'TimeSync'
nAgents = 50  # number of UAVs during training 
saveDirRoot = 'experiments' 
saveDir = os.path.join(saveDirRoot, thisFilename) 
saveDir = saveDir + '-%03d-' % nAgents + "20231128143805"
if not os.path.exists(saveDir):
    raise Exception("error in finding data folder!")
    
saveDir = os.path.join(saveDir, "savedData")

savedEndToEndData = os.path.join(saveDir, os.listdir(saveDir)[0])
savedLayerWiseData = os.path.join(saveDir, os.listdir(saveDir)[1])

updateTime = 0.01 # clock update time

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
    
    
    os.listdir(savedLayerWiseData)
    

    
    #%%
    offsetTest = offsetTest[:, :, :, :]
    skewTest = skewTest[:, :, :, :]
    avgOffset = np.mean(offsetTest, axis = 3) # nSamples x tSamples x 1
    avgSkew = np.mean(skewTest/10, axis= 3) # nSamples x tSamples x 1, change unit from 10ppm to 100ppm               
    
    diffOffset = offsetTest - np.tile(np.expand_dims(avgOffset, 3), (1, 1, 1, nAgents)) # nSamples x tSamples x 1 x nAgents
    diffSkew = skewTest/10 - np.tile(np.expand_dims(avgSkew, 3), (1, 1, 1, nAgents)) # nSamples x tSamples x 1 x nAgents
    
    diffOffset = np.sum(diffOffset**2, 2) # nSamples x tSamples x nAgents
    diffSkew = np.sum(diffSkew**2, 2) # nSamples x tSamples x nAgents
    
    diffOffsetAvg = np.mean(diffOffset, axis = 2) # nSamples x tSamples
    diffSkewAvg = np.mean(diffSkew, axis = 2) # nSamples x tSamples
    
    costPerSample = np.sum(diffOffsetAvg, axis = 1) + np.sum(diffSkewAvg, axis = 1)*updateTime # nSamples
    
    cost = np.mean(costPerSample) # scalar
    
    print(cost)  
    
    #%%
    
    yy = lossTrain.reshape((lossTrain.shape[0], 1200))
    
    zz = accValid.reshape((accValid.shape[0], 240))    

plt.figure()
plt.rcParams["figure.figsize"] = (6.4,4.8)            
for i in range(lossTrain.shape[0]):
    plt.plot(yy[i,:])               


plt.figure()
plt.rcParams["figure.figsize"] = (6.4,4.8)            
for i in range(lossTrain.shape[0]):
    plt.plot(zz[i,:])               


# for i in range(lossTrain.shape[0]):
#     for j in range(lossTrain.shape[1]):
        
#         plt.figure()
#         plt.rcParams["figure.figsize"] = (6.4,4.8)            
#         plt.plot(np.reshape(lossTrain[i, j, :, :], (lossTrain.shape[2] * lossTrain.shape[3])))

#         plt.figure()
#         plt.rcParams["figure.figsize"] = (6.4,4.8)            
#         plt.plot(np.reshape(accValid[i, j, :, :], (accValid.shape[2] * accValid.shape[3])))       
    
    
    