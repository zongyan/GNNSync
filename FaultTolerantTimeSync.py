# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 19:28:24 2024

@author: yan
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import copy

import torch; torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim

import utils.dataTools as dataTools
import modules.architecturesTime as architTime
import modules.model as model
import modules.training as training
import modules.evaluation as evaluation

from utils.miscTools import loadSeed

#%%
saveDirRoot = 'experiments'
dataFolder = os.listdir(saveDirRoot)
for i in range(len(dataFolder)):
    saveDir = os.path.join(saveDirRoot, dataFolder[i])
    loadSeed('./' + saveDirRoot + '/' + dataFolder[i]) # loading the states and seed
        
# the following is for temperature use
folderName = "TimeSync-050-20240320204410"
saveDir = os.path.join(saveDirRoot, folderName)
loadSeed('./experiments' + '/' + folderName) # loading the states and seed
#%%

nAgents = 50  # number of UAVs during training
useGPU = True
commRadius = 2. # communication radius
repelDist = 1. # minimum distance before activating repelling function
nTest = 20 # number of testing samples
duration = 10. # simulation duration 
updateTime = 0.01 # clock update time
adjustTime = 0.01 # clock adjustment time
initVelValue = 3. # initial velocities: [-initVelValue, initVelValue]
initMinDist = 0.1 # initial minimum distance between any two UAVs
accelMax = 10. # maximum acceleration value
normalizeGraph = True # normalise wireless communication graph

optimAlg = 'ADAM' 
learningRate = 0.0005
beta1 = 0.9 # default value in pytorch adam
beta2 = 0.999 # default value in pytorch adam
lossFunction = nn.MSELoss
trainer = training.Trainer
evaluator = evaluation.evaluate

nDAggersValues = [1]

nonlinearity = nn.Tanh

layerWiseTraining = True
endToEndTraining = not layerWiseTraining
layerWise = [endToEndTraining]

savingSeeds = False
attackMode = 1 # 1 -- attacking mode 1; 2 -- attacking mode 2
evalModel = True
#%%
'''
In the initial manuscript, we only study the fault tolerant control
performance of the proposed GNN structure, which is GNNThree at the 
Experiments 2 and 2.1. Experiment 2 means the Laplacian matrix, and
Experiment 2.1 uses the adjacency matrix.
'''

modelList = []

# GNN, NO heat kernel, Tanh and readout layer (for comparison)
hParamsGNNFour = {}
hParamsGNNFour['name'] = 'GNNFour'
hParamsGNNFour['archit'] = architTime.LocalGNN_DB
hParamsGNNFour['device'] = 'cuda:0' if (useGPU and torch.cuda.is_available()) else 'cpu'
hParamsGNNFour['dimNodeSignals'] = [2, 64, 64, 64, 2] # features per layer
hParamsGNNFour['nFilterTaps'] = [2, 2, 2, 1] # number of filter taps
hParamsGNNFour['bias'] = True
hParamsGNNFour['nonlinearity'] = nonlinearity
hParamsGNNFour['dimReadout'] = [ ] 
hParamsGNNFour['dimEdgeFeatures'] = 1 # scalar edge weights
hParamsGNNFour['heatKernel'] = False
modelList += [hParamsGNNFour['name']]

trainingOptions = {}

'''ONLY for hidden layer parameters [at the layer-wise training] '''
paramsLayerWiseTrainGNNFour = {}
paramsLayerWiseTrainGNNFour['name'] = 'GNNFour'
paramsLayerWiseTrainGNNFour['dimNodeSignals'] = [ ] # features per hidden layer
paramsLayerWiseTrainGNNFour['nFilterTaps'] = [ ] # number of filter taps for each hidden layer
paramsLayerWiseTrainGNNFour['bias'] = True
paramsLayerWiseTrainGNNFour['nonlinearity'] = nonlinearity # nonlinearity for each hidden layer
paramsLayerWiseTrainGNNFour['dimReadout'] = [ ]
paramsLayerWiseTrainGNNFour['dimEdgeFeatures'] = 1 # scalar edge weights

#%%
if useGPU and torch.cuda.is_available():
    torch.cuda.empty_cache()

print("Selected devices:")
for thisModel in modelList:
    hParamsDict = eval('hParams' + thisModel)
    print("\t%s: %s" % (thisModel, hParamsDict['device']))

print("Generating dummy training data", end = '')
print("...", flush = True)

data = dataTools.AerialSwarm(nAgents, commRadius, repelDist,
            nTest, 1, 1, nTest,
            duration, updateTime, adjustTime, 0, saveDir, 
            initVelValue, initMinDist, accelMax, True)

#%%
modelsGNN = {}

print("Model initialisation...", flush = True)    

for thisModel in modelList:
    hParamsDict = copy.deepcopy(eval('hParams' + thisModel))
    thisName = hParamsDict.pop('name')
    callArchit = hParamsDict.pop('archit')
    thisDevice = hParamsDict.pop('device')
    print("\tInitialising %s..." % thisName, end = ' ',flush = True)

    thisOptimAlg = optimAlg
    thisLearningRate = learningRate
    thisBeta1 = beta1
    thisBeta2 = beta2

    thisArchit = callArchit(**hParamsDict) # initialise the GNN structure
    thisArchit.to(thisDevice)

    thisOptim = optim.Adam(thisArchit.parameters(),
                           lr = learningRate,
                           betas = (beta1, beta2))

    thisLossFunction = lossFunction()
    thisTrainer = trainer
    thisEvaluator = evaluator
    modelCreated = model.Model(thisArchit,
                               thisLossFunction,
                               thisOptim,
                               thisTrainer,
                               thisEvaluator,
                               thisDevice,
                               thisName,
                               nDAggersValues,
                               layerWise,
                               saveDir)
        
    modelsGNN[thisName] = modelCreated
    print("OK")

initModelsGNN = copy.deepcopy(modelsGNN)    
trainedModelsGNN = [copy.deepcopy([copy.deepcopy(initModelsGNN) for k in range(len(nDAggersValues))]) for j in range(len(layerWise))]

print("Configuring and loading the trained model parameters%s..." % thisModel)
for thisModel in modelsGNN.keys():
    
    paramsLayerWiseTrain = copy.deepcopy(eval('paramsLayerWiseTrain' + thisModel))    
    paramsLayerWiseTrain.pop('name')
    
    for nDAggersVal in nDAggersValues:
        
        for val in layerWise:
            
            modelsGNN[thisModel] = copy.deepcopy(initModelsGNN[thisModel])
            thisTrainVars = modelsGNN[thisModel].configure(data, 1, 1, \
                                                        nDAggersVal, 1, 1, \
                                                            paramsLayerWiseTrain, val, \
                                                                lossFunction, learningRate, beta1, beta2, evalModel, **trainingOptions)

            trainedModelsGNN[layerWise.index(val)][nDAggersValues.index(nDAggersVal)][thisModel] = copy.deepcopy(modelsGNN[thisModel])
            
    print(" ")

print("Generating testing data", end = '')
dataTest = dataTools.AerialSwarm(nAgents, commRadius, repelDist,
                1, 1, 1, nTest,
                duration, updateTime, adjustTime, attackMode, saveDir, 
                initVelValue, initMinDist, accelMax, savingSeeds)
print("...", flush = True)

for thisModel in list(modelsGNN.keys()):
    
    for nDAggersVal in nDAggersValues:
            
        for val in layerWise:
            modelsGNN[thisModel] = copy.deepcopy(trainedModelsGNN[layerWise.index(val)][nDAggersValues.index(nDAggersVal)][thisModel])
            modelsGNN[thisModel].evaluate(dataTest, nDAggersVal, val)
