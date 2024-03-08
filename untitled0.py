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
addedLayerNum = np.array([9, 4, 4, 4, 9, 4, 4, 4])
saveDirRoot = 'experiments'
dataFolder = os.listdir(saveDirRoot)

testName = "TimeSync-050-20240301232924"

#%%
nAgents = 50  # number of UAVs during training 

loadSeed('./experiments/2nd_simulation_1') # loading the states and seed

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

evalModel = True

#%%
modelList = []

hParamsGNNThree = {}
hParamsGNNThree['name'] = 'GNNThree'
hParamsGNNThree['archit'] = architTime.LocalGNN_DB
hParamsGNNThree['device'] = 'cuda:0' if (useGPU and torch.cuda.is_available()) else 'cpu'
hParamsGNNThree['dimNodeSignals'] = [2, 64, 2] # features per layer
hParamsGNNThree['nFilterTaps'] = [2, 1] # number of filter taps
hParamsGNNThree['bias'] = True
hParamsGNNThree['nonlinearity'] = nonlinearity
hParamsGNNThree['dimReadout'] = [ ] 
hParamsGNNThree['dimEdgeFeatures'] = 1 # scalar edge weights
hParamsGNNThree['heatKernel'] = True
modelList += [hParamsGNNThree['name']]

trainingOptions = {}

'''ONLY for hidden layer parameters [at the layer-wise training] '''
paramsLayerWiseTrainGNNThree = {}
paramsLayerWiseTrainGNNThree['name'] = 'GNNThree'
paramsLayerWiseTrainGNNThree['dimNodeSignals'] = [ ] # features per hidden layer
paramsLayerWiseTrainGNNThree['nFilterTaps'] = [ ] # number of filter taps for each hidden layer
paramsLayerWiseTrainGNNThree['bias'] = True
paramsLayerWiseTrainGNNThree['nonlinearity'] = nonlinearity # nonlinearity for each hidden layer
paramsLayerWiseTrainGNNThree['dimReadout'] = [ ]
paramsLayerWiseTrainGNNThree['dimEdgeFeatures'] = 1 # scalar edge weights

#%%
if useGPU and torch.cuda.is_available():
    torch.cuda.empty_cache()

print("Selected devices:")
for thisModel in modelList:
    hParamsDict = eval('hParams' + thisModel)
    print("\t%s: %s" % (thisModel, hParamsDict['device']))

print("Generating dummy training data", end = '')
print("...", flush = True)

savingSeeds = True
data = dataTools.AerialSwarm(nAgents, commRadius, repelDist,
            nTest, 1, 1, nTest, # no care about testing, re-generating the dataset for testing
            duration, updateTime, adjustTime, 
            initVelValue, initMinDist, accelMax, savingSeeds)
savingSeeds = False

#%%
modelsGNN = {}
    
saveDir = os.path.join(saveDirRoot, testName)

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
                duration, updateTime, adjustTime,
                initVelValue, initMinDist, accelMax, savingSeeds)
print("...", flush = True)

for thisModel in list(modelsGNN.keys()):
    
    for nDAggersVal in nDAggersValues:
            
        for val in layerWise:
            modelsGNN[thisModel] = copy.deepcopy(trainedModelsGNN[layerWise.index(val)][nDAggersValues.index(nDAggersVal)][thisModel]) # 这里的代码是有问题的，按理说这里的model是训练出来的结果，至少框架结构如此
            modelsGNN[thisModel].evaluate(dataTest, nDAggersVal, val)
