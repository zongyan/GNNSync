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
import torch.nn.functional as F

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

testName = "TimeSync-050-20240103162844"

#%%
nAgents = 50  # number of UAVs during training 

loadSeed('./experiments/test1') # loading the states and seed in Test 2

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

nDAggersValues = [1] # [1, 10, 20, 30, 40, 50]

nonlinearity = nn.Tanh

layerWiseTraining = True
endToEndTraining = not layerWiseTraining
layerWise = [layerWiseTraining, endToEndTraining]

evalModel = True

#%%
modelList = []

hParamsGNNOne = {}
hParamsGNNOne['name'] = 'GNNOne'
hParamsGNNOne['archit'] = architTime.LocalGNN_DB
hParamsGNNOne['device'] = 'cuda:0' if (useGPU and torch.cuda.is_available()) else 'cpu'
hParamsGNNOne['dimNodeSignals'] = [2, 32, 32, 32, 2] # features per layer
hParamsGNNOne['nFilterTaps'] = [1, 1, 1, 1] # number of filter taps
hParamsGNNOne['bias'] = True
hParamsGNNOne['nonlinearity'] = nonlinearity
hParamsGNNOne['dimReadout'] = [ ] 
hParamsGNNOne['dimEdgeFeatures'] = 1 # scalar edge weights
modelList += [hParamsGNNOne['name']]

hParamsGNNTwo = {}
hParamsGNNTwo['name'] = 'GNNTwo'
hParamsGNNTwo['archit'] = architTime.LocalGNN_DB
hParamsGNNTwo['device'] = 'cuda:0' if (useGPU and torch.cuda.is_available()) else 'cpu'
hParamsGNNTwo['dimNodeSignals'] = [2, 32, 32, 32, 2] # features per layer
hParamsGNNTwo['nFilterTaps'] = [1, 1, 1, 1] # number of filter taps
hParamsGNNTwo['bias'] = True
hParamsGNNTwo['nonlinearity'] = nonlinearity
hParamsGNNTwo['dimReadout'] = [2] 
hParamsGNNTwo['dimEdgeFeatures'] = 1 # scalar edge weights
modelList += [hParamsGNNTwo['name']]

hParamsGNNThree = {}
hParamsGNNThree['name'] = 'GNNThree'
hParamsGNNThree['archit'] = architTime.LocalGNN_DB
hParamsGNNThree['device'] = 'cuda:0' if (useGPU and torch.cuda.is_available()) else 'cpu'
hParamsGNNThree['dimNodeSignals'] = [2, 32, 32, 32] # features per layer
hParamsGNNThree['nFilterTaps'] = [1, 1, 1] # number of filter taps
hParamsGNNThree['bias'] = True
hParamsGNNThree['nonlinearity'] = nonlinearity
hParamsGNNThree['dimReadout'] = [2] 
hParamsGNNThree['dimEdgeFeatures'] = 1 # scalar edge weights
modelList += [hParamsGNNThree['name']]

hParamsGNNFour = {}
hParamsGNNFour['name'] = 'GNNFour'
hParamsGNNFour['archit'] = architTime.LocalGNN_DB
hParamsGNNFour['device'] = 'cuda:0' if (useGPU and torch.cuda.is_available()) else 'cpu'
hParamsGNNFour['dimNodeSignals'] = [2, 32, 32, 2] # features per layer
hParamsGNNFour['nFilterTaps'] = [1, 1, 1] # number of filter taps
hParamsGNNFour['bias'] = True
hParamsGNNFour['nonlinearity'] = nonlinearity
hParamsGNNFour['dimReadout'] = [ ] 
hParamsGNNFour['dimEdgeFeatures'] = 1 # scalar edge weights
modelList += [hParamsGNNFour['name']]

hParamsGNNFive = {}
hParamsGNNFive['name'] = 'GNNFive'
hParamsGNNFive['archit'] = architTime.LocalGNN_DB
hParamsGNNFive['device'] = 'cuda:0' if (useGPU and torch.cuda.is_available()) else 'cpu'
hParamsGNNFive['dimNodeSignals'] = [2, 32, 32, 2] # features per layer
hParamsGNNFive['nFilterTaps'] = [1, 1, 1] # number of filter taps
hParamsGNNFive['bias'] = True
hParamsGNNFive['nonlinearity'] = nonlinearity
hParamsGNNFive['dimReadout'] = [2] 
hParamsGNNFive['dimEdgeFeatures'] = 1 # scalar edge weights
modelList += [hParamsGNNFive['name']]

hParamsGNNSix = {}
hParamsGNNSix['name'] = 'GNNSix'
hParamsGNNSix['archit'] = architTime.LocalGNN_DB
hParamsGNNSix['device'] = 'cuda:0' if (useGPU and torch.cuda.is_available()) else 'cpu'
hParamsGNNSix['dimNodeSignals'] = [2, 32, 32] # features per layer
hParamsGNNSix['nFilterTaps'] = [1, 1] # number of filter taps
hParamsGNNSix['bias'] = True
hParamsGNNSix['nonlinearity'] = nonlinearity
hParamsGNNSix['dimReadout'] = [2] 
hParamsGNNSix['dimEdgeFeatures'] = 1 # scalar edge weights
modelList += [hParamsGNNSix['name']]

trainingOptions = {}

'''ONLY for hidden layer parameters [at the layer-wise training] '''
paramsLayerWiseTrainGNNOne = {}
paramsLayerWiseTrainGNNOne['name'] = 'GNNOne'
paramsLayerWiseTrainGNNOne['dimNodeSignals'] = [ ] # features per hidden layer
paramsLayerWiseTrainGNNOne['nFilterTaps'] = [ ] # number of filter taps for each hidden layer
paramsLayerWiseTrainGNNOne['bias'] = True
paramsLayerWiseTrainGNNOne['nonlinearity'] = nonlinearity # nonlinearity for each hidden layer
paramsLayerWiseTrainGNNOne['dimReadout'] = [ ]
paramsLayerWiseTrainGNNOne['dimEdgeFeatures'] = 1 # scalar edge weights

paramsLayerWiseTrainGNNTwo = {}
paramsLayerWiseTrainGNNTwo['name'] = 'GNNTwo'
paramsLayerWiseTrainGNNTwo['dimNodeSignals'] = [ ] # features per hidden layer
paramsLayerWiseTrainGNNTwo['nFilterTaps'] = [ ] # number of filter taps for each hidden layer
paramsLayerWiseTrainGNNTwo['bias'] = True
paramsLayerWiseTrainGNNTwo['nonlinearity'] = nonlinearity # nonlinearity for each hidden layer
paramsLayerWiseTrainGNNTwo['dimReadout'] = [32, 32, 32, 32, 32, 32]
paramsLayerWiseTrainGNNTwo['dimEdgeFeatures'] = 1 # scalar edge weights

paramsLayerWiseTrainGNNThree = {}
paramsLayerWiseTrainGNNThree['name'] = 'GNNThree'
paramsLayerWiseTrainGNNThree['dimNodeSignals'] = [ ] # features per hidden layer
paramsLayerWiseTrainGNNThree['nFilterTaps'] = [ ] # number of filter taps for each hidden layer
paramsLayerWiseTrainGNNThree['bias'] = True
paramsLayerWiseTrainGNNThree['nonlinearity'] = nonlinearity # nonlinearity for each hidden layer
paramsLayerWiseTrainGNNThree['dimReadout'] = [32, 32, 32, 32, 32, 32]
paramsLayerWiseTrainGNNThree['dimEdgeFeatures'] = 1 # scalar edge weights

paramsLayerWiseTrainGNNFour = {}
paramsLayerWiseTrainGNNFour['name'] = 'GNNFour'
paramsLayerWiseTrainGNNFour['dimNodeSignals'] = [ ] # features per hidden layer
paramsLayerWiseTrainGNNFour['nFilterTaps'] = [ ] # number of filter taps for each hidden layer
paramsLayerWiseTrainGNNFour['bias'] = True
paramsLayerWiseTrainGNNFour['nonlinearity'] = nonlinearity # nonlinearity for each hidden layer
paramsLayerWiseTrainGNNFour['dimReadout'] = [ ]
paramsLayerWiseTrainGNNFour['dimEdgeFeatures'] = 1 # scalar edge weights

paramsLayerWiseTrainGNNFive = {}
paramsLayerWiseTrainGNNFive['name'] = 'GNNFive'
paramsLayerWiseTrainGNNFive['dimNodeSignals'] = [ ] # features per hidden layer
paramsLayerWiseTrainGNNFive['nFilterTaps'] = [ ] # number of filter taps for each hidden layer
paramsLayerWiseTrainGNNFive['bias'] = True
paramsLayerWiseTrainGNNFive['nonlinearity'] = nonlinearity # nonlinearity for each hidden layer
paramsLayerWiseTrainGNNFive['dimReadout'] = [32, 32, 32, 32, 32, 32]
paramsLayerWiseTrainGNNFive['dimEdgeFeatures'] = 1 # scalar edge weights

paramsLayerWiseTrainGNNSix = {}
paramsLayerWiseTrainGNNSix['name'] = 'GNNSix'
paramsLayerWiseTrainGNNSix['dimNodeSignals'] = [ ] # features per hidden layer
paramsLayerWiseTrainGNNSix['nFilterTaps'] = [ ] # number of filter taps for each hidden layer
paramsLayerWiseTrainGNNSix['bias'] = True
paramsLayerWiseTrainGNNSix['nonlinearity'] = nonlinearity # nonlinearity for each hidden layer
paramsLayerWiseTrainGNNSix['dimReadout'] = [32, 32, 32, 32, 32, 32]
paramsLayerWiseTrainGNNSix['dimEdgeFeatures'] = 1 # scalar edge weights

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

#%%




#%%
# addedLayerNum = np.array([9, 4, 4, 4, 9, 4, 4, 4])
# addedLayerName = np.array(['1 layers', '2 layers', '3 layers', '4 layers', '5 layers', '6 layers', '7 layers', '8 layers', '9 layers'])
# gnnName = np.array(['GNNOne', 'GNNTwo', 'GNNThree', 'GNNFour', 'GNNFive', 'GNNSix', 'GNNSeven', 'GNNEight'])
# saveDirRoot = 'experiments' 
# dataFolder = os.listdir(saveDirRoot)

# for j in range(len(dataFolder)):
    
#     saveDir = os.path.join(saveDirRoot, os.listdir(saveDirRoot)[j]) 
#     if not os.path.exists(saveDir):
#         raise Exception("error in finding data folder!")    
#     saveDir = os.path.join(saveDir, "savedData")
#     savedEndToEndData = os.path.join(saveDir, os.listdir(saveDir)[0])
#     savedLayerWiseData = os.path.join(saveDir, os.listdir(saveDir)[1])
    
#     print("Computing the cost of each best %s model in the layer-wise training..." %(gnnName[j]))    
#     for i in range(addedLayerNum[j]):
        
#         testingData = np.load(os.path.join(savedLayerWiseData, os.listdir(savedLayerWiseData)[i]), allow_pickle=True)    
        
#         testingData = np.load('experiments\\TimeSync-050-20231128143805\\savedArchits\\baseGNNOne-nDAggers-1-1-EndToEnd-0-GSO-[2, 32, 2]-Readout-[].npz', \
#                               allow_pickle=True)     
            
#         nLayers = copy.deepcopy(testingData['lastL'])
#         dimNodeSignals = copy.deepcopy(testingData['lastF'])
#         nFilterTaps = copy.deepcopy(testingData['lastK'])
#         dimEdgeFeatures = copy.deepcopy(testingData['lastE'])
#         bias = copy.deepcopy(testingData['lastBias'])
#         nonlinearity = copy.deepcopy(testingData['lastSigma'])
#         dimReadout = copy.deepcopy(testingData['lastReadout'])

#         import os
#         from pathlib import Path
        
#         yy = sorted(Path('experiments\\TimeSync-050-20231128143805\\savedModels\\endToEndTraining').iterdir(), key=os.path.getmtime)

#         os.listdir('experiments\\TimeSync-050-20231128143805\\savedArchits')
        
#         os.listdir('experiments\\TimeSync-050-20231128143805\\savedModels\\endToEndTraining')        
        
#         yy = os.listdir('experiments\\TimeSync-050-20231128143805\\savedModels\\endToEndTraining')        
                
#         mm = []
        
#         for element in yy:            
#             if np.int64(str(element)[89]) == 0:
#                 mm.append(str(element)[69:])

#         for i in range(len(mm)):
#             if list(reversed(mm))[i][-9:-5] == 'Best':
#                 print(i)
#                 break
            
        
        
            
        
#         xx = os.listdir('experiments\\TimeSync-050-20231128143805\\savedModels\\layerWiseTraining')        
        
#         xx[0][21]
#         xx[0][-9:-5]
        
#         if layerWiseTraining == True:                          
#             # reload best model for layer-wise training
#             self.model.load(layerWiseTraining, nDAggers, bestL, bestIteration, bestEpoch, bestBatch, label = 'Best')


#             if layerWiseTraining == True:
#                 saveFile = os.path.join(saveModelDir, self.name + '-LayerWise-' + str(l) + '-DAgger-' + str(iteration) + '-' + str(nDAggers) + '-Epoch-' + str(epoch) + '-Batch-' + str(batch))
#             else:
#                 saveFile = os.path.join(saveModelDir, self.name + '-EndToEnd-' + str(l) + '-DAgger-' + str(iteration) + '-' + str(nDAggers) + '-Epoch-' + str(epoch) + '-Batch-' + str(batch))
            
#         architLoadFile = saveFile + '-Archit-' + label +'.ckpt' # list(reversed(mm))[i+1]
#         optimLoadFile = saveFile + '-Optim-' + label + '.ckpt' # list(reversed(mm))[i]
#         self.archit.load_state_dict(torch.load(architLoadFile))
#         self.optim.load_state_dict(torch.load(optimLoadFile))

# #%% calculating the output values (in the spatial domain) after the tanh function
# m = nn.Tanh()
# input = torch.randn(2)
# oupt = m(input)
# print(oupt)

# oupt = F.tanh(input)
# print(oupt)

# #%% calculating the values in the spectral domain

# A = [[0,1,0,1,0]
#     ,[1,0,1,0,0]
#     ,[0,1,0,1,0]
#     ,[1,0,1,0,1]
#     ,[0,0,0,1,0]]

# yyy = np.array([1, 2, 3, 4, 5])

# A = np.array(A)
# D = np.sum(A,axis=1)
# L = np.diag(D) - A  # Non-Normalized Laplacian

# eigen_values, eigen_vectors = np.linalg.eigh(L)

# zzz = np.matmul(eigen_vectors, yyy)


# #%%
# plt.figure()
# plt.rcParams["figure.figsize"] = (6.4,4.8)            
# plt.plot(zzz, linestyle='--', marker='o', color='b', label='line with marker')