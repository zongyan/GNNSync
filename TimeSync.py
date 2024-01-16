from IPython import get_ipython
get_ipython().magic('clear')
get_ipython().magic('reset -f')

import time
time.sleep(10) # wait 10 seconds

#%%
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime
import copy

import torch; torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim

import utils.dataTools as dataTools
import modules.architecturesTime as architTime
import modules.model as model
import modules.training as training
import modules.evaluation as evaluation

from utils.miscTools import saveSeed
from utils.miscTools import loadSeed
    
# Start measuring time
startRunTime = datetime.datetime.now()

#%%
thisFilename = 'TimeSync'
nAgents = 50  # number of UAVs during training 
saveDirRoot = 'experiments' 
saveDir = os.path.join(saveDirRoot, thisFilename) 

today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
saveDir = saveDir + '-%03d-' % nAgents + today
if not os.path.exists(saveDir):
    os.makedirs(saveDir)

savingSeeds = False
loadingSeeds = not savingSeeds

if savingSeeds:
    torchState = torch.get_rng_state() # PyTorch seeds
    torchSeed = torch.initial_seed() # PyTorch seeds
    numpyState = np.random.RandomState().get_state() # Numpy seeds
    
    randomStates = []
    randomStates.append({})
    randomStates[0]['module'] = 'numpy'
    randomStates[0]['state'] = numpyState
    randomStates.append({})
    randomStates[1]['module'] = 'torch'
    randomStates[1]['state'] = torchState
    randomStates[1]['seed'] = torchSeed
    
    saveSeed(randomStates, saveDirRoot)
else:    
    loadSeed(saveDirRoot)

useGPU = True
commRadius = 2. # communication radius
repelDist = 1. # minimum distance before activating repelling function
nTrain = 400 # number of training samples
nDAgger = nTrain
nValid = 20 # number of valid samples
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

nEpochs = 60 # number of epochs
batchSize = 20 # batch size
validationInterval = 5 # how many training steps to do the validation
nDAggersValues = [1] # [1, 10, 20, 30, 40, 50]
# nDAggers = 1 # 1 means no DAgger
expertProb = 0.9
aggregationSize = nDAgger

nonlinearityHidden = torch.tanh
nonlinearityOutput = torch.tanh
nonlinearity = nn.Tanh

layerWiseTraining = True
endToEndTraining = not layerWiseTraining
layerWise = [layerWiseTraining, endToEndTraining]

printInterval = 1 # after how many training steps, print the partial results
                  #   0 means to never print partial results while training

modelList = []

# hParamsGNNOne = {}
# hParamsGNNOne['name'] = 'GNNOne'
# hParamsGNNOne['archit'] = architTime.LocalGNN_DB
# hParamsGNNOne['device'] = 'cuda:0' if (useGPU and torch.cuda.is_available()) else 'cpu'
# hParamsGNNOne['dimNodeSignals'] = [2, 32, 32, 32, 2] # features per layer
# hParamsGNNOne['nFilterTaps'] = [1, 1, 1, 1] # number of filter taps
# hParamsGNNOne['bias'] = True
# hParamsGNNOne['nonlinearity'] = nonlinearity
# hParamsGNNOne['dimReadout'] = [ ] 
# hParamsGNNOne['dimEdgeFeatures'] = 1 # scalar edge weights
# modelList += [hParamsGNNOne['name']]

hParamsGNNTwo = {}
hParamsGNNTwo['name'] = 'GNNTwo'
hParamsGNNTwo['archit'] = architTime.LocalGNN_DB
hParamsGNNTwo['device'] = 'cuda:0' if (useGPU and torch.cuda.is_available()) else 'cpu'
hParamsGNNTwo['dimNodeSignals'] = [2, 32, 32, 32, 2] # features per layer
hParamsGNNTwo['nFilterTaps'] = [1, 1, 1, 1] # number of filter taps
hParamsGNNTwo['bias'] = True
hParamsGNNTwo['nonlinearity'] = nonlinearity
hParamsGNNTwo['dimReadout'] = [2, 2] 
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
hParamsGNNThree['dimReadout'] = [2, 2] 
hParamsGNNThree['dimEdgeFeatures'] = 1 # scalar edge weights
modelList += [hParamsGNNThree['name']]

# hParamsGNNFour = {}
# hParamsGNNFour['name'] = 'GNNFour'
# hParamsGNNFour['archit'] = architTime.LocalGNN_DB
# hParamsGNNFour['device'] = 'cuda:0' if (useGPU and torch.cuda.is_available()) else 'cpu'
# hParamsGNNFour['dimNodeSignals'] = [2, 32, 32, 2] # features per layer
# hParamsGNNFour['nFilterTaps'] = [1, 1, 1] # number of filter taps
# hParamsGNNFour['bias'] = True
# hParamsGNNFour['nonlinearity'] = nonlinearity
# hParamsGNNFour['dimReadout'] = [ ] 
# hParamsGNNFour['dimEdgeFeatures'] = 1 # scalar edge weights
# modelList += [hParamsGNNFour['name']]

hParamsGNNFive = {}
hParamsGNNFive['name'] = 'GNNFive'
hParamsGNNFive['archit'] = architTime.LocalGNN_DB
hParamsGNNFive['device'] = 'cuda:0' if (useGPU and torch.cuda.is_available()) else 'cpu'
hParamsGNNFive['dimNodeSignals'] = [2, 32, 32, 2] # features per layer
hParamsGNNFive['nFilterTaps'] = [1, 1, 1] # number of filter taps
hParamsGNNFive['bias'] = True
hParamsGNNFive['nonlinearity'] = nonlinearity
hParamsGNNFive['dimReadout'] = [2, 2] 
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
hParamsGNNSix['dimReadout'] = [2, 2] 
hParamsGNNSix['dimEdgeFeatures'] = 1 # scalar edge weights
modelList += [hParamsGNNSix['name']]

trainingOptions = {}
trainingOptions['printInterval'] = printInterval
trainingOptions['validationInterval'] = validationInterval

'''ONLY for hidden layer parameters [at the layer-wise training] '''
# paramsLayerWiseTrainGNNOne = {}
# paramsLayerWiseTrainGNNOne['name'] = 'GNNOne'
# paramsLayerWiseTrainGNNOne['dimNodeSignals'] = [ ] # features per hidden layer
# paramsLayerWiseTrainGNNOne['nFilterTaps'] = [ ] # number of filter taps for each hidden layer
# paramsLayerWiseTrainGNNOne['bias'] = True
# paramsLayerWiseTrainGNNOne['nonlinearity'] = nonlinearity # nonlinearity for each hidden layer
# paramsLayerWiseTrainGNNOne['dimReadout'] = [ ]
# paramsLayerWiseTrainGNNOne['dimEdgeFeatures'] = 1 # scalar edge weights

paramsLayerWiseTrainGNNTwo = {}
paramsLayerWiseTrainGNNTwo['name'] = 'GNNTwo'
paramsLayerWiseTrainGNNTwo['dimNodeSignals'] = [ ] # features per hidden layer
paramsLayerWiseTrainGNNTwo['nFilterTaps'] = [ ] # number of filter taps for each hidden layer
paramsLayerWiseTrainGNNTwo['bias'] = True
paramsLayerWiseTrainGNNTwo['nonlinearity'] = nonlinearity # nonlinearity for each hidden layer
paramsLayerWiseTrainGNNTwo['dimReadout'] = [32, 32, 32]
paramsLayerWiseTrainGNNTwo['dimEdgeFeatures'] = 1 # scalar edge weights

paramsLayerWiseTrainGNNThree = {}
paramsLayerWiseTrainGNNThree['name'] = 'GNNThree'
paramsLayerWiseTrainGNNThree['dimNodeSignals'] = [ ] # features per hidden layer
paramsLayerWiseTrainGNNThree['nFilterTaps'] = [ ] # number of filter taps for each hidden layer
paramsLayerWiseTrainGNNThree['bias'] = True
paramsLayerWiseTrainGNNThree['nonlinearity'] = nonlinearity # nonlinearity for each hidden layer
paramsLayerWiseTrainGNNThree['dimReadout'] = [32, 32, 32]
paramsLayerWiseTrainGNNThree['dimEdgeFeatures'] = 1 # scalar edge weights

# paramsLayerWiseTrainGNNFour = {}
# paramsLayerWiseTrainGNNFour['name'] = 'GNNFour'
# paramsLayerWiseTrainGNNFour['dimNodeSignals'] = [ ] # features per hidden layer
# paramsLayerWiseTrainGNNFour['nFilterTaps'] = [ ] # number of filter taps for each hidden layer
# paramsLayerWiseTrainGNNFour['bias'] = True
# paramsLayerWiseTrainGNNFour['nonlinearity'] = nonlinearity # nonlinearity for each hidden layer
# paramsLayerWiseTrainGNNFour['dimReadout'] = [ ]
# paramsLayerWiseTrainGNNFour['dimEdgeFeatures'] = 1 # scalar edge weights

paramsLayerWiseTrainGNNFive = {}
paramsLayerWiseTrainGNNFive['name'] = 'GNNFive'
paramsLayerWiseTrainGNNFive['dimNodeSignals'] = [ ] # features per hidden layer
paramsLayerWiseTrainGNNFive['nFilterTaps'] = [ ] # number of filter taps for each hidden layer
paramsLayerWiseTrainGNNFive['bias'] = True
paramsLayerWiseTrainGNNFive['nonlinearity'] = nonlinearity # nonlinearity for each hidden layer
paramsLayerWiseTrainGNNFive['dimReadout'] = [32, 32, 32]
paramsLayerWiseTrainGNNFive['dimEdgeFeatures'] = 1 # scalar edge weights

paramsLayerWiseTrainGNNSix = {}
paramsLayerWiseTrainGNNSix['name'] = 'GNNSix'
paramsLayerWiseTrainGNNSix['dimNodeSignals'] = [ ] # features per hidden layer
paramsLayerWiseTrainGNNSix['nFilterTaps'] = [ ] # number of filter taps for each hidden layer
paramsLayerWiseTrainGNNSix['bias'] = True
paramsLayerWiseTrainGNNSix['nonlinearity'] = nonlinearity # nonlinearity for each hidden layer
paramsLayerWiseTrainGNNSix['dimReadout'] = [32, 32, 32]
paramsLayerWiseTrainGNNSix['dimEdgeFeatures'] = 1 # scalar edge weights

#%%
if useGPU and torch.cuda.is_available():
    torch.cuda.empty_cache()

print("Selected devices:")
for thisModel in modelList:
    hParamsDict = eval('hParams' + thisModel)
    print("\t%s: %s" % (thisModel, hParamsDict['device']))
#%%
print("Generating data", end = '')
print("...", flush = True)

data = dataTools.AerialSwarm(nAgents, commRadius,repelDist,
            nTrain, nDAgger, nValid, 1, # no care about testing, re-generating the dataset for testing
            duration, updateTime, adjustTime, 
            initVelValue, initMinDist, accelMax, savingSeeds)

print("Preview data", end = '')
print("...", flush = True)
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

#%%
for thisModel in modelsGNN.keys():
    print("Training model %s..." % thisModel)
    
    paramsLayerWiseTrain = copy.deepcopy(eval('paramsLayerWiseTrain' + thisModel))    
    paramsLayerWiseTrain.pop('name')
    
    for nDAggersVal in nDAggersValues:
        
        for val in layerWise:
            
            modelsGNN[thisModel] = copy.deepcopy(initModelsGNN[thisModel])
            thisTrainVars = modelsGNN[thisModel].train(data, nEpochs, batchSize, \
                                                        nDAggersVal, expertProb, aggregationSize, \
                                                            paramsLayerWiseTrain, val, \
                                                                lossFunction, learningRate, beta1, beta2, **trainingOptions)

            trainedModelsGNN[layerWise.index(val)][nDAggersValues.index(nDAggersVal)][thisModel] = copy.deepcopy(modelsGNN[thisModel])

#%%
dataTest = dataTools.AerialSwarm(nAgents, commRadius, repelDist,
                1, 1, 1, nTest, # no care about training nor validation
                duration, updateTime, adjustTime,
                initVelValue, initMinDist, accelMax, savingSeeds)

offsetTest = dataTest.getData('offset', 'test')
skewTest = dataTest.getData('skew', 'test')
commGraphTest = dataTest.getData('commGraph', 'test')

dataTest.evaluate(offsetTest, skewTest, 1)
dataTest.evaluate(offsetTest[:,-1:,:,:], skewTest[:,-1:,:,:], 1)

for thisModel in list(modelsGNN.keys()):
    
    for nDAggersVal in nDAggersValues:
            
        for val in layerWise:
            modelsGNN[thisModel] = copy.deepcopy(trainedModelsGNN[layerWise.index(val)][nDAggersValues.index(nDAggersVal)][thisModel])
            modelsGNN[thisModel].evaluate(dataTest, nDAggersVal, val)

#%%
endRunTime = datetime.datetime.now()

totalRunTime = abs(endRunTime - startRunTime)
totalRunTimeH = int(divmod(totalRunTime.total_seconds(), 3600)[0])
totalRunTimeM, totalRunTimeS = \
               divmod(totalRunTime.total_seconds() - totalRunTimeH * 3600., 60)
totalRunTimeM = int(totalRunTimeM)
print("Simulation started: %s" %startRunTime.strftime("%Y/%m/%d %H:%M:%S"))
print("Simulation ended:   %s" % endRunTime.strftime("%Y/%m/%d %H:%M:%S"))
print("Total time: %dh %dm %.2fs" % (totalRunTimeH,
                                     totalRunTimeM,
                                     totalRunTimeS))    

#%%
saveDataDir = os.path.join(saveDir,'savedData')
saveArchitDir = os.path.join(saveDir,'savedArchits')
    
if layerWiseTraining == True:
    saveDataDir = os.path.join(saveDataDir,'layerWiseTraining')
    saveArchitFile = os.path.join(saveArchitDir, 'LayerWiseTraining')
elif endToEndTraining == True:
    saveDataDir = os.path.join(saveDataDir,'endToEndTraining')
    saveArchitFile = os.path.join(saveArchitDir, 'endToEndTraining')

for thisModel in modelsGNN.keys():
    
    for nDAggersVal in nDAggersValues:
        
        for val in layerWise:
            
            modelsGNN[thisModel] = copy.deepcopy(trainedModelsGNN[layerWise.index(val)][nDAggersValues.index(nDAggersVal)][thisModel])
        
            paramsLayerWiseTrain = modelsGNN[thisModel].trainer[layerWise.index(val)][nDAggersValues.index(nDAggersVal)].trainingOptions['paramsLayerWiseTrain']
            layerWiseTraining = modelsGNN[thisModel].trainer[layerWise.index(val)][nDAggersValues.index(nDAggersVal)].trainingOptions['layerWiseTraining']
            nDAggers = modelsGNN[thisModel].trainer[layerWise.index(val)][nDAggersValues.index(nDAggersVal)].trainingOptions['nDAggers']
        
            paramsNameLayerWiseTrain = list(paramsLayerWiseTrain)    
            layerWiseTrainL = len(paramsLayerWiseTrain[paramsNameLayerWiseTrain[1]])
            layerWiseTraindimReadout = paramsLayerWiseTrain[paramsNameLayerWiseTrain[4]]
        
            bestTraining = np.load(saveArchitFile + '-' + str(modelsGNN[thisModel].name) + '-nDAggers-' + str(nDAggers) + '.npz') # the data file loaded from the example folder
        
            historicalBestL = np.int64(bestTraining['historicalBestL'])
            historicalBestIteration = np.int64(bestTraining['historicalBestIteration'])
            historicalBestEpoch = np.int64(bestTraining['historicalBestEpoch'])
            historicalBestBatch = np.int64(bestTraining['historicalBestBatch'])
            
            maximumLayerWiseNum = max(np.array((layerWiseTrainL, len(layerWiseTraindimReadout))))     
        
            l = 0
            while l < maximumLayerWiseNum + 1:
            
                if layerWiseTraining == True:
                    saveDataFile = os.path.join(saveDataDir, modelsGNN[thisModel].name + '-LayerWise-' + str(historicalBestL[l]) + '-DAgger-' + str(historicalBestIteration[l]) + '-' + str(nDAggers) + '-Epoch-' + str(historicalBestEpoch[l]) + '-Batch-' + str(historicalBestBatch[l]))
                elif endToEndTraining == True:
                    saveDataFile = os.path.join(saveDataDir, modelsGNN[thisModel].name + '-EndToEnd-' + str(historicalBestL[l]) + '-DAgger-' + str(historicalBestIteration[l]) + '-' + str(nDAggers) + '-Epoch-' + str(historicalBestEpoch[l]) + '-Batch-' + str(historicalBestBatch[l]))
            
                gnn_test = np.load(saveDataFile + '.npz') # the data file loaded from the example folder
            
                matplotlib.rc('figure', max_open_warning = 0)
                
                offsetTest = gnn_test['offsetTestBest']
                skewTest = gnn_test['skewTestBest']
                adjlTest = gnn_test['adjTestBest']
                stateTest = gnn_test['stateTestBest']
                commGraphTest = gnn_test['commGraphTestBest']
            
                lossTrain = gnn_test['lossTrain']
                accValid = gnn_test['accValid']    
                        
                # plot the velocity of all agents via the GNN method
                for i in range(0, nTest, 11):
                    plt.figure()
                    plt.rcParams["figure.figsize"] = (6.4,4.8)
                    for j in range(0, nAgents, 1):
                        # the input and output features are two dimensions, which means that one 
                        # dimension is for x-axis velocity, the other one is for y-axis velocity 
                        plt.plot(np.arange(0, np.int32(duration/updateTime), 1), offsetTest[i, :, 0, j]) 
                        plt.xlim((0, np.int32(duration/updateTime)))
                        plt.xticks(np.arange(0, np.int32(duration/updateTime)+1, 2/updateTime), np.arange(0, np.int32(duration/adjustTime)+1, 2/adjustTime))
                    # end for 
                    plt.xlabel(r'$time (s)$')
                    plt.ylabel(r'${\bf \theta}_{gnn}$')
                    plt.title(r'${\bf \theta}_{gnn}$ for ' + str(50)+ ' agents (gnn controller)')
                    plt.grid()
                    plt.show()    
                # end for
                
                # plot the velocity of all agents via the centralised optimal controller
                for i in range(0, nTest, 11):
                    plt.figure()
                    plt.rcParams["figure.figsize"] = (6.4,4.8)
                    for j in range(0, nAgents, 1):
                        # the input and output features are two dimensions, which means that one 
                        # dimension is for x-axis velocity, the other one is for y-axis velocity 
                        plt.plot(np.arange(0, np.int32(duration/updateTime), 1), skewTest[i, :, 0, j]) 
                        plt.xlim((0, np.int32(duration/updateTime)))
                        plt.xticks(np.arange(0, np.int32(duration/updateTime)+1, 2/updateTime), np.arange(0, np.int32(duration/adjustTime)+1,2/adjustTime))
                    # end for 
                    plt.xlabel(r'$time (s)$')
                    plt.ylabel(r'${\bf \gamma}_{gnn}$')
                    plt.title(r'$\bf \gamma_{gnn}$ for ' + str(50)+ ' agents (centralised controller)')
                    plt.grid()
                    plt.show()    
                # end for
                
                offsetTest = offsetTest[:, 350:-1, :, :]
                skewTest = skewTest[:, 350:-1, :, :]
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
                            
                l = l + 1
        
for i in range(lossTrain.shape[0]):
    for j in range(lossTrain.shape[1]):
        
        plt.figure()
        plt.rcParams["figure.figsize"] = (6.4,4.8)            
        plt.plot(np.reshape(lossTrain[i, j, :, :], (lossTrain.shape[2] * lossTrain.shape[3])))

        plt.figure()
        plt.rcParams["figure.figsize"] = (6.4,4.8)            
        plt.plot(np.reshape(accValid[i, j, :, :], (accValid.shape[2] * accValid.shape[3])))   