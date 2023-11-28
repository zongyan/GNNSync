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

savingSeeds = True
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
nDAggersValues = [1, 10, 20, 30, 40, 50]
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

hParamsbaseGNNOne = {}
hParamsbaseGNNOne['name'] = 'baseGNNOne'
hParamsbaseGNNOne['archit'] = architTime.LocalGNN_DB
hParamsbaseGNNOne['device'] = 'cuda:0' if (useGPU and torch.cuda.is_available()) else 'cpu'
hParamsbaseGNNOne['dimNodeSignals'] = [2, 32, 2] # features per layer
hParamsbaseGNNOne['nFilterTaps'] = [1, 1] # number of filter taps
hParamsbaseGNNOne['bias'] = True
hParamsbaseGNNOne['nonlinearity'] = nonlinearity
hParamsbaseGNNOne['dimReadout'] = [ ] 
hParamsbaseGNNOne['dimEdgeFeatures'] = 1 # scalar edge weights
modelList += [hParamsbaseGNNOne['name']]

hParamsbaseGNNTwo = {}
hParamsbaseGNNTwo['name'] = 'baseGNNTwo'
hParamsbaseGNNTwo['archit'] = architTime.LocalGNN_DB
hParamsbaseGNNTwo['device'] = 'cuda:0' if (useGPU and torch.cuda.is_available()) else 'cpu'
hParamsbaseGNNTwo['dimNodeSignals'] = [2, 32, 2] # features per layer
hParamsbaseGNNTwo['nFilterTaps'] = [2, 2] # number of filter taps
hParamsbaseGNNTwo['bias'] = True
hParamsbaseGNNTwo['nonlinearity'] = nonlinearity
hParamsbaseGNNTwo['dimReadout'] = [ ] 
hParamsbaseGNNTwo['dimEdgeFeatures'] = 1 # scalar edge weights
modelList += [hParamsbaseGNNTwo['name']]

hParamsbaseGNNThree = {}
hParamsbaseGNNThree['name'] = 'baseGNNThree'
hParamsbaseGNNThree['archit'] = architTime.LocalGNN_DB
hParamsbaseGNNThree['device'] = 'cuda:0' if (useGPU and torch.cuda.is_available()) else 'cpu'
hParamsbaseGNNThree['dimNodeSignals'] = [2, 32, 2] # features per layer
hParamsbaseGNNThree['nFilterTaps'] = [3, 3] # number of filter taps
hParamsbaseGNNThree['bias'] = True
hParamsbaseGNNThree['nonlinearity'] = nonlinearity
hParamsbaseGNNThree['dimReadout'] = [ ] 
hParamsbaseGNNThree['dimEdgeFeatures'] = 1 # scalar edge weights
modelList += [hParamsbaseGNNThree['name']]

hParamsbaseGNNFour = {}
hParamsbaseGNNFour['name'] = 'baseGNNFour'
hParamsbaseGNNFour['archit'] = architTime.LocalGNN_DB
hParamsbaseGNNFour['device'] = 'cuda:0' if (useGPU and torch.cuda.is_available()) else 'cpu'
hParamsbaseGNNFour['dimNodeSignals'] = [2, 32, 2] # features per layer
hParamsbaseGNNFour['nFilterTaps'] = [4, 4] # number of filter taps
hParamsbaseGNNFour['bias'] = True
hParamsbaseGNNFour['nonlinearity'] = nonlinearity
hParamsbaseGNNFour['dimReadout'] = [ ] 
hParamsbaseGNNFour['dimEdgeFeatures'] = 1 # scalar edge weights
modelList += [hParamsbaseGNNFour['name']]

hParamsbaseGNNFive = {}
hParamsbaseGNNFive['name'] = 'baseGNNFive'
hParamsbaseGNNFive['archit'] = architTime.LocalGNN_DB
hParamsbaseGNNFive['device'] = 'cuda:0' if (useGPU and torch.cuda.is_available()) else 'cpu'
hParamsbaseGNNFive['dimNodeSignals'] = [2, 64, 2] # features per layer
hParamsbaseGNNFive['nFilterTaps'] = [1, 1] # number of filter taps
hParamsbaseGNNFive['bias'] = True
hParamsbaseGNNFive['nonlinearity'] = nonlinearity
hParamsbaseGNNFive['dimReadout'] = [ ] 
hParamsbaseGNNFive['dimEdgeFeatures'] = 1 # scalar edge weights
modelList += [hParamsbaseGNNFive['name']]

hParamsbaseGNNSix = {}
hParamsbaseGNNSix['name'] = 'baseGNNSix'
hParamsbaseGNNSix['archit'] = architTime.LocalGNN_DB
hParamsbaseGNNSix['device'] = 'cuda:0' if (useGPU and torch.cuda.is_available()) else 'cpu'
hParamsbaseGNNSix['dimNodeSignals'] = [2, 64, 2] # features per layer
hParamsbaseGNNSix['nFilterTaps'] = [2, 2] # number of filter taps
hParamsbaseGNNSix['bias'] = True
hParamsbaseGNNSix['nonlinearity'] = nonlinearity
hParamsbaseGNNSix['dimReadout'] = [ ] 
hParamsbaseGNNSix['dimEdgeFeatures'] = 1 # scalar edge weights
modelList += [hParamsbaseGNNSix['name']]

hParamsbaseGNNSeven = {}
hParamsbaseGNNSeven['name'] = 'baseGNNSeven'
hParamsbaseGNNSeven['archit'] = architTime.LocalGNN_DB
hParamsbaseGNNSeven['device'] = 'cuda:0' if (useGPU and torch.cuda.is_available()) else 'cpu'
hParamsbaseGNNSeven['dimNodeSignals'] = [2, 64, 2] # features per layer
hParamsbaseGNNSeven['nFilterTaps'] = [3, 3] # number of filter taps
hParamsbaseGNNSeven['bias'] = True
hParamsbaseGNNSeven['nonlinearity'] = nonlinearity
hParamsbaseGNNSeven['dimReadout'] = [ ] 
hParamsbaseGNNSeven['dimEdgeFeatures'] = 1 # scalar edge weights
modelList += [hParamsbaseGNNSeven['name']]

hParamsbaseGNNEight = {}
hParamsbaseGNNEight['name'] = 'baseGNNEight'
hParamsbaseGNNEight['archit'] = architTime.LocalGNN_DB
hParamsbaseGNNEight['device'] = 'cuda:0' if (useGPU and torch.cuda.is_available()) else 'cpu'
hParamsbaseGNNEight['dimNodeSignals'] = [2, 64, 2] # features per layer
hParamsbaseGNNEight['nFilterTaps'] = [4, 4] # number of filter taps
hParamsbaseGNNEight['bias'] = True
hParamsbaseGNNEight['nonlinearity'] = nonlinearity
hParamsbaseGNNEight['dimReadout'] = [ ] 
hParamsbaseGNNEight['dimEdgeFeatures'] = 1 # scalar edge weights
modelList += [hParamsbaseGNNEight['name']]

trainingOptions = {}
trainingOptions['printInterval'] = printInterval
trainingOptions['validationInterval'] = validationInterval

'''ONLY for hidden layer parameters [at the layer-wise training] '''
paramsLayerWiseTrainbaseGNNOne = {}
paramsLayerWiseTrainbaseGNNOne['name'] = 'baseGNNOne'
paramsLayerWiseTrainbaseGNNOne['dimNodeSignals'] = [32, 32, 32, 32, 32, 32, 32, 32, 32] # features per hidden layer
paramsLayerWiseTrainbaseGNNOne['nFilterTaps'] = [1, 1, 1, 1, 1, 1, 1, 1, 1] # number of filter taps for each hidden layer
paramsLayerWiseTrainbaseGNNOne['bias'] = True
paramsLayerWiseTrainbaseGNNOne['nonlinearity'] = nonlinearity # nonlinearity for each hidden layer
paramsLayerWiseTrainbaseGNNOne['dimReadout'] = [ ]
paramsLayerWiseTrainbaseGNNOne['dimEdgeFeatures'] = 1 # scalar edge weights

paramsLayerWiseTrainbaseGNNTwo = {}
paramsLayerWiseTrainbaseGNNTwo['name'] = 'baseGNNTwo'
paramsLayerWiseTrainbaseGNNTwo['dimNodeSignals'] = [32, 32, 32, 32] # features per hidden layer
paramsLayerWiseTrainbaseGNNTwo['nFilterTaps'] = [2, 2, 2, 2] # number of filter taps for each hidden layer
paramsLayerWiseTrainbaseGNNTwo['bias'] = True
paramsLayerWiseTrainbaseGNNTwo['nonlinearity'] = nonlinearity # nonlinearity for each hidden layer
paramsLayerWiseTrainbaseGNNTwo['dimReadout'] = [ ]
paramsLayerWiseTrainbaseGNNTwo['dimEdgeFeatures'] = 1 # scalar edge weights

paramsLayerWiseTrainbaseGNNThree = {}
paramsLayerWiseTrainbaseGNNThree['name'] = 'baseGNNThree'
paramsLayerWiseTrainbaseGNNThree['dimNodeSignals'] = [32, 32, 32, 32] # features per hidden layer
paramsLayerWiseTrainbaseGNNThree['nFilterTaps'] = [3, 3, 3, 3] # number of filter taps for each hidden layer
paramsLayerWiseTrainbaseGNNThree['bias'] = True
paramsLayerWiseTrainbaseGNNThree['nonlinearity'] = nonlinearity # nonlinearity for each hidden layer
paramsLayerWiseTrainbaseGNNThree['dimReadout'] = [ ]
paramsLayerWiseTrainbaseGNNThree['dimEdgeFeatures'] = 1 # scalar edge weights

paramsLayerWiseTrainbaseGNNFour = {}
paramsLayerWiseTrainbaseGNNFour['name'] = 'baseGNNFour'
paramsLayerWiseTrainbaseGNNFour['dimNodeSignals'] = [32, 32, 32, 32] # features per hidden layer
paramsLayerWiseTrainbaseGNNFour['nFilterTaps'] = [4, 4, 4, 4] # number of filter taps for each hidden layer
paramsLayerWiseTrainbaseGNNFour['bias'] = True
paramsLayerWiseTrainbaseGNNFour['nonlinearity'] = nonlinearity # nonlinearity for each hidden layer
paramsLayerWiseTrainbaseGNNFour['dimReadout'] = [ ]
paramsLayerWiseTrainbaseGNNFour['dimEdgeFeatures'] = 1 # scalar edge weights

paramsLayerWiseTrainbaseGNNFive = {}
paramsLayerWiseTrainbaseGNNFive['name'] = 'baseGNNFive'
paramsLayerWiseTrainbaseGNNFive['dimNodeSignals'] = [64, 64, 64, 64, 64, 64, 64, 64, 64] # features per hidden layer
paramsLayerWiseTrainbaseGNNFive['nFilterTaps'] = [1, 1, 1, 1, 1, 1, 1, 1, 1] # number of filter taps for each hidden layer
paramsLayerWiseTrainbaseGNNFive['bias'] = True
paramsLayerWiseTrainbaseGNNFive['nonlinearity'] = nonlinearity # nonlinearity for each hidden layer
paramsLayerWiseTrainbaseGNNFive['dimReadout'] = [ ]
paramsLayerWiseTrainbaseGNNFive['dimEdgeFeatures'] = 1 # scalar edge weights

paramsLayerWiseTrainbaseGNNSix = {}
paramsLayerWiseTrainbaseGNNSix['name'] = 'baseGNNSix'
paramsLayerWiseTrainbaseGNNSix['dimNodeSignals'] = [64, 64, 64, 64] # features per hidden layer
paramsLayerWiseTrainbaseGNNSix['nFilterTaps'] = [2, 2, 2, 2] # number of filter taps for each hidden layer
paramsLayerWiseTrainbaseGNNSix['bias'] = True
paramsLayerWiseTrainbaseGNNSix['nonlinearity'] = nonlinearity # nonlinearity for each hidden layer
paramsLayerWiseTrainbaseGNNSix['dimReadout'] = [ ]
paramsLayerWiseTrainbaseGNNSix['dimEdgeFeatures'] = 1 # scalar edge weights

paramsLayerWiseTrainbaseGNNSeven = {}
paramsLayerWiseTrainbaseGNNSeven['name'] = 'baseGNNSeven'
paramsLayerWiseTrainbaseGNNSeven['dimNodeSignals'] = [64, 64, 64, 64] # features per hidden layer
paramsLayerWiseTrainbaseGNNSeven['nFilterTaps'] = [3, 3, 3, 3] # number of filter taps for each hidden layer
paramsLayerWiseTrainbaseGNNSeven['bias'] = True
paramsLayerWiseTrainbaseGNNSeven['nonlinearity'] = nonlinearity # nonlinearity for each hidden layer
paramsLayerWiseTrainbaseGNNSeven['dimReadout'] = [ ]
paramsLayerWiseTrainbaseGNNSeven['dimEdgeFeatures'] = 1 # scalar edge weights

paramsLayerWiseTrainbaseGNNEight = {}
paramsLayerWiseTrainbaseGNNEight['name'] = 'baseGNNEight'
paramsLayerWiseTrainbaseGNNEight['dimNodeSignals'] = [64, 64, 64, 64] # features per hidden layer
paramsLayerWiseTrainbaseGNNEight['nFilterTaps'] = [4, 4, 4, 4] # number of filter taps for each hidden layer
paramsLayerWiseTrainbaseGNNEight['bias'] = True
paramsLayerWiseTrainbaseGNNEight['nonlinearity'] = nonlinearity # nonlinearity for each hidden layer
paramsLayerWiseTrainbaseGNNEight['dimReadout'] = [ ]
paramsLayerWiseTrainbaseGNNEight['dimEdgeFeatures'] = 1 # scalar edge weights

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

offsetTest = dataTest.getData('offset', 'train')
skewTest = dataTest.getData('skew', 'train')
commGraphTest = dataTest.getData('commGraph', 'train')

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