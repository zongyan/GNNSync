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
from copy import deepcopy

import torch; torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim

import utils.dataTools as dataTools
import modules.architecturesTime as architTime
import modules.model as model
import modules.training as training
import modules.evaluation as evaluation

# Start measuring time
startRunTime = datetime.datetime.now()

#%%
thisFilename = 'TimeSync'
nAgents = 30  # number of UAVs during training 
saveDirRoot = 'experiments' 
saveDir = os.path.join(saveDirRoot, thisFilename) 

today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
saveDir = saveDir + '-%03d-' % nAgents + today
if not os.path.exists(saveDir):
    os.makedirs(saveDir)

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

nEpochs = 30 # number of epochs
batchSize = 32 # batch size
validationInterval = 5 # how many training steps to do the validation
nDAggersValues = [1, 20, 30]
nDAggers = 1 # 1 means no DAgger
expertProb = 0.9
aggregationSize = nDAgger

nonlinearityHidden = torch.tanh
nonlinearityOutput = torch.tanh
nonlinearity = nn.Tanh

layerWiseTraining = True
endToEndTraining = not layerWiseTraining

printInterval = 1 # after how many training steps, print the partial results
                  #   0 means to never print partial results while training

modelList = []

hParamsbaseGNN = {}
hParamsbaseGNN['name'] = 'baseGNN'
hParamsbaseGNN['archit'] = architTime.LocalGNN_DB
hParamsbaseGNN['device'] = 'cuda:0' if (useGPU and torch.cuda.is_available()) else 'cpu'
hParamsbaseGNN['dimNodeSignals'] = [2, 16, 2] # features per layer
hParamsbaseGNN['nFilterTaps'] = [1, 1] # number of filter taps
hParamsbaseGNN['bias'] = True
hParamsbaseGNN['nonlinearity'] = nonlinearity
hParamsbaseGNN['dimReadout'] = [ ] 
hParamsbaseGNN['dimEdgeFeatures'] = 1 # scalar edge weights
modelList += [hParamsbaseGNN['name']]

trainingOptions = {}
trainingOptions['printInterval'] = printInterval
trainingOptions['validationInterval'] = validationInterval

'''ONLY for hidden layer parameters [at the layer-wise training] '''
paramsLayerWiseTrain = {}
paramsLayerWiseTrain['dimNodeSignals'] = [16, 16, 16] # features per hidden layer
paramsLayerWiseTrain['nFilterTaps'] = [1, 1, 1] # number of filter taps for each hidden layer
paramsLayerWiseTrain['bias'] = True
paramsLayerWiseTrain['nonlinearity'] = nonlinearity # nonlinearity for each hidden layer
paramsLayerWiseTrain['dimReadout'] = [ ]
paramsLayerWiseTrain['dimEdgeFeatures'] = 1 # scalar edge weights

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
            initVelValue, initMinDist, accelMax,
            normalizeGraph)

print("Preview data", end = '')
print("...", flush = True)
#%%
modelsGNN = {}

print("Model initialisation...", flush = True)

for thisModel in modelList:
    hParamsDict = deepcopy(eval('hParams' + thisModel))
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
                               saveDir)
    modelsGNN[thisName] = modelCreated
    print("OK")
    
initModelsGNN = deepcopy(modelsGNN)

trainedModelsGNN = [ ]
for i in range(len(nDAggersValues)):
    trainedModelsGNN.append(initModelsGNN)

#%%
i = 0
for thisModel in modelsGNN.keys():
    print("Training model %s..." % thisModel)
    
    for nDAggersVal in nDAggersValues:
        if nDAggersValues.index(nDAggersVal) == 0:
            thisTrainVars = modelsGNN[thisModel].train(data, nEpochs, batchSize, \
                                                        nDAggersVal, expertProb, aggregationSize, \
                                                            paramsLayerWiseTrain, layerWiseTraining, endToEndTraining, \
                                                                lossFunction, learningRate, beta1, beta2, **trainingOptions)
        else:
            modelsGNN[thisModel] = deepcopy(initModelsGNN[thisModel])
            thisTrainVars = modelsGNN[thisModel].train(data, nEpochs, batchSize, \
                                                        nDAggersVal, expertProb, aggregationSize, \
                                                            paramsLayerWiseTrain, layerWiseTraining, endToEndTraining, \
                                                                lossFunction, learningRate, beta1, beta2, **trainingOptions)            
        
        trainedModelsGNN[i + nDAggersValues.index(nDAggersVal)] = deepcopy(modelsGNN[thisModel])
    
    i = i + len(nDAggersValues)
#%%
dataTest = dataTools.AerialSwarm(nAgents, commRadius, repelDist,
                1, 1, 1, nTest, # no care about training nor validation
                duration, updateTime, adjustTime,
                initVelValue, initMinDist, accelMax)

offsetTest = dataTest.getData('offset', 'train')
skewTest = dataTest.getData('skew', 'train')
commGraphTest = dataTest.getData('commGraph', 'train')

dataTest.evaluate(offsetTest, skewTest, 1)
dataTest.evaluate(offsetTest[:,-1:,:,:], skewTest[:,-1:,:,:], 1)

for thisModel in list(modelsGNN.keys()):
    
    for nDAggersVal in nDAggersValues:
        modelsGNN[thisModel] = deepcopy(trainedModelsGNN[list(modelsGNN.keys()).index(thisModel) + nDAggersValues.index(nDAggersVal)])
        modelsGNN[thisModel].evaluate(dataTest, nDAggersVal)  
    
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

        modelsGNN[thisModel] = deepcopy(trainedModelsGNN[list(modelsGNN.keys()).index(thisModel) + nDAggersValues.index(nDAggersVal)])
        
        paramsLayerWiseTrain = modelsGNN[thisModel].trainer[nDAggersValues.index(nDAggersVal)].trainingOptions['paramsLayerWiseTrain']
        layerWiseTraining = modelsGNN[thisModel].trainer[nDAggersValues.index(nDAggersVal)].trainingOptions['layerWiseTraining']
        endToEndTraining = modelsGNN[thisModel].trainer[nDAggersValues.index(nDAggersVal)].trainingOptions['endToEndTraining']
        nDAggers = modelsGNN[thisModel].trainer[nDAggersValues.index(nDAggersVal)].trainingOptions['nDAggers']
    
        paramsNameLayerWiseTrain = list(paramsLayerWiseTrain)    
        layerWiseTrainL = len(paramsLayerWiseTrain[paramsNameLayerWiseTrain[1]])
        layerWiseTraindimReadout = paramsLayerWiseTrain[paramsNameLayerWiseTrain[4]]
    
        bestTraining = np.load(saveArchitFile + '-nDAggers-' + str(nDAggers) + '.npz') # the data file loaded from the example folder
    
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