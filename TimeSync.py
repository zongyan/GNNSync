import os
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
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

#%%

thisFilename = 'TimeSync'
nAgents = 50 # number of UAVs during training 
saveDirRoot = 'experiments' 
saveDir = os.path.join(saveDirRoot, thisFilename) 

today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
saveDir = saveDir + '-%03d-' % nAgents + today
if not os.path.exists(saveDir):
    os.makedirs(saveDir)

useGPU = True
commRadius = 2. # communication radius
repelDist = 1. # minimum distance before activating repelling potential
nTrain = 400 # number of training samples
nValid = 20 # number of valid samples
nTest = 50 # number of testing samples
duration = 2. # trajectory duration 
samplingTime = 0.01 # sampling time
initVelValue = 3. # initial velocities: [-initVelValue, initVelValue]
initMinDist = 0.1 # minimum distance between any two UAVs
accelMax = 10. # maximum acceleration value

optimAlg = 'ADAM' 
learningRate = 0.0005 
beta1 = 0.9  
beta2 = 0.999 
lossFunction = nn.MSELoss
trainer = training.Trainer
evaluator = evaluation.evaluate

nEpochs = 30 # number of epochs
batchSize = 20 # batch size
validationInterval = 5 # how many training steps to do the validation

nonlinearityHidden = torch.tanh
nonlinearityOutput = torch.tanh
nonlinearity = nn.Tanh

modelList = []

hParamsGCNN = {}

hParamsGCNN['name'] = 'GCNN'
# Chosen architecture
hParamsGCNN['archit'] = architTime.LocalGNN_DB
hParamsGCNN['device'] = 'cuda:0' if (useGPU and torch.cuda.is_available()) else 'cpu'

# Graph convolutional parameters
hParamsGCNN['dimNodeSignals'] = [2, 16] # Features per layer
hParamsGCNN['nFilterTaps'] = [2] # Number of filter taps
hParamsGCNN['bias'] = True # Decide whether to include a bias term
# Nonlinearity
hParamsGCNN['nonlinearity'] = nonlinearity # Selected nonlinearity
    # is affected by the summary
# Readout layer: local linear combination of features
hParamsGCNN['dimReadout'] = [2] # Dimension of the fully connected
    # layers after the GCN layers (map); this fully connected layer
    # is applied only at each node, without any further exchanges nor 
    # considering all nodes at once, making the architecture entirely
    # local.
# Graph structure
hParamsGCNN['dimEdgeFeatures'] = 1 # Scalar edge weights

modelList += [hParamsGCNN['name']]

printInterval = 1 # after how many training steps, print the partial results
                  #   0 means to never print partial results while training

#%%
if useGPU and torch.cuda.is_available():
    torch.cuda.empty_cache()

print("Selected devices:")
for thisModel in modelList:
    hParamsDict = eval('hParams' + thisModel)
    print("\t%s: %s" % (thisModel, hParamsDict['device']))
    
trainingOptions = {}

trainingOptions['printInterval'] = printInterval
trainingOptions['validationInterval'] = validationInterval

#%%
print("Generating data", end = '')
print("...", flush = True)

data = dataTools.AerialSwarm(
            # Structure
            nAgents,
            commRadius,
            repelDist,
            # Samples
            nTrain,
            nValid,
            1, # We do not care about testing, we will re-generate the dataset for testing
            # Time
            duration,
            samplingTime,
            # Initial conditions
            initVelValue = initVelValue,
            initMinDist = initMinDist,
            accelMax = accelMax)

print("Preview data", end = '')
print("...", flush = True)

#%%
modelsGNN = {}

print("Model initialization...", flush = True)

for thisModel in modelList:

    hParamsDict = deepcopy(eval('hParams' + thisModel))

    thisName = hParamsDict.pop('name')
    callArchit = hParamsDict.pop('archit')
    thisDevice = hParamsDict.pop('device')

    print("\tInitializing %s..." % thisName, end = ' ',flush = True)

    thisOptimAlg = optimAlg
    thisLearningRate = learningRate
    thisBeta1 = beta1
    thisBeta2 = beta2

    thisArchit = callArchit(**hParamsDict)
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
                               saveDir)

    modelsGNN[thisName] = modelCreated

    print("OK")

#%%
for thisModel in modelsGNN.keys():

    print("Training model %s..." % thisModel)
        
    for m in modelList:
        if m in thisModel:
            modelName = m

    thisTrainVars = modelsGNN[thisModel].train(data,
                                               nEpochs,
                                               batchSize)

#%%
dataTest = dataTools.AerialSwarm(
                # Structure
                nAgents,
                commRadius,
                repelDist,
                # Samples
                1, # We don't care about training
                1, # nor validation
                nTest,
                # Time
                duration,
                samplingTime,
                # Initial conditions
                initVelValue = initVelValue,
                initMinDist = initMinDist,
                accelMax = accelMax)

posTest = dataTest.getData('offset', 'train')
velTest = dataTest.getData('skew', 'train')
commGraphTest = dataTest.getData('commGraph', 'train')

dataTest.evaluate(thetaOffset = posTest, gammaSkew = velTest)

dataTest.evaluate(thetaOffset = posTest[:,-1:,:,:], gammaSkew = velTest[:,-1:,:,:])              

#%%

gnn_test = np.load('./gnn_test.npz') # the data file loaded from the example folder

matplotlib.rc('figure', max_open_warning = 0)

offsetTest = gnn_test['posTestBest']
skewTest = gnn_test['velTestBest']
adjlTest = gnn_test['accelTestBest']
stateTest = gnn_test['stateTestBest']
commGraphTest = gnn_test['commGraphTestBest']

# plot the velocity of all agents via the GNN method
for i in range(0, 20, 3):
    plt.figure()
    plt.rcParams["figure.figsize"] = (6.4,4.8)
    for j in range(0, nAgents, 1):
        # the input and output features are two dimensions, which means that one 
        # dimension is for x-axis velocity, the other one is for y-axis velocity 
        plt.plot(np.arange(0, 200, 1), offsetTest[i, :, 0, j]) 
        # networks 4, 6, 7, 8, 9, 10, 12, 14, 15, 16, 17, 19 converge
    # end for 
    plt.xlabel(r'$time (s)$')
    plt.ylabel(r'${\bf \theta}_{gnn}$')
    plt.title(r'${\bf \theta}_{gnn}$ for ' + str(50)+ ' agents (gnn controller)')
    plt.grid()
    plt.show()    
# end for

# plot the velocity of all agents via the centralised optimal controller
for i in range(0, 20, 3):
    plt.figure()
    plt.rcParams["figure.figsize"] = (6.4,4.8)
    for j in range(0, nAgents, 1):
        # the input and output features are two dimensions, which means that one 
        # dimension is for x-axis velocity, the other one is for y-axis velocity 
        plt.plot(np.arange(0, 200, 1), skewTest[i, :, 0, j]) 
        # networks 4, 6, 7, 8, 9, 10, 12, 14, 15, 16, 17, 19 converge
    # end for 
    plt.xlabel(r'$time (s)$')
    plt.ylabel(r'${\bf \gamma}_{gnn}$')
    plt.title(r'$\bf \gamma_{gnn}$ for ' + str(50)+ ' agents (centralised controller)')
    plt.grid()
    plt.show()    
# end for

