#%% importing 

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import datetime
from copy import deepcopy

import torch; torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim

import utils.dataTools as dataTools
import utils.graphML as gml
import modules.architecturesTime as architTime
import modules.model as model
import modules.training as training
import modules.evaluation as evaluation

from utils.miscTools import writeVarValues

startRunTime = datetime.datetime.now()

#%%parameters setting

thisFilename = 'TimeSync' # the general name of all related files

nNodes = 50 # the number of nodes at training time

saveDirRoot = 'experiments' # the relative location
saveDir = os.path.join(saveDirRoot, thisFilename) # dir where to save all results from each run

# create .txt to store the values of the setting parameters.
# append date and time to avoid several runs of overwritting each other.
today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

saveDir = saveDir + '-%03d-' % nNodes + today

if not os.path.exists(saveDir): # create directory
    os.makedirs(saveDir)
# end if 

# create the file where all the (hyper)parameters and results will be saved.
varsFile = os.path.join(saveDir,'hyperparameters.txt')
with open(varsFile, 'w+') as file:
    file.write('%s\n\n' % datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
# end with 

# save seeds for reproducibility
# pytorch seeds
torchState = torch.get_rng_state()
torchSeed = torch.initial_seed()
# numpy seeds
numpyState = np.random.RandomState().get_state()

randomStates = [] # create a list 
randomStates.append({}) # create a dictionary in the list 
randomStates[0]['module'] = 'numpy'
randomStates[0]['state'] = numpyState
randomStates.append({}) # create a dictionary in the list
randomStates[1]['module'] = 'torch'
randomStates[1]['state'] = torchState
randomStates[1]['seed'] = torchSeed

# save the random seeds of both torch and numpy as a pickle file for reproduction
pathToSeed = os.path.join(saveDir, 'randomSeedUsed.pkl')
with open(pathToSeed, 'wb') as seedFile:
    pickle.dump({'randomStates': randomStates}, seedFile)
    
# # load the random seeds for both torch and numpy
# loadDir = saveDir
# pathToSeed = os.path.join(loadDir, 'randomSeedUsed.pkl')
# with open(pathToSeed, 'rb') as seedFile:
#     randomStates = pickle.load(seedFile)
#     randomStates = randomStates['randomStates']
# for module in randomStates: # a listing consist of two dictionaries
#     thisModule = module['module']
#     if thisModule == 'numpy':
#         np.random.RandomState().set_state(module['state'])
#     elif thisModule == 'torch':
#         torch.set_rng_state(module['state'])
#         torch.manual_seed(module['seed'])    
#     # end if 
# # end for

########
# Data #
########

nTrain = 400 # number of training samples
nValid = 20 # number of valid samples
nTest = 20 # number of testing samples
duration = 2. # simulation duration, unit: second 
samplingTimeScale = 0.01 # sampling timescale, unit: second, according to Giorgi2011
initOffsetValue = 6e+5 # initial clock offset = 600 us
initSkewValue = 50 # initial clock skew = 50 ppm
gainOffset = np.array([0.5, 0.5, 0.5, 0.5]) # K1, K2, K3 and K4 for offset correction
gainSkew = np.array([0.5, 0.5, 0.5, 0.5]) # K1, K2, K3 and K4 for skew correction

varValues = {'nNodes': nNodes, 'nTrain': nTrain, 'nValid': nValid, 'nTest': nTest, \
             'duration': duration, 'samplingTimeScale': samplingTimeScale, \
                 'initOffsetValue': initOffsetValue, 'initSkewValue': initSkewValue}

# save values:
with open(varsFile, 'a+') as file:
    for key in varValues.keys():
        file.write('%s = %s\n' % (key, varValues[key]))
    file.write('\n')    

############
# Training #
############

# model training options
optimAlg = 'ADAM'
learningRate = 0.0005 
beta1 = 0.9 # beta1 for adam
beta2 = 0.999 # beta2 for adam only

# loss function
lossFunction = nn.MSELoss

# training algorithm
trainer = training.Trainer

# evaluation algorithm
evaluator = evaluation.evaluate

# overall training options
nEpochs = 1 # number of epochs
batchSize = 20 # batch size
doLearningRateDecay = False # learning rate decay
learningRateDecayRate = 0.9 # rate 
learningRateDecayPeriod = 1 # how many epochs after which update the lr
validationInterval = 5 # how many training steps to do the validation

del varValues
varValues = {'optimisationAlgorithm': optimAlg, 'learningRate': learningRate, \
             'beta1': beta1, 'beta2': beta2, 'lossFunction': lossFunction, \
                 'trainer': trainer, 'evaluator': evaluator, 'nEpochs': nEpochs, \
                     'batchSize': batchSize, 'doLearningRateDecay': doLearningRateDecay, \
                         'learningRateDecayRate': learningRateDecayRate, 'learningRateDecayPeriod': learningRateDecayPeriod, \
                             'validationInterval': validationInterval}
    
with open(varsFile, 'a+') as file:
    for key in varValues.keys():
        file.write('%s = %s\n' % (key, varValues[key]))
    file.write('\n')    
    
#################
# Architectures #
#################

nonlinearityHidden = torch.tanh
nonlinearityOutput = torch.tanh
nonlinearity = nn.Tanh # chosen nonlinearity for nonlinear architectures

modelList = []
    
hParamsGNN = {} # hyperparameters (hParams) for GNN 

hParamsGNN['name'] = 'GNN'
hParamsGNN['archit'] = architTime.LocalGNN_DB
hParamsGNN['device'] = 'cuda:0' if (torch.cuda.is_available()) else 'cpu'
hParamsGNN['dimNodeSignals'] = [2, 32] # features per layer
hParamsGNN['nFilterTaps'] = [3] # number of filter taps, i.e. three-hop neighbours' infor 
hParamsGNN['bias'] = True # decide whether to include a bias term
hParamsGNN['nonlinearity'] = nonlinearity
                                            
hParamsGNN['dimReadout'] = [2] # Dimension of the fully connected layers after 
                               # the GCN layers (map); this fully connected layer
                               # is applied only at each node, without any 
                               # further exchanges nor  considering all nodes at 
                               # once, making the architecture entirely local.
hParamsGNN['dimEdgeFeatures'] = 1 # scalar edge weights

with open(varsFile, 'a+') as file:
    for key in hParamsGNN.keys():
        file.write('%s = %s\n' % (key, hParamsGNN[key]))
    file.write('\n')

modelList += [hParamsGNN['name']]

###########
# Logging #
###########

printInterval = 1 # after how many training steps, print the partial results
                  # 0 means to never print partial results while training

del varValues
varValues = {'printInterval': printInterval}
    
with open(varsFile, 'a+') as file:
    for key in varValues.keys():
        file.write('%s = %s\n' % (key, varValues[key]))
    file.write('\n')    

#%% setup 

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("Selected devices: ")
for thisModel in modelList:
    hParamsDict = eval('hParams' + thisModel)
    print("\t%s: %s" % (thisModel, hParamsDict['device']))
# end for     

####################
# Training options #
####################

# Training phase. It has a lot of options that are input through a
# dictionary of arguments.
# The value of these options was decided above with the rest of the parameters.
# This just creates a dictionary necessary to pass to the train function.

trainingOptions = {}

trainingOptions['printInterval'] = printInterval

if doLearningRateDecay:
    trainingOptions['learningRateDecayRate'] = learningRateDecayRate
    trainingOptions['learningRateDecayPeriod'] = learningRateDecayPeriod
# end if 
trainingOptions['validationInterval'] = validationInterval

trainingOptsPerModel= {}
    
#%% data handling 

############
# Datasets #
############

print("Generating data", end = '')
print("...", flush = True)

# Generate the dataset
data = dataTools.initClockNetwk(
            # Structure
            nNodes,
            # Samples
            nTrain,
            nValid,
            1, # no need care about testing, will re-generate the dataset for testing
            # Time
            duration, samplingTimeScale,
            # Initial conditions
            initOffsetValue, initSkewValue,
            gainOffset, gainSkew, 
            netwkType='digraph')

###########
# Preview #
###########

print("Preview data", end = '')
print("...", flush = True)

#%% Models initialisation 

# This is the dictionary where we store the models (in a model.Model
# class).
modelsGNN = {}

# If a new model is to be created, it should be called for here.

print("Model initialisation...", flush = True)

for thisModel in modelList:

    # Get the corresponding parameter dictionary
    hParamsDict = deepcopy(eval('hParams' + thisModel))
    # and training options
    trainingOptsPerModel[thisModel] = deepcopy(trainingOptions)

    thisName = hParamsDict.pop('name')
    callArchit = hParamsDict.pop('archit')
    thisDevice = hParamsDict.pop('device')

    print("\tInitialising %s..." % thisName,
          end = ' ',flush = True)

    ##############
    # Parameters #
    ##############

    # Optimiser options
    thisOptimAlg = optimAlg
    thisLearningRate = learningRate
    thisBeta1 = beta1
    thisBeta2 = beta2

    ################
    # Architecture #
    ################

    thisArchit = callArchit(**hParamsDict)
    thisArchit.to(thisDevice)

    #############
    # Optimiser #
    #############

    if thisOptimAlg == 'ADAM':
        thisOptim = optim.Adam(thisArchit.parameters(),
                               lr = learningRate,
                               betas = (beta1, beta2))
    elif thisOptimAlg == 'SGD':
        thisOptim = optim.SGD(thisArchit.parameters(),
                              lr = learningRate)
    elif thisOptimAlg == 'RMSprop':
        thisOptim = optim.RMSprop(thisArchit.parameters(),
                                  lr = learningRate, alpha = beta1)
    # end if 

    ########
    # Loss #
    ########

    thisLossFunction = lossFunction()
    
    ###########
    # Trainer #
    ###########

    thisTrainer = trainer
    
    #############
    # Evaluator #
    #############

    thisEvaluator = evaluator

    #########
    # Model #
    #########

    modelCreated = model.Model(thisArchit,
                               thisLossFunction,
                               thisOptim,
                               thisTrainer,
                               thisEvaluator,
                               thisDevice,
                               thisName,
                               saveDir)

    modelsGNN[thisName] = modelCreated

    del varValues
    varValues = {'name': thisName, 'thisOptimizationAlgorithm': thisOptimAlg, \
                 'thisTrainer': thisTrainer, 'thisEvaluator': thisEvaluator, \
                     'thisLearningRate': thisLearningRate, 'thisBeta1': thisBeta1, \
                         'thisBeta2': thisBeta2}
        
    with open(varsFile, 'a+') as file:
        for key in varValues.keys():
            file.write('%s = %s\n' % (key, varValues[key]))
        file.write('\n')    
        
    print("OK")

# end for
#%% Training 

############
# Training #
############

print("")

for thisModel in modelsGNN.keys():

    print("Training model %s..." % thisModel)
        
    for m in modelList:
        if m in thisModel:
            modelName = m

    modelsGNN[thisModel].train(data, nEpochs, batchSize, **trainingOptsPerModel[m])

#%% Evaluation 

# Now that the model has been trained, we evaluate them on the test
# samples.

# We have two versions of each model to evaluate: the one obtained
# at the best result of the validation step, and the last trained model.
    
    
print("")
print("[%3d Agents] Generating test set" % nNodes,
      end = '')
print("...", flush = True)

# Load the data, which will give a specific split
dataTest = dataTools.initClockNetwk(
                # Structure
                nNodes,
                # Samples
                1, # We don't care about training
                1, # nor validation
                nTest,
                # Time
                duration, samplingTimeScale,
                # Initial conditions
                initOffsetValue, initSkewValue,
                gainOffset, gainSkew,
                netwkType='digraph')

###############
# Centralised #
###############

offsetTest = dataTest.getData('offset', 'test')
skewTest = dataTest.getData('skew', 'test')

print("[%3d Agents] Preview data"  % nNodes, end = '')
print("...", flush = True)

# Get the cost for the centralised dynamic controller

# Centralised dynamic controller
costCtrlFull = dataTest.evaluate(offset = offsetTest, skew = skewTest)

del offsetTest, skewTest

##########
# Models #
##########

for thisModel in modelsGNN.keys():

    print("[%3d Agents] Evaluating model %s" % (nNodes, thisModel), end = '')
    print("...", flush = True)
                
    modelsGNN[thisModel].evaluate(dataTest)

#%% Plot 

# Finish measuring time
endRunTime = datetime.datetime.now()

totalRunTime = abs(endRunTime - startRunTime)
totalRunTimeH = int(divmod(totalRunTime.total_seconds(), 3600)[0])
totalRunTimeM, totalRunTimeS = \
               divmod(totalRunTime.total_seconds() - totalRunTimeH * 3600., 60)
totalRunTimeM = int(totalRunTimeM)

print(" ")
print("Simulation started: %s" %startRunTime.strftime("%Y/%m/%d %H:%M:%S"))
print("Simulation ended:   %s" % endRunTime.strftime("%Y/%m/%d %H:%M:%S"))
print("Total time: %dh %dm %.2fs" % (totalRunTimeH,
                                     totalRunTimeM,
                                     totalRunTimeS))
    
# And save this info into the .txt file as well
with open(varsFile, 'a+') as file:
    file.write("Simulation started: %s\n" % 
                                     startRunTime.strftime("%Y/%m/%d %H:%M:%S"))
    file.write("Simulation ended:   %s\n" % 
                                       endRunTime.strftime("%Y/%m/%d %H:%M:%S"))
    file.write("Total time: %dh %dm %.2fs" % (totalRunTimeH,
                                              totalRunTimeM,
                                              totalRunTimeS))

#%%

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

gnn_test = np.load('./gnn_test.npz') # the data file loaded from the example folder

offsetTest = gnn_test['offsetTestBest']
skewTest = gnn_test['skewTestBest']
offsetCorrectionTest = gnn_test['offsetCorrectionTestBest']
skewCorrectionTest = gnn_test['skewCorrectionTestBest']
networkTopology = gnn_test['networkTopologyTest']

offsetCtrl, skewCtrl, offsetCorrectionCtrl, skewCorrectionCtrl = \
    data.computeViaCentralisedDynamicController(data.nNodes, nTest, 
                                                offsetTest[:,0,:,:], skewTest[:,0,:,:],
                                                data.gainOffset, data.gainSkew,
                                                data.duration, data.samplingTimeScale)    
        
# plot the velocity of all agents via the GNN method
for i in range(0, 1, 1):
    plt.figure()
    plt.rcParams["figure.figsize"] = (6.4,4.8)
    for j in range(0, 50, 1):
        # the input and output features are two dimensions, which means that one 
        # dimension is for x-axis velocity, the other one is for y-axis velocity 
        plt.plot(np.arange(0, 200, 1), offsetTest[i, :, 0, j]) 
        # networks 4, 6, 7, 8, 9, 10, 12, 14, 15, 16, 17, 19 converge
    # end for 
    plt.xlabel(r'$time (s)$')
    plt.ylabel(r'$\|{\bf v}_{in}\|_2$')
    plt.title(r'$\bf v_{gnn}$ for ' + str(50)+ ' agents (gnn controller)')
    plt.grid()
    plt.show()    
# end for

# plot the velocity of all agents via the centralised optimal controller
for i in range(0, 1, 1):
    plt.figure()
    plt.rcParams["figure.figsize"] = (6.4,4.8)
    for j in range(0, 50, 1):
        # the input and output features are two dimensions, which means that one 
        # dimension is for x-axis velocity, the other one is for y-axis velocity 
        plt.plot(np.arange(0, 200, 1), offsetCtrl[i, :, 0, j]) 
        # networks 4, 6, 7, 8, 9, 10, 12, 14, 15, 16, 17, 19 converge
    # end for 
    plt.xlabel(r'$time (s)$')
    plt.ylabel(r'$\|{\bf v}_{in}\|_2$')
    plt.title(r'$\bf v_{cc}$ for ' + str(50)+ ' agents (centralised controller)')
    plt.grid()
    plt.show()    
# end for

# plot the velocity difference of all agents by using the centralised optimal controller and GNN methods
for i in range(0, 1, 1):
    plt.figure()
    plt.rcParams["figure.figsize"] = (6.4,4.8)
    for j in range(0, 50, 1):
        # the input and output features are two dimensions, which means that one 
        # dimension is for x-axis velocity, the other one is for y-axis velocity 
        plt.plot(np.arange(0, 200, 1), offsetTest[i, :, 0, j] - offsetCtrl[i, :, 0, j]) 
    # end for 
    plt.xlabel(r'$time (s)$')
    plt.ylabel(r'$\|{\bf v}_{in}\|_2$')
    plt.title(r'$\|{\bf v}_{gnn}\|_2 - \|{\bf v}_{cc}\|_2$')    
    plt.grid()
    plt.show()    
# end for
    