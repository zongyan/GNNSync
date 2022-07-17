# Created on Tue Jul 12 16:18:24 2022
# Yan Zong, y.zong@cranfield.ac.uk

#%%######################################################
###                  IMPORTING                  #########
#########################################################

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

#%%###############################################################
###                  SETTING PARAMETERS                  #########
##################################################################

thisFilename = 'TimeSync' # the general name of all related files

nNodes = 50 # the number of nodes at training time

saveDirRoot = 'experiments' # the relative location
saveDir = os.path.join(saveDirRoot, thisFilename) # dir where to save all results from each run

# create .txt to store the values of the setting parameters
today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# append date and time of the run to the directory, to avoid several runs of
# overwritting each other.
saveDir = saveDir + '-%03d-' % nNodes + today
# create directory
if not os.path.exists(saveDir):
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
# collect all random states
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
    
# load the random seeds for both torch and numpy
# loadDir = saveDir
# pathToSeed = os.path.join(loadDir, 'randomSeedUsed.pkl')
# with open(pathToSeed, 'rb') as seedFile:
#     randomStates = pickle.load(seedFile)
#     randomStates = randomStates['randomStates']
# for module in randomStates:
#     thisModule = module['module']
#     if thisModule == 'numpy':
#         np.random.RandomState().set_state(module['state'])
#     elif thisModule == 'torch':
#         torch.set_rng_state(module['state'])
#         torch.manual_seed(module['seed'])    
#     # end if 
# # end for

########
# DATA #
########

nNodesMax = nNodes # Maximum number of agents to test the solution
nSimPoints = 1 # Number of simulations between nNodes and nNodesMax -- NOT use
               # At test time, the architectures trained on nNodes will be tested on a
               # varying number of agents, starting at nNodes all the way to nNodesMax;
               # the number of simulations for different number of agents is given by
               # nSimPoints, i.e. if nNodes = 50, nNodesMax = 100 and nSimPoints = 3, 
               # then the architectures are trained on 50, 75 and 100 agents.
commRadius = 2. # Communication radius -- NOT use
repelDist = 1. # Minimum distance before activating repelling potential -- NOT use
nTrain = 400 # number of training samples
nValid = 20 # number of valid samples
nTest = 20 # number of testing samples
duration = 2. # simulation duration
samplingTimeScale = 0.01 # sampling timescale, according to Giorgi2011
initGeometry = 'circular' # Geometry of initial positions -- NOT use
initVelValue = 3. # Initial velocities are samples from an interval -- NEED to modify according to MATLAB
    # [-initVelValue, initVelValue]
initMinDist = 0.1 # No two agents are located at a distance less than this -- NOT use
accelMax = 10. # This is the maximum value of acceleration allowed -- DO we need this 

varValues = {'nNodes': nNodes, 'nNodesMax': nNodesMax, 'nSimPoints': nSimPoints, \
             'commRadius': commRadius, 'repelDist': repelDist, 'nTrain': nTrain, \
                 'nValid': nValid, 'nTest': nTest, 'duration': duration, \
                     'samplingTimeScale': samplingTimeScale, 'initGeometry': initGeometry, \
                         'initVelValue': initVelValue, 'initMinDist': initMinDist, 'accelMax': accelMax}

# save values:
with open(varsFile, 'a+') as file:
    for key in varValues.keys():
        file.write('%s = %s\n' % (key, varValues[key]))
    file.write('\n')    

############
# TRAINING #
############

# model training options
optimAlg = 'ADAM' # Options: 'SGD', 'ADAM', 'RMSprop'
learningRate = 0.0005 # In all options
beta1 = 0.9 # beta1 if 'ADAM', alpha if 'RMSprop'
beta2 = 0.999 # ADAM option only

# loss function
lossFunction = nn.MSELoss

# training algorithm
trainer = training.TrainerFlocking

# evaluation algorithm
evaluator = evaluation.evaluateFlocking

# overall training options
nEpochs = 1 # Number of epochs
batchSize = 20 # Batch size
doLearningRateDecay = False # Learning rate decay
learningRateDecayRate = 0.9 # Rate
learningRateDecayPeriod = 1 # How many epochs after which update the lr
validationInterval = 5 # How many training steps to do the validation

del varValues
varValues = {'optimizationAlgorithm': optimAlg, 'learningRate': learningRate, \
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
# ARCHITECTURES #
#################

nonlinearityHidden = torch.tanh
nonlinearityOutput = torch.tanh
nonlinearity = nn.Tanh # chosen nonlinearity for nonlinear architectures

modelList = []
    
hParamsLocalGNN = {} # hyperparameters (hParams) for the Local GNN (LclGNN)

hParamsLocalGNN['name'] = 'LocalGNN'
hParamsLocalGNN['archit'] = architTime.LocalGNN_DB
hParamsLocalGNN['device'] = 'cuda:0' if (torch.cuda.is_available()) else 'cpu'
hParamsLocalGNN['dimNodeSignals'] = [6, 64] # Features per layer
hParamsLocalGNN['nFilterTaps'] = [3] # Number of filter taps
hParamsLocalGNN['bias'] = True # Decide whether to include a bias term
hParamsLocalGNN['nonlinearity'] = nonlinearity # Selected nonlinearity
                                               # is affected by the summary
hParamsLocalGNN['dimReadout'] = [2] # Dimension of the fully connected
                                    # layers after the GCN layers (map); this fully connected layer
                                    # is applied only at each node, without any further exchanges nor 
                                    # considering all nodes at once, making the architecture entirely
                                    # local.
hParamsLocalGNN['dimEdgeFeatures'] = 1 # Scalar edge weights

with open(varsFile, 'a+') as file:
    for key in hParamsLocalGNN.keys():
        file.write('%s = %s\n' % (key, hParamsLocalGNN[key]))
    file.write('\n')

modelList += [hParamsLocalGNN['name']]

###########
# LOGGING #
###########

# Parameters:
printInterval = 1 # After how many training steps, print the partial results
#   0 means to never print partial results while training
xAxisMultiplierTrain = 10 # How many training steps in between those shown in
    # the plot, i.e., one training step every xAxisMultiplierTrain is shown.
xAxisMultiplierValid = 2 # How many validation steps in between those shown,
    # same as above.
figSize = 5 # Overall size of the figure that contains the plot
lineWidth = 2 # Width of the plot lines
markerShape = 'o' # Shape of the markers
markerSize = 3 # Size of the markers

del varValues
varValues = {'printInterval': printInterval, 'figSize': figSize, \
                 'lineWidth': lineWidth, 'markerShape': markerShape, 'markerSize': markerSize}
    
with open(varsFile, 'a+') as file:
    for key in varValues.keys():
        file.write('%s = %s\n' % (key, varValues[key]))
    file.write('\n')    

#%%##################################################################
#                                                                   #
#                    SETUP                                          #
#                                                                   #
#####################################################################

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("Selected devices:")
for thisModel in modelList:
    hParamsDict = eval('hParams' + thisModel)
    print("\t%s: %s" % (thisModel, hParamsDict['device']))
    
#\\\ Number of agents at test time
nNodesTest = np.linspace(nNodes, nNodesMax, num = nSimPoints,dtype = np.int64) # -- NOT use
nNodesTest = np.unique(nNodesTest).tolist() # -- NOT use
nSimPoints = len(nNodesTest) # -- NOT use, but need to add in the future
writeVarValues(varsFile, {'nNodesTest': nNodesTest}) # Save list # -- NOT use

#\\\ Save variables during evaluation.
# We will save all the evaluations obtained for each of the trained models.
# The first list is one for each value of nNodes that we want to simulate 
# (i.e. these are test results, so if we test for different number of agents,
# we need to save the results for each of them). Each element in the list will
# be a dictionary (i.e. for each testing case, we have a dictionary).
# It basically is a dictionary, containing a list. The key of the
# dictionary determines the model, then the first list index determines
# which split realization. Then, this will be converted to numpy to compute
# mean and standard deviation (across the split dimension).
# We're saving the cost of the full trajectory, as well as the cost at the end
# instant.
costBestFull = [None] * nSimPoints # -- NOT use
costBestEnd = [None] * nSimPoints # -- NOT use
costLastFull = [None] * nSimPoints # -- NOT use
costLastEnd = [None] * nSimPoints # -- NOT use
costOptFull = [None] * nSimPoints # -- NOT use
costOptEnd = [None] * nSimPoints # -- NOT use
for n in range(nSimPoints):
    costBestFull[n] = {} # Accuracy for the best model (full trajectory)
    costBestEnd[n] = {} # Accuracy for the best model (end time)
    costLastFull[n] = {} # Accuracy for the last model
    costLastEnd[n] = {} # Accuracy for the last model
    for thisModel in modelList: # Create an element for each split realization,
        costBestFull[n][thisModel] = [None]
        costBestEnd[n][thisModel] = [None]
        costLastFull[n][thisModel] = [None]
        costLastEnd[n][thisModel] = [None]
    costOptFull[n] = [None] # Accuracy for optimal controller
    costOptEnd[n] = [None] # Accuracy for optimal controller

####################
# TRAINING OPTIONS #
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
trainingOptions['validationInterval'] = validationInterval

# And in case each model has specific training options (aka 'DAGger'), then
# we create a separate dictionary per model.

trainingOptsPerModel= {}

# Create relevant dirs: we need directories to save the videos of the dataset
# that involve the optimal centralized controllers, and we also need videos
# for the learned trajectory of each model. Note that all of these depend on
# each realization, so we will be saving videos for each realization.
# Here, we create all those directories.
datasetTrajectoryDir = os.path.join(saveDir,'datasetTrajectories')
if not os.path.exists(datasetTrajectoryDir):
    os.makedirs(datasetTrajectoryDir)
    
datasetTrainTrajectoryDir = os.path.join(datasetTrajectoryDir,'train')
if not os.path.exists(datasetTrainTrajectoryDir):
    os.makedirs(datasetTrainTrajectoryDir)
    
datasetTestTrajectoryDir = os.path.join(datasetTrajectoryDir,'test')
if not os.path.exists(datasetTestTrajectoryDir):
    os.makedirs(datasetTestTrajectoryDir)

datasetTestAgentTrajectoryDir = [None] * nSimPoints
for n in range(nSimPoints):    
    datasetTestAgentTrajectoryDir[n] = os.path.join(datasetTestTrajectoryDir,
                                                    '%03d' % nNodesTest[n])
    
#%%##################################################################
#                                                                   #
#                    DATA HANDLING                                  #
#                                                                   #
#####################################################################

############
# DATASETS #
############

print("Generating data", end = '')
print("...", flush = True)

#   Generate the dataset
data = dataTools.Flocking(
            # Structure
            nNodes,
            commRadius,
            repelDist,
            # Samples
            nTrain,
            nValid,
            1, # We do not care about testing, we will re-generate the
               # dataset for testing
            # Time
            duration,
            samplingTimeScale,
            # Initial conditions
            initGeometry = initGeometry,
            initVelValue = initVelValue,
            initMinDist = initMinDist,
            accelMax = accelMax)

###########
# PREVIEW #
###########

print("Preview data", end = '')
print("...", flush = True)

#%%##################################################################
#                                                                   #
#                    MODELS INITIALIZATION                          #
#                                                                   #
#####################################################################

# This is the dictionary where we store the models (in a model.Model
# class).
modelsGNN = {}

# If a new model is to be created, it should be called for here.

print("Model initialization...", flush = True)

for thisModel in modelList:

    # Get the corresponding parameter dictionary
    hParamsDict = deepcopy(eval('hParams' + thisModel))
    # and training options
    trainingOptsPerModel[thisModel] = deepcopy(trainingOptions)

    # Now, this dictionary has all the hyperparameters that we need to pass
    # to the architecture, but it also has the 'name' and 'archit' that
    # we do not need to pass them. So we are going to get them out of
    # the dictionary
    thisName = hParamsDict.pop('name')
    callArchit = hParamsDict.pop('archit')
    thisDevice = hParamsDict.pop('device')
    # If there's a specific DAGger type, pop it out now
    if 'DAGgerType' in hParamsDict.keys() \
                                    and 'probExpert' in hParamsDict.keys():
        trainingOptsPerModel[thisModel]['probExpert'] = \
                                              hParamsDict.pop('probExpert')
        trainingOptsPerModel[thisModel]['DAGgerType'] = \
                                              hParamsDict.pop('DAGgerType')

    # If more than one graph or data realization is going to be carried out,
    # we are going to store all of thos models separately, so that any of
    # them can be brought back and studied in detail.

    print("\tInitializing %s..." % thisName,
          end = ' ',flush = True)

    ##############
    # PARAMETERS #
    ##############

    #\\\ Optimizer options
    #   (If different from the default ones, change here.)
    thisOptimAlg = optimAlg
    thisLearningRate = learningRate
    thisBeta1 = beta1
    thisBeta2 = beta2

    ################
    # ARCHITECTURE #
    ################

    thisArchit = callArchit(**hParamsDict)
    thisArchit.to(thisDevice)

    #############
    # OPTIMIZER #
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

    ########
    # LOSS #
    ########

    thisLossFunction = lossFunction()
    
    ###########
    # TRAINER #
    ###########

    thisTrainer = trainer
    
    #############
    # EVALUATOR #
    #############

    thisEvaluator = evaluator

    #########
    # MODEL #
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

    writeVarValues(varsFile,
                   {'name': thisName,
                    'thisOptimizationAlgorithm': thisOptimAlg,
                    'thisTrainer': thisTrainer,
                    'thisEvaluator': thisEvaluator,
                    'thisLearningRate': thisLearningRate,
                    'thisBeta1': thisBeta1,
                    'thisBeta2': thisBeta2})

    print("OK")

#%%##################################################################
#                                                                   #
#                    TRAINING                                       #
#                                                                   #
#####################################################################


############
# TRAINING #
############

print("")

for thisModel in modelsGNN.keys():

    print("Training model %s..." % thisModel)
        
    for m in modelList:
        if m in thisModel:
            modelName = m

    modelsGNN[thisModel].train(data, nEpochs, batchSize, **trainingOptsPerModel[m])

#%%##################################################################
#                                                                   #
#                    EVALUATION                                     #
#                                                                   #
#####################################################################

# Now that the model has been trained, we evaluate them on the test
# samples.

# We have two versions of each model to evaluate: the one obtained
# at the best result of the validation step, and the last trained model.
    
for n in range(nSimPoints):
    
    print("")
    print("[%3d Agents] Generating test set" % nNodesTest[n],
          end = '')
    print("...", flush = True)

    #   Load the data, which will give a specific split
    dataTest = dataTools.Flocking(
                    # Structure
                    nNodesTest[n],
                    commRadius,
                    repelDist,
                    # Samples
                    1, # We don't care about training
                    1, # nor validation
                    nTest,
                    # Time
                    duration,
                    samplingTimeScale,
                    # Initial conditions
                    initGeometry = initGeometry,
                    initVelValue = initVelValue,
                    initMinDist = initMinDist,
                    accelMax = accelMax)

    ###########
    # OPTIMAL #
    ###########
    
    #\\\ PREVIEW
    #\\\\\\\\\\\
    
    # Save videos for the optimal trajectories of the test set (before it
    # was for the otpimal trajectories of the training set)
    
    posTest = dataTest.getData('pos', 'test')
    velTest = dataTest.getData('vel', 'test')
    commGraphTest = dataTest.getData('commGraph', 'test')

    print("[%3d Agents] Preview data"  % nNodesTest[n], end = '')
    print("...", flush = True)
    
    #\\\ EVAL
    #\\\\\\\\
    
    # Get the cost for the optimal trajectories
    
    # Full trajectory
    costOptFull[n] = dataTest.evaluate(vel = velTest)
    
    # Last time instant
    costOptEnd[n] = dataTest.evaluate(vel = velTest[:,-1:,:,:])
    
    del posTest, velTest, commGraphTest
    
    ##########
    # MODELS #
    ##########

    for thisModel in modelsGNN.keys():

        print("[%3d Agents] Evaluating model %s" % \
                                 (nNodesTest[n], thisModel), end = '')
        print("...", flush = True)
            
        addKW = {}
        addKW['graphNo'] = nNodesTest[n]
            
        thisEvalVars = modelsGNN[thisModel].evaluate(dataTest, **addKW)

        thisCostBestFull = thisEvalVars['costBestFull']
        thisCostBestEnd = thisEvalVars['costBestEnd']
        thisCostLastFull = thisEvalVars['costLastFull']
        thisCostLastEnd = thisEvalVars['costLastEnd']
        
        # Find which model to save the results (when having multiple
        # realizations)
        for m in modelList:
            if m in thisModel:
                costBestFull[n][m] = thisCostBestFull
                costBestEnd[n][m] = thisCostBestEnd
                costLastFull[n][m] = thisCostLastFull
                costLastEnd[n][m] = thisCostLastEnd

#%%##################################################################
#                                                                   #
#                    PLOT                                           #
#                                                                   #
#####################################################################




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



posTest = gnn_test['posTestBest']
velTest = gnn_test['velTestBest']
accelTest = gnn_test['accelTestBest']
stateTest = gnn_test['stateTestBest']
commGraphTest = gnn_test['commGraphTestBest']

posOptim, velOptim, accelOptim = data.computeOptimalTrajectory(posTest[:,0,:,:], \
                                                               posTest[:,0,:,:], \
                                                                   duration=data.duration, \
                                                                       samplingTime=data.samplingTime, \
                                                                           repelDist=data.repelDist, \
                                                                               accelMax=data.accelMax)

# plot the velocity of all agents via the GNN method
for i in range(0, 1, 1):
    plt.figure()
    plt.rcParams["figure.figsize"] = (6.4,4.8)
    for j in range(0, 50, 1):
        # the input and output features are two dimensions, which means that one 
        # dimension is for x-axis velocity, the other one is for y-axis velocity 
        plt.plot(np.arange(0, 200, 1), np.sqrt(velTest[i, :, 0, j]**2 + velTest[i, :, 1, j]**2)) 
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
        plt.plot(np.arange(0, 200, 1), np.sqrt(velOptim[i, :, 0, j]**2 + velOptim[i, :, 1, j]**2)) 
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
        vel_temp = np.sqrt(velTest[i, :, 0, j] ** 2 + velTest[i, :, 1, j] ** 2) \
            - np.sqrt(velOptim[i, :, 0, j] ** 2 + velOptim[i, :, 1, j] ** 2)                
        plt.plot(np.arange(0, 200, 1), vel_temp) 
        # networks 4, 6, 7, 8, 9, 10, 12, 14, 15, 16, 17, 19 converge
    # end for 
    plt.xlabel(r'$time (s)$')
    plt.ylabel(r'$\|{\bf v}_{in}\|_2$')
    plt.title(r'$\|{\bf v}_{gnn}\|_2 - \|{\bf v}_{cc}\|_2$')    
    plt.grid()
    plt.show()    
# end for
    