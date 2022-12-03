"""****************************************************************************
// * File:        This file is a part of GNNSync.
// * Created on:  11/11/2022
// * Author:      Yan Zong (y.zong@nuaa.edu.cn)
// *
// * Copyright:   (C) 2022 Nanjing University of Aeronautics and Astronautics
// *
// *              GNNSync is free software; you can redistribute it and/or 
// *              modify it under the terms of the GNU General Public License 
// *              as published by the Free Software Foundation; either version 
// *              3 of the License, or (at your option) any later version.
// *
// *              GNNSync is distributed in the hope that it will be useful, 
// *              but WITHOUT ANY WARRANTY; without even the implied warranty 
// *              of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See 
// *              the GNU General Public License for more details.
// *
// * Funding:     This work was financed by the xx 
// *              xx, China
// * 
// * Description: Learn decentralized controllers for time synchronisation.  
// *              There is a team of UAVs that start flying at random velocities, 
// *              and all the UAVs coordinate so that they can fly together 
// *              while avoiding collisions by using the centralised expert 
// *              controller. We learn a decentralized controller by using 
// *              imitation learning, in order to realise clock synchronsiation 
// *              in a UAVs swarm system.
// * Outputs:     - Text file with all the hyperparameters selected for the run 
// *                and the corresponding results (hyperparameters.txt)
// *              - Pickle file with the random seeds of both torch and numpy 
// *                for accurate reproduction of results (randomSeedUsed.pkl)
// *              - The parameters of the trained models, for both the Best and 
// *                the Last instance of each model (savedModels/)
// *              - The figures of loss and evaluation through the training 
// *                iterations for each model (figs/ and trainVars/)
// *              - Videos for some of the trajectories in the dataset, 
// *                following the optimal centralized controller 
// *                (datasetTrajectories/)
// *              - Videos for some of the learned trajectories following the 
// *                controles learned by each model (learnedTrajectories/)
// *************************************************************************"""
#%%##################################################################
#                                                                   #
#                    IMPORTING                                      #
#                                                                   #
#####################################################################

#\\\ Standard libraries:
import os
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
# matplotlib.rcParams['text.latex.preamble']=[r'\usepackage{amsmath}']
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

# Start measuring time
startRunTime = datetime.datetime.now()

#%%##################################################################
#                                                                   #
#                    SETTING PARAMETERS                             #
#                                                                   #
#####################################################################

thisFilename = 'flockingGNN' # This is the general name of all related files

nAgents = 50 # Number of agents at training time

saveDirRoot = 'experiments' # In this case, relative location
saveDir = os.path.join(saveDirRoot, thisFilename) # Dir where to save all
    # the results from each run

#\\\ Create .txt to store the values of the setting parameters for easier
# reference when running multiple experiments
today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# Append date and time of the run to the directory, to avoid several runs of
# overwritting each other.
saveDir = saveDir + '-%03d-' % nAgents + today
# Create directory
if not os.path.exists(saveDir):
    os.makedirs(saveDir)

# Create the file where all the (hyper)parameters and results will be saved.
varsFile = os.path.join(saveDir,'hyperparameters.txt')
with open(varsFile, 'w+') as file:
    file.write('%s\n\n' % datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))

########
# DATA #
########

useGPU = True # If true, and GPU is available, use it.

nAgentsMax = nAgents # Maximum number of agents to test the solution
nSimPoints = 1 # Number of simulations between nAgents and nAgentsMax
    # At test time, the architectures trained on nAgents will be tested on a
    # varying number of agents, starting at nAgents all the way to nAgentsMax;
    # the number of simulations for different number of agents is given by
    # nSimPoints, i.e. if nAgents = 50, nAgentsMax = 100 and nSimPoints = 3, 
    # then the architectures are trained on 50, 75 and 100 agents.
commRadius = 2. # Communication radius
repelDist = 1. # Minimum distance before activating repelling potential
nTrain = 400 # Number of training samples
nValid = 20 # Number of valid samples
nTest = 50 # Number of testing samples
duration = 2. # Duration of the trajectory
samplingTime = 0.01 # Sampling time
initGeometry = 'circular' # Geometry of initial positions
initVelValue = 3. # Initial velocities are samples from an interval
    # [-initVelValue, initVelValue]
initMinDist = 0.1 # No two agents are located at a distance less than this
accelMax = 10. # This is the maximum value of acceleration allowed

############
# TRAINING #
############

#\\\ Individual model training options
optimAlg = 'ADAM' # Options: 'SGD', 'ADAM', 'RMSprop'
learningRate = 0.0005 # In all options
beta1 = 0.9 # beta1 if 'ADAM', alpha if 'RMSprop'
beta2 = 0.999 # ADAM option only

#\\\ Loss function choice
lossFunction = nn.MSELoss

#\\\ Training algorithm
trainer = training.TrainerFlocking

#\\\ Evaluation algorithm
evaluator = evaluation.evaluateFlocking

#\\\ Overall training options
probExpert = 0.993 # Probability of choosing the expert in DAGger
#DAGgerType = 'fixedBatch' # 'replaceTimeBatch', 'randomEpoch'
nEpochs = 30 # Number of epochs
batchSize = 20 # Batch size
doLearningRateDecay = False # Learning rate decay
learningRateDecayRate = 0.9 # Rate
learningRateDecayPeriod = 1 # How many epochs after which update the lr
validationInterval = 5 # How many training steps to do the validation

#################
# ARCHITECTURES #
#################

# In this section, we determine the (hyper)parameters of models that we are
# going to train. This only sets the parameters. The architectures need to be
# created later below. Do not forget to add the name of the architecture
# to modelList.

# If the hyperparameter dictionary is called 'hParams' + name, then it can be
# picked up immediately later on, and there's no need to recode anything after
# the section 'Setup' (except for setting the number of nodes in the 'N'
# variable after it has been coded).

# The name of the keys in the hyperparameter dictionary have to be the same
# as the names of the variables in the architecture call, because they will
# be called by unpacking the dictionary.

#nFeatures = 32 # Number of features in all architectures
#nFilterTaps = 4 # Number of filter taps in all architectures
# [[The hyperparameters are for each architecture, and they were chosen 
#   following the results of the hyperparameter search]]
nonlinearityHidden = torch.tanh
nonlinearityOutput = torch.tanh
nonlinearity = nn.Tanh # Chosen nonlinearity for nonlinear architectures

# Select desired architectures
doLocalFlt = False # Local filter (no nonlinearity)
doLocalGNN = True # Local GNN (include nonlinearity)
doDlAggGNN = False
doGraphRNN = False

modelList = []
   
#\\\\\\\\\\\\\\\\\
#\\\ LOCAL GNN \\\
#\\\\\\\\\\\\\\\\\

if doLocalGNN:

    #\\\ Basic parameters for the Local GNN architecture

    hParamsLocalGNN = {} # Hyperparameters (hParams) for the Local GNN (LclGNN)

    hParamsLocalGNN['name'] = 'LocalGNN'
    # Chosen architecture
    hParamsLocalGNN['archit'] = architTime.LocalGNN_DB
    hParamsLocalGNN['device'] = 'cuda:0' if (useGPU and torch.cuda.is_available()) else 'cpu'

    # Graph convolutional parameters
    hParamsLocalGNN['dimNodeSignals'] = [2, 16] # Features per layer
    hParamsLocalGNN['nFilterTaps'] = [2] # Number of filter taps
    hParamsLocalGNN['bias'] = True # Decide whether to include a bias term
    # Nonlinearity
    hParamsLocalGNN['nonlinearity'] = nonlinearity # Selected nonlinearity
        # is affected by the summary
    # Readout layer: local linear combination of features
    hParamsLocalGNN['dimReadout'] = [2] # Dimension of the fully connected
        # layers after the GCN layers (map); this fully connected layer
        # is applied only at each node, without any further exchanges nor 
        # considering all nodes at once, making the architecture entirely
        # local.
    # Graph structure
    hParamsLocalGNN['dimEdgeFeatures'] = 1 # Scalar edge weights

    modelList += [hParamsLocalGNN['name']]
    

    


###########
# LOGGING #
###########

# Options:
doPrint = True # Decide whether to print stuff while running



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
videoSpeed = 0.5 # Slow down by half to show transitions
nVideos = 3 # Number of videos to save

#%%##################################################################
#                                                                   #
#                    SETUP                                          #
#                                                                   #
#####################################################################

#\\\ If CUDA is selected, empty cache:
if useGPU and torch.cuda.is_available():
    torch.cuda.empty_cache()

#\\\ Notify of processing units
if doPrint:
    print("Selected devices:")
    for thisModel in modelList:
        hParamsDict = eval('hParams' + thisModel)
        print("\t%s: %s" % (thisModel, hParamsDict['device']))
    
#\\\ Number of agents at test time
nAgentsTest = np.linspace(nAgents, nAgentsMax, num = nSimPoints,dtype = np.int64)
nAgentsTest = np.unique(nAgentsTest).tolist()
nSimPoints = len(nAgentsTest)

#\\\ Save variables during evaluation.
# We will save all the evaluations obtained for each of the trained models.
# The first list is one for each value of nAgents that we want to simulate 
# (i.e. these are test results, so if we test for different number of agents,
# we need to save the results for each of them). Each element in the list will
# be a dictionary (i.e. for each testing case, we have a dictionary).
# It basically is a dictionary, containing a list. The key of the
# dictionary determines the model, then the first list index determines
# which split realization. Then, this will be converted to numpy to compute
# mean and standard deviation (across the split dimension).
# We're saving the cost of the full trajectory, as well as the cost at the end
# instant.
costBestFull = [None] * nSimPoints
costBestEnd = [None] * nSimPoints
costLastFull = [None] * nSimPoints
costLastEnd = [None] * nSimPoints
costOptFull = [None] * nSimPoints
costOptEnd = [None] * nSimPoints
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

####################
# TRAINING OPTIONS #
####################

# Training phase. It has a lot of options that are input through a
# dictionary of arguments.
# The value of these options was decided above with the rest of the parameters.
# This just creates a dictionary necessary to pass to the train function.

trainingOptions = {}

if doPrint:
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
                                                    '%03d' % nAgentsTest[n])

#%%##################################################################
#                                                                   #
#                    DATA SPLIT REALIZATION                         #
#                                                                   #
#####################################################################

# Start generating a new data realization for each number of total realizations

for realization in range(nRealizations):

    # On top of the rest of the training options, we pass the identification
    # of this specific data split realization.

    if doPrint:
        print("", flush = True)

    #%%##################################################################
    #                                                                   #
    #                    DATA HANDLING                                  #
    #                                                                   #
    #####################################################################

    ############
    # DATASETS #
    ############

    if doPrint:
        print("Generating data", end = '')
        print("...", flush = True)

    #   Generate the dataset
    data = dataTools.Flocking(
                # Structure
                nAgents,
                commRadius,
                repelDist,
                # Samples
                nTrain,
                nValid,
                1, # We do not care about testing, we will re-generate the
                   # dataset for testing
                # Time
                duration,
                samplingTime,
                # Initial conditions
                initGeometry = initGeometry,
                initVelValue = initVelValue,
                initMinDist = initMinDist,
                accelMax = accelMax)

    ###########
    # PREVIEW #
    ###########
    
    if doPrint:
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

    if doPrint:
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

        if doPrint:
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

        if doPrint:
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

        if doPrint:
            print("Training model %s..." % thisModel)
            
        for m in modelList:
            if m in thisModel:
                modelName = m

        thisTrainVars = modelsGNN[thisModel].train(data,
                                                   nEpochs,
                                                   batchSize,
                                                   **trainingOptsPerModel[m])

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
        
        if doPrint:
            print("")
            print("[%3d Agents] Generating test set" % nAgentsTest[n],
                  end = '')
            print("...", flush = True)

        #   Load the data, which will give a specific split
        dataTest = dataTools.Flocking(
                        # Structure
                        nAgentsTest[n],
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
        
        posTest = dataTest.getData('offset', 'train')
        velTest = dataTest.getData('skew', 'train')
        commGraphTest = dataTest.getData('commGraph', 'train')
    
        if doPrint:
            print("[%3d Agents] Preview data"  % nAgentsTest[n], end = '')
            print("...", flush = True)
        
        #\\\ EVAL
        #\\\\\\\\
        
        # Get the cost for the optimal trajectories
        
        # Full trajectory
        costOptFull[n][realization] = dataTest.evaluate(thetaOffset = posTest, gammaSkew = velTest)
        
        # Last time instant
        costOptEnd[n][realization] = dataTest.evaluate(thetaOffset = posTest[:,-1:,:,:], gammaSkew = velTest[:,-1:,:,:])
                
        del posTest, velTest, commGraphTest
        
        ##########
        # MODELS #
        ##########
    
        for thisModel in modelsGNN.keys():
    
            if doPrint:
                print("[%3d Agents] Evaluating model %s" % \
                                         (nAgentsTest[n], thisModel), end = '')
                print("...", flush = True)
                
            addKW = {}
            addKW['nVideos'] = nVideos
            addKW['graphNo'] = nAgentsTest[n]
                
            thisEvalVars = modelsGNN[thisModel].evaluate(dataTest, **addKW)
    
            thisCostBestFull = thisEvalVars['costBestFull']
            thisCostBestEnd = thisEvalVars['costBestEnd']
            thisCostLastFull = thisEvalVars['costLastFull']
            thisCostLastEnd = thisEvalVars['costLastEnd']            
    
            # Find which model to save the results (when having multiple
            # realizations)
            for m in modelList:
                if m in thisModel:
                    costBestFull[n][m][realization] = thisCostBestFull
                    costBestEnd[n][m][realization] = thisCostBestEnd
                    costLastFull[n][m][realization] = thisCostLastFull
                    costLastEnd[n][m][realization] = thisCostLastEnd

#%%

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

gnn_test = np.load('./gnn_test.npz') # the data file loaded from the example folder

matplotlib.rc('figure', max_open_warning = 0)

offsetTest = gnn_test['posTestBest']
skewTest = gnn_test['velTestBest']
adjlTest = gnn_test['accelTestBest']
stateTest = gnn_test['stateTestBest']
commGraphTest = gnn_test['commGraphTestBest']

# posOptim, velOptim, accelOptim = data.computeOptimalTrajectory(posTest[:,0,:,:], \
#                                                                posTest[:,0,:,:], \
#                                                                    duration=data.duration, \
#                                                                        samplingTime=data.samplingTime, \
#                                                                            repelDist=data.repelDist, \
#                                                                                accelMax=data.accelMax)

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

