import os
import torch
import pickle
import numpy as np

def evaluateFlocking(model, data, **kwargs):
                    
    if 'graphNo' in kwargs.keys():
        graphNo = kwargs['graphNo']
    else:
        graphNo = -1

    if 'realizationNo' in kwargs.keys():
        if 'graphNo' in kwargs.keys():
            realizationNo = kwargs['realizationNo']
        else:
            graphNo = kwargs['realizationNo']
            realizationNo = -1
    else:
        realizationNo = -1

    #\\\\\\\\\\\\\\\\\\\\
    #\\\ TRAJECTORIES \\\
    #\\\\\\\\\\\\\\\\\\\\

    ########
    # DATA #
    ########

    # Initial data
    initPosTest = data.getData('initPos', 'test')
    initVelTest = data.getData('initVel', 'test')

    ##############
    # BEST MODEL #
    ##############

    model.load(label = 'Best')

    print("\tComputing learned trajectory for best model...", end = ' ', flush = True)

    posTestBest, velTestBest, accelTestBest, stateTestBest, commGraphTestBest = \
        data.computeTrajectory(initPosTest, initVelTest, data.duration, archit = model.archit)
        
    SavedPath ='./gnn_test.npz'
    np.savez(SavedPath, posTestBest=posTestBest, velTestBest=velTestBest, \
             accelTestBest=accelTestBest, stateTestBest=stateTestBest, \
                 commGraphTestBest=commGraphTestBest)
    print("\tSaved the test data to the following path: ./gnn_test.npz...", end = ' ')
    print("OK", flush = True)

    ##############
    # LAST MODEL #
    ##############

    model.load(label = 'Last')

    print("\tComputing learned trajectory for last model...",
          end = ' ', flush = True)

    posTestLast, \
    velTestLast, \
    accelTestLast, \
    stateTestLast, \
    commGraphTestLast = \
        data.computeTrajectory(initPosTest, initVelTest, data.duration, archit = model.archit)

    print("OK")

    ###########
    # PREVIEW #
    ###########

    learnedTrajectoriesDir = os.path.join(model.saveDir,
                                          'learnedTrajectories')
    
    if not os.path.exists(learnedTrajectoriesDir):
        os.mkdir(learnedTrajectoriesDir)
    
    if graphNo > -1:
        learnedTrajectoriesDir = os.path.join(learnedTrajectoriesDir,
                                              '%03d' % graphNo)
        if not os.path.exists(learnedTrajectoriesDir):
            os.mkdir(learnedTrajectoriesDir)
    if realizationNo > -1:
        learnedTrajectoriesDir = os.path.join(learnedTrajectoriesDir,
                                              '%03d' % realizationNo)
        if not os.path.exists(learnedTrajectoriesDir):
            os.mkdir(learnedTrajectoriesDir)

    learnedTrajectoriesDir = os.path.join(learnedTrajectoriesDir, model.name)

    if not os.path.exists(learnedTrajectoriesDir):
        os.mkdir(learnedTrajectoriesDir)

    print("\tPreview data...",
          end = ' ', flush = True)

    print("OK", flush = True)

    #\\\\\\\\\\\\\\\\\\
    #\\\ EVALUATION \\\
    #\\\\\\\\\\\\\\\\\\
        
    evalVars = {}
    evalVars['costBestFull'] = data.evaluate(vel = velTestBest)
    evalVars['costBestEnd'] = data.evaluate(vel = velTestBest[:,-1:,:,:])
    evalVars['costLastFull'] = data.evaluate(vel = velTestLast)
    evalVars['costLastEnd'] = data.evaluate(vel = velTestLast[:,-1:,:,:])

    return evalVars