import os
import numpy as np

def evaluateFlocking(model, data, **kwargs):
    """
    evaluateClassif: evaluate a model using the flocking cost of velocity 
        variacne of the team
    
    Input:
        model (model class): class from Modules.model
        data (data class): the data class that generates the flocking data
        doPrint (optional; bool, default: True): if True prints results
        nVideos (optional; int, default: 3): number of videos to save
        graphNo (optional): identify the run with a number
        realizationNo (optional): identify the run with another number
    
    Output:
        evalVars (dict):
            'costBestFull': cost of the best model over the full trajectory
            'costBestEnd': cost of the best model at the end of the trajectory
            'costLastFull': cost of the last model over the full trajectory
            'costLastEnd': cost of the last model at the end of the trajectory
    """
    
    if 'doPrint' in kwargs.keys():
        doPrint = kwargs['doPrint']
    else:
        doPrint = True
        
    if 'nVideos' in kwargs.keys():
        nVideos = kwargs['nVideos']
    else:
        nVideos = 3
        
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
    initPosTest = data.getData('initOffset', 'test')
    initVelTest = data.getData('initSkew', 'test')
    graphTest = data.getData('commGraph','test')   
    clockNoiseTest = data.getData('clockNoise','test')   
    measurementNoiseTest = data.getData('packetExchangeDelay','test')   
    processingNoiseTest = data.getData('processingDelay','test')   
                    
    ##############
    # BEST MODEL #
    ##############

    model.load(label = 'Best')

    if doPrint:
        print("\tComputing learned trajectory for best model...",
              end = ' ', flush = True)

    posTestBest, \
    velTestBest, \
    accelTestBest, \
    stateTestBest, \
    commGraphTestBest = \
        data.computeTrajectory(initPosTest, initVelTest, measurementNoiseTest, processingNoiseTest, clockNoiseTest, graphTest, data.duration,
                               archit = model.archit)
                
    SavedPath ='./gnn_test.npz'
    np.savez(SavedPath, posTestBest=posTestBest, velTestBest=velTestBest, \
             accelTestBest=accelTestBest, stateTestBest=stateTestBest, \
                 commGraphTestBest=commGraphTestBest)
    print("\tSaved the test data to the following path: ./gnn_test.npz...", end = ' ')
    print("OK", flush = True)        
    
    # posTestBest2, \
    # velTestBest2, \
    # accelTestBest2, \
    # stateTestBest2, \
    # commGraphTestBest2 = \
    #     data.computeTrajectory2(initPosTest, initVelTest, measurementNoiseTest, processingNoiseTest, graphTest, data.duration,
    #                            archit = model.archit)
        
    # SavedPath ='./gnn_test2.npz'
    # np.savez(SavedPath, posTestBest=posTestBest2, velTestBest=velTestBest2, \
    #          accelTestBest=accelTestBest2, stateTestBest=stateTestBest2, \
    #              commGraphTestBest=commGraphTestBest2)
    # print("\tSaved the test data to the following path: ./gnn_test2.npz...", end = ' ')
    # print("OK", flush = True)                

    if doPrint:
        print("OK")

    ##############
    # LAST MODEL #
    ##############

    model.load(label = 'Last')

    if doPrint:
        print("\tComputing learned trajectory for last model...",
              end = ' ', flush = True)

    posTestLast, \
    velTestLast, \
    accelTestLast, \
    stateTestLast, \
    commGraphTestLast = \
        data.computeTrajectory(initPosTest, initVelTest, measurementNoiseTest, processingNoiseTest, clockNoiseTest, graphTest, data.duration,
                               archit = model.archit)

    if doPrint:
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

    if doPrint:
        print("\tPreview data...",
              end = ' ', flush = True)

    if doPrint:
        print("OK", flush = True)

    #\\\\\\\\\\\\\\\\\\
    #\\\ EVALUATION \\\
    #\\\\\\\\\\\\\\\\\\
        
    evalVars = {}
    evalVars['costBestFull'] = data.evaluate(thetaOffset = posTestBest, gammaSkew = velTestBest)
    evalVars['costBestEnd'] = data.evaluate(thetaOffset = posTestBest[:,-1:,:,:], gammaSkew = velTestBest[:,-1:,:,:])
    evalVars['costLastFull'] = data.evaluate(thetaOffset = posTestLast, gammaSkew = velTestLast)
    evalVars['costLastEnd'] = data.evaluate(thetaOffset = posTestLast[:,-1:,:,:], gammaSkew = velTestLast[:,-1:,:,:])

    return evalVars