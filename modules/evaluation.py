import numpy as np
import os

def evaluate(model, trainer, data, **kwargs):
    initPosTest = data.getData('initOffset', 'test')
    initVelTest = data.getData('initSkew', 'test')
    graphTest = data.getData('commGraph','test')   
    clockNoiseTest = data.getData('clockNoise','test')   
    measurementNoiseTest = data.getData('packetExchangeDelay','test')   
    processingNoiseTest = data.getData('processingDelay','test')   
    
    paramsLayerWiseTrain = trainer.trainingOptions['paramsLayerWiseTrain']
    layerWiseTraining = trainer.trainingOptions['layerWiseTraining']
    endToEndTraining = trainer.trainingOptions['endToEndTraining']
    nDAggers = trainer.trainingOptions['nDAggers']

    paramsNameLayerWiseTrain = list(paramsLayerWiseTrain)    
    layerWiseTrainL = len(paramsLayerWiseTrain[paramsNameLayerWiseTrain[1]])
    layerWiseTraindimReadout = paramsLayerWiseTrain[paramsNameLayerWiseTrain[4]]

    saveArchitDir = os.path.join(model.saveDir,'savedArchits')

    if layerWiseTraining == True:
        saveFile = os.path.join(saveArchitDir, 'LayerWiseTraining')
    elif endToEndTraining == True:
        saveFile = os.path.join(saveArchitDir, 'endToEndTraining')

    bestTraining = np.load(saveFile + '.npz') # the data file loaded from the example folder

    historicalBestL = bestTraining['historicalBestL']
    historicalBestIteration = bestTraining['historicalBestIteration']
    historicalBestEpoch = bestTraining['historicalBestEpoch']
    historicalBestBatch = bestTraining['historicalBestBatch']
    
    assert len(historicalBestL) == layerWiseTrainL + 1    
    assert len(historicalBestL) == len(historicalBestIteration)
    assert len(historicalBestL) == len(historicalBestEpoch)
    assert len(historicalBestL) == len(historicalBestBatch) 

    maximumLayerWiseNum = max(np.array((layerWiseTrainL, len(layerWiseTraindimReadout))))       
    
    l = 0
    while l < maximumLayerWiseNum + 1:

        model.load(layerWiseTraining, endToEndTraining, \
               nDAggers, historicalBestL[l], historicalBestIteration[l], historicalBestEpoch[l], historicalBestBatch[l], label = 'Best')        
                       
        print("\tComputing learned time synchronisation for best model...",
              end = ' ', flush = True)
    
        offsetTestBest, \
        skewTestBest, \
        adjTestBest, \
        stateTestBest, \
        commGraphTestBest = \
            data.computeTrajectory(initPosTest, initVelTest, \
                                   measurementNoiseTest, processingNoiseTest, clockNoiseTest, 
                                   graphTest, data.duration,
                                   archit = model.archit)    
    
        saveDataDir = os.path.join(model.saveDir,'savedData')
        if not os.path.exists(saveDataDir):
            os.makedirs(saveDataDir)
    
        if layerWiseTraining == True:
            saveDataDir = os.path.join(saveDataDir,'layerWiseTraining')
        elif endToEndTraining == True:
            saveDataDir = os.path.join(saveDataDir,'endToEndTraining')        
        if not os.path.exists(saveDataDir):
            os.makedirs(saveDataDir)        
    
        if layerWiseTraining == True:
            saveFile = os.path.join(saveDataDir, model.name + '-LayerWise-' + str(historicalBestL[l]) + '-DAgger-' + str(historicalBestIteration[l]) + '-' + str(nDAggers) + '-Epoch-' + str(historicalBestEpoch[l]) + '-Batch-' + str(historicalBestBatch[l]))
        elif endToEndTraining == True:
            saveFile = os.path.join(saveDataDir, model.name + '-EndToEnd-' + str(historicalBestL[l]) + '-DAgger-' + str(historicalBestIteration[l]) + '-' + str(nDAggers) + '-Epoch-' + str(historicalBestEpoch[l]) + '-Batch-' + str(historicalBestBatch[l]))
        
        saveFile = saveFile + '.npz'
        np.savez(saveFile, offsetTestBest=offsetTestBest, skewTestBest=skewTestBest, \
                 adjTestBest=adjTestBest, stateTestBest=stateTestBest, \
                     commGraphTestBest=commGraphTestBest, \
                         bestL = historicalBestL[l], bestIteration = historicalBestIteration[l], bestEpoch = historicalBestEpoch[l], bestBatch = historicalBestBatch[l], \
                             lossTrain = trainer.lossTrain, accValid = trainer.accValid)
            
        l = l + 1
    
    print("OK")

    # model.load(label = 'Last')

    # print("\tComputing learned time synchronisation for last model...",
    #       end = ' ', flush = True)

    # offsetTestLast, \
    # skewTestLast, \
    # adjTestLast, \
    # stateTestLast, \
    # commGraphTestLast = \
    #     data.computeTrajectory(initPosTest, initVelTest,\
    #                            measurementNoiseTest, processingNoiseTest, clockNoiseTest, \
    #                            graphTest, data.duration,
    #                            archit = model.archit)

    # print("OK")
