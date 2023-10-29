import numpy as np
import os

def evaluate(model, trainer, data, **kwargs):
    initPosTest = data.getData('initOffset', 'test')
    initVelTest = data.getData('initSkew', 'test')
    graphTest = data.getData('commGraph','test')   
    clockNoiseTest = data.getData('clockNoise','test')   
    measurementNoiseTest = data.getData('packetExchangeDelay','test')   
    processingNoiseTest = data.getData('processingDelay','test')   
                            
    layerWiseTraining = trainer.trainingOptions['layerWiseTraining']
    endToEndTraining = trainer.trainingOptions['endToEndTraining']
    nDAggers = trainer.trainingOptions['nDAggers']
    bestL = trainer.bestLayerWiseIteration
    bestIteration = trainer.bestDAggerIteration
    bestEpoch = trainer.bestEpochIteration
    bestBatch = trainer.bestBatchIteration
    
    model.load(layerWiseTraining, endToEndTraining, \
               nDAggers, bestL, bestIteration, bestEpoch, bestBatch, label = 'Best')
        
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
        saveFile = os.path.join(saveDataDir, model.name + '-LayerWise-' + str(bestL) + '-DAgger-' + str(bestIteration) + '-' + str(nDAggers) + '-Epoch-' + str(bestEpoch) + '-Batch-' + str(bestBatch))
    elif endToEndTraining == True:
        saveFile = os.path.join(saveDataDir, model.name + '-EndToEnd-' + str(bestL) + '-DAgger-' + str(bestIteration) + '-' + str(nDAggers) + '-Epoch-' + str(bestEpoch) + '-Batch-' + str(bestBatch))
    
    saveFile = saveFile + '.npz'
    np.savez(saveFile, offsetTestBest=offsetTestBest, skewTestBest=skewTestBest, \
             adjTestBest=adjTestBest, stateTestBest=stateTestBest, \
                 commGraphTestBest=commGraphTestBest, \
                     lossTrain=trainer.lossTrain, accValid=trainer.accValid)
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

    print("OK")
