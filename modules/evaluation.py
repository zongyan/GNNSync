import numpy as np
import os
import torch.nn as nn
import torch.optim as optim

import utils.graphML as gml

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
    learningRate = trainer.trainingOptions['learningRate']
    beta1 = trainer.trainingOptions['beta1']
    beta2 = trainer.trainingOptions['beta2']

    paramsNameLayerWiseTrain = list(paramsLayerWiseTrain)    
    layerWiseTrainL = len(paramsLayerWiseTrain[paramsNameLayerWiseTrain[1]])
    layerWiseTraindimReadout = paramsLayerWiseTrain[paramsNameLayerWiseTrain[4]]

    saveArchitDir = os.path.join(model.saveDir,'savedArchits')

    if layerWiseTraining == True:
        saveFile = os.path.join(saveArchitDir, 'LayerWiseTraining')
    elif endToEndTraining == True:
        saveFile = os.path.join(saveArchitDir, 'endToEndTraining')

    trainingFile = np.load(saveFile + '.npz', allow_pickle=True) # the data file loaded from the example folder
        
    historicalL = np.int64(trainingFile['historicalL'])
    historicalF = np.int64(trainingFile['historicalF'])
    historicalK = np.int64(trainingFile['historicalK'])
    historicalE = np.int64(trainingFile['historicalE'])
    historicalBias = trainingFile['historicalBias']
    historicalSigma = trainingFile['historicalSigma']
    historicalReadout = np.int64(trainingFile['historicalReadout'])
    historicalNumReadoutLayer = np.int64(trainingFile['historicalNumReadoutLayer'])
    
    historicalBestL = np.int64(trainingFile['historicalBestL'])
    historicalBestIteration = np.int64(trainingFile['historicalBestIteration'])
    historicalBestEpoch = np.int64(trainingFile['historicalBestEpoch'])
    historicalBestBatch = np.int64(trainingFile['historicalBestBatch'])
    
    assert len(historicalBestL) == layerWiseTrainL + 1    
    assert len(historicalBestL) == len(historicalBestIteration)
    assert len(historicalBestL) == len(historicalBestEpoch)
    assert len(historicalBestL) == len(historicalBestBatch) 

    maximumLayerWiseNum = max(np.array((layerWiseTrainL, len(layerWiseTraindimReadout))))       

    modules = [name for name, _ in model.archit.named_parameters()]
    layers = [name[0:3] for name in modules if len(name)<=13]
    layers = layers + [name[0:7] for name in modules if len(name)>13]
    
    l = 0
    sumNumLayerF = 0
    sumNumLayerK = 0
    sumNumReadoutLayer = 0
    while l < maximumLayerWiseNum + 1:
            
        trainedModelL = historicalL[l]
        trainedModelF = historicalF[sumNumLayerF:sumNumLayerF+trainedModelL+1]
        trainedModelK = historicalK[sumNumLayerK:sumNumLayerK+trainedModelL]
        trainedModelE = historicalE[l]
        trainedModelBias = historicalBias[l]
        trainedModelSigma = historicalSigma[l]        
        trainedModelNumReadoutLayer = historicalNumReadoutLayer[l]
        trainedModelReadout = historicalReadout[sumNumReadoutLayer:sumNumReadoutLayer+trainedModelNumReadoutLayer]
        
        sumNumLayerF = sumNumLayerF + trainedModelL + 1
        sumNumLayerK = sumNumLayerK + trainedModelL
        sumNumReadoutLayer = sumNumReadoutLayer + trainedModelNumReadoutLayer
        
        evaluationGFL = []
        evaluationGFL.append(gml.GraphFilter_DB(trainedModelF[0], trainedModelF[1], trainedModelK[0], trainedModelE, trainedModelBias))        

        for i in range(1, trainedModelL):
            evaluationGFL.append(trainedModelSigma())
            evaluationGFL.append(gml.GraphFilter_DB(trainedModelF[i], trainedModelF[i+1], trainedModelK[i], trainedModelE, trainedModelBias))                

        model.archit.GFL = nn.Sequential(*evaluationGFL) # graph filtering layers
        
        evaluationFC = []
        if len(trainedModelReadout) > 0:
            evaluationFC.append(trainedModelSigma())            
            evaluationFC.append(nn.Linear(trainedModelF[-1], trainedModelReadout[0], bias = trainedModelBias))
            for i in range(trainedModelNumReadoutLayer-1):
                evaluationFC.append(trainedModelSigma)
                evaluationFC.append(nn.Linear(trainedModelReadout[i], trainedModelReadout[i+1], bias = trainedModelBias))

            model.archit.Readout = nn.Sequential(*evaluationFC) # readout layers        

        model.archit.L = trainedModelL      
        model.archit.F = trainedModelF
        model.archit.K = trainedModelK
        model.archit.E = trainedModelE
        model.archit.bias = trainedModelBias
        model.archit.sigma = trainedModelSigma
        model.archit.dimReadout = trainedModelReadout
        
        model.optim = optim.Adam(model.archit.parameters(),
                                lr = learningRate,
                                betas = (beta1, beta2))
        
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
