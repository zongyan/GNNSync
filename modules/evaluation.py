import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
import copy 

import utils.graphML as gml

def evaluate(model, trainer, data, evalModel, **kwargs):
    initPosTest = data.getData('initOffset', 'test')
    initVelTest = data.getData('initSkew', 'test')
    graphTest = data.getData('commGraph','test')   
    clockNoiseTest = data.getData('clockNoise','test')   
    measurementNoiseTest = data.getData('packetExchangeDelay','test')   
    processingNoiseTest = data.getData('processingDelay','test')   
    
    attackCenterTest = data.getData('attackCenter','test')
    attackRadiusTest = data.getData('attackRadius','test')
    numAttackedNodesTest = data.getData('numAttackedNodes','test')
    attackNodesIndexTest = data.getData('attackNodesIndex','test')

    thisAttackGraphTest = (copy.deepcopy(graphTest)).reshape((graphTest.shape[0], graphTest.shape[1], 1, graphTest.shape[2], graphTest.shape[3]))
    attackGraphTest = np.repeat(thisAttackGraphTest, attackRadiusTest.shape[2], axis=2)
    
    for instant in range(attackRadiusTest.shape[0]):
        for i in range(attackRadiusTest.shape[2]):
            # print("\t attacking radius: %.1f... " %(attackRadiusTest[i,i,i,i]), flush = True)            
            for t in range(attackRadiusTest.shape[1]):        
                thisAttackNodes = np.int64(attackNodesIndexTest[instant,t,i,0:np.int64(numAttackedNodesTest[instant,t,i])])                
                attackGraphTest[instant, t, i, thisAttackNodes, :] = np.zeros((np.int64(numAttackedNodesTest[instant,t,i]), 50)) # we remove communication link due to attacks
    
    paramsLayerWiseTrain = trainer.trainingOptions['paramsLayerWiseTrain']
    layerWiseTraining = trainer.trainingOptions['layerWiseTraining']
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
    else:
        saveFile = os.path.join(saveArchitDir, 'endToEndTraining')

    trainingFile = np.load(saveFile + '-' + str(model.name) + '-nDAggers-' + str(nDAggers) + '.npz', allow_pickle=True) # the data file loaded from the example folder
    
    historicalL = np.int64(trainingFile['historicalL'])
    historicalF = np.int64(trainingFile['historicalF'])
    historicalK = np.int64(trainingFile['historicalK'])
    historicalE = np.int64(trainingFile['historicalE'])
    historicalBias = trainingFile['historicalBias']
    historicalSigma = trainingFile['historicalSigma']
    historicalReadout = np.int64(trainingFile['historicalReadout'])
    historicalHeatKernel = np.int64(trainingFile['historicalHeatKernel'])    
    historicalNumReadoutLayer = np.int64(trainingFile['historicalNumReadoutLayer'])
    
    historicalBestL = np.int64(trainingFile['historicalBestL'])
    historicalBestIteration = np.int64(trainingFile['historicalBestIteration'])
    historicalBestEpoch = np.int64(trainingFile['historicalBestEpoch'])
    historicalBestBatch = np.int64(trainingFile['historicalBestBatch'])
    
    if (len(historicalBestL) != layerWiseTrainL + 1):        
        assert len(historicalBestL) == len(layerWiseTraindimReadout) + 1
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
        trainedModelHeatKernel = historicalHeatKernel[l]        
        trainedModelReadout = historicalReadout[sumNumReadoutLayer:sumNumReadoutLayer+trainedModelNumReadoutLayer]
        
        sumNumLayerF = sumNumLayerF + trainedModelL + 1
        sumNumLayerK = sumNumLayerK + trainedModelL
        sumNumReadoutLayer = sumNumReadoutLayer + trainedModelNumReadoutLayer
        
        evaluationGFL = []
        evaluationGFL.append(gml.GraphFilter_DB(trainedModelF[0], trainedModelF[1], trainedModelK[0], trainedModelE, trainedModelBias, trainedModelHeatKernel))        

        for i in range(1, trainedModelL):
            # evaluationGFL.append(trainedModelSigma())
            evaluationGFL.append(gml.GraphFilter_DB(trainedModelF[i], trainedModelF[i+1], trainedModelK[i], trainedModelE, trainedModelBias, trainedModelHeatKernel))                

        model.archit.GFL = nn.Sequential(*evaluationGFL) # graph filtering layers
        
        evaluationFC = []
        if len(trainedModelReadout) > 0:
            evaluationFC.append(trainedModelSigma())            
            evaluationFC.append(nn.Linear(trainedModelF[-1], trainedModelReadout[0], bias = trainedModelBias))
            for i in range(trainedModelNumReadoutLayer-1):
                # evaluationFC.append(trainedModelSigma())
                evaluationFC.append(nn.Linear(trainedModelReadout[i], trainedModelReadout[i+1], bias = trainedModelBias))

            model.archit.Readout = nn.Sequential(*evaluationFC) # readout layers        

        model.archit.L = trainedModelL      
        model.archit.F = trainedModelF
        model.archit.K = trainedModelK
        model.archit.E = trainedModelE
        model.archit.bias = trainedModelBias
        model.archit.sigma = trainedModelSigma
        model.archit.dimReadout = trainedModelReadout
        model.archit.heatKernel = trainedModelHeatKernel        
        
        model.optim = optim.Adam(model.archit.parameters(),
                                lr = learningRate,
                                betas = (beta1, beta2))
        
        model.load(layerWiseTraining, \
               nDAggers, historicalBestL[l], historicalBestIteration[l], historicalBestEpoch[l], historicalBestBatch[l], label = 'Best')        
        
        print("\tComputing learned time synchronisation for best %s model with NO attacks..." %(model.name), 
              flush = True)
        
        offsetTestBest, \
        skewTestBest, \
        adjTestBest, \
        stateTestBest, \
        commGraphTestBest = \
            data.computeTrajectory(initPosTest, initVelTest, \
                                    measurementNoiseTest, processingNoiseTest, clockNoiseTest, 
                                    graphTest, data.duration,
                                    archit = model.archit)    
        
        offset = offsetTestBest[:, :, :, :]
        skew = skewTestBest[:, :, :, :]
        avgOffset = np.mean(offset, axis = 3) # nSamples x tSamples x 1
        avgSkew = np.mean(skew/10, axis = 3) # nSamples x tSamples x 1, change unit from 10ppm to 100ppm               
        
        diffOffset = offset - np.tile(np.expand_dims(avgOffset, 3), (1, 1, 1, 50)) # nSamples x tSamples x 1 x nAgents
        diffSkew = skew/10 - np.tile(np.expand_dims(avgSkew, 3), (1, 1, 1, 50)) # nSamples x tSamples x 1 x nAgents
        
        diffOffset = np.sum(diffOffset**2, 2) # nSamples x tSamples x nAgents
        diffSkew = np.sum(diffSkew**2, 2) # nSamples x tSamples x nAgents
        
        diffOffsetAvg = np.mean(diffOffset, axis = 2) # nSamples x tSamples
        diffSkewAvg = np.mean(diffSkew, axis = 2) # nSamples x tSamples
        
        costPerSample = np.sum(diffOffsetAvg, axis = 1) + np.sum(diffSkewAvg, axis = 1)*0.01 # nSamples
        
        cost = np.mean(costPerSample) # scalar
        print("\tThe cost of time sync for best model with NO attacks: %.4f" %(cost), flush = True)

        print("\tComputing learned time synchronisation for best %s model under attacks..." %(model.name), 
              flush = True)
                
        for i in range(attackRadiusTest.shape[2]):
            print("\tAttacking radius: %.1f... " %(attackRadiusTest[i,i,i,i]), flush = True)            

            attackOffsetTestBest, \
            attackSkewTestBest, \
            attackAdjTestBest, \
            attackStateTestBest, \
            attackCommGraphTestBest = \
                data.computeTrajectory(initPosTest, initVelTest, \
                                       measurementNoiseTest, processingNoiseTest, clockNoiseTest, 
                                       attackGraphTest[:, :, i, :, :], data.duration,
                                       archit = model.archit)    
            
            attackOffset = copy.deepcopy(attackOffsetTestBest)
            attackSkew = copy.deepcopy(attackSkewTestBest)
            
            for instant in range(attackRadiusTest.shape[0]):
                for t in range(attackRadiusTest.shape[1]):        
                    thisAttackNodes = np.int64(attackNodesIndexTest[instant,t,i,0:np.int64(numAttackedNodesTest[instant,t,i])])                
                
                    thisAttackOffsetTestBest = copy.deepcopy(attackOffsetTestBest[instant, t, :, :])
                    thisAttackOffsetTestBest[:, thisAttackNodes] = np.zeros((attackOffsetTestBest.shape[2], np.int64(numAttackedNodesTest[instant,t,i])))
                    attackOffset[instant, t, :, :] = copy.deepcopy(thisAttackOffsetTestBest)
                    
                    thisAttackSkewTestBest = copy.deepcopy(attackSkewTestBest[instant, t, :, :])
                    thisAttackSkewTestBest[:, thisAttackNodes] = np.zeros((attackSkewTestBest.shape[2], np.int64(numAttackedNodesTest[instant,t,i])))                
                    attackSkew[instant, t, :, :] = copy.deepcopy(thisAttackSkewTestBest)

            attackAvgOffset = np.mean(attackOffset, axis = 3) # nSamples x tSamples x 1
            attackAvgSkew = np.mean(attackSkew/10, axis = 3) # nSamples x tSamples x 1, change unit from 10ppm to 100ppm               
            
            attackDiffOffset = attackOffset - np.tile(np.expand_dims(attackAvgOffset, 3), (1, 1, 1, 50)) # nSamples x tSamples x 1 x nAgents
            attackDiffSkew = attackSkew/10 - np.tile(np.expand_dims(attackAvgSkew, 3), (1, 1, 1, 50)) # nSamples x tSamples x 1 x nAgents

            for instant in range(attackRadiusTest.shape[0]):
                for t in range(attackRadiusTest.shape[1]):        
                    thisAttackNodes = np.int64(attackNodesIndexTest[instant,t,i,0:np.int64(numAttackedNodesTest[instant,t,i])])                
                
                    thisAttackDiffOffset = copy.deepcopy(attackDiffOffset[instant, t, :, :])
                    thisAttackDiffOffset[:, thisAttackNodes] = np.zeros((attackDiffOffset.shape[2], np.int64(numAttackedNodesTest[instant,t,i])))
                    attackDiffOffset[instant, t, :, :] = copy.deepcopy(thisAttackDiffOffset)
                    
                    thisAttackDiffSkew = copy.deepcopy(attackDiffSkew[instant, t, :, :])
                    thisAttackDiffSkew[:, thisAttackNodes] = np.zeros((attackDiffSkew.shape[2], np.int64(numAttackedNodesTest[instant,t,i])))                
                    attackDiffSkew[instant, t, :, :] = copy.deepcopy(thisAttackDiffSkew)
            
            attackDiffOffset = np.sum(attackDiffOffset**2, 2) # nSamples x tSamples x nAgents
            attackDiffSkew = np.sum(attackDiffSkew**2, 2) # nSamples x tSamples x nAgents
            
            attackDiffOffsetAvg = np.mean(attackDiffOffset, axis = 2) # nSamples x tSamples
            attackDiffSkewAvg = np.mean(attackDiffSkew, axis = 2) # nSamples x tSamples
            
            attackCostPerSample = np.sum(attackDiffOffsetAvg, axis = 1) + np.sum(attackDiffSkewAvg, axis = 1)*0.01 # nSamples
            
            attackCost = np.mean(attackCostPerSample) # scalar
            print("\tThe cost of time sync for best model under attacks: %.4f" %(attackCost), flush = True)   
        
        if (evalModel == False):
            saveDataDir = os.path.join(model.saveDir,'savedData')
            if not os.path.exists(saveDataDir):
                os.makedirs(saveDataDir)
        
            if layerWiseTraining == True:
                saveDataDir = os.path.join(saveDataDir,'layerWiseTraining')
            else:
                saveDataDir = os.path.join(saveDataDir,'endToEndTraining')        
            if not os.path.exists(saveDataDir):
                os.makedirs(saveDataDir)        
        
            if layerWiseTraining == True:
                saveFile = os.path.join(saveDataDir, model.name + '-LayerWise-' + str(historicalBestL[l]) + '-DAgger-' + str(historicalBestIteration[l]) + '-' + str(nDAggers) + '-Epoch-' + str(historicalBestEpoch[l]) + '-Batch-' + str(historicalBestBatch[l]))
            else:
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
