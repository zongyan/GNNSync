import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
import copy 
from collections import Counter
from scipy import io

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

    totalAttackNodeIndex = []
    
    if data.attackMode == 1 or data.attackMode == 2: # attacking occurs
        for instant in range(attackRadiusTest.shape[0]):
            eachAttackNodeIndex = []
            for i in range(attackRadiusTest.shape[2]):
                thisAttackNodeIndex = []
                
                for t in range(attackRadiusTest.shape[1]):        
                    thisAttackNodes = np.int64(attackNodesIndexTest[instant, t, i, 0:np.int64(numAttackedNodesTest[instant,t,i])])
                    thisAttackNodeIndex.append(thisAttackNodes)
                    for element in thisAttackNodes:                    
                        attackGraphTest[instant, t, i, element, :] = np.zeros((50)) # we remove communication link due to attacks
                        attackGraphTest[instant, t, i, :, element] = np.zeros((50)) # we remove communication link due to attacks                
                
                thisAttackNodeIdxs = thisAttackNodeIndex[0]
                for idx in range(1, len(thisAttackNodeIndex)):    
                    thisAttackNodeIdxs = np.concatenate([thisAttackNodeIdxs, thisAttackNodeIndex[idx]])
                
                dicAttackNodeIdxs = Counter(thisAttackNodeIdxs)
                sortedAttackNodeIdxs = sorted(dicAttackNodeIdxs.items(), reverse=True)          
                attackedNodes = []
                for idx in range(len(sortedAttackNodeIdxs)):
                    attackedNodes.append(sortedAttackNodeIdxs[idx][0])
                
                attackedNodes = np.array(attackedNodes)     
                eachAttackNodeIndex.append(attackedNodes)
            
            totalAttackNodeIndex.append(eachAttackNodeIndex)
    
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

        print("\tComputing learned time synchronisation for the expert model with NO attacks...", 
              flush = True)
        
        offsetExpertTest = data.getData('offset', 'test')
        skewExpertTest = data.getData('skew', 'test')
        
        offset = copy.deepcopy(offsetExpertTest)
        skew = copy.deepcopy(skewExpertTest)
        avgOffset = np.mean(offset, axis = 3) # nSamples x tSamples x 1
        avgSkew = np.mean(skew/10, axis = 3) # nSamples x tSamples x 1, change unit from 10ppm to 100ppm               
        
        diffOffset = offset - np.tile(np.expand_dims(avgOffset, 3), (1, 1, 1, data.nAgents)) # nSamples x tSamples x 1 x nAgents
        diffSkew = skew/10 - np.tile(np.expand_dims(avgSkew, 3), (1, 1, 1, data.nAgents)) # nSamples x tSamples x 1 x nAgents
        
        diffOffset = np.sum(diffOffset**2, 2) # nSamples x tSamples x nAgents
        diffSkew = np.sum(diffSkew**2, 2) # nSamples x tSamples x nAgents
        
        diffOffsetAvg = np.mean(diffOffset, axis = 2) # nSamples x tSamples
        diffSkewAvg = np.mean(diffSkew, axis = 2) # nSamples x tSamples
        
        costPerSample = np.sum(diffOffsetAvg, axis = 1) + np.sum(diffSkewAvg, axis = 1)*(0.01**2) # nSamples
        
        cost = np.mean(costPerSample) # scalar
        print("\tThe cost of time sync for the expert model with NO attacks: %.4f" %(cost), flush = True)            

        print("\tComputing learned time synchronisation for distributed controller with NO attacks...", flush = True)
        
        offsetTestBestDistributed, \
            skewTestBestDistributed, \
                adjTestBestDistributed = \
                    data.computeDistributedCtrlTrajectory(initPosTest, initVelTest, \
                                                          measurementNoiseTest, processingNoiseTest, clockNoiseTest, \
                                                              graphTest, data.duration)

        offset = copy.deepcopy(offsetTestBestDistributed)
        skew = copy.deepcopy(skewTestBestDistributed)
        avgOffset = np.mean(offset, axis = 3) # nSamples x tSamples x 1
        avgSkew = np.mean(skew/10, axis = 3) # nSamples x tSamples x 1, change unit from 10ppm to 100ppm               
        
        diffOffset = offset - np.tile(np.expand_dims(avgOffset, 3), (1, 1, 1, data.nAgents)) # nSamples x tSamples x 1 x nAgents
        diffSkew = skew/10 - np.tile(np.expand_dims(avgSkew, 3), (1, 1, 1, data.nAgents)) # nSamples x tSamples x 1 x nAgents
        
        diffOffset = np.sum(diffOffset**2, 2) # nSamples x tSamples x nAgents
        diffSkew = np.sum(diffSkew**2, 2) # nSamples x tSamples x nAgents
        
        diffOffsetAvg = np.mean(diffOffset, axis = 2) # nSamples x tSamples
        diffSkewAvg = np.mean(diffSkew, axis = 2) # nSamples x tSamples
        
        costPerSample = np.sum(diffOffsetAvg, axis = 1) + np.sum(diffSkewAvg, axis = 1)*(0.01**2) # nSamples
        
        cost = np.mean(costPerSample) # scalar
        print("\tThe cost of time sync for distributed controller with NO attacks: %.4f" %(cost), flush = True)
            
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
        
        offset = copy.deepcopy(offsetTestBest)
        skew = copy.deepcopy(skewTestBest)
        avgOffset = np.mean(offset, axis = 3) # nSamples x tSamples x 1
        avgSkew = np.mean(skew/10, axis = 3) # nSamples x tSamples x 1, change unit from 10ppm to 100ppm               
        
        diffOffset = offset - np.tile(np.expand_dims(avgOffset, 3), (1, 1, 1, data.nAgents)) # nSamples x tSamples x 1 x nAgents
        diffSkew = skew/10 - np.tile(np.expand_dims(avgSkew, 3), (1, 1, 1, data.nAgents)) # nSamples x tSamples x 1 x nAgents
        
        diffOffset = np.sum(diffOffset**2, 2) # nSamples x tSamples x nAgents
        diffSkew = np.sum(diffSkew**2, 2) # nSamples x tSamples x nAgents
        
        diffOffsetAvg = np.mean(diffOffset, axis = 2) # nSamples x tSamples
        diffSkewAvg = np.mean(diffSkew, axis = 2) # nSamples x tSamples
        
        costPerSample = np.sum(diffOffsetAvg, axis = 1) + np.sum(diffSkewAvg, axis = 1)*(0.01**2) # nSamples
        
        cost = np.mean(costPerSample) # scalar
        print("\tThe cost of time sync for best model with NO attacks: %.4f" %(cost), flush = True)

        if data.attackMode == 1 or data.attackMode == 2: # attacking occurs
            
            print("\tComputing learned time synchronisation for the expert model under attacks...", flush = True)
            
            for i in range(attackRadiusTest.shape[2]):
                print("\tAttacking radius: %.1f... " %(attackRadiusTest[i,i,i,i]), flush = True)            
                            
                thisAttackNodeIndex = attackNodesIndexTest[:,:,i,:]
                thisNumAttackedNodes = numAttackedNodesTest[:,:,i]
                
                attackOffsetTestBest, \
                attackSkewTestBest, \
                attackAdjTestBest = \
                    data.computeExpertTrajectory(initPosTest, initVelTest, \
                                           measurementNoiseTest, processingNoiseTest, clockNoiseTest, \
                                               attackGraphTest[:, :, i, :, :], data.duration, thisAttackNodeIndex, thisNumAttackedNodes, \
                                                   archit = model.archit)    
    
                attackOffset = copy.deepcopy(attackOffsetTestBest)
                attackSkew = copy.deepcopy(attackSkewTestBest)
    
                attackCostsPerSample = []
                for instant in range(attackRadiusTest.shape[0]):
                    thisTotalAttackNodeIndex = totalAttackNodeIndex[instant][i]
    
                    for element in thisTotalAttackNodeIndex:
                        attackOffset[instant, :, :, element] = np.zeros((attackOffset.shape[1], attackOffset.shape[2]))
                        attackSkew[instant, :, :, element] = np.zeros((attackOffset.shape[1], attackOffset.shape[2]))
    
                    thisAttackOffset = copy.deepcopy(attackOffset[instant, :, :, :])
                    thisAttackSkew = copy.deepcopy(attackSkew[instant, :, :, :])
                        
                    for element in thisTotalAttackNodeIndex:
                        thisAttackOffset = np.delete(thisAttackOffset, element, axis=2)
                        thisAttackSkew = np.delete(thisAttackSkew, element, axis=2) 
                    
                    thisAttackAvgOffset = np.mean(thisAttackOffset, axis = 2) # nSamples x tSamples x 1
                    thisAttackAvgSkew = np.mean(thisAttackSkew/10, axis = 2) # nSamples x tSamples x 1, change unit from 10ppm to 100ppm   
                    
                    thisAttackDiffOffset = thisAttackOffset - np.tile(np.expand_dims(thisAttackAvgOffset, 2), (1, 1, 50-np.int64(len(thisTotalAttackNodeIndex)))) # tSamples x 1 x nAgents
                    thisAttackDiffSkew = thisAttackSkew/10 - np.tile(np.expand_dims(thisAttackAvgSkew, 2), (1, 1, 50-np.int64(len(thisTotalAttackNodeIndex)))) # tSamples x 1 x nAgents                
    
                    thisAttackDiffOffset = np.sum(thisAttackDiffOffset**2, 1) # tSamples x nAgents
                    thisAttackDiffSkew = np.sum(thisAttackDiffSkew**2, 1) # tSamples x nAgents
                    
                    thisAttackDiffOffsetAvg = np.mean(thisAttackDiffOffset, axis = 1) # nSamples x tSamples
                    thisAttackDiffSkewAvg = np.mean(thisAttackDiffSkew, axis = 1) # nSamples x tSamples
                    
                    thisAttackCostPerSample = np.sum(thisAttackDiffOffsetAvg, axis = 0) + np.sum(thisAttackDiffSkewAvg, axis = 0)*(0.01**2) # nSamples
                    
                    attackCostsPerSample.append(thisAttackCostPerSample)
                    
                attackCost = np.mean(np.array(attackCostsPerSample)) # scalar
    
                print("\tThe cost of time sync for expert controller under attacks: %.4f" %(attackCost), flush = True)

            print("\tComputing learned time synchronisation for the distributed controller under attacks...", flush = True)
            
            for i in range(attackRadiusTest.shape[2]):
                print("\tAttacking radius: %.1f... " %(attackRadiusTest[i,i,i,i]), flush = True)            
                            
                thisAttackNodeIndex = attackNodesIndexTest[:,:,i,:]
                thisNumAttackedNodes = numAttackedNodesTest[:,:,i]
                
                attackOffsetTestBestDistributed, \
                    attackSkewTestBestDistributed, \
                        attackAdjTestBestDistributed = \
                            data.computeDistributedCtrlTrajectory(initPosTest, initVelTest, \
                                                                  measurementNoiseTest, processingNoiseTest, clockNoiseTest, \
                                                                      attackGraphTest[:, :, i, :, :], data.duration)

                attackOffsetDistributed = copy.deepcopy(attackOffsetTestBestDistributed)
                attackSkewDistributed = copy.deepcopy(attackSkewTestBestDistributed)
    
                attackCostsPerSample = []
                for instant in range(attackRadiusTest.shape[0]):
                    thisTotalAttackNodeIndex = totalAttackNodeIndex[instant][i]
    
                    for element in thisTotalAttackNodeIndex:
                        attackOffsetDistributed[instant, :, :, element] = np.zeros((attackOffsetDistributed.shape[1], attackOffsetDistributed.shape[2]))
                        attackSkewDistributed[instant, :, :, element] = np.zeros((attackOffsetDistributed.shape[1], attackOffsetDistributed.shape[2]))
    
                    thisAttackOffset = copy.deepcopy(attackOffsetDistributed[instant, :, :, :])
                    thisAttackSkew = copy.deepcopy(attackSkewDistributed[instant, :, :, :])
                        
                    for element in thisTotalAttackNodeIndex:
                        thisAttackOffset = np.delete(thisAttackOffset, element, axis=2)
                        thisAttackSkew = np.delete(thisAttackSkew, element, axis=2) 
                    
                    thisAttackAvgOffset = np.mean(thisAttackOffset, axis = 2) # nSamples x tSamples x 1
                    thisAttackAvgSkew = np.mean(thisAttackSkew/10, axis = 2) # nSamples x tSamples x 1, change unit from 10ppm to 100ppm   
                    
                    thisAttackDiffOffset = thisAttackOffset - np.tile(np.expand_dims(thisAttackAvgOffset, 2), (1, 1, 50-np.int64(len(thisTotalAttackNodeIndex)))) # tSamples x 1 x nAgents
                    thisAttackDiffSkew = thisAttackSkew/10 - np.tile(np.expand_dims(thisAttackAvgSkew, 2), (1, 1, 50-np.int64(len(thisTotalAttackNodeIndex)))) # tSamples x 1 x nAgents                
    
                    thisAttackDiffOffset = np.sum(thisAttackDiffOffset**2, 1) # tSamples x nAgents
                    thisAttackDiffSkew = np.sum(thisAttackDiffSkew**2, 1) # tSamples x nAgents
                    
                    thisAttackDiffOffsetAvg = np.mean(thisAttackDiffOffset, axis = 1) # nSamples x tSamples
                    thisAttackDiffSkewAvg = np.mean(thisAttackDiffSkew, axis = 1) # nSamples x tSamples
                    
                    thisAttackCostPerSample = np.sum(thisAttackDiffOffsetAvg, axis = 0) + np.sum(thisAttackDiffSkewAvg, axis = 0)*(0.01**2) # nSamples
                    
                    attackCostsPerSample.append(thisAttackCostPerSample)
                    
                attackCost = np.mean(np.array(attackCostsPerSample)) # scalar
    
                print("\tThe cost of time sync for distributed controller under attacks: %.4f" %(attackCost), flush = True)
    
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
    
                attackCostsPerSample = []
                for instant in range(attackRadiusTest.shape[0]):
                    thisTotalAttackNodeIndex = totalAttackNodeIndex[instant][i]
    
                    for element in thisTotalAttackNodeIndex:
                        attackOffset[instant, :, :, element] = np.zeros((attackOffset.shape[1], attackOffset.shape[2]))
                        attackSkew[instant, :, :, element] = np.zeros((attackOffset.shape[1], attackOffset.shape[2]))
    
                    thisAttackOffset = copy.deepcopy(attackOffset[instant, :, :, :])
                    thisAttackSkew = copy.deepcopy(attackSkew[instant, :, :, :])
                        
                    for element in thisTotalAttackNodeIndex:
                        thisAttackOffset = np.delete(thisAttackOffset, element, axis=2)
                        thisAttackSkew = np.delete(thisAttackSkew, element, axis=2) 
                    
                    thisAttackAvgOffset = np.mean(thisAttackOffset, axis = 2) # nSamples x tSamples x 1
                    thisAttackAvgSkew = np.mean(thisAttackSkew/10, axis = 2) # nSamples x tSamples x 1, change unit from 10ppm to 100ppm   
                    
                    thisAttackDiffOffset = thisAttackOffset - np.tile(np.expand_dims(thisAttackAvgOffset, 2), (1, 1, 50-np.int64(len(thisTotalAttackNodeIndex)))) # tSamples x 1 x nAgents
                    thisAttackDiffSkew = thisAttackSkew/10 - np.tile(np.expand_dims(thisAttackAvgSkew, 2), (1, 1, 50-np.int64(len(thisTotalAttackNodeIndex)))) # tSamples x 1 x nAgents                
    
                    thisAttackDiffOffset = np.sum(thisAttackDiffOffset**2, 1) # tSamples x nAgents
                    thisAttackDiffSkew = np.sum(thisAttackDiffSkew**2, 1) # tSamples x nAgents
                    
                    thisAttackDiffOffsetAvg = np.mean(thisAttackDiffOffset, axis = 1) # nSamples x tSamples
                    thisAttackDiffSkewAvg = np.mean(thisAttackDiffSkew, axis = 1) # nSamples x tSamples
                    
                    thisAttackCostPerSample = np.sum(thisAttackDiffOffsetAvg, axis = 0) + np.sum(thisAttackDiffSkewAvg, axis = 0)*(0.01**2) # nSamples
                    
                    attackCostsPerSample.append(thisAttackCostPerSample)
                    
                attackCost = np.mean(np.array(attackCostsPerSample)) # scalar
    
                print("\tThe cost of time sync for best model under attacks: %.4f" %(attackCost), flush = True)   
                
                saveAttackDir = os.path.join(model.saveDir,'savedAttacks')
                if not os.path.exists(saveAttackDir):
                    os.makedirs(saveAttackDir)
    
                saveFile = os.path.join(saveAttackDir, 'AttackMode-' + str(data.attackMode) + '-Radius-' + str(attackRadiusTest[i,i,i,i]) + '-GNN-Results')
                
                np.savez(saveFile+'.npz', processedAttackOffset=attackOffset, processedAttackSkew=attackSkew, \
                         attackOffsetDistributed=attackOffsetDistributed, attackSkewDistributed=attackSkewDistributed, \
                             attackOffsetTestBest=attackOffsetTestBest, attackSkewTestBest=attackSkewTestBest, \
                                 attackAdjTestBest=attackAdjTestBest, \
                                     attackCenterTest=attackCenterTest, attackRadiusTest=attackRadiusTest, \
                                         numAttackedNodesTest=numAttackedNodesTest, attackNodesIndexTest=attackNodesIndexTest, \
                                             attackGraphTest=attackGraphTest)
    
                mdic = {"processedAttackOffset": attackOffset, \
                        "processedAttackSkew": attackSkew, \
                        "attackOffsetDistributed": attackOffsetDistributed, \
                        "attackSkewDistributed": attackSkewDistributed, \
                        "attackOffsetTestBest": attackOffsetTestBest, \
                        "attackSkewTestBest": attackSkewTestBest, \
                        "attackAdjTestBest": attackAdjTestBest, \
                        "attackCenterTest": attackCenterTest, \
                        "attackRadiusTest": attackRadiusTest, \
                        "numAttackedNodesTest": numAttackedNodesTest, \
                        "attackNodesIndexTest": attackNodesIndexTest, \
                        "attackGraphTest": attackGraphTest}            
                io.savemat(saveFile+'.mat', mdic)
        
        if (evalModel == False) or (data.attackMode == 0):
            saveDataDir = os.path.join(model.saveDir,'savedData')
            if not os.path.exists(saveDataDir):
                os.makedirs(saveDataDir)
        
            if layerWiseTraining == True:
                saveDataDir = os.path.join(saveDataDir,'layerWiseTraining')
            else:
                saveDataDir = os.path.join(saveDataDir,'endToEndTraining')        
            if not os.path.exists(saveDataDir):
                os.makedirs(saveDataDir)        
            
            if (data.attackMode == 0):            
                if layerWiseTraining == True:
                    saveFile = os.path.join(saveDataDir, model.name + '-nUAVs-' + str(data.nAgents) + '-LayerWise-' + str(historicalBestL[l]) + '-DAgger-' + str(historicalBestIteration[l]) + '-' + str(nDAggers) + '-Epoch-' + str(historicalBestEpoch[l]) + '-Batch-' + str(historicalBestBatch[l]))
                else:
                    saveFile = os.path.join(saveDataDir, model.name + '-nUAVs-' + str(data.nAgents) + '-EndToEnd-' + str(historicalBestL[l]) + '-DAgger-' + str(historicalBestIteration[l]) + '-' + str(nDAggers) + '-Epoch-' + str(historicalBestEpoch[l]) + '-Batch-' + str(historicalBestBatch[l]))
            elif (data.attackMode == 1) or (data.attackMode == 2):
                if layerWiseTraining == True:
                    saveFile = os.path.join(saveDataDir, model.name + '-LayerWise-' + str(historicalBestL[l]) + '-DAgger-' + str(historicalBestIteration[l]) + '-' + str(nDAggers) + '-Epoch-' + str(historicalBestEpoch[l]) + '-Batch-' + str(historicalBestBatch[l]))
                else:
                    saveFile = os.path.join(saveDataDir, model.name + '-EndToEnd-' + str(historicalBestL[l]) + '-DAgger-' + str(historicalBestIteration[l]) + '-' + str(nDAggers) + '-Epoch-' + str(historicalBestEpoch[l]) + '-Batch-' + str(historicalBestBatch[l]))
            else:
                raise Exception("unknown attack mode is found!")                                    
            
            saveFile = saveFile + '.npz'
            np.savez(saveFile, offsetTestBest=offsetTestBest, skewTestBest=skewTestBest, \
                      adjTestBest=adjTestBest, stateTestBest=stateTestBest, \
                          commGraphTestBest=commGraphTestBest, \
                              bestL = historicalBestL[l], bestIteration = historicalBestIteration[l], bestEpoch = historicalBestEpoch[l], bestBatch = historicalBestBatch[l])
                
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
