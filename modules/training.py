import os
import torch
import numpy as np
import datetime
from pathlib import Path
import torch.nn as nn
import torch.optim as optim

import utils.graphML as gml
import modules.architecturesTime as architTime
                    
class Trainer:   
    def __init__(self, model, data, nEpochs, batchSize, \
                 nDAggers, expertProb, aggregationSize, \
                     paramsLayerWiseTrain, layerWiseTraining, \
                         lossFunction, learningRate, beta1, beta2, **kwargs):
                
        self.model = model
        self.data = data
        
        if 'printInterval' in kwargs.keys():
            printInterval = kwargs['printInterval']
            if printInterval > 0:
                doPrint = True
            else:
                doPrint = False
        else:
            doPrint = True
            printInterval = (data.nTrain//batchSize)//5

        if 'validationInterval' in kwargs.keys():
            validationInterval = kwargs['validationInterval']
        else:
            validationInterval = data.nTrain//batchSize
                    
        nTrain = data.nTrain # size of the training set

        if nTrain < batchSize:
            nBatches = 1
            batchSize = [nTrain]
        elif nTrain % batchSize != 0:
            nBatches = np.ceil(nTrain/batchSize).astype(np.int64)
            batchSize = [batchSize] * nBatches
            while sum(batchSize) != nTrain:
                batchSize[-1] -= 1
        else:
            nBatches = np.int64(nTrain/batchSize)
            batchSize = [batchSize] * nBatches
        batchIndex = np.cumsum(batchSize).tolist()
        batchIndex = [0] + batchIndex
        
        self.trainingOptions = {}
        self.trainingOptions['doPrint'] = doPrint
        self.trainingOptions['printInterval'] = printInterval
        self.trainingOptions['validationInterval'] = validationInterval
        self.trainingOptions['batchIndex'] = batchIndex
        self.trainingOptions['batchSize'] = batchSize
        self.trainingOptions['nEpochs'] = nEpochs
        self.trainingOptions['nBatches'] = nBatches
        self.trainingOptions['nDAggers'] = nDAggers
        self.trainingOptions['expertProb'] = expertProb
        self.trainingOptions['aggSize'] = aggregationSize
        self.trainingOptions['paramsLayerWiseTrain'] = paramsLayerWiseTrain
        self.trainingOptions['layerWiseTraining'] = layerWiseTraining
        self.trainingOptions['lossFunction'] = lossFunction
        self.trainingOptions['learningRate'] = learningRate
        self.trainingOptions['beta1'] = beta1
        self.trainingOptions['beta2'] = beta2
        
    def train(self):        
        printInterval = self.trainingOptions['printInterval']
        validationInterval = self.trainingOptions['validationInterval']
        batchIndex = self.trainingOptions['batchIndex']
        batchSize = self.trainingOptions['batchSize']
        nEpochs = self.trainingOptions['nEpochs']
        nBatches = self.trainingOptions['nBatches']
        nDAggers = self.trainingOptions['nDAggers']
        expertProb = self.trainingOptions['expertProb']
        aggSize = self.trainingOptions['aggSize']
        paramsLayerWiseTrain = self.trainingOptions['paramsLayerWiseTrain']        
        layerWiseTraining = self.trainingOptions['layerWiseTraining']
        lossFunction = self.trainingOptions['lossFunction']
        learningRate = self.trainingOptions['learningRate']
        beta1 = self.trainingOptions['beta1']
        beta2 = self.trainingOptions['beta2']
                
        paramsNameLayerWiseTrain = list(paramsLayerWiseTrain)
        
        layerWiseTrainL = len(paramsLayerWiseTrain[paramsNameLayerWiseTrain[1]])

        assert len(paramsLayerWiseTrain[paramsNameLayerWiseTrain[0]]) == len(paramsLayerWiseTrain[paramsNameLayerWiseTrain[1]])        
                        
        layerWiseTrainF = paramsLayerWiseTrain[paramsNameLayerWiseTrain[0]]
        layerWiseTrainK = paramsLayerWiseTrain[paramsNameLayerWiseTrain[1]]
        layerWiseTrainE = paramsLayerWiseTrain[paramsNameLayerWiseTrain[5]]
        layerWiseTrainBias = paramsLayerWiseTrain[paramsNameLayerWiseTrain[2]]
        layerWiseTrainSigma = paramsLayerWiseTrain[paramsNameLayerWiseTrain[3]]
        layerWiseTraindimReadout = paramsLayerWiseTrain[paramsNameLayerWiseTrain[4]]
                
        nTrain = self.data.nTrain
        thisDevice = self.model.device
        
        epoch = 0 # epoch counter
        iteration = 1 # DAgger counter        
        l = 0 # adding layer counter
                      
        timeTrain = []
        timeValid = []

        xTrainOrig, yTrainOrig = self.data.getSamples('train')
        StrainOrig = self.data.getData('commGraph', 'train')
        
        adjStep = [*range(0, int(self.data.duration/self.data.updateTime), \
                int(self.data.adjustTime/self.data.updateTime))]
                
        modules = [name for name, _ in self.model.archit.named_parameters()]
        layers = [name[0:3] for name in modules if len(name)<=13]
        layers = layers + [name[0:7] for name in modules if len(name)>13]  
        
        """
        if the number of graph filter layers is less than 2, layer-wise training
        in graph filter layers will raise bugs, (see the first 'append the 
        layer-wise training layer' thisArchit.F[-2] part)
        """
        assert len(self.model.archit.K) >= 2
        
        maximumLayerWiseNum = max(np.array((layerWiseTrainL, len(layerWiseTraindimReadout))))

        # store the gnn architecture in the layer-wise training
        historicalL = []
        historicalF = []
        historicalK = []
        historicalE = []
        historicalBias = []
        historicalSigma = []
        historicalReadout = []
        historicalNumReadoutLayer = []        
        
        # store the layer-wise, DAgger, epoch, and batch iteration for each best GNN model
        historicalBestL = []
        historicalBestIteration = []
        historicalBestEpoch = []
        historicalBestBatch = []        
        
        if maximumLayerWiseNum != 0:
            self.lossTrain = np.zeros(((maximumLayerWiseNum+1), nDAggers, nEpochs, nBatches))
            self.accValid = np.zeros(((maximumLayerWiseNum+1), nDAggers, nEpochs, np.int64(nBatches/validationInterval)))        
        else:
            self.lossTrain = np.zeros((nDAggers, nEpochs, nBatches))
            self.accValid = np.zeros((nDAggers, nEpochs, np.int64(nBatches/validationInterval)))                    
        
        l = 0 # layer wise training counter
        while l < maximumLayerWiseNum + 1:
            
            print("\tdimGFL: % 2s, numTap: % 2s, dimReadout: %2s " % (
                str(list(self.model.archit.F)), str(list(self.model.archit.K)), str(list(np.int64(self.model.archit.dimReadout)))
                ), end = ' ')
            print("")

            xTrainAll = xTrainOrig[:,adjStep,:,:]
            yTrainAll = yTrainOrig[:,adjStep,:,:]
            sTrainAll = StrainOrig[:,adjStep,:,:]    
            
            thisLoss = self.model.loss
            thisOptim = self.model.optim
                        
            iteration = 0 # DAgger counter
            while iteration < nDAggers:
                
                epoch = 0 # epoch counter
                while epoch < nEpochs:
                    
                    randomPermutation = np.random.permutation(nTrain)
                    idxEpoch = [int(i) for i in randomPermutation]
                    
                    lossTrain = []
                    evalValid = []
                            
                    batch = 0 # batch counter
                    while batch < nBatches:
                        
                        thisBatchIndices = idxEpoch[batchIndex[batch]
                                                    : batchIndex[batch+1]]
        
                        xTrain = xTrainAll[thisBatchIndices]
                        yTrain = yTrainAll[thisBatchIndices]
                        Strain = sTrainAll[thisBatchIndices]
        
                        xTrain = torch.tensor(xTrain, device = thisDevice)
                        Strain = torch.tensor(Strain, device = thisDevice)
                        yTrain = torch.tensor(yTrain, device = thisDevice)
        
                        startTime = datetime.datetime.now()
        
                        self.model.archit.zero_grad()
        
                        yHatTrain = self.model.archit(xTrain, Strain)
        
                        lossValueTrain = thisLoss(yHatTrain, yTrain)
        
                        lossValueTrain.backward()
        
                        thisOptim.step()
        
                        endTime = datetime.datetime.now()
        
                        timeElapsed = abs(endTime - startTime).total_seconds()
        
                        # Logging values
                        # lossTrainTB = lossValueTrain.item()
                        # Save values
                        lossTrain += [lossValueTrain.item()]
                        timeTrain += [timeElapsed]
        
                        if printInterval > 0:
                            if (epoch * nBatches + batch) % printInterval == 0:                                
                                if layerWiseTraining == True:                          
                                    print("\t(LayerWise: %3d, DAgger: %3d, Epoch: %2d, Batch: %3d) Loss: %7.4f" % (
                                            l, iteration, epoch, batch, lossValueTrain.item()), end = ' ')
                                else:
                                    print("\t(EndToEnd: %3d, DAgger: %3d, Epoch: %2d, Batch: %3d) Loss: %7.4f" % (
                                            l, iteration, epoch, batch, lossValueTrain.item()), end = ' ')
                                print("")
        
                        del xTrain
                        del Strain
                        del yTrain
                        del lossValueTrain
    
                        if (epoch * nBatches + batch) % validationInterval == 0:                    
                            startTime = datetime.datetime.now()
                                        
                            initThetaValid = self.data.getData('initOffset','valid')
                            initGammaValid = self.data.getData('initSkew','valid')
                            graphValid = self.data.getData('commGraph','valid')                       
                            clockNoiseValid = self.data.getData('clockNoise','valid')                        
                            measurementNoiseValid = self.data.getData('packetExchangeDelay','valid')    
                            processingNoiseValid = self.data.getData('processingDelay','valid')    
                            
                            offsetTestValid, skewTestValid, _, _, _ = self.data.computeTrajectory(
                                    initThetaValid, initGammaValid, measurementNoiseValid, 
                                    processingNoiseValid, clockNoiseValid, graphValid, self.data.duration,
                                    archit = self.model.archit, doPrint = False)
                            
                            accValid = self.data.evaluate(thetaOffset=offsetTestValid, 
                                                          gammaSkew=skewTestValid)
        
                            endTime = datetime.datetime.now()
        
                            timeElapsed = abs(endTime - startTime).total_seconds()
        
                            # evalValidTB = accValid
                            # Save values
                            evalValid += [accValid]
                            timeValid += [timeElapsed]
                            
                            if layerWiseTraining == True:                          
                                print("\t(LayerWise: %3d, DAgger: %3d, Epoch: %2d, Batch: %3d) Valid Accuracy: %8.4f" % (
                                        l, iteration, epoch, batch, accValid), end = ' ')
                            else:
                                print("\t(EndToEnd: %3d, DAgger: %3d, Epoch: %2d, Batch: %3d) Valid Accuracy: %8.4f" % (
                                        l, iteration, epoch, batch, accValid), end = ' ')
                            print("")
        
                            if iteration == 0 and epoch == 0 and batch == 0:
                                bestScore = accValid
                                bestL, bestIteration, bestEpoch, bestBatch = l, iteration, epoch, batch
                                self.model.save(layerWiseTraining, nDAggers, l, iteration, epoch, batch, label = 'Best')
                            else:
                                thisValidScore = accValid
                                if thisValidScore < bestScore:
                                    bestScore = thisValidScore
                                    bestL, bestIteration, bestEpoch, bestBatch = l, iteration, epoch, batch
                                    print("\t=> New best achieved: %.4f" % \
                                              (bestScore))
                                    self.model.save(layerWiseTraining, nDAggers, l, iteration, epoch, batch, label = 'Best')
                                    # initialBest = False
        
                            del initThetaValid
                            del initGammaValid                                                        
                            
                        batch += 1 # end of batch, and increase batch count
                        
                    if maximumLayerWiseNum != 0:
                        self.lossTrain[l, iteration, epoch, :] = np.asarray(lossTrain, dtype=np.float64)
                        self.accValid[l, iteration, epoch, :] = np.asarray(evalValid, dtype=np.float64)
                    else:                        
                        self.lossTrain[iteration, epoch, :] = np.asarray(lossTrain, dtype=np.float64)
                        self.accValid[iteration, epoch, :] = np.asarray(evalValid, dtype=np.float64)                       
                        
                    epoch += 1 # end of epoch, increase epoch count
        
                self.model.save(layerWiseTraining, nDAggers, l, iteration, epoch, batch, label = 'Last') # training over, save the last model
        
                if nEpochs == 0:
                    bestL, bestIteration, bestEpoch, bestBatch = l, iteration, epoch, batch
                    self.model.save(layerWiseTraining, nDAggers, l, iteration, epoch, batch, label = 'Best')
                    self.model.save(layerWiseTraining, nDAggers, l, iteration, epoch, batch, label = 'Last')
                    print("\nWARNING: No training. Best and Last models are the same.\n")
                
                if nEpochs > 0:
                    if layerWiseTraining == True:                          
                        print("\t=> Best validation achieved (LayerWise: %3d, DAgger: %3d, Epoch: %2d, Batch: %3d): %.4f" % (
                                bestL, bestIteration, bestEpoch, bestBatch, bestScore))
                    else:
                        print("\t=> Best validation achieved (EndToEnd: %3d, DAgger: %3d, Epoch: %2d, Batch: %3d): %.4f" % (
                                bestL, bestIteration, bestEpoch, bestBatch, bestScore))
                
                '''ToDo: if the 'adjustTime' is not the same as the 'updateTime', 
                         we may need to re-write the DAgger part'''
                assert self.data.adjustTime == self.data.updateTime
                tSamples = len(adjStep)            
                
                ##### Data Aggregation #####
                # Compute the prob expert
                chooseExpertProb = expertProb ** iteration
                chooseExpertControl = np.random.binomial(1, chooseExpertProb, tSamples)
    
                initThetaDAgger = self.data.getData('initOffset','dagger')
                initGammaDAgger = self.data.getData('initSkew','dagger')
                clockNoiseDAgger = self.data.getData('clockNoise','dagger')                        
                measurementNoiseDAgger = self.data.getData('packetExchangeDelay','dagger')    
                processingNoiseDAgger = self.data.getData('processingDelay','dagger')
                graphDAgger = self.data.getData('commGraph','dagger')
    
                aggTheta  = np.zeros((aggSize, tSamples, 1, self.data.nAgents), dtype = np.float64)
                aggGamma  = np.zeros((aggSize, tSamples, 1, self.data.nAgents), dtype = np.float64)
                aggAdjust = np.zeros((aggSize, tSamples, 2, self.data.nAgents), dtype=np.float64)
                aggState  = np.zeros((aggSize, tSamples, 2, self.data.nAgents), dtype=np.float64)
                
                # position and velocity are under the optimal expert controller, 
                # quadrotor position is used to obtain the communication graph, 
                # no need to re-calculate again
                aggTheta[:,0,:,:] = initThetaDAgger
                aggGamma[:,0,:,:] = initGammaDAgger
                
                for t in range(1, tSamples-1):
                                    
                    # choose the expeter strategy 
                    if chooseExpertControl[t] == 1:
    
                        aggTheta[:,t,:,:], aggGamma[:,t,:,:], aggAdjust[:,t-1,:,:] = \
                            self.data.computeSingleStepOptimalTrajectory(aggTheta[:,t-1,:,:], aggGamma[:,t-1,:,:], \
                                                                         measurementNoiseDAgger[:,t-1,:,:], processingNoiseDAgger[:,t-1,:,:], clockNoiseDAgger[:,t-1,:,:], \
                                                                             t, self.data.duration, self.data.updateTime, self.data.adjustTime)
            
                    # If not, we compute a new trajectory based on the given architecture
                    elif chooseExpertControl[t] == 0:
    
                        aggTheta, aggGamma, aggAdjust, aggState = \
                            self.data.computeSingleStepTrajectory(aggTheta, aggGamma, aggAdjust, aggState, \
                                                                  measurementNoiseDAgger, processingNoiseDAgger, clockNoiseDAgger, \
                                                                      graphDAgger, t, self.data.duration, archit = self.model.archit)                    
                                
                        _, _, _, aggInput = self.data.computeSingleStepTrajectory(aggTheta, aggGamma, aggAdjust, aggState, \
                                                                  measurementNoiseDAgger, processingNoiseDAgger, clockNoiseDAgger, \
                                                                      graphDAgger, t+1, self.data.duration, archit = self.model.archit)
    
                        _, _, aggOutput = self.data.computeSingleStepOptimalTrajectory(aggTheta[:,t,:,:], aggGamma[:,t,:,:], \
                                                                         measurementNoiseDAgger[:,t,:,:], processingNoiseDAgger[:,t,:,:], clockNoiseDAgger[:,t,:,:], \
                                                                             t+1, self.data.duration, self.data.updateTime, self.data.adjustTime)
    
                        # store the expert controlling strategy                    
                        if 'xDAgg' not in globals():
                            xDAgg = np.expand_dims(aggInput[:,t,:,:], axis=1) 
                        else:
                            xDAgg = np.append(xDAgg, np.expand_dims(aggInput[:,t,:,:], axis=1), axis=1)
                            
                        if 'sDAgg' not in globals():
                            sDAgg = np.expand_dims(graphDAgger[:,t,:,:], axis=1)
                        else:
                            sDAgg= np.append(sDAgg, np.expand_dims(graphDAgger[:,t,:,:], axis=1), axis=1)                        
                                
                        if 'yDAgg' not in globals():
                            yDAgg = np.expand_dims(aggOutput, axis=1)
                        else:
                            yDAgg = np.append(yDAgg, np.expand_dims(aggOutput, axis=1), axis=1)                    
    
                    else:
                        raise Exception('Warning: error occurs in choosing expert or gnn control!')
                
                if 'xDAgg' in globals() and 'sDAgg' in globals() and 'yDAgg' in globals():
                    xTrainAll = np.concatenate((xTrainAll, xDAgg), axis=1) # np.append uses np.concatenate   
                    sTrainAll = np.concatenate((sTrainAll, sDAgg), axis=1)
                    yTrainAll = np.concatenate((yTrainAll, yDAgg), axis=1)
                    
                    del xDAgg, sDAgg, yDAgg
                
                iteration = iteration + 1 # end of DAgger, increase iteration count

            if layerWiseTraining == True:                          
                # reload best model for layer-wise training
                self.model.load(layerWiseTraining, nDAggers, bestL, bestIteration, bestEpoch, bestBatch, label = 'Best')
            else:
                pass
            
            # store the layer-wise, DAgger, epoch, and batch iteration for each best GNN model
            historicalBestL = np.append(historicalBestL, bestL)
            historicalBestIteration = np.append(historicalBestIteration, bestIteration)
            historicalBestEpoch = np.append(historicalBestEpoch, bestEpoch)
            historicalBestBatch = np.append(historicalBestBatch, bestBatch)
            
            # store the gnn architecture in the layer-wise training                             
            historicalL = np.append(historicalL, self.model.archit.L)
            historicalF = np.append(historicalF, self.model.archit.F)
            historicalK = np.append(historicalK, self.model.archit.K)
            historicalE = np.append(historicalE, self.model.archit.E)
            historicalBias = np.append(historicalBias, self.model.archit.bias)
            historicalSigma = np.append(historicalSigma, self.model.archit.sigma)
            historicalReadout = np.append(historicalReadout, self.model.archit.dimReadout)
            historicalNumReadoutLayer = np.append(historicalNumReadoutLayer, len(self.model.archit.dimReadout))
            
            lastL = self.model.archit.L
            lastF = self.model.archit.F
            lastK = self.model.archit.K
            lastE = self.model.archit.E
            lastBias = self.model.archit.bias
            lastSigma = self.model.archit.sigma
            lastReadout = self.model.archit.dimReadout
            
            if ("GFL" in layers) and (l < layerWiseTrainL):
                
                thisGraphFilterLayers = self.model.archit.GFL
                # preserve the output layer
                lastGraphFilterLayer = thisGraphFilterLayers[-1]
                
                originalArchitF = self.model.archit.F
                originalArchitK = self.model.archit.K
                originalArchitL = self.model.archit.L
                
                self.model.archit.F = np.append(np.append(self.model.archit.F[0:-1], layerWiseTrainF[l]), self.model.archit.F[-1])
                self.model.archit.K = np.append(self.model.archit.K, layerWiseTrainK[l])
                self.model.archit.L = len(self.model.archit.K)
                
                layerWiseGFL = [] 
                for i in range(len(thisGraphFilterLayers) - 1):                

                    # set parameters of all layers except the output layer to non-trainable
                    origLayer = thisGraphFilterLayers[i]
                    
                    if layerWiseTraining == True:
                    
                        for param in origLayer.parameters():
                            param.requires_grad = False
            
                        # append the original layer
                        layerWiseGFL.append(origLayer)

                    else:

                        if (i % 2) == 0:
                            layerWiseGFL.append(gml.GraphFilter_DB(originalArchitF[np.int64(i/2)], originalArchitF[np.int64((i/2) + 1)], originalArchitK[np.int64(i/2)], self.model.archit.E, self.model.archit.bias))
                        else:
                            layerWiseGFL.append(nn.Tanh())
        
                # append the layer-wise training layer
                layerWiseGFL.append(gml.GraphFilter_DB(originalArchitF[-2], layerWiseTrainF[l], layerWiseTrainK[l], layerWiseTrainE, layerWiseTrainBias))
                layerWiseGFL.append(nn.Tanh())
                 
                # add the original final output layer
                layerWiseGFL.append(gml.GraphFilter_DB(layerWiseTrainF[l], originalArchitF[-1], originalArchitK[-1], self.model.archit.E, self.model.archit.bias))
                
                architTime.LocalGNN_DB.gflLayerWiseInit(self.model.archit, layerWiseGFL) # graph filtering layers for layer-wise training            
            
            if ("Readout" in layers) and (l < len(layerWiseTraindimReadout)):
                
                thisReadoutLayers = self.model.archit.Readout
                # preserve the output layer
                lastReadoutLayer = thisReadoutLayers[-1]
                
                layerWiseFC = []
                for i in range(len(thisReadoutLayers) - 1): 

                    # set parameters of all layers except the output layer to non-trainable
                    origLayer = thisReadoutLayers[i]
                        
                    if layerWiseTraining == True:              
                    
                        for param in origLayer.parameters():
                            param.requires_grad = False
                        
                        # append the original layer
                        layerWiseFC.append(origLayer)

                    else:

                        if (i % 2) == 0:
                            layerWiseFC.append(nn.Tanh())  
                        else:
                            layerWiseFC.append(nn.Linear(origLayer.in_features, origLayer.out_features, bias = self.model.archit.bias))
    
                # append the original layer
                layerWiseFC.append(nn.Linear(lastReadoutLayer.in_features, layerWiseTraindimReadout[l], bias = layerWiseTrainBias))            
                layerWiseFC.append(nn.Tanh())
                
                # add the original final output layer
                layerWiseFC.append(nn.Linear(layerWiseTraindimReadout[l], lastReadoutLayer.out_features, bias = self.model.archit.bias))
                
                self.model.archit.dimReadout = np.append(layerWiseTraindimReadout[0:l+1], self.model.archit.dimReadout[-1])
                architTime.LocalGNN_DB.readoutLayerWiseInit(self.model.archit, layerWiseFC) # readout layer for layer-wise training  
                
                if layerWiseTraining == True:
                    nn.init.xavier_uniform_(self.model.archit.Readout[-3].weight)
                    nn.init.zeros_(self.model.archit.Readout[-3].bias)
                    nn.init.xavier_uniform_(self.model.archit.Readout[-1].weight)
                    nn.init.zeros_(self.model.archit.Readout[-1].bias)                    
                else:
                    for i in range(np.int64((len(self.model.archit.Readout) + 1)/2)):
                        nn.init.xavier_uniform_(self.model.archit.Readout[np.int64(2*i+1)].weight)
                        nn.init.zeros_(self.model.archit.Readout[np.int64(2*i+1)].bias)
               
            del thisLoss
            del thisOptim
            
            self.model.loss = lossFunction()
            self.model.optim = optim.Adam(self.model.archit.parameters(),
                                    lr = learningRate,
                                    betas = (beta1, beta2))            
            self.model.archit.to(self.model.device)
            
            saveArchitDir = os.path.join(self.model.saveDir,'savedArchits')
            if not os.path.exists(saveArchitDir):
                os.makedirs(saveArchitDir)

            if layerWiseTraining == True:
                saveFile = os.path.join(saveArchitDir, str(self.model.name) + '-nDAggers-' + str(iteration) + '-' + str(nDAggers) + '-LayerWise-' + str(l) + '-GSO-' + str(list(lastF)) + '-Readout-' + str(list(np.int64(lastReadout))))
            else:
                saveFile = os.path.join(saveArchitDir, str(self.model.name) + '-nDAggers-' + str(iteration) + '-' + str(nDAggers) + '-EndToEnd-' + str(l) + '-GSO-' + str(list(lastF)) + '-Readout-' + str(list(np.int64(lastReadout))))

            np.savez(saveFile+'.npz', lastL=lastL, lastF=lastF, \
                     lastK=lastK, lastE=lastE, \
                         lastBias=lastBias, lastSigma=lastSigma, \
                             lastReadout=lastReadout)               
                
            l = l + 1
            
        if layerWiseTraining == True:
            saveFile = os.path.join(saveArchitDir, 'LayerWiseTraining')
        else:
            saveFile = os.path.join(saveArchitDir, 'endToEndTraining')
        
        np.savez(saveFile + '-' + str(self.model.name) + '-nDAggers-' + str(nDAggers) + '.npz', historicalL=historicalL, historicalF=historicalF, \
                 historicalK=historicalK, historicalE=historicalE, \
                     historicalBias=historicalBias, historicalSigma=historicalSigma, \
                         historicalReadout=historicalReadout, historicalNumReadoutLayer=historicalNumReadoutLayer, \
                             historicalBestL = historicalBestL, historicalBestIteration = historicalBestIteration, \
                                 historicalBestEpoch = historicalBestEpoch, historicalBestBatch = historicalBestBatch)

#%%
    def configuration(self):        
        paramsLayerWiseTrain = self.trainingOptions['paramsLayerWiseTrain']        
        layerWiseTraining = self.trainingOptions['layerWiseTraining']
                
        paramsNameLayerWiseTrain = list(paramsLayerWiseTrain)
        
        layerWiseTrainL = len(paramsLayerWiseTrain[paramsNameLayerWiseTrain[1]])

        assert len(paramsLayerWiseTrain[paramsNameLayerWiseTrain[0]]) == len(paramsLayerWiseTrain[paramsNameLayerWiseTrain[1]])        
                        
        layerWiseTrainF = paramsLayerWiseTrain[paramsNameLayerWiseTrain[0]]
        layerWiseTrainK = paramsLayerWiseTrain[paramsNameLayerWiseTrain[1]]
        layerWiseTrainE = paramsLayerWiseTrain[paramsNameLayerWiseTrain[5]]
        layerWiseTrainBias = paramsLayerWiseTrain[paramsNameLayerWiseTrain[2]]
        layerWiseTrainSigma = paramsLayerWiseTrain[paramsNameLayerWiseTrain[3]]
        layerWiseTraindimReadout = paramsLayerWiseTrain[paramsNameLayerWiseTrain[4]]
        
        l = 0 # adding layer counter
                      
        modules = [name for name, _ in self.model.archit.named_parameters()]
        layers = [name[0:3] for name in modules if len(name)<=13]
        layers = layers + [name[0:7] for name in modules if len(name)>13]  
        
        """
        if the number of graph filter layers is less than 2, layer-wise training
        in graph filter layers will raise bugs, (see the first 'append the 
        layer-wise training layer' thisArchit.F[-2] part)
        """
        assert len(self.model.archit.K) >= 2
        
        maximumLayerWiseNum = max(np.array((layerWiseTrainL, len(layerWiseTraindimReadout))))
        
        l = 0 # layer wise training counter
        while l < maximumLayerWiseNum + 1:
                        
            saveDir = os.path.join(self.model.saveDir, 'savedModels')
            if (self.trainingOptions['layerWiseTraining'] == True):
                                    
                print("\tLoading best layer-wise training %s model parameters..." % self.model.name) 
                
                saveDir = os.path.join(saveDir, 'layerWiseTraining')
                
                historicalModels = sorted(Path(saveDir).iterdir(), key=os.path.getmtime)
                                        
                thisHistoricalModels = []
    
                for element in historicalModels:            
                    if self.model.name == str(element)[70:(70+len(self.model.name))] and np.int64(str(element)[70+len(self.model.name)+len('-LayerWise-')]) == l:
                        thisHistoricalModels.append(str(element)[70:])
                
                for i in range(len(thisHistoricalModels)):
                    if list(reversed(thisHistoricalModels))[i][-9:-5] == 'Best':
                        thisBestModelArchit = list(reversed(thisHistoricalModels))[i+1]                    
                        break                    
                
                architLoadFile = os.path.join(saveDir, thisBestModelArchit) 
                self.model.archit.load_state_dict(torch.load(architLoadFile))                        
            else:
                print("\tLoading best end-to-end training %s model parameters..." % self.model.name)
                
                saveDir = os.path.join(saveDir, 'endToEndTraining')                    

                historicalModels = sorted(Path(saveDir).iterdir(), key=os.path.getmtime)
                                        
                thisHistoricalModels = []
                
                for element in historicalModels:                                    
                    if self.model.name == str(element)[69:(69+len(self.model.name))] and np.int64(str(element)[69+len(self.model.name)+len('-EndToEnd-')]) == l:                        
                        thisHistoricalModels.append(str(element)[69:])
                
                for i in range(len(thisHistoricalModels)):
                    if list(reversed(thisHistoricalModels))[i][-9:-5] == 'Best':
                        thisBestModelArchit = list(reversed(thisHistoricalModels))[i+1]   
                        break
                
                architLoadFile = os.path.join(saveDir, thisBestModelArchit) 
                self.model.archit.load_state_dict(torch.load(architLoadFile))
            
            if ("GFL" in layers) and (l < layerWiseTrainL):
                
                thisGraphFilterLayers = self.model.archit.GFL
                # preserve the output layer
                lastGraphFilterLayer = thisGraphFilterLayers[-1]
                
                originalArchitF = self.model.archit.F
                originalArchitK = self.model.archit.K
                originalArchitL = self.model.archit.L
                
                self.model.archit.F = np.append(np.append(self.model.archit.F[0:-1], layerWiseTrainF[l]), self.model.archit.F[-1])
                self.model.archit.K = np.append(self.model.archit.K, layerWiseTrainK[l])
                self.model.archit.L = len(self.model.archit.K)
                
                layerWiseGFL = [] 
                for i in range(len(thisGraphFilterLayers) - 1):                

                    # set parameters of all layers except the output layer to non-trainable
                    origLayer = thisGraphFilterLayers[i]
                    
                    if layerWiseTraining == True:
                    
                        for param in origLayer.parameters():
                            param.requires_grad = False
            
                        # append the original layer
                        layerWiseGFL.append(origLayer)

                    else:

                        if (i % 2) == 0:
                            layerWiseGFL.append(gml.GraphFilter_DB(originalArchitF[np.int64(i/2)], originalArchitF[np.int64((i/2) + 1)], originalArchitK[np.int64(i/2)], self.model.archit.E, self.model.archit.bias))
                        else:
                            layerWiseGFL.append(nn.Tanh())
        
                # append the layer-wise training layer
                layerWiseGFL.append(gml.GraphFilter_DB(originalArchitF[-2], layerWiseTrainF[l], layerWiseTrainK[l], layerWiseTrainE, layerWiseTrainBias))
                layerWiseGFL.append(nn.Tanh())
                 
                # add the original final output layer
                layerWiseGFL.append(gml.GraphFilter_DB(layerWiseTrainF[l], originalArchitF[-1], originalArchitK[-1], self.model.archit.E, self.model.archit.bias))
                
                architTime.LocalGNN_DB.gflLayerWiseInit(self.model.archit, layerWiseGFL) # graph filtering layers for layer-wise training            
            
            if ("Readout" in layers) and (l < len(layerWiseTraindimReadout)):
                
                thisReadoutLayers = self.model.archit.Readout
                # preserve the output layer
                lastReadoutLayer = thisReadoutLayers[-1]
                
                layerWiseFC = []
                for i in range(len(thisReadoutLayers) - 1): 

                    # set parameters of all layers except the output layer to non-trainable
                    origLayer = thisReadoutLayers[i]
                        
                    if layerWiseTraining == True:              
                    
                        for param in origLayer.parameters():
                            param.requires_grad = False
                        
                        # append the original layer
                        layerWiseFC.append(origLayer)

                    else:

                        if (i % 2) == 0:
                            layerWiseFC.append(nn.Tanh())  
                        else:
                            layerWiseFC.append(nn.Linear(origLayer.in_features, origLayer.out_features, bias = self.model.archit.bias))
    
                # append the original layer
                layerWiseFC.append(nn.Linear(lastReadoutLayer.in_features, layerWiseTraindimReadout[l], bias = layerWiseTrainBias))            
                layerWiseFC.append(nn.Tanh())
                
                # add the original final output layer
                layerWiseFC.append(nn.Linear(layerWiseTraindimReadout[l], lastReadoutLayer.out_features, bias = self.model.archit.bias))
                
                self.model.archit.dimReadout = np.append(layerWiseTraindimReadout[0:l+1], self.model.archit.dimReadout[-1])
                architTime.LocalGNN_DB.readoutLayerWiseInit(self.model.archit, layerWiseFC) # readout layer for layer-wise training  
                
                if layerWiseTraining == True:
                    nn.init.xavier_uniform_(self.model.archit.Readout[-3].weight)
                    nn.init.zeros_(self.model.archit.Readout[-3].bias)
                    nn.init.xavier_uniform_(self.model.archit.Readout[-1].weight)
                    nn.init.zeros_(self.model.archit.Readout[-1].bias)                    
                else:
                    for i in range(np.int64((len(self.model.archit.Readout) + 1)/2)):
                        nn.init.xavier_uniform_(self.model.archit.Readout[np.int64(2*i+1)].weight)
                        nn.init.zeros_(self.model.archit.Readout[np.int64(2*i+1)].bias)
               
            self.model.archit.to(self.model.device)
                
            l = l + 1
