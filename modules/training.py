import os
import torch
import numpy as np
import datetime
import torch.nn as nn

import utils.graphML as gml
import modules.architecturesTime as architTime
                    
class Trainer:   
    def __init__(self, model, data, nEpochs, batchSize, nDAggers, expertProb, aggregationSize, paramsLayerWiseTrain, layerWiseTraining, endToEndTraining, **kwargs):
                
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
        self.trainingOptions['endToEndTraining'] = endToEndTraining        
        
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
        endToEndTraining = self.trainingOptions['endToEndTraining']

        assert layerWiseTraining == (not endToEndTraining)        
                
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
        thisArchit = self.model.archit
        thisLoss = self.model.loss
        thisOptim = self.model.optim
        thisDevice = self.model.device
        
        epoch = 0 # epoch counter
        iteration = 1 # DAgger counter        
        l = 0 # adding layer counter
        
        lossTrain = []
        evalValid = []
        timeTrain = []
        timeValid = []

        xTrainOrig, yTrainOrig = self.data.getSamples('train')
        StrainOrig = self.data.getData('commGraph', 'train')
        
        adjStep = [*range(0, int(self.data.duration/self.data.updateTime), \
                int(self.data.adjustTime/self.data.updateTime))]
        
        xTrainAll = xTrainOrig[:,adjStep,:,:]
        yTrainAll = yTrainOrig[:,adjStep,:,:]
        sTrainAll = StrainOrig[:,adjStep,:,:]      
        
        modules = [name for name, _ in thisArchit.named_parameters()]
        layers = [name[0:3] for name in modules if len(name)<=13]
        layers = layers + [name[0:7] for name in modules if len(name)>13]  
        
        """
        if the number of graph filter layers is less than 2, layer-wise training
        in graph filter layers will raise bugs, (see the first 'append the 
        layer-wise training layer' thisArchit.F[-2] part)
        """
        assert len(thisArchit.K) >= 2
        
        maximumLayerWiseNum = max(np.array((layerWiseTrainL, len(layerWiseTraindimReadout))))

        # store the gnn architecture in the layer-wise training
        historicalL = []
        historicalF = []
        historicalK = []
        historicalE = []
        historicalBias = []
        historicalSigma = []
        historicalReadout = []
        
        l = 0 # layer wise training counter
        while l < maximumLayerWiseNum + 1:
            
            print("\tdimGFL: % 2s, numTap: % 2s, dimReadout: %2s " % (
                str(list(thisArchit.F)), str(list(thisArchit.K)), str(list(np.int64(thisArchit.dimReadout)))
                ), end = ' ')
            print("")
            
            iteration = 0 # DAgger counter
            while iteration < nDAggers:
                
                epoch = 0 # epoch counter
                while epoch < nEpochs:
                    
                    randomPermutation = np.random.permutation(nTrain)
                    idxEpoch = [int(i) for i in randomPermutation]
                            
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
        
                        thisArchit.zero_grad()
        
                        yHatTrain = thisArchit(xTrain, Strain)
        
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
                                print("\t(LayerWise: %3d, DAgger: %3d, Epoch: %2d, Batch: %3d) %7.4f" % (
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
                                    archit = thisArchit, doPrint = False)
                            
                            accValid = self.data.evaluate(thetaOffset=offsetTestValid, 
                                                          gammaSkew=skewTestValid)
        
                            endTime = datetime.datetime.now()
        
                            timeElapsed = abs(endTime - startTime).total_seconds()
        
                            # evalValidTB = accValid
                            # Save values
                            evalValid += [accValid]
                            timeValid += [timeElapsed]
        
                            print("\t(LayerWise: %3d, DAgger: %3d, Epoch: %2d, Batch: %3d) %8.4f" % (
                                    l, iteration, epoch, batch, accValid), end = ' ')
                            print("[VALIDATION]")
        
                            if epoch == 0 and batch == 0:
                                bestScore = accValid
                                bestL, bestIteration, bestEpoch, bestBatch = l, iteration, epoch, batch
                                self.model.save(layerWiseTraining, endToEndTraining, l, iteration, epoch, batch, label = 'Best')
                            else:
                                thisValidScore = accValid
                                if thisValidScore < bestScore:
                                    bestScore = thisValidScore
                                    bestL, bestIteration, bestEpoch, bestBatch = l, iteration, epoch, batch
                                    print("\t=> New best achieved: %.4f" % \
                                              (bestScore))
                                    self.model.save(layerWiseTraining, endToEndTraining, l, iteration, epoch, batch, label = 'Best')
                                    # initialBest = False
        
                            del initThetaValid
                            del initGammaValid
                            
                        batch += 1 # end of batch, and increase batch count
        
                    epoch += 1 # end of epoch, increase epoch count
        
                self.model.save(layerWiseTraining, endToEndTraining, l, iteration, epoch, batch, label = 'Last') # training over, save the last model
        
                if nEpochs == 0:
                    bestL, bestIteration, bestEpoch, bestBatch = l, iteration, epoch, batch
                    self.model.save(layerWiseTraining, endToEndTraining, l, iteration, epoch, batch, label = 'Best')
                    self.model.save(layerWiseTraining, endToEndTraining, l, iteration, epoch, batch, label = 'Last')
                    print("\nWARNING: No training. Best and Last models are the same.\n")
        
                self.model.load(layerWiseTraining, endToEndTraining, bestL, bestIteration, bestEpoch, bestBatch, label = 'Best') # reload best model for evaluation
        
                if nEpochs > 0:
                    print("\t=> Best validation achieved (E: %d, B: %d): %.4f" % (
                            bestEpoch, bestBatch, bestScore))
                
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
                                                                      graphDAgger, t, self.data.duration, archit = thisArchit)                    
                                
                        _, _, _, aggInput = self.data.computeSingleStepTrajectory(aggTheta, aggGamma, aggAdjust, aggState, \
                                                                  measurementNoiseDAgger, processingNoiseDAgger, clockNoiseDAgger, \
                                                                      graphDAgger, t+1, self.data.duration, archit = thisArchit)
    
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

            # store the gnn architecture in the layer-wise training                             
            historicalL = np.append(historicalL, thisArchit.L)
            historicalF = np.append(historicalF, thisArchit.F)
            historicalK = np.append(historicalK, thisArchit.K)
            historicalE = np.append(historicalE, thisArchit.E)
            historicalBias = np.append(historicalBias, thisArchit.bias)
            historicalSigma = np.append(historicalSigma, thisArchit.sigma)
            historicalReadout = np.append(historicalReadout, thisArchit.dimReadout)
            
            lastL = thisArchit.L
            lastF = thisArchit.F
            lastK = thisArchit.K
            lastE = thisArchit.E
            lastBias = thisArchit.bias
            lastSigma = thisArchit.sigma
            lastReadout = thisArchit.dimReadout
            
            if ("GFL" in layers) and (l < layerWiseTrainL):
                
                thisGraphFilterLayers = thisArchit.GFL
                # preserve the output layer
                lastGraphFilterLayer = thisGraphFilterLayers[-1]
                
                layerWiseGFL = [] 
                for i in range(len(thisGraphFilterLayers) - 1):                

                    # set parameters of all layers except the output layer to non-trainable
                    origLayer = thisGraphFilterLayers[i]
                    
                    if layerWiseTraining == True:
                    
                        for param in origLayer.parameters():
                            param.requires_grad = False
            
                        # append the original layer
                        layerWiseGFL.append(origLayer)

                    elif endToEndTraining == True:

                        if (i % 2) == 0:
                            layerWiseGFL.append(gml.GraphFilter_DB(thisArchit.F[np.int64(i/2)], thisArchit.F[np.int64((i/2) + 1)], thisArchit.K[np.int64(i/2)], thisArchit.E, thisArchit.bias))
                        else:
                            layerWiseGFL.append(thisArchit.sigma())                        

                    else:
                        print("\nWARNING: no training method is found.\n")  
        
                # append the layer-wise training layer
                layerWiseGFL.append(gml.GraphFilter_DB(thisArchit.F[-2], layerWiseTrainF[l], layerWiseTrainK[l], layerWiseTrainE, layerWiseTrainBias))
                layerWiseGFL.append(layerWiseTrainSigma())
                 
                # add the original final output layer
                layerWiseGFL.append(gml.GraphFilter_DB(layerWiseTrainF[l], thisArchit.F[-1], thisArchit.K[-1], thisArchit.E, thisArchit.bias))
                
                thisArchit.F = np.append(np.append(thisArchit.F[0:-1], layerWiseTrainF[l]), thisArchit.F[-1])
                thisArchit.K = np.append(thisArchit.K, layerWiseTrainK[l])
                thisArchit.L = len(thisArchit.K)
                architTime.LocalGNN_DB.gflLayerWiseInit(thisArchit, layerWiseGFL) # graph filtering layers for layer-wise training            
            
            if ("Readout" in layers) and (l < len(layerWiseTraindimReadout)):
                
                thisReadoutLayers = thisArchit.Readout
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

                    elif endToEndTraining == True:

                        if (i % 2) == 0:
                            layerWiseFC.append(thisArchit.sigma())  
                        else:
                            layerWiseFC.append(nn.Linear(origLayer.in_features, origLayer.out_features, bias = thisArchit.bias))
                            
                    else:
                        print("\nWARNING: no training method is found.\n")                             
    
                # append the original layer
                layerWiseFC.append(nn.Linear(lastReadoutLayer.in_features, layerWiseTraindimReadout[l], bias = layerWiseTrainBias))            
                layerWiseFC.append(layerWiseTrainSigma())
                
                # add the original final output layer
                layerWiseFC.append(nn.Linear(layerWiseTraindimReadout[l], lastReadoutLayer.out_features, bias = thisArchit.bias))
                
                thisArchit.dimReadout = np.append(np.append(np.append(thisArchit.dimReadout[0], thisArchit.dimReadout[1:-1]), layerWiseTraindimReadout[l]), thisArchit.dimReadout[-1])                                                
                architTime.LocalGNN_DB.readoutLayerWiseInit(thisArchit, layerWiseFC) # readout layer for layer-wise training  
            
            thisArchit.to(self.model.device)
            
            saveArchitDir = os.path.join(self.model.saveDir,'savedArchits')
            if not os.path.exists(saveArchitDir):
                os.makedirs(saveArchitDir)

            if layerWiseTraining == True:
                saveFile = os.path.join(saveArchitDir, 'LayerWise-' + str(l) + '-GSO-' + str(list(lastF)) + '-Readout-' + str(list(np.int64(lastReadout))))
            elif endToEndTraining == True:
                saveFile = os.path.join(saveArchitDir, 'EndToEnd-' + str(l) + '-GSO-' + str(list(lastF)) + '-Readout-' + str(list(np.int64(lastReadout))))                                

            np.savez(saveFile+'.npz', historicalL=lastL, historicalF=lastF, \
                     historicalK=lastK, historicalE=lastE, \
                         historicalBias=lastBias, historicalSigma=lastSigma, \
                             historicalReadout=lastReadout)
            
            l = l + 1
            
        saveFile = os.path.join(saveArchitDir, 'LayerWiseTraining')            
        np.savez(saveFile+'.npz', historicalL=historicalL, historicalF=historicalF, \
                 historicalK=historicalK, historicalE=historicalE, \
                     historicalBias=historicalBias, historicalSigma=historicalSigma, \
                         historicalReadout=historicalReadout)            