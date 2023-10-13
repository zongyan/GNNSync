import torch
import numpy as np
import datetime
import torch.nn as nn

import utils.graphML as gml
                    
class Trainer:   
    def __init__(self, model, data, nEpochs, batchSize, nDAggers, expertProb, aggregationSize, nLayers, hParamsbaseGNN, paramsLayerWiseTrain, **kwargs):
                
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
        self.trainingOptions['nLayers'] = nLayers
        self.trainingOptions['hParamsbaseGNN'] = hParamsbaseGNN
        self.trainingOptions['paramsLayerWiseTrain'] = paramsLayerWiseTrain
        
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
        nLayers = self.trainingOptions['nLayers']
        hParamsbaseGNN = self.trainingOptions['hParamsbaseGNN']
        paramsLayerWiseTrain = self.trainingOptions['paramsLayerWiseTrain']
                
        paramsNameLayerWiseTrain = list(paramsLayerWiseTrain)
        
        layerWiseTrainL = len(paramsLayerWiseTrain[paramsNameLayerWiseTrain[1]])

        assert nLayers == layerWiseTrainL
        assert len(paramsLayerWiseTrain[paramsNameLayerWiseTrain[0]]) == len(paramsLayerWiseTrain[paramsNameLayerWiseTrain[1]])        
                        
        layerWiseTrainF = paramsNameLayerWiseTrain[0]
        layerWiseTrainK = paramsNameLayerWiseTrain[1]
        layerWiseTrainE = paramsNameLayerWiseTrain[5]
        layerWiseTrainBias = paramsNameLayerWiseTrain[2]
        layerWiseTrainSigma = paramsNameLayerWiseTrain[3]
        layerWiseTraindimReadout = paramsNameLayerWiseTrain[4]
        
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

        while l < layerWiseTrainL: # number of layers added to neural network
            
            modules = [name for name, _ in thisArchit.named_parameters()]
            layers = [name[0:3] for name in modules if len(name)<=13]
            layers = layers + [name[0:7] for name in modules if len(name)>13]
            
            if "GFL" in layers:
                thisGraphFilterLayers = thisArchit.GFL
                # preserve the output layer
                lastGraphFilterLayer = thisGraphFilterLayers[-1]
                
                layerWiseGFL = [] 
                for i in range(len(thisGraphFilterLayers) - 1):                
                    # set parameters of all layers except the output layer to non-trainable
                    origLayer = thisGraphFilterLayers[i]
                    for param in origLayer.parameters():
                        param.requires_grad = False        

                    # append the original layer
                    layerWiseGFL.append(origLayer)

                # append the layer-wise training layer
                layerWiseGFL.append(layerWiseTrainSigma)                
                layerWiseGFL.append(gml.GraphFilter_DB(layerWiseTrainF[l], layerWiseTrainF[l+1], layerWiseTrainK[l], layerWiseTrainE, layerWiseTrainBias))
                 
                # add the original final output layer
                layerWiseGFL.append(lastGraphFilterLayer)

                self.GFL = nn.Sequential(*layerWiseGFL) # graph filtering layers for layer-wise training
                
            if "Readout" in layers:
                thisReadoutLayers = thisArchit.Readout
                # preserve the output layer
                lastReadoutLayer = thisReadoutLayers[-1]
                
                layerWiseFC = []
                for i in range(len(thisReadoutLayers) - 1):                
                    # set parameters of all layers except the output layer to non-trainable
                    origLayer = thisReadoutLayers[i]
                    for param in origLayer.parameters():
                        param.requires_grad = False      
                        
                    # append the original layer
                    layerWiseFC.append(origLayer)

                # append the original layer
                layerWiseFC.append(nn.Linear(layerWiseTraindimReadout[l], layerWiseTraindimReadout[l+1], bias = layerWiseTrainBias))            
                layerWiseFC.append(layerWiseTrainSigma)
                
                # add the original final output layer                
                layerWiseFC.append(lastReadoutLayer)

                self.Readout = nn.Sequential(*layerWiseFC) # readout layer for layer-wise training

            l = l + 1 # increase layer count
            
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
                            print("\t(E: %2d, B: %3d) %7.4f - %6.4fs" % (
                                    epoch+1, batch+1,
                                    lossValueTrain.item(), timeElapsed),
                                end = ' ')
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
                                                      gammaSkew=skewTestValid) # 这个数值太大了，感觉不太合理
    
                        endTime = datetime.datetime.now()
    
                        timeElapsed = abs(endTime - startTime).total_seconds()
    
                        # evalValidTB = accValid
                        # Save values
                        evalValid += [accValid]
                        timeValid += [timeElapsed]
    
                        print("\t(E: %2d, B: %3d) %8.4f - %6.4fs" % (
                                epoch+1, batch+1,
                                accValid, 
                                timeElapsed), end = ' ')
                        print("[VALIDATION", end = '')
                        print(" (%s)]" % self.model.name)
    
                        if epoch == 0 and batch == 0:
                            bestScore = accValid
                            bestEpoch, bestBatch = epoch, batch
                            self.model.save(label = 'Best')
                        else:
                            thisValidScore = accValid
                            if thisValidScore < bestScore:
                                bestScore = thisValidScore
                                bestEpoch, bestBatch = epoch, batch
                                print("\t=> New best achieved: %.4f" % \
                                          (bestScore))
                                self.model.save(label = 'Best')
                                # initialBest = False
    
                        del initThetaValid
                        del initGammaValid
                        
                    batch += 1 # end of batch, and increase batch count
    
                epoch += 1 # end of epoch, increase epoch count
    
            self.model.save(label = 'Last') # training over, save the last model
    
            if nEpochs == 0:
                self.model.save(label = 'Best')
                self.model.save(label = 'Last')
                print("\nWARNING: No training. Best and Last models are the same.\n")
    
            self.model.load(label = 'Best') # reload best model for evaluation
    
            if nEpochs > 0:
                print("\t=> Best validation achieved (E: %d, B: %d): %.4f" % (
                        bestEpoch + 1, bestBatch + 1, bestScore))
            
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
            print("\t done.", flush=True)            
            
            