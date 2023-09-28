import torch
import numpy as np
import datetime
                    
class Trainer:   
    def __init__(self, model, data, nEpochs, batchSize, nDAggers, expertProb, aggregationSize, **kwargs):
                
        self.model = model # 很好奇呀，这个model，也没有传递这个参数呀，怎么就自动的传动过来了呢。
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
        
        nTrain = self.data.nTrain
        thisArchit = self.model.archit
        thisLoss = self.model.loss
        thisOptim = self.model.optim
        thisDevice = self.model.device
        
        epoch = 0 # epoch counter
        iteration = 1 # DAgger counter        

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
        StrainAll = StrainOrig[:,adjStep,:,:]        

        while iteration < nDAggers:
            
            while epoch < nEpochs:
                randomPermutation = np.random.permutation(nTrain)
                idxEpoch = [int(i) for i in randomPermutation]
                        
                batch = 0 # batch counter
                while batch < nBatches:
                    thisBatchIndices = idxEpoch[batchIndex[batch]
                                                : batchIndex[batch+1]]
    
                    xTrain = xTrainAll[thisBatchIndices]
                    yTrain = yTrainAll[thisBatchIndices]
                    Strain = StrainAll[thisBatchIndices]
    
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

            measureNoise, 
            processNoise, 
            clkNoise,
            initThetaValid = self.data.getData('initOffset','dagger')
            initGammaValid = self.data.getData('initSkew','dagger')
            graphValid = self.data.getData('commGraph','dagger')                       
            clockNoiseValid = self.data.getData('clockNoise','dagger')                        
            measurementNoiseValid = self.data.getData('packetExchangeDelay','dagger')    
            processingNoiseValid = self.data.getData('processingDelay','dagger')                
                        
            aggTheta  = np.zeros((aggSize, tSamples, 1, self.data.nAgents), dtype = np.float64)
            aggGamma  = np.zeros((aggSize, tSamples, 1, self.data.nAgents), dtype = np.float64)
            aggAdjust = np.zeros((aggSize, tSamples, 2, self.data.nAgents), dtype=np.float64)
            aggState  = np.zeros((aggSize, tSamples, 2, self.data.nAgents), dtype=np.float64)
                        
            aggPos = np.zeros((aggSize, tSamples, 2, nAgents), dtype=np.float64)
            aggVel = np.zeros((aggSize, tSamples, 2, nAgents), dtype=np.float64)
            aggAccel = np.zeros((aggSize, tSamples, 2, nAgents), dtype=np.float64)
                        
            for t in range(1, tSamples):
                
                aggPos[:,t,:,:], aggVel[:,t,:,:], aggAccel[:,t,:,:] = self.data.computeSingleStepOptimalSpatialTrajectory(aggPos[:,t-1,:,:], aggVel[:,t-1,:,:], self.data.duration, self.data.updateTime, self.data.adjustTime, self.data.repelDist, self.data.accelMax)
                
                # choose the expeter strategy 
                if chooseExpertControl[t] == 1:

                    aggTheta[:,t,:,:], aggGamma[:,t,:,:], aggAdjust[:,t,:,:] = self.data.computeSingleStepOptimalTemporalTrajectory(aggTheta[:,t-1,:,:], aggGamma[:,t-1,:,:], measureNoise, processNoise, clkNoise, self.data.duration, self.data.updateTime, self.data.adjustTime)
        
                # If not, we compute a new trajectory based on the given architecture
                elif chooseExpertControl[t] == 0:

                    aggTheta[:,t,:,:], aggGamma[:,t,:,:], aggAdjust[:,t,:,:], aggState[:,t,:,:] = self.data.computeSingleStepTrajectory(aggTheta[:,t-1,:,:], aggGamma[:,t-1,:,:], aggAdjust[:,t-1,:,:], aggState[:,t-1,:,:], \
                                                                                                    measureNoise, processNoise, clkNoise, graph, \
                                                                                                        t, self.data.updateTime, self.data.duration)                    

                else:
                    raise Exception('Warning: error occurs in choosing expert or gnn control!')           
                    
                # store the expert controlling strategy
                    
                if 'xDAgg' not in globals():
                    xDAgg = aggInput
                else:
                    xDAgg = np.append(xDAgg, aggInput, axis=1)
                    
                if 'sDAgg' not in globals():
                    sDAgg = aggInput
                else:
                    sDAgg= np.append(xDAgg, aggInput, axis=1)                        
                        
                if 'yDAgg' not in globals():
                    yDAgg = np.expand_dims(aggExpertTorqueThrust[:, :, k], axis=1) # (nDAgg, 1, 4)
                else:
                    yDAgg = np.append(yDAgg, np.expand_dims(aggExpertTorqueThrust[:, :, k], axis=1), axis=1)                    

                    _, _, clockCorrection = self.data.computeSingleStepOptimalTemporalTrajectory(aggTheta[:,t,:,:], aggGamma[:,t,:,:], measureNoise, processNoise, clkNoise, self.data.duration, self.data.updateTime, self.data.adjustTime)
