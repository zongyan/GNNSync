import torch
import numpy as np
import datetime
                    
class Trainer:   
    def __init__(self, model, data, nEpochs, batchSize, **kwargs):
                
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
        
    def train(self):        
        printInterval = self.trainingOptions['printInterval']
        validationInterval = self.trainingOptions['validationInterval']
        batchIndex = self.trainingOptions['batchIndex']
        batchSize = self.trainingOptions['batchSize']
        nEpochs = self.trainingOptions['nEpochs']
        nBatches = self.trainingOptions['nBatches']
        
        nTrain = self.data.nTrain
        thisArchit = self.model.archit
        thisLoss = self.model.loss
        thisOptim = self.model.optim
        thisDevice = self.model.device
        
        epoch = 0 # epoch counter

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
                                                  gammaSkew=skewTestValid)

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

    
