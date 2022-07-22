import torch
import numpy as np
import os
import pickle
import datetime

from utils.dataTools import invertTensorEW

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
        # end if 

        if ('learningRateDecayRate' in kwargs.keys()) and ('learningRateDecayPeriod' in kwargs.keys()):
            doLearningRateDecay = True
            learningRateDecayRate = kwargs['learningRateDecayRate']
            learningRateDecayPeriod = kwargs['learningRateDecayPeriod']
        else:
            doLearningRateDecay = False
        # end if 

        if 'validationInterval' in kwargs.keys():
            validationInterval = kwargs['validationInterval']
        else:
            validationInterval = data.nTrain//batchSize
        # end if 

        if 'earlyStoppingLag' in kwargs.keys():
            doEarlyStopping = True
            earlyStoppingLag = kwargs['earlyStoppingLag']
        else:
            doEarlyStopping = False
            earlyStoppingLag = 0
        # end if 
                    
        nTrain = data.nTrain # size of the training set

        if nTrain < batchSize:
            nBatches = 1
            batchSize = [nTrain]
        elif nTrain % batchSize != 0:
            nBatches = np.ceil(nTrain/batchSize).astype(np.int64)
            batchSize = [batchSize] * nBatches
            while sum(batchSize) != nTrain:
                batchSize[-1] -= 1
            # end while
        else:
            nBatches = np.int(nTrain/batchSize)
            batchSize = [batchSize] * nBatches
        # end if 
        
        batchIndex = np.cumsum(batchSize).tolist()
        batchIndex = [0] + batchIndex
        
        self.trainingOptions = {}
        self.trainingOptions['doPrint'] = doPrint
        self.trainingOptions['printInterval'] = printInterval
        self.trainingOptions['doLearningRateDecay'] = doLearningRateDecay
        if doLearningRateDecay:
            self.trainingOptions['learningRateDecayRate'] = learningRateDecayRate
            self.trainingOptions['learningRateDecayPeriod'] = learningRateDecayPeriod
        # end if 
        self.trainingOptions['validationInterval'] = validationInterval
        self.trainingOptions['doEarlyStopping'] = doEarlyStopping
        self.trainingOptions['earlyStoppingLag'] = earlyStoppingLag
        self.trainingOptions['batchIndex'] = batchIndex
        self.trainingOptions['batchSize'] = batchSize
        self.trainingOptions['nEpochs'] = nEpochs
        self.trainingOptions['nBatches'] = nBatches

    def train(self):
        
        printInterval = self.trainingOptions['printInterval']
        doLearningRateDecay = self.trainingOptions['doLearningRateDecay']
        if doLearningRateDecay:
            learningRateDecayRate=self.trainingOptions['learningRateDecayRate']
            learningRateDecayPeriod=self.trainingOptions['learningRateDecayPeriod']
        # end if 
        validationInterval = self.trainingOptions['validationInterval']
        doEarlyStopping = self.trainingOptions['doEarlyStopping']
        earlyStoppingLag = self.trainingOptions['earlyStoppingLag']
        batchIndex = self.trainingOptions['batchIndex']
        batchSize = self.trainingOptions['batchSize']
        nEpochs = self.trainingOptions['nEpochs']
        nBatches = self.trainingOptions['nBatches']
        
        nTrain = self.data.nTrain
        thisArchit = self.model.archit
        thisLoss = self.model.loss
        thisOptim = self.model.optim
        thisDevice = self.model.device
        
        if doLearningRateDecay:
            learningRateScheduler = torch.optim.lr_scheduler.StepLR(self.optim,
                    learningRateDecayPeriod, learningRateDecayRate)
        # end if 

        epoch = 0 # epoch counter
        lagCount = 0 # lag counter for early stopping

        xTrainAll, yTrainAll = self.data.getSamples('train')
        StrainAll = self.data.getData('commNetwk', 'train')
        initOffsetTrainAll = self.data.getData('initOffset', 'train')
        initSkewTrainAll = self.data.getData('initSkew', 'train')     
        initOffsetSkewTrainAll = np.concatenate((initOffsetTrainAll, initSkewTrainAll), axis = 2)        

        while (epoch < nEpochs) and (lagCount < earlyStoppingLag or (not doEarlyStopping)):
            randomPermutation = np.random.permutation(nTrain)
            idxEpoch = [int(i) for i in randomPermutation]

            # Learning decay
            if doLearningRateDecay:
                learningRateScheduler.step()
                print("Epoch %d, learning rate = %.8f" % (epoch+1,
                      learningRateScheduler.optim.param_groups[0]['lr']))
            # end if 
                    
            batch = 0 # batch counter
            while (batch < nBatches) and (lagCount<earlyStoppingLag or (not doEarlyStopping)):
                          
                thisBatchIndices = idxEpoch[batchIndex[batch] : batchIndex[batch+1]]

                xTrain = xTrainAll[thisBatchIndices]
                yTrain = yTrainAll[thisBatchIndices]
                Strain = StrainAll[thisBatchIndices]
                initOffsetSkewTrain = initOffsetSkewTrainAll[thisBatchIndices]                   

                xTrain = torch.tensor(xTrain, device = thisDevice)
                Strain = torch.tensor(Strain, device = thisDevice)
                yTrain = torch.tensor(yTrain, device = thisDevice)
                initOffsetSkewTrain = torch.tensor(initOffsetSkewTrain, device = thisDevice)

                startTime = datetime.datetime.now()

                thisArchit.zero_grad()

                yHatTrain = thisArchit(xTrain, Strain)

                lossValueTrain = thisLoss(yHatTrain, yTrain)

                lossValueTrain.backward()

                thisOptim.step()

                endTime = datetime.datetime.now()

                timeElapsed = abs(endTime - startTime).total_seconds()

                # Print:
                if printInterval > 0:
                    if (epoch * nBatches + batch) % printInterval == 0:
                        print("\t(E: %2d, B: %3d) %7.4f - %6.4fs" % (
                                epoch+1, batch+1,
                                lossValueTrain.item(), timeElapsed),
                            end = ' ')
                        print("")


                # Delete variables to free space in CUDA memory
                del xTrain
                del Strain
                del yTrain
                del initOffsetSkewTrain
                del lossValueTrain

                if (epoch * nBatches + batch) % validationInterval == 0:
                    
                    startTime = datetime.datetime.now()
                                        
                    # Initial data
                    initOffsetValid = self.data.getData('initOffset','valid')
                    initSkewValid = self.data.getData('initSkew','valid')
                    networkTopologyValid = self.data.getData('commNetwk','valid')                    

                    offsetTestValid, skewTestValid, _, _ = self.data.computeTimeSynchronisation(initOffsetValid, initSkewValid, networkTopologyValid, 
                                   self.data.duration, thisArchit, displayProgress=False)
                                        
                    accValid = self.data.evaluate(offset = offsetTestValid, skew = skewTestValid)

                    endTime = datetime.datetime.now()

                    timeElapsed = abs(endTime - startTime).total_seconds()

                    print("\t(E: %2d, B: %3d) %8.4f - %6.4fs" % (
                            epoch+1, batch+1,
                            accValid, 
                            timeElapsed), end = ' ')
                    print("[VALIDATION", end = '')
                    print(" (%s)]" % self.model.name)

                    if epoch == 0 and batch == 0:
                        bestScore = accValid
                        bestEpoch, bestBatch = epoch, batch
                        # Save this model as the best (so far)
                        self.model.save(label = 'Best')
                        # Start the counter
                        if doEarlyStopping:
                            initialBest = True
                    else:
                        thisValidScore = accValid
                        if thisValidScore < bestScore:
                            bestScore = thisValidScore
                            bestEpoch, bestBatch = epoch, batch
                            print("\t=> New best achieved: %.4f" % \
                                      (bestScore))
                            self.model.save(label = 'Best')
                            # Now that we have found a best that is not the
                            # initial one, we can start counting the lag (if
                            # needed)
                            initialBest = False
                            # If we achieved a new best, then we need to reset
                            # the lag count.
                            if doEarlyStopping:
                                lagCount = 0
                        # If we didn't achieve a new best, increase the lag
                        # count.
                        # Unless it was the initial best, in which case we
                        # haven't found any best yet, so we shouldn't be doing
                        # the early stopping count.
                        elif doEarlyStopping and not initialBest:
                            lagCount += 1

                    # Delete variables to free space in CUDA memory
                    del initOffsetValid
                    del initSkewValid
                    del networkTopologyValid

                #\\\\\\\
                #\\\ END OF BATCH:
                #\\\\\\\

                #\\\ Increase batch count:
                batch += 1

            #\\\\\\\
            #\\\ END OF EPOCH:
            #\\\\\\\

            #\\\ Increase epoch count:
            epoch += 1

        #\\\ Save models:
        self.model.save(label = 'Last')

        if nEpochs == 0:
            self.model.save(label = 'Best')
            self.model.save(label = 'Last')
            print("\nWARNING: No training. Best and Last models are the same.\n")

        self.model.load(label = 'Best')

        #\\\ Print out best:
        if nEpochs > 0:
            print("\t=> Best validation achieved (E: %d, B: %d): %.4f" % (
                    bestEpoch + 1, bestBatch + 1, bestScore))
    