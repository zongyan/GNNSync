import torch
import numpy as np
import os
import pickle
import datetime

from utils.dataTools import invertTensorEW
                    
class Trainer:   
    def __init__(self, model, data, nEpochs, batchSize, **kwargs):
        
        #\\\ Store model
        
        self.model = model
        self.data = data
        
        ####################################
        # ARGUMENTS (Store chosen options) #
        ####################################
        
        # Training Options:
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
        
        # No training case:
        if nEpochs == 0:
            doSaveVars = False
            doLogging = False
            # If there's no training happening, there's nothing to report about
            # training losses and stuff.
            
        ###########################################
        # DATA INPUT (pick up on data parameters) #
        ###########################################

        nTrain = data.nTrain # size of the training set

        # Number of batches: If the desired number of batches does not split the
        # dataset evenly, we reduce the size of the last batch (the number of
        # samples in the last batch).
        # The variable batchSize is a list of length nBatches (number of
        # batches), where each element of the list is a number indicating the
        # size of the corresponding batch.
        if nTrain < batchSize:
            nBatches = 1
            batchSize = [nTrain]
        elif nTrain % batchSize != 0:
            nBatches = np.ceil(nTrain/batchSize).astype(np.int64)
            batchSize = [batchSize] * nBatches
            # If the sum of all batches so far is not the total number of
            # graphs, start taking away samples from the last batch (remember
            # that we used ceiling, so we are overshooting with the estimated
            # number of batches)
            while sum(batchSize) != nTrain:
                batchSize[-1] -= 1
        # If they fit evenly, then just do so.
        else:
            nBatches = np.int(nTrain/batchSize)
            batchSize = [batchSize] * nBatches
        # batchIndex is used to determine the first and last element of each
        # batch.
        # If batchSize is, for example [20,20,20] meaning that there are three
        # batches of size 20 each, then cumsum will give [20,40,60] which
        # determines the last index of each batch: up to 20, from 20 to 40, and
        # from 40 to 60. We add the 0 at the beginning so that
        # batchIndex[b]:batchIndex[b+1] gives the right samples for batch b.
        batchIndex = np.cumsum(batchSize).tolist()
        batchIndex = [0] + batchIndex
        
        ###################
        # SAVE ATTRIBUTES #
        ###################

        self.trainingOptions = {}
        # self.trainingOptions['doLogging'] = doLogging
        # self.trainingOptions['logger'] = logger
        # self.trainingOptions['doSaveVars'] = doSaveVars
        # self.trainingOptions['doPrint'] = doPrint
        self.trainingOptions['printInterval'] = printInterval
        # self.trainingOptions['doLearningRateDecay'] = doLearningRateDecay
        # if doLearningRateDecay:
            # self.trainingOptions['learningRateDecayRate'] = \
                                                         # learningRateDecayRate
            # self.trainingOptions['learningRateDecayPeriod'] = \
                                                         # learningRateDecayPeriod
        self.trainingOptions['validationInterval'] = validationInterval
        # self.trainingOptions['doEarlyStopping'] = doEarlyStopping
        # self.trainingOptions['earlyStoppingLag'] = earlyStoppingLag
        self.trainingOptions['batchIndex'] = batchIndex
        self.trainingOptions['batchSize'] = batchSize
        self.trainingOptions['nEpochs'] = nEpochs
        self.trainingOptions['nBatches'] = nBatches
        # self.trainingOptions['graphNo'] = graphNo
        # self.trainingOptions['realizationNo'] = realizationNo
        
    def train(self):
        
        # Get back the training options
        printInterval = self.trainingOptions['printInterval']
        validationInterval = self.trainingOptions['validationInterval']
        batchIndex = self.trainingOptions['batchIndex']
        batchSize = self.trainingOptions['batchSize']
        nEpochs = self.trainingOptions['nEpochs']
        nBatches = self.trainingOptions['nBatches']
        
        # Get the values we need
        nTrain = self.data.nTrain
        thisArchit = self.model.archit
        thisLoss = self.model.loss
        thisOptim = self.model.optim
        thisDevice = self.model.device
        
        # Initialize counters (since we give the possibility of early stopping,
        # we had to drop the 'for' and use a 'while' instead):
        epoch = 0 # epoch counter
        lagCount = 0 # lag counter for early stopping

        lossTrain = []
        evalValid = []
        timeTrain = []
        timeValid = []

        # Get original dataset
        xTrainOrig, yTrainOrig = self.data.getSamples('train')
        StrainOrig = self.data.getData('commGraph', 'train')
        initVelTrainAll = self.data.getData('initSkew', 'train')

        # And save it as the original "all samples"
        xTrainAll = xTrainOrig
        yTrainAll = yTrainOrig
        StrainAll = StrainOrig

        # If it is:
        #   'randomEpoch' assigns always the original training set at the
        #       beginning of each epoch, so it is reset by using the variable
        #       Orig, instead of the variable all
        #   'replaceTimeBatch' keeps working only in the All variables, so
        #       every epoch updates the previous dataset, and never goes back
        #       to the original dataset (i.e. there is no Orig involved in
        #       the 'replaceTimeBatch' DAGger)
        #   'fixedBatch': it takes All = Orig from the beginning and then it
        #       doesn't matter becuase it always acts by creating a new
        #       batch with "corrected" trajectories for the learned policies

        while epoch < nEpochs:
            # The condition will be zero (stop), whenever one of the items of
            # the 'and' is zero. Therefore, we want this to stop only for epoch
            # counting when we are NOT doing early stopping. This can be
            # achieved if the second element of the 'and' is always 1 (so that
            # the first element, the epoch counting, decides). In order to
            # force the second element to be one whenever there is not early
            # stopping, we have an or, and force it to one. So, when we are not
            # doing early stopping, the variable 'not doEarlyStopping' is 1,
            # and the result of the 'or' is 1 regardless of the lagCount. When
            # we do early stopping, then the variable 'not doEarlyStopping' is
            # 0, and the value 1 for the 'or' gate is determined by the lag
            # count.
            # ALTERNATIVELY, we could just keep 'and lagCount<earlyStoppingLag'
            # and be sure that lagCount can only be increased whenever
            # doEarlyStopping is True. But I somehow figured out that would be
            # harder to maintain (more parts of the code to check if we are
            # accidentally increasing lagCount).

            # Randomize dataset for each epoch
            randomPermutation = np.random.permutation(nTrain)
            # Convert a numpy.array of numpy.int into a list of actual int.
            idxEpoch = [int(i) for i in randomPermutation]
                    
            # Initialize counter
            batch = 0 # batch counter
            while batch < nBatches:
                          
                # Extract the adequate batch
                thisBatchIndices = idxEpoch[batchIndex[batch]
                                            : batchIndex[batch+1]]
                # Get the samples
                xTrain = xTrainAll[thisBatchIndices]
                yTrain = yTrainAll[thisBatchIndices]
                Strain = StrainAll[thisBatchIndices]
                initVelTrain = initVelTrainAll[thisBatchIndices]

                # Now that we have our dataset, move it to tensor and device
                # so we can use it
                xTrain = torch.tensor(xTrain, device = thisDevice)
                Strain = torch.tensor(Strain, device = thisDevice)
                yTrain = torch.tensor(yTrain, device = thisDevice)
                initVelTrain = torch.tensor(initVelTrain, device = thisDevice)

                # Start measuring time
                startTime = datetime.datetime.now()

                # Reset gradients
                thisArchit.zero_grad()

                # Obtain the output of the GNN
                yHatTrain = thisArchit(xTrain, Strain)

                # Compute loss
                lossValueTrain = thisLoss(yHatTrain, yTrain)

                # Compute gradients
                lossValueTrain.backward()

                # Optimize
                thisOptim.step()

                # Finish measuring time
                endTime = datetime.datetime.now()

                timeElapsed = abs(endTime - startTime).total_seconds()

                # Logging values
                lossTrainTB = lossValueTrain.item()
                # Save values
                lossTrain += [lossValueTrain.item()]
                timeTrain += [timeElapsed]

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
                del initVelTrain
                del lossValueTrain

                # logger.scalar_summary(mode = 'Training',
                #                       epoch = epoch * nBatches + batch,
                #                       **{'lossTrain': lossTrainTB})

                #\\\\\\\
                #\\\ VALIDATION
                #\\\\\\\

                if (epoch * nBatches + batch) % validationInterval == 0:
                    
                    # Start measuring time
                    startTime = datetime.datetime.now()
                    
                    # Create trajectories
                    
                    # Initial data
                    initThetaValid = self.data.getData('initOffset','valid')
                    initGammaValid = self.data.getData('initSkew','valid')
                    graphValid = self.data.getData('commGraph','valid')                       
                    clockNoiseValid = self.data.getData('clockNoise','valid')                        
                    measurementNoiseValid = self.data.getData('packetExchangeDelay','valid')    
                    processingNoiseValid = self.data.getData('processingDelay','valid')    
                    
                    # Compute trajectories
                    offsetTestValid, skewTestValid, _, _, _ = self.data.computeTrajectory(
                            initThetaValid, initGammaValid, measurementNoiseValid, 
                            processingNoiseValid, clockNoiseValid, graphValid, self.data.duration,
                            archit = thisArchit, doPrint = False)
                    
                    # Compute evaluation
                    accValid = self.data.evaluate(thetaOffset=offsetTestValid, gammaSkew=skewTestValid)

                    # Finish measuring time
                    endTime = datetime.datetime.now()

                    timeElapsed = abs(endTime - startTime).total_seconds()

                    evalValidTB = accValid
                    # Save values
                    evalValid += [accValid]
                    timeValid += [timeElapsed]

                    print("\t(E: %2d, B: %3d) %8.4f - %6.4fs" % (
                            epoch+1, batch+1,
                            accValid, 
                            timeElapsed), end = ' ')
                    print("[VALIDATION", end = '')
                    print(" (%s)]" % self.model.name)

                    # logger.scalar_summary(mode = 'Validation',
                    #                   epoch = epoch * nBatches + batch,
                    #                   **{'evalValid': evalValidTB})

                    # No previous best option, so let's record the first trial
                    # as the best option
                    if epoch == 0 and batch == 0:
                        bestScore = accValid
                        bestEpoch, bestBatch = epoch, batch
                        # Save this model as the best (so far)
                        self.model.save(label = 'Best')
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
                        # If we didn't achieve a new best, increase the lag
                        # count.
                        # Unless it was the initial best, in which case we
                        # haven't found any best yet, so we shouldn't be doing
                        # the early stopping count.

                    # Delete variables to free space in CUDA memory
                    del initThetaValid
                    del initGammaValid

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

        #################
        # TRAINING OVER #
        #################

        # Now, if we didn't do any training (i.e. nEpochs = 0), then the last is
        # also the best.
        if nEpochs == 0:
            self.model.save(label = 'Best')
            self.model.save(label = 'Last')
            print("\nWARNING: No training. Best and Last models are the same.\n")

        # After training is done, reload best model before proceeding to
        # evaluation:
        self.model.load(label = 'Best')

        if nEpochs > 0:
            print("\t=> Best validation achieved (E: %d, B: %d): %.4f" % (
                    bestEpoch + 1, bestBatch + 1, bestScore))

    
