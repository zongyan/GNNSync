import os
import torch
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import scipy
from collections import Counter

import utils.graphTools as graph

zeroTolerance = 1e-9 # values below this number are zero

"""
change the dataType of variable 'x' into 'dataType'. If dataType cannot be 
recognised, we change it to numpy. Only converts between numpy and torch.
"""
def changeDataType(x, dataType):
    # determine the data type the variable 
    if 'dtype' in dir(x):
        varType = x.dtype
    
    if 'numpy' in repr(dataType):
        if 'torch' in repr(varType):
            x = x.cpu().numpy().astype(dataType)
        elif 'numpy' in repr(type(x)):
            x = x.astype(dataType)
    elif 'torch' in repr(dataType):
        if 'torch' in repr(varType):
            x = x.type(dataType)
        elif 'numpy' in repr(type(x)):
            x = torch.tensor(x, dtype = dataType)

    return x

'''
elementwise inversion of a tensor where the 0 elements are kept as zero.
Warning: Creates a copy of the tensor
'''
def invertTensorEW(x):    
    xInv = x.copy()
    xInv[x < zeroTolerance] = 1.
    xInv = 1./xInv # elementwise inversion
    xInv[x < zeroTolerance] = 0.
    
    return xInv

'''
Internal supraclass from which all datasets inherit. There are certain functions 
that all data classes must have: getSamples(), to() and astype(). Thus, we 
create a class from which the data can inherit this basic one.
'''
class _data:   
    def __init__(self):
        self.dataType = None
        self.device = None
        self.nTrain = None
        self.nValid = None
        self.nTest = None
        self.samples = {}
        self.samples['train'] = {}
        self.samples['train']['signals'] = None
        self.samples['train']['targets'] = None
        self.samples['dagger'] = {}
        self.samples['dagger']['signals'] = None
        self.samples['dagger']['targets'] = None        
        self.samples['valid'] = {}
        self.samples['valid']['signals'] = None
        self.samples['valid']['targets'] = None
        self.samples['test'] = {}
        self.samples['test']['signals'] = None
        self.samples['test']['targets'] = None
        
    def getSamples(self, samplesType, *args):
        # samplesType: train, valid, test
        # args: 0 args, give back all
        # args: 1 arg: if int, give that number of samples, chosen at random
        # args: 1 arg: if list (or np.array), give those samples precisely.

        assert samplesType=='train' or samplesType=='dagger' or  samplesType=='valid' or samplesType=='test'
        assert len(args) <= 1
        
        if len(args)==0:
            x = self.samples[samplesType]['signals']
            y = self.samples[samplesType]['targets']
        elif len(args)==1:
            if type(args[0]) == int:
                nSamples = x.shape[0] # total number of samples
                assert args[0] <= nSamples
                selectedIndices = np.random.choice(nSamples,size=args[0],replace=False)
                xSelected = x[selectedIndices]
                y = y[selectedIndices]
            else:
                xSelected = x[args[0]]
                y = y[args[0]]
                
            if len(xSelected.shape)<len(x.shape):
                if 'torch' in self.dataType:
                    x = xSelected.unsqueeze(0)
                else:
                    x = np.expand_dims(xSelected,axis=0)
            else:
                x = xSelected

        return x, y

    def to(self, device):
        # Only change the type of the attribute (i.e. samples), and also can 
        # only be done if they are torch tensors
        if 'torch' in repr(self.dataType):
            for key in self.samples.keys():
                for secondKey in self.samples[key].keys():
                    self.samples[key][secondKey] \
                                      = self.samples[key][secondKey].to(device)

            if device is not self.device:
                self.device = device
            
class AerialSwarm(_data):    
    def __init__(self, nAgents, commRadius, repelDist,
                 nTrain, nDAgger, nValid, nTest,
                 duration, updateTime, adjustTime,
                 initVelValue=3.,initMinDist=0.1,
                 accelMax=10., savingSeeds=True, 
                 initOffsetValue=1., initSkewValue=0.,
                 meanOffsetValue=0., stdOffsetValue=0.5,
                 meanSkewValue=0., stdSkewValue=5.,
                 sigmaMeasureOffsetValue=0., sigmaProcessOffsetValue=0.,                  
                 normalizeGraph=True, doPrint=True,
                 dataType=np.float64, device='cpu'):
        
        super().__init__()
        self.nAgents = nAgents
        self.commRadius = commRadius
        self.repelDist = repelDist

        self.nTrain = nTrain
        self.nDAgger = nDAgger
        self.nValid = nValid
        self.nTest = nTest
        nSamples = nTrain + nDAgger + nValid + nTest

        self.initVelValue = initVelValue
        self.initMinDist = initMinDist
        self.accelMax = accelMax
        
        self.initOffsetValue = initOffsetValue # x100 us
        self.meanOffsetValue = meanOffsetValue # x100 us
        self.stdOffsetValue = stdOffsetValue # x100 us
        
        self.initSkewValue = initSkewValue # x10 ppm        
        self.meanSkewValue = meanSkewValue # x10 ppm
        self.stdSkewValue = stdSkewValue # x10 ppm

        self.sigmaMeasureOffsetValue = sigmaMeasureOffsetValue # x100us 
        self.sigmaProcessOffsetValue = sigmaProcessOffsetValue # x100us
        
        self.duration = float(duration)
        self.updateTime = updateTime
        self.adjustTime = adjustTime        

        self.normalizeGraph = normalizeGraph
        self.dataType = dataType
        self.device = device

        self.doPrint = doPrint
        
        self.initOffset = None
        self.initSkew = None
        self.offset = None
        self.skew = None
        self.clockNoise = None        
        self.packetExchangeDelay = None
        self.processingDelay = None

        self.adj = None
        self.commGraph = None
        self.state = None
        
        if savingSeeds:
            if self.doPrint:
                print("\tComputing initial conditions...", end = ' ', flush = True)
            
            if nTrain == 400 or nTrain == 20:            
                initPosAll, initVelAll, \
                    initOffsetAll, initSkewAll = self.computeInitialConditions(
                                                  self.nAgents, nSamples, 
                                                  self.commRadius, self.initMinDist,                                                                                           
                                                  self.initOffsetValue, self.initSkewValue,
                                                  self.meanOffsetValue, self.stdOffsetValue,
                                                  self.meanSkewValue, self.stdSkewValue,
                                                  xMaxInitVel=self.initVelValue,
                                                  yMaxInitVel=self.initVelValue)            
                
                np.savez(os.path.join('experiments', 'flockingGNN') + 'initTrainValues' + '.npz', \
                         initPosAll=initPosAll, initVelAll=initVelAll, \
                             initOffsetAll=initOffsetAll, initSkewAll=initSkewAll)            
            else:
                initPosAll, initVelAll, \
                    initOffsetAll, initSkewAll = self.computeInitialConditions(
                                                  self.nAgents, nSamples, 
                                                  self.commRadius, self.initMinDist,                                                                                           
                                                  self.initOffsetValue, self.initSkewValue,
                                                  self.meanOffsetValue, self.stdOffsetValue,
                                                  self.meanSkewValue, self.stdSkewValue,
                                                  xMaxInitVel=self.initVelValue,
                                                  yMaxInitVel=self.initVelValue)            
                
                np.savez(os.path.join('experiments', 'flockingGNN') + 'initTestValues' + '.npz', \
                         initPosAll=initPosAll, initVelAll=initVelAll, \
                             initOffsetAll=initOffsetAll, initSkewAll=initSkewAll)                    
                
        else:

            if self.doPrint:
                print("\tLoading initial conditions...", end = ' ', flush = True)            
            
            if nTrain == 400 or nTrain == 20:
                initValuesFile = np.load(os.path.join('experiments', 'flockingGNN') + 'initTrainValues' + '.npz', allow_pickle=True) # the data file loaded from the example folder
                
                initPosAll = initValuesFile['initPosAll']
                initVelAll = initValuesFile['initVelAll']
                initOffsetAll = initValuesFile['initOffsetAll']
                initSkewAll = initValuesFile['initSkewAll']
            else:
                initValuesFile = np.load(os.path.join('experiments', 'flockingGNN') + 'initTestValues' + '.npz', allow_pickle=True) # the data file loaded from the example folder
                
                initPosAll = initValuesFile['initPosAll']
                initVelAll = initValuesFile['initVelAll']
                initOffsetAll = initValuesFile['initOffsetAll']
                initSkewAll = initValuesFile['initSkewAll']
                
        self.initOffset = {}
        self.initSkew = {}

        if self.doPrint:
            print("OK", flush = True)     
            print("\tComputing delays...", end = ' ', flush = True)
            
        clockNoiseAll, packetExchangeDelayAll, \
            processingDelayAll = self.computeNoises(self.nAgents, nSamples, 
                                                    self.duration, self.updateTime,
                                                    self.sigmaMeasureOffsetValue,
                                                    self.sigmaProcessOffsetValue, 
                                                    sigmaOffsetVal=0., 
                                                    sigmaSkewVal=0.)
                      
        self.clockNoise = {}   
        self.packetExchangeDelay = {}
        self.processingDelay = {}                      
        
        if self.doPrint:
            print("OK", flush = True)
            print("\tComputing the optimal trajectories...",
                  end=' ', flush=True)
        
        posAll, velAll, accelAll, \
            offsetAll, skewAll, adjAll = self.computeOptimalTrajectory(
                                        initPosAll, initVelAll, initOffsetAll, initSkewAll, 
                                        packetExchangeDelayAll, processingDelayAll, clockNoiseAll,
                                        self.duration, self.updateTime, self.adjustTime, self.repelDist,
                                        accelMax = self.accelMax)
        
        self.offset = {}
        self.skew = {}
        self.adj = {}
        
        if self.doPrint:
            print("OK", flush = True)
            print("\tComputing the communication graphs...",
                  end=' ', flush=True)
        
        commGraphAll = self.computeCommunicationGraph(posAll, self.commRadius,
                                                      self.normalizeGraph)
        
        self.commGraph = {}
        
        if self.doPrint:
            print("OK", flush = True)
            print("\tComputing the agent states...", end = ' ', flush = True)
        
        stateAll = self.computeStates(offsetAll, skewAll, commGraphAll)
        
        self.state = {}
        
        if self.doPrint:
            print("OK", flush = True)
        
        self.samples['train']['signals'] = stateAll[0:self.nTrain].copy()
        self.samples['train']['targets'] = adjAll[0:self.nTrain].copy()
        self.initOffset['train'] = initOffsetAll[0:self.nTrain]
        self.initSkew['train'] = initSkewAll[0:self.nTrain]
        self.clockNoise['train'] = clockNoiseAll[0:self.nTrain]        
        self.packetExchangeDelay['train'] = packetExchangeDelayAll[0:self.nTrain]
        self.processingDelay['train'] = processingDelayAll[0:self.nTrain]
        self.offset['train'] = offsetAll[0:self.nTrain]
        self.skew['train'] = skewAll[0:self.nTrain]
        self.adj['train'] = adjAll[0:self.nTrain]
        self.commGraph['train'] = commGraphAll[0:self.nTrain]
        self.state['train'] = stateAll[0:self.nTrain]

        startSample = self.nTrain
        endSample = self.nTrain + self.nDAgger
        self.samples['dagger']['signals']=stateAll[startSample:endSample].copy()
        self.samples['dagger']['targets']=adjAll[startSample:endSample].copy()
        self.initOffset['dagger'] = initOffsetAll[startSample:endSample]
        self.initSkew['dagger'] = initSkewAll[startSample:endSample]
        self.clockNoise['dagger'] = clockNoiseAll[startSample:endSample]            
        self.packetExchangeDelay['dagger'] = packetExchangeDelayAll[startSample:endSample]
        self.processingDelay['dagger'] = processingDelayAll[startSample:endSample]
        self.offset['dagger'] = offsetAll[startSample:endSample]
        self.skew['dagger'] = skewAll[startSample:endSample]
        self.adj['dagger'] = adjAll[startSample:endSample]
        self.commGraph['dagger'] = commGraphAll[startSample:endSample]
        self.state['dagger'] = stateAll[startSample:endSample]

        startSample = self.nTrain + self.nDAgger
        endSample = self.nTrain + self.nDAgger + self.nValid
        self.samples['valid']['signals']=stateAll[startSample:endSample].copy()
        self.samples['valid']['targets']=adjAll[startSample:endSample].copy()
        self.initOffset['valid'] = initOffsetAll[startSample:endSample]
        self.initSkew['valid'] = initSkewAll[startSample:endSample]
        self.clockNoise['valid'] = clockNoiseAll[startSample:endSample]            
        self.packetExchangeDelay['valid'] = packetExchangeDelayAll[startSample:endSample]
        self.processingDelay['valid'] = processingDelayAll[startSample:endSample]
        self.offset['valid'] = offsetAll[startSample:endSample]
        self.skew['valid'] = skewAll[startSample:endSample]
        self.adj['valid'] = adjAll[startSample:endSample]
        self.commGraph['valid'] = commGraphAll[startSample:endSample]
        self.state['valid'] = stateAll[startSample:endSample]

        startSample = self.nTrain + self.nDAgger + self.nValid
        endSample = self.nTrain + self.nDAgger + self.nValid + self.nTest
        self.samples['test']['signals']=stateAll[startSample:endSample].copy()
        self.samples['test']['targets']=adjAll[startSample:endSample].copy()
        self.initOffset['test'] = initOffsetAll[startSample:endSample]
        self.initSkew['test'] = initSkewAll[startSample:endSample]
        self.clockNoise['test'] = clockNoiseAll[startSample:endSample]                    
        self.packetExchangeDelay['test'] = packetExchangeDelayAll[startSample:endSample]
        self.processingDelay['test'] = processingDelayAll[startSample:endSample]
        self.offset['test'] = offsetAll[startSample:endSample]
        self.skew['test'] = skewAll[startSample:endSample]
        self.adj['test'] = adjAll[startSample:endSample]
        self.commGraph['test'] = commGraphAll[startSample:endSample]
        self.state['test'] = stateAll[startSample:endSample]        
        
        self.astype(self.dataType)
        self.to(self.device)
        
    def astype(self, dataType):
        datasetType = ['train', 'valid', 'test']
        for key in datasetType:
            self.initOffset[key] = changeDataType(self.initOffset[key], dataType)
            self.initSkew[key] = changeDataType(self.initSkew[key], dataType)
            self.offset[key] = changeDataType(self.offset[key], dataType)
            self.skew[key] = changeDataType(self.skew[key], dataType)
            self.adj[key] = changeDataType(self.adj[key], dataType)
            self.commGraph[key] = changeDataType(self.commGraph[key], dataType)
            self.state[key] = changeDataType(self.state[key], dataType)
        
    def to(self, device):
        
        if 'torch' in repr(self.dataType):
            datasetType = ['train', 'valid', 'test']
            for key in datasetType:
                self.initOffset[key].to(device)
                self.initSkew[key].to(device)
                self.clockNoise[key].to(device)
                self.packetExchangeDelay[key].to(device)
                self.processingDelay[key].to(device)
                self.offset[key].to(device)
                self.skew[key].to(device)
                self.adj[key].to(device)
                self.commGraph[key].to(device)
                self.state[key].to(device)
            
            super().to(device) # call the parent
        
    def computeStates(self, theta, gamma, graphMatrix, **kwargs):
        # input: ###
        # theta: nSamples x tSamples x 1 x nAgents
        # gamma: nSamples x tSamples x 1 x nAgents
        # graphMatrix: nSaples x tSamples x nAgents x nAgents
        
        # output: ###
        # state: nSamples x tSamples x 2 x nAgents
        
        if 'doPrint' in kwargs.keys():
            doPrint = kwargs['doPrint']
        else:
            doPrint = self.doPrint
                 
        assert len(theta.shape) == len(gamma.shape) == len(graphMatrix.shape) == 4
        nSamples = theta.shape[0]
        tSamples = theta.shape[1]
        assert theta.shape[2] == 1
        nAgents = theta.shape[3]
        assert gamma.shape[0] == graphMatrix.shape[0] == nSamples
        assert gamma.shape[1] == graphMatrix.shape[1] == tSamples
        assert gamma.shape[2] == 1
        assert gamma.shape[3] == graphMatrix.shape[2] == graphMatrix.shape[3] == nAgents
                
        maxTimeSamples = 200 
        maxBatchSize = 100 
        
        if nSamples < maxBatchSize:
            nBatches = 1
            batchSize = [nSamples]
        elif nSamples % maxBatchSize != 0:
            nBatches = nSamples // maxBatchSize + 1
            batchSize = [maxBatchSize] * nBatches
            batchSize[-1] = nSamples - sum(batchSize[0:-1])
        else:
            nBatches = int(nSamples/maxBatchSize)
            batchSize = [maxBatchSize] * nBatches

        batchIndex = np.cumsum(batchSize).tolist()
        batchIndex = [0] + batchIndex
        
        state = np.zeros((nSamples, tSamples, 2, nAgents))
        
        for b in range(nBatches):
            thetaBatch = theta[batchIndex[b]:batchIndex[b+1]]
            gammaBatch = gamma[batchIndex[b]:batchIndex[b+1]]            
            graphMatrixBatch = graphMatrix[batchIndex[b]:batchIndex[b+1]]
        
            if tSamples > maxTimeSamples:
                for t in range(tSamples):
                    offsetDiff, _ = self.computeDifferences(thetaBatch[:,t,:,:])
                    # offsetDiff: batchSize[b] x 1 x nAgents x nAgents
                    skewDiff, _ = self.computeDifferences(gammaBatch[:,t,:,:])
                    # skewDiff: batchSize[b] x 1 x nAgents x nAgents
                    
                    graphMatrixTime = (np.abs(graphMatrixBatch[:,t,:,:])>zeroTolerance).astype(theta.dtype)
                    # graphMatrix: batchSize[b] x nAgents x nAgents                    
                    graphMatrixTime = np.expand_dims(graphMatrixTime, 1)
                    # graphMatrix: batchSize[b] x 1 x nAgents x nAgents
                    
                    offsetDiff = offsetDiff * graphMatrixTime # element-wise multiplication 
                    skewDiff = skewDiff * graphMatrixTime # element-wise multiplication
                    
                    stateOffset = np.sum(offsetDiff, axis = 3)
                    # stateOffset: batchSize[b] x 1 x nAgents
                    stateSkew = np.sum(skewDiff, axis = 3)
                    # stateOffset: batchSize[b] x 1 x nAgents
                    
                    state[batchIndex[b]:batchIndex[b+1],t,:,:] = np.concatenate((stateOffset, stateSkew), axis = 1)
                    # batchSize[b] x 2 x nAgents
                    
                    if doPrint:
                        percentageCount = int(100*(t+1+b*tSamples)\
                                                          /(nBatches*tSamples))
                        
                        if t == 0 and b == 0:
                            print("%3d%%" % percentageCount,
                                  end = '', flush = True)
                        else:
                            print('\b \b' * 4 + "%3d%%" % percentageCount,
                                  end = '', flush = True)                
            else:
                offsetDiff, _ = self.computeDifferences(thetaBatch)
                # posDiff: batchSize[b] x tSamples x 1 x nAgents x nAgents
                skewDiff, _ = self.computeDifferences(gammaBatch)
                # velDiff: batchSize[b] x tSamples x 1 x nAgents x nAgents
                
                graphMatrixBatch = (np.abs(graphMatrixBatch) > zeroTolerance).astype(theta.dtype)
                # graphMatrixBatch: batchSize[b] x tSamples x nAgents x nAgents
                
                graphMatrixBatch = np.expand_dims(graphMatrixBatch, 2)
                # graphMatrix: batchSize[b] x tSamples x 1 x nAgents x nAgents
                
                offsetDiff = offsetDiff * graphMatrixBatch # element-wise multiplication
                skewDiff = skewDiff * graphMatrixBatch # element-wise multiplication
                
                stateOffset = np.sum(offsetDiff, axis = 4)
                # stateOffset: batchSize[b] x tSamples x 1 x nAgents
                stateSkew = np.sum(skewDiff, axis = 4)
                # stateSkew: batchSize[b] x tSamples x 1 x nAgents
                
                state[batchIndex[b]:batchIndex[b+1]] = np.concatenate((stateOffset, stateSkew), axis = 2)
                # state: batchSize[b] x tSamples x 2 x nAgents

                if doPrint:
                    percentageCount = int(100*(b+1)/nBatches)
                    
                    if b == 0:
                        print("%3d%%" % percentageCount,
                              end = '', flush = True)
                    else:
                        print('\b \b' * 4 + "%3d%%" % percentageCount,
                              end = '', flush = True)                        
                                                                                      
        if doPrint:
            print('\b \b' * 4, end = '', flush = True)        
        
        return state
        
    def computeCommunicationGraph(self, pos, commRadius, normalizeGraph,
                                  **kwargs):
        # input: ###
        # pos: nSamples x tSamples x 2 x nAgents
        
        # output: ###
        # graphMatrix: nSamples x tSamples x nAgents x nAgents
        
        assert commRadius>0
        assert len(pos.shape) == 4
        nSamples = pos.shape[0]
        tSamples = pos.shape[1]
        assert pos.shape[2] == 2
        nAgents = pos.shape[3]
        
        '''
        ToDo: Do we still need the 'kernelType'? The kernelType='gaussian'
        in this function
        '''
        kernelType = 'gaussian'
        kernelScale = 1.
                
        if 'doPrint' in kwargs.keys():
            doPrint = kwargs['doPrint']
        else:
            doPrint = self.doPrint
                
        maxTimeSamples = 200 
        maxBatchSize = 100
        
        if nSamples < maxBatchSize:
            nBatches = 1
            batchSize = [nSamples]
        elif nSamples % maxBatchSize != 0:
            nBatches = nSamples // maxBatchSize + 1
            batchSize = [maxBatchSize] * nBatches
            batchSize[-1] = nSamples - sum(batchSize[0:-1])
        else:
            nBatches = int(nSamples/maxBatchSize)
            batchSize = [maxBatchSize] * nBatches
        batchIndex = np.cumsum(batchSize).tolist()
        batchIndex = [0] + batchIndex
        
        graphMatrix = np.zeros((nSamples, tSamples, nAgents, nAgents))
        distanceSquare = np.zeros((nSamples, tSamples, nAgents, nAgents))
        
        for b in range(nBatches):
            posBatch = pos[batchIndex[b]:batchIndex[b+1]]
                
            if tSamples > maxTimeSamples:
                for t in range(tSamples):                    
                    _, distSq = self.computeDifferences(posBatch[:,t,:,:])

                    graphMatrixTime = np.exp(-distSq)
                    graphMatrixTime[distSq > (commRadius ** 2)] = 0.
                    graphMatrixTime[:,\
                                    np.arange(0,nAgents),np.arange(0,nAgents)]\
                                                                           = 0.

                    graphMatrixTime = (graphMatrixTime > zeroTolerance)\
                                                      .astype(distSq.dtype)
                                                              
                    if normalizeGraph:
                        isSymmetric = np.allclose(graphMatrixTime,
                                                  np.transpose(graphMatrixTime,
                                                               axes = [0,2,1]))
                    
                        adjacencyMatrixTime = graphMatrixTime
                        degreeMatrixTime = np.sum(graphMatrixTime, axis=2)
                        
                        laplacianMatrixTime = np.zeros((degreeMatrixTime.shape[0], nAgents, nAgents))
                        
                        for i in range(degreeMatrixTime.shape[0]):
                            laplacianMatrixTime[i, :, :] = np.diag(degreeMatrixTime[i, :]) - adjacencyMatrixTime[i, :, :]  # Non-Normalized laplacian matrix
    
                        graphMatrixTime = laplacianMatrixTime

                        if isSymmetric:
                            W = np.linalg.eigvalsh(graphMatrixTime)
                        else:
                            W = np.linalg.eigvals(graphMatrixTime)
                        
                        maxEigenvalue = np.max(np.real(W), axis = 1) # Todo: need check axis
                        maxEigenvalue=maxEigenvalue.reshape((batchSize[b],1,1))

                        graphMatrixTime = graphMatrixTime / maxEigenvalue
                        
                    graphMatrix[batchIndex[b]:batchIndex[b+1],t,:,:] = graphMatrixTime
                    distanceSquare[batchIndex[b]:batchIndex[b+1],t,:,:] = distSq
                    
                    if doPrint:
                        percentageCount = int(100*(t+1+b*tSamples)\
                                                          /(nBatches*tSamples))
                        
                        if t == 0 and b == 0:
                            print("%3d%%" % percentageCount,
                                  end = '', flush = True)
                        else:
                            print('\b \b' * 4 + "%3d%%" % percentageCount,
                                  end = '', flush = True)
                
            else:
                _, distSq = self.computeDifferences(posBatch)

                graphMatrixBatch = np.exp(-distSq)

                graphMatrixBatch[distSq > (commRadius ** 2)] = 0.
                graphMatrixBatch[:,:,
                                 np.arange(0,nAgents),np.arange(0,nAgents)] =0.
                graphMatrixBatch = (graphMatrixBatch > zeroTolerance)\
                                                      .astype(distSq.dtype)
                    
                if normalizeGraph:
                    isSymmetric = np.allclose(graphMatrixBatch,
                                              np.transpose(graphMatrixBatch,
                                                            axes = [0,1,3,2]))

                    adjacencyMatrixBatch = graphMatrixBatch
                    degreeMatrixBatch = np.sum(graphMatrixBatch, axis=3)
                    
                    laplacianMatrixBatch = np.zeros((degreeMatrixBatch.shape[0], degreeMatrixBatch.shape[1], nAgents, nAgents))
                    
                    for i in range(degreeMatrixBatch.shape[0]):
                        for j in range(degreeMatrixBatch.shape[0]):
                            laplacianMatrixBatch[i, j, :, :] = np.diag(degreeMatrixBatch[i, j, :]) - adjacencyMatrixBatch[i, j, :, :]  # Non-Normalized laplacian matrix

                    graphMatrixBatch = laplacianMatrixBatch

                    if isSymmetric:
                        W = np.linalg.eigvalsh(graphMatrixBatch)
                        # WW = np.linalg.eigvalsh(normalisedLaplacianMatrixTime)                        
                    else:
                        W = np.linalg.eigvals(graphMatrixBatch)
                      
                    maxEigenvalue = np.max(np.real(W), axis = 2) # Todo: need check axis
                    maxEigenvalue = maxEigenvalue.reshape((batchSize[b],
                                                            tSamples,
                                                            1, 1))

                    graphMatrixBatch = graphMatrixBatch / maxEigenvalue
                    
                graphMatrix[batchIndex[b]:batchIndex[b+1]] = graphMatrixBatch
                distanceSquare[batchIndex[b]:batchIndex[b+1]] = distSq
                
                if doPrint:
                    percentageCount = int(100*(b+1)/nBatches)
                    
                    if b == 0:
                        print("%3d%%" % percentageCount,
                              end = '', flush = True)
                    else:
                        print('\b \b' * 4 + "%3d%%" % percentageCount,
                              end = '', flush = True)
                    
        if doPrint:
            print('\b \b' * 4, end = '', flush = True)        
        
        maximumDistanceSquare = np.max(distanceSquare, axis=3)
        maxDistanceSquare = np.max(maximumDistanceSquare, axis=2)
        maxDistSquare = np.max(maxDistanceSquare, axis=1)        
        maxDist = np.sqrt(maxDistSquare)

        thisMaxDist = Counter(list(np.rint(maxDist)))
        sortedRadius = sorted(thisMaxDist.items(), reverse=True)          

        radiusRange = np.arange(start=1, stop=sortedRadius[0][0]+0.5, step=0.5)
        attackRadius = np.zeros((pos.shape[0], radiusRange.shape[0], pos.shape[1], pos.shape[3]))

        for index in range(len(radiusRange)):
            thisAttackRadius = np.reshape(np.repeat(radiusRange[index], pos.shape[0] * pos.shape[1] * pos.shape[3]), (pos.shape[0], pos.shape[1], pos.shape[3]))            
            attackRadius[:, index, :, :] = thisAttackRadius

        meanCenter = np.zeros((pos.shape[0], pos.shape[1], pos.shape[2]))

        for i in range(pos.shape[1]):            
            meanCenter[:, i, :] = np.mean(pos[:, i, :, :], axis=2)
            
        attackCenter = np.repeat(meanCenter.reshape((pos.shape[0], pos.shape[1], pos.shape[2], 1)), pos.shape[3], axis=3)            

        attackNodesIndex = np.zeros((pos.shape[0], pos.shape[1], radiusRange.shape[0], pos.shape[3]))
        
        distanceAttacker = (pos - attackCenter)**2        
        for index in range(len(radiusRange)):
            thisIsAttack = distanceAttacker[:, :, 0, :] + distanceAttacker[:, :, 1, :] <= attackRadius[:, index, :, :] ** 2        
            thisNumAttackedNodes = np.sum(thisIsAttack, axis=2)    

            thisAttackNodesIndex = np.zeros((pos.shape[0], pos.shape[1], pos.shape[3]), dtype=np.int8)
        
            for i in range(thisIsAttack.shape[0]):
                for j in range(thisIsAttack.shape[1]):
                    thisAttackNodesIndex[i, j, 0:thisNumAttackedNodes[i,j]] = np.array([index for index, element in enumerate(thisIsAttack[i, j,:].tolist()) if element == True])                        
            
            attackNodesIndex[:, :, index, :] = thisAttackNodesIndex
            
        return graphMatrix
    
    def getData(self, name, samplesType, *args):
        # samplesType: train, valid, test
        # args: 0 args, give back all
        # args: 1 arg: if int, give that number of samples, chosen at random
        # args: 1 arg: if list (or np.array), give those samples precisely.
                
        assert samplesType=='train' or samplesType=='dagger' or  samplesType=='valid' or samplesType=='test'
        assert len(args) <= 1
        assert name in dir(self)

        thisDataDict = getattr(self, name)
        assert type(thisDataDict) is dict
        assert samplesType in thisDataDict.keys()
        
        thisData = thisDataDict[samplesType]
        thisDataDims = len(thisData.shape)
        
        assert thisDataDims > 1
        
        if len(args) == 1:
            if type(args[0]) == int:
                nSamples = thisData.shape[0] 
                assert args[0] <= nSamples
                selectedIndices = np.random.choice(nSamples, size=args[0], replace=False)
                thisData = thisData[selectedIndices]
            else:
                thisData = thisData[args[0]]
                
            if len(thisData.shape) < thisDataDims:
                if 'torch' in repr(thisData.dtype):
                    thisData = thisData.unsqueeze(0)
                else:
                    thisData = np.expand_dims(thisData, axis=0)

        return thisData
        
    def evaluate(self, thetaOffset=None, gammaSkew=None, updateTime=None):
        # input: ###
        # thetaOffset: nSamples x tSamples x 1 x nAgents
        # gammaSkew: nSamples x tSamples x 1 x nAgents
        
        # output: ###
        # cost: scalar
        
        if updateTime is None:
            updateTime = self.updateTime    
        
        assert len(thetaOffset.shape) == len(gammaSkew.shape) == 4
        nSamples = thetaOffset.shape[0]
        tSamples = thetaOffset.shape[1]
        assert thetaOffset.shape[2] == 1
        nAgents = thetaOffset.shape[3]
       
        assert nSamples == gammaSkew.shape[0]
        assert tSamples == gammaSkew.shape[1]
        assert gammaSkew.shape[2] == 1
        assert nAgents == gammaSkew.shape[3]        
        
        if 'torch' in repr(thetaOffset.dtype):
            avgOffset = torch.mean(thetaOffset, dim = 3) # nSamples x tSamples x 1
            avgSkew = torch.mean(gammaSkew/10, dim = 3) # nSamples x tSamples x 1, change unit from 10ppm to 100ppm            

            diffOffset = thetaOffset - avgOffset.unsqueeze(3) # nSamples x tSamples x 1 x nAgents
            diffSkew = gammaSkew/10 - avgSkew.unsqueeze(3) # nSamples x tSamples x 1 x nAgents         
            
            diffOffset = torch.sum(diffOffset**2, dim = 2) # nSamples x tSamples x nAgents
            diffSkew = torch.sum(diffSkew**2, dim = 2) # nSamples x tSamples x nAgents        
            
            diffOffsetAvg = torch.mean(diffOffset, dim = 2) # nSamples x tSamples
            diffSkewAvg = torch.mean(diffSkew, dim = 2) # nSamples x tSamples            

            costPerSample = torch.sum(diffOffsetAvg, dim = 1) + torch.sum(diffSkewAvg, dim = 1)*(updateTime**2) # nSamples            

            cost = torch.mean(costPerSample) # scalar            
        else:            
            avgOffset = np.mean(thetaOffset, axis = 3) # nSamples x tSamples x 1
            avgSkew = np.mean(gammaSkew/10, axis= 3) # nSamples x tSamples x 1, change unit from 10ppm to 100ppm               

            diffOffset = thetaOffset - np.tile(np.expand_dims(avgOffset, 3), (1, 1, 1, nAgents)) # nSamples x tSamples x 1 x nAgents
            diffSkew = gammaSkew/10 - np.tile(np.expand_dims(avgSkew, 3), (1, 1, 1, nAgents)) # nSamples x tSamples x 1 x nAgents
            
            diffOffset = np.sum(diffOffset**2, 2) # nSamples x tSamples x nAgents
            diffSkew = np.sum(diffSkew**2, 2) # nSamples x tSamples x nAgents
            
            diffOffsetAvg = np.mean(diffOffset, axis = 2) # nSamples x tSamples
            diffSkewAvg = np.mean(diffSkew, axis = 2) # nSamples x tSamples

            costPerSample = np.sum(diffOffsetAvg, axis = 1) + np.sum(diffSkewAvg, axis = 1)*updateTime # nSamples

            cost = np.mean(costPerSample) # scalar        
        return cost
       
    def computeTrajectory(self, initTheta, initGamma, 
                           measureNoise, processNoise, clkNoise,
                           graph, duration, **kwargs):
        # input: ###
        # initTheta: nSamples x 1 x nAgents
        # initGamma: nSamples x 1 x nAgents
        # measureNoise: nSamples x tSamples x 2 x nAgents
        # processNoise: nSamples x tSamples x 2 x nAgents
        # clkNoise: nSamples x tSamples x 2 x nAgents
        # graph: nSamples x tSamples x nAgents x nAgents
        
        # output: ###
        # theta: nSamples x tSamples x 1 x nAgents
        # gamma： nSamples x tSamples x 1 x nAgents
        # adjust： nSamples x tSamples x 2 x nAgents
        # state： nSamples x tSamples x 2 x nAgents
        # graph： nSamples x tSamples x nAgents x nAgents
        
        assert len(initTheta.shape) == 3
        batchSize = initTheta.shape[0]
        assert initTheta.shape[1] == 1
        nAgents = initTheta.shape[2]
        
        assert len(initGamma.shape) == 3
        assert initGamma.shape[0] == batchSize
        assert initGamma.shape[1] == 1
        assert initGamma.shape[2] == nAgents
        
        assert len(graph.shape) == 4
        assert graph.shape[0] == batchSize
        assert graph.shape[1] == int(duration/self.updateTime)
        assert graph.shape[2] == nAgents
        assert graph.shape[3] == nAgents       

        '''
        ToDo: since this function is numpy, but the torch is used during 
        training. There exists exchange beteween numpy and torch, which slows 
        down training speed [torch -> numpy -> torch]. 
        '''
        if 'torch' in repr(initTheta.dtype):
            assert 'torch' in repr(initGamma.dtype)
            useTorch = True
            device = initTheta.device
            assert initGamma.device == device
        else:
            useTorch = False
        
        time = np.arange(0, duration, self.updateTime)
        tSamples = len(time)
       
        assert 'archit' in kwargs.keys()
        archit = kwargs['archit']
        architDevice = list(archit.parameters())[0].device
            
        if 'doPrint' in kwargs.keys():
            doPrint = kwargs['doPrint']
        else:
            doPrint = self.doPrint # Use default      
            
        theta = np.zeros((batchSize, tSamples, 1, nAgents), dtype = np.float64)
        gamma = np.zeros((batchSize, tSamples, 1, nAgents), dtype = np.float64)
        adjust = np.zeros((batchSize, tSamples, 2, nAgents), dtype=np.float64)
        state = np.zeros((batchSize, tSamples, 2, nAgents), dtype=np.float64)
        
        beforeAct = []
        afterAct = []        
            
        if useTorch:
            theta[:,0,:,:] = initTheta.cpu().numpy()
            gamma[:,0,:,:] = initGamma.cpu().numpy()
        else:
            theta[:,0,:,:] = initTheta.copy()
            gamma[:,0,:,:] = initGamma.copy()
            
        if doPrint:
            percentageCount = int(100/tSamples)
            print("%3d%%" % percentageCount, end = '', flush = True)            
        
        for t in range(1, tSamples):
            thisOffset = np.expand_dims(theta[:,t-1,:,:], 1) \
                         + np.expand_dims(measureNoise[:,t-1,0,:], (1, 2))
            thisSkew = np.expand_dims(gamma[:,t-1,:,:], 1) \
                         + np.expand_dims(measureNoise[:,t-1,1,:], (1, 2))
            thisGraph = np.expand_dims(graph[:,t-1,:,:], 1)

            thisState = self.computeStates(thisOffset, thisSkew, thisGraph, doPrint=False)
            state[:,t-1,:,:] = thisState.squeeze(1)
            
            x = torch.tensor(state[:,0:t,:,:], device = architDevice)
            S = torch.tensor(graph[:,0:t,:,:], device = architDevice)
            with torch.no_grad():
                thisAdjust = archit(x, S)
            thisAdjust = thisAdjust.cpu().numpy()[:,-1,:,:]
            adjust[:,t-1,:,:] = thisAdjust
            
            # if archit.evalModel == True:
            #     beforeAct.append(archit.beforeActivation) # values before the activation function
            #     afterAct.append(archit.afterActivation) # values from the activation function
            
            if self.updateTime == self.adjustTime:                            
                theta[:,t,:,:] = theta[:,t-1,:,:] + gamma[:,t-1,:,:] * self.updateTime \
                                                  + (1/nAgents) * np.expand_dims(adjust[:,t-1,0,:], 1) \
                                                  + np.expand_dims(clkNoise[:,t-1,0,:], 1) \
                                                  - np.expand_dims(processNoise[:,t-1,0,:], 1)
                gamma[:,t,:,:] = gamma[:,t-1,:,:] + (1/nAgents) * np.expand_dims(adjust[:,t-1,1,:], 1)\
                                                  + np.expand_dims(clkNoise[:,t-1,1,:], 1)                                              
            else:
                if int(t % (self.adjustTime/self.updateTime)) == 0:                                
                    theta[:,t,:,:] = theta[:,t-1,:,:] + gamma[:,t-1,:,:] * self.updateTime \
                                                      + (1/nAgents) * np.expand_dims(adjust[:,t-1,0,:], 1) \
                                                      + np.expand_dims(clkNoise[:,t-1,0,:], 1) \
                                                      - np.expand_dims(processNoise[:,t-1,0,:], 1)
                    gamma[:,t,:,:] = gamma[:,t-1,:,:] + (1/nAgents) * np.expand_dims(adjust[:,t-1,1,:], 1)\
                                                      + np.expand_dims(clkNoise[:,t-1,1,:], 1)                                        
                else: 
                    theta[:,t,:,:] = theta[:,t-1,:,:] + gamma[:,t-1,:,:] * self.updateTime \
                                                      + np.expand_dims(clkNoise[:,t-1,0,:], 1) \
                                                      - np.expand_dims(processNoise[:,t-1,0,:], 1)
                    gamma[:,t,:,:] = gamma[:,t-1,:,:] + np.expand_dims(clkNoise[:,t-1,1,:], 1)                    
            
            if doPrint:
                percentageCount = int(100*(t+1)/tSamples)
                print('\b \b' * 4 + "%3d%%" % percentageCount,
                      end = '', flush = True)

        thisOffset = np.expand_dims(theta[:,-1,:,:], 1) \
                        + np.expand_dims(measureNoise[:,-1,0,:], (1, 2))
        thisSkew = np.expand_dims(gamma[:,-1,:,:], 1) \
                         + np.expand_dims(measureNoise[:,-1,1,:], (1, 2))            
        thisGraph = np.expand_dims(graph[:,-1,:,:], 1)

        thisState = self.computeStates(thisOffset, thisSkew, thisGraph, doPrint=False)
        state[:,-1,:,:] = thisState.squeeze(1)

        x = torch.tensor(state).to(architDevice)
        S = torch.tensor(graph).to(architDevice)
        with torch.no_grad():
            thisAdjust = archit(x, S)
        thisAdjust = thisAdjust.cpu().numpy()[:,-1,:,:]
        adjust[:,-1,:,:] = thisAdjust
                
        if doPrint:
            print('\b \b' * 4, end = '', flush = True)

        if useTorch:            
            theta = torch.tensor(theta).to(device)
            gamma = torch.tensor(gamma).to(device)
            adjust = torch.tensor(adjust).to(device)
            
        # if archit.evalModel == True:
            
        #     laplacianMatrix = np.zeros((graph.shape[0], tSamples, nAgents, nAgents), dtype = np.float64)
        #     eigenValues = np.zeros((graph.shape[0], tSamples, nAgents), dtype = np.float64)
        #     eigenVectors = np.zeros((graph.shape[0], tSamples, nAgents, nAgents), dtype = np.float64)
        #     beforeActValues = np.zeros((graph.shape[0], tSamples, beforeAct[0][0].shape[1], nAgents), dtype = np.float64) # values before the activation function
        #     afterActValues = np.zeros((graph.shape[0], tSamples, afterAct[0][0].shape[1], nAgents), dtype = np.float64) # values after the activation function

        #     for t in range(1, tSamples-1):
        #         thisBeforeAct = beforeAct[t][0] # values before the activation function
        #         thisAfterAct = afterAct[t][0] # values after the activation function    
                
        #         thisGraph = graph[:,t,:,:]
        #         thisGraph[thisGraph > zeroTolerance] = 1. # reset the normalised adjacency matrix to the normal graph matrix
        #         thisAdjacencyMatrix = thisGraph
        #         thisDegreeMatrix = np.sum(thisAdjacencyMatrix, axis=2)
                
        #         for i in range(thisDegreeMatrix.shape[0]): # in the experiment instant dimension                    
        #             laplacianMatrix[i, t, :, :] = np.diag(thisDegreeMatrix[i, :]) - thisAdjacencyMatrix[i, :, :]  # Non-Normalized laplacian matrix
        #             # The column eigenvectors[:, i] is the normalized eigenvector corresponding to the eigenvalue eigenvalues[i]
        #             eigenValues[i, t, :], eigenVectors[i, t, :, :] = np.linalg.eigh(laplacianMatrix[i, t, :, :])

        #             assert thisBeforeAct.shape[1] == thisAfterAct.shape[1]
        #             for j in range(thisBeforeAct.shape[1]): # in the feature dimension                    
        #                 beforeActValues[i, t, j, :] = np.matmul(np.transpose(eigenVectors[i, t, :, :]), np.float64(thisBeforeAct[i, j, :])) # values before the activation function
        #                 afterActValues[i, t, j, :] = np.matmul(np.transpose(eigenVectors[i, t, :, :]), np.float64(thisAfterAct[i, j, :])) # values after the activation function

        #     for i in range(0, thisBeforeAct.shape[1]):
        #         plt.figure()
        #         plt.rcParams["figure.figsize"] = (6.4,4.8)
        #         for t in range(0, tSamples-1):
        #             plt.vlines(eigenValues[0, t, :], 0, beforeActValues[0, t, i, :], color='#D3D3D3', alpha=0.4)
        #             plt.scatter(eigenValues[0, t, :], beforeActValues[0, t, i, :], color='#D3D3D3', alpha=0.4)
                    
        #             plt.vlines(eigenValues[0, t, :], 0, afterActValues[0, t, i, :], color='#7BC8F6', alpha=0.4)
        #             plt.scatter(eigenValues[0, t, :], afterActValues[0, t, i, :], color='#7BC8F6', alpha=0.4)
            
        return theta, gamma, adjust, state, graph     

    def computeSingleStepTrajectory(self, theta, gamma, adjust, state,
                                measureNoise, processNoise, clkNoise,
                           graph, step, duration, **kwargs):
        # input: ###
        # theta: nSamples x tSamples x 1 x nAgents
        # gamma: nSamples x tSamples x 1 x nAgents
        # adjust： nSamples x tSamples x 2 x nAgents
        # state： nSamples x tSamples x 2 x nAgents        
        # measureNoise: nSamples x tSamples x 2 x nAgents
        # processNoise: nSamples x tSamples x 2 x nAgents
        # clkNoise: nSamples x tSamples x 2 x nAgents
        # graph: nSamples x tSamples x nAgents x nAgents
        
        # output: ###
        # theta: nSamples x tSamples x 1 x nAgents
        # gamma： nSamples x tSamples x 1 x nAgents
        # adjust： nSamples x tSamples x 2 x nAgents
        # state： nSamples x tSamples x 2 x nAgents
        # graph： nSamples x tSamples x nAgents x nAgents
        
        assert len(theta.shape) == 4
        batchSize = theta.shape[0]
        assert theta.shape[1] == int(duration/self.updateTime)
        assert theta.shape[2] == 1
        nAgents = theta.shape[3]
        
        assert len(gamma.shape) == 4
        assert gamma.shape[0] == batchSize
        assert gamma.shape[1] == int(duration/self.updateTime)        
        assert gamma.shape[2] == 1
        assert gamma.shape[3] == nAgents
        
        assert len(adjust.shape) == 4
        assert adjust.shape[0] == batchSize
        assert adjust.shape[1] == int(duration/self.updateTime)        
        assert adjust.shape[2] == 2
        assert adjust.shape[3] == nAgents     
        
        assert len(state.shape) == 4
        assert state.shape[0] == batchSize
        assert state.shape[1] == int(duration/self.updateTime)        
        assert state.shape[2] == 2
        assert state.shape[3] == nAgents        
        
        assert len(graph.shape) == 4
        assert graph.shape[0] == batchSize
        assert graph.shape[1] == int(duration/self.updateTime)
        assert graph.shape[2] == nAgents
        assert graph.shape[3] == nAgents       

        '''
        ToDo: since this function is numpy, but the torch is used during 
        training. There exists exchange beteween numpy and torch, which slows 
        down training speed [torch -> numpy -> torch]. 
        '''
        if 'torch' in repr(theta.dtype):
            assert 'torch' in repr(gamma.dtype)
            useTorch = True
            device = theta.device
            assert gamma.device == device
        else:
            useTorch = False
        
        time = np.arange(0, duration, self.updateTime)
        tSamples = len(time)
       
        assert 'archit' in kwargs.keys()
        archit = kwargs['archit']
        architDevice = list(archit.parameters())[0].device
            
        if 'doPrint' in kwargs.keys():
            doPrint = kwargs['doPrint']
        else:
            doPrint = self.doPrint # Use default      

        thisOffset = np.expand_dims(theta[:,step-1,:,:], 1) \
                     + np.expand_dims(measureNoise[:,step-1,0,:], (1, 2))
        thisSkew = np.expand_dims(gamma[:,step-1,:,:], 1) \
                     + np.expand_dims(measureNoise[:,step-1,1,:], (1, 2))
        thisGraph = np.expand_dims(graph[:,step-1,:,:], 1)

        thisState = self.computeStates(thisOffset, thisSkew, thisGraph, doPrint=False)
        state[:,step-1,:,:] = thisState.squeeze(1)
        
        x = torch.tensor(state[:,0:step,:,:], device = architDevice)
        S = torch.tensor(graph[:,0:step,:,:], device = architDevice)
        with torch.no_grad():
            thisAdjust = archit(x, S)
        thisAdjust = thisAdjust.cpu().numpy()[:,-1,:,:]
        adjust[:,step,:,:] = thisAdjust

        if self.updateTime == self.adjustTime:                            
            theta[:,step,:,:] = theta[:,step-1,:,:] + gamma[:,step-1,:,:] * self.updateTime \
                                              + (1/nAgents) * np.expand_dims(adjust[:,step-1,0,:], 1) \
                                              + np.expand_dims(clkNoise[:,step-1,0,:], 1) \
                                              - np.expand_dims(processNoise[:,step-1,0,:], 1)
            gamma[:,step,:,:] = gamma[:,step-1,:,:] + (1/nAgents) * np.expand_dims(adjust[:,step-1,1,:], 1)\
                                              + np.expand_dims(clkNoise[:,step-1,1,:], 1)                                              
        else:
            if int(step % (self.adjustTime/self.updateTime)) == 0:                                
                theta[:,step,:,:] = theta[:,step-1,:,:] + gamma[:,step-1,:,:] * self.updateTime \
                                                  + (1/nAgents) * np.expand_dims(adjust[:,step-1,0,:], 1) \
                                                  + np.expand_dims(clkNoise[:,step-1,0,:], 1) \
                                                  - np.expand_dims(processNoise[:,step-1,0,:], 1)
                gamma[:,step,:,:] = gamma[:,step-1,:,:] + (1/nAgents) * np.expand_dims(adjust[:,step-1,1,:], 1)\
                                                  + np.expand_dims(clkNoise[:,step-1,1,:], 1)                                        
            else: 
                theta[:,step,:,:] = theta[:,step-1,:,:] + gamma[:,step-1,:,:] * self.updateTime \
                                                  + np.expand_dims(clkNoise[:,step-1,0,:], 1) \
                                                  - np.expand_dims(processNoise[:,step-1,0,:], 1)
                gamma[:,step,:,:] = gamma[:,step-1,:,:] + np.expand_dims(clkNoise[:,step-1,1,:], 1)                    

        if useTorch:            
            theta = torch.tensor(theta).to(device)
            gamma = torch.tensor(gamma).to(device)
            adjust = torch.tensor(adjust).to(device)
            
        return theta, gamma, adjust, state
    
    def computeDifferences(self, u):        
        # input u shape: ###
        #   nSamples x tSamples x 2 x nAgents or 
        #   nSamples x tSamples x 1 x nAgents or 
        #   nSamples x 2 x nAgents or 
        #   nSamples x 1 x nAgents 
        
        # output shape: ###
        # uDiff [elementwise difference u_i - u_j]:
        #   nSamples x tSamples x 2 x nAgents x nAgents or 
        #   nSamples x tSamples x 1 x nAgents x nAgents or   
        #   nSamples x 2 x nAgents x nAgents or 
        #   nSamples x 1 x nAgents x nAgents                
        # uDistSq [squared ||u_i - u_j||^2]:
        #   nSamples x tSamples x nAgents x nAgents or 
        #   nSamples x nAgents x nAgents        
        
        assert len(u.shape) == 3 or len(u.shape) == 4
        if len(u.shape) == 3:
            u = np.expand_dims(u, 1)
            hasTimeDim = False
        else:
            hasTimeDim = True
        
        nSamples = u.shape[0]
        tSamples = u.shape[1]
        uFeatureDim = u.shape[2]
        nAgents = u.shape[3]
        
        if uFeatureDim == 2:
            uCol_x = u[:,:,0,:].reshape((nSamples, tSamples, nAgents, 1))
            uRow_x = u[:,:,0,:].reshape((nSamples, tSamples, 1, nAgents))
            uDiff_x = uCol_x - uRow_x # nSamples x tSamples x nAgents x nAgents

            uCol_y = u[:,:,1,:].reshape((nSamples, tSamples, nAgents, 1))
            uRow_y = u[:,:,1,:].reshape((nSamples, tSamples, 1, nAgents))
            uDiff_y = uCol_y - uRow_y # nSamples x tSamples x nAgents x nAgents

            uDistSq = uDiff_x ** 2 + uDiff_y ** 2

            uDiff_x = np.expand_dims(uDiff_x, 2)
            uDiff_y = np.expand_dims(uDiff_y, 2)

            uDiff = np.concatenate((uDiff_x, uDiff_y), 2) # nSamples x tSamples x 2 x nAgents x nAgents                                    
        elif uFeatureDim == 1: 
            uCol = u[:,:,0,:].reshape((nSamples, tSamples, nAgents, 1))
            uRow = u[:,:,0,:].reshape((nSamples, tSamples, 1, nAgents))
            uDiff = uCol - uRow # nSamples x tSamples x nAgents x nAgents

            uDistSq = uDiff ** 2 

            uDiff = np.expand_dims(uDiff, 2) # nSamples x tSamples x 1 x nAgents x nAgents                                    
        else:
            raise Exception("unexpected feature dimension is found!")
                
        if not hasTimeDim:
            uDistSq = uDistSq.squeeze(1) # nSamples x nAgents x nAgents
            uDiff = uDiff.squeeze(1) # nSamples x 2 x nAgents x nAgents            

        return uDiff, uDistSq
        
    def computeOptimalTrajectory(self, initPos, initVel, 
                                 initOffsetTheta, initSkewGamma, 
                                 measureNoise, processNoise, clkNoise, 
                                 duration, updateTime, adjustTime,
                                 repelDist, accelMax = 100.):
        # input: ######
        # initPos: nSamples x 2 x nAgents 
        # initVel: nSamples x 2 x nAgents
        # initOffsetTheta: nSamples x 1 x nAgents
        # initSkewGamma: nSamples x 1 x nAgents
        # measureNoise: nSamples x tSamples x 2 x nAgents 
        # processNoise: nSamples x tSamples x 2 x nAgents
        # clkNoise: nSamples x tSamples x 2 x nAgents
        
        # output: ######        
        # pos: nSamples x tSamples x 2 x nAgents 
        # vel: nSamples x tSamples x 2 x nAgents 
        # accel: nSamples x tSamples x 2 x nAgents  
        # theta: nSamples x tSamples x 1 x nAgents 
        # gamma: nSamples x tSamples x 1 x nAgents 
        # clockCorrection: nSamples x tSamples x 2 x nAgents 
        
        assert len(initPos.shape) == len(initVel.shape) == 3
        nSamples = initPos.shape[0]
        assert initPos.shape[1] == initVel.shape[1] == 2
        nAgents = initPos.shape[2]
        assert initVel.shape[0] == nSamples
        assert initVel.shape[2] == nAgents

        assert len(initOffsetTheta.shape) == len(initSkewGamma.shape) == 3
        assert nSamples == initOffsetTheta.shape[0]
        assert initOffsetTheta.shape[1] == initSkewGamma.shape[1] == 1
        assert nAgents == initOffsetTheta.shape[2]
        assert initSkewGamma.shape[0] == nSamples
        assert initSkewGamma.shape[2] == nAgents
       
        time = np.arange(0, duration, updateTime)
        tSamples = len(time)
        
        pos = np.zeros((nSamples, tSamples, 2, nAgents))
        vel = np.zeros((nSamples, tSamples, 2, nAgents))
        accel = np.zeros((nSamples, tSamples, 2, nAgents))        
        theta = np.zeros((nSamples, tSamples, 1, nAgents)) # offset 
        gamma = np.zeros((nSamples, tSamples, 1, nAgents)) # skew 
        deltaTheta = np.zeros((nSamples, tSamples, 1, nAgents)) # offset adjustment        
        deltaGamma = np.zeros((nSamples, tSamples, 1, nAgents)) # skew adjustment               
        
        pos[:,0,:,:] = initPos
        vel[:,0,:,:] = initVel        
        theta[:,0,:,:] = initOffsetTheta
        gamma[:,0,:,:] = initSkewGamma        
        
        if self.doPrint:
            percentageCount = int(100/tSamples)
            print("%3d%%" % percentageCount, end = '', flush = True)        

        for t in range(1, tSamples):
            ### Compute the optimal UAVs trajectories ###            
            ijDiffPos, ijDistSq = self.computeDifferences(pos[:,t-1,:,:])
            #   ijDiffPos: nSamples x 2 x nAgents x nAgents
            #   ijDistSq:  nSamples x nAgents x nAgents
            
            ijDiffVel, _ = self.computeDifferences(vel[:,t-1,:,:])
            #   ijDiffVel: nSamples x 2 x nAgents x nAgents

            repelMask = (ijDistSq < (repelDist**2)).astype(ijDiffPos.dtype)
            ijDiffPos = ijDiffPos * np.expand_dims(repelMask, 1)
            ijDistSqInv = invertTensorEW(ijDistSq)
            ijDistSqInv = np.expand_dims(ijDistSqInv, 1)

            accel[:,t-1,:,:] = \
                    -np.sum(ijDiffVel, axis = 3) \
                    +2* np.sum(ijDiffPos * (ijDistSqInv ** 2 + ijDistSqInv),
                               axis = 3)
                    
            thisAccel = accel[:,t-1,:,:].copy()
            thisAccel[accel[:,t-1,:,:] > accelMax] = accelMax
            thisAccel[accel[:,t-1,:,:] < -accelMax] = -accelMax
            accel[:,t-1,:,:] = thisAccel
            
            ### Compute the optimal clock offset and skew correction values ###
            ijDiffOffset, _ = self.computeDifferences(theta[:,t-1,:,:] \
                                                      + np.expand_dims(measureNoise[:,t-1,0,:], 1))
            #   ijDiffOffset: nSamples x 1 x nAgents x nAgents

            ijDiffSkew, _ = self.computeDifferences(gamma[:,t-1,:,:] \
                                                    + np.expand_dims(measureNoise[:,t-1,1,:], 1))
            #   ijDiffVel: nSamples x 1 x nAgents x nAgents

            deltaTheta[:,t-1,:,:] = -0.5*np.sum(ijDiffOffset, axis = 3)                                
            deltaGamma[:,t-1,:,:] = -0.5*np.sum(ijDiffSkew, axis = 3)                  

            ### Update the values ###
            vel[:,t,:,:] = vel[:,t-1,:,:] + accel[:,t-1,:,:] * updateTime
            pos[:,t,:,:] = pos[:,t-1,:,:] + vel[:,t-1,:,:] * updateTime \
                                          + accel[:,t-1,:,:] * (updateTime ** 2)/2             
            
            if updateTime == adjustTime:                            
                theta[:,t,:,:] = theta[:,t-1,:,:] + gamma[:,t-1,:,:] * updateTime \
                                                  + (1/self.nAgents) * deltaTheta[:,t-1,:,:] \
                                                  + np.expand_dims(clkNoise[:,t-1,0,:], 1) \
                                                  - np.expand_dims(processNoise[:,t-1,0,:], 1)
                gamma[:,t,:,:] = gamma[:,t-1,:,:] + (1/self.nAgents) * deltaGamma[:,t-1,:,:] \
                                                  + np.expand_dims(clkNoise[:,t-1,1,:], 1)
            else:                                
                if int(t % (adjustTime/updateTime)) == 0:
                    theta[:,t,:,:] = theta[:,t-1,:,:] + gamma[:,t-1,:,:] * updateTime \
                                                      + (1/self.nAgents) * deltaTheta[:,t-1,:,:] \
                                                      + np.expand_dims(clkNoise[:,t-1,0,:], 1) \
                                                      - np.expand_dims(processNoise[:,t-1,0,:], 1)
                    gamma[:,t,:,:] = gamma[:,t-1,:,:] + (1/self.nAgents) * deltaGamma[:,t-1,:,:] \
                                                      + np.expand_dims(clkNoise[:,t-1,1,:], 1)                                    
                else: 
                    theta[:,t,:,:] = theta[:,t-1,:,:] + gamma[:,t-1,:,:] * updateTime \
                                                      + np.expand_dims(clkNoise[:,t-1,0,:], 1) \
                                                      - np.expand_dims(processNoise[:,t-1,0,:], 1)
                    gamma[:,t,:,:] = gamma[:,t-1,:,:] + np.expand_dims(clkNoise[:,t-1,1,:], 1)
                                                      
            if self.doPrint:
                percentageCount = int(100*(t+1)/tSamples)
                print('\b \b' * 4 + "%3d%%" % percentageCount,
                      end = '', flush = True)
        
        clockCorrection = np.concatenate((deltaTheta,deltaGamma),axis=2)
        
        if self.doPrint:
            print('\b \b' * 4, end = '', flush = True)

        return pos, vel, accel, theta, gamma, clockCorrection
    
    def computeSingleStepOptimalTrajectory(self, offset, skew, 
                                                   measureNoise, processNoise, clkNoise, 
                                                   step, duration, updateTime, adjustTime):
        # input: ######
        # pos: nSamples x 2 x nAgents 
        # vel: nSamples x 2 x nAgents
        # offset: nSamples x 1 x nAgents
        # skew: nSamples x 1 x nAgents
        # measureNoise: nSamples x 2 x nAgents 
        # processNoise: nSamples x 2 x nAgents
        # clkNoise: nSamples x 2 x nAgents
        
        # output: ######        
        # pos: nSamples x 2 x nAgents 
        # vel: nSamples x 2 x nAgents 
        # accel: nSamples x 2 x nAgents  
        # theta: nSamples x 1 x nAgents 
        # gamma: nSamples x 1 x nAgents 
        # clockCorrection: nSamples x 2 x nAgents 

        assert len(offset.shape) == len(skew.shape) == 3
        nSamples = offset.shape[0]
        assert offset.shape[1] == skew.shape[1] == 1
        nAgents = offset.shape[2]
        assert skew.shape[0] == nSamples
        assert skew.shape[2] == nAgents
                          
        ### Compute the optimal clock offset and skew correction values ###
        ijDiffOffset, _ = self.computeDifferences(offset + np.expand_dims(measureNoise[:, 0, :], 1))
        #   ijDiffOffset: nSamples x 1 x nAgents x nAgents

        ijDiffSkew, _ = self.computeDifferences(skew + np.expand_dims(measureNoise[:, 1, :], 1))
        #   ijDiffVel: nSamples x 1 x nAgents x nAgents

        deltaTheta = -0.5*np.sum(ijDiffOffset, axis = 3)                                
        deltaGamma = -0.5*np.sum(ijDiffSkew, axis = 3)                  

        ### Update the values ###            
        if updateTime == adjustTime:                            
            nextOffset = offset + skew * updateTime + (1/self.nAgents) * deltaTheta \
                                              + np.expand_dims(clkNoise[:, 0, :], 1) \
                                              - np.expand_dims(processNoise[:, 0, :], 1)
            nextSkew = skew + (1/self.nAgents) * deltaGamma + np.expand_dims(clkNoise[:, 1, :], 1)
        else:                                
            if int(step % (adjustTime/updateTime)) == 0:
                nextOffset = offset + skew * updateTime + (1/self.nAgents) * deltaTheta \
                                                  + np.expand_dims(clkNoise[:, 0, :], 1) \
                                                  - np.expand_dims(processNoise[:, 0, :], 1)
                nextSkew = skew + (1/self.nAgents) * deltaGamma + np.expand_dims(clkNoise[:, 1, :], 1)                                    
            else: 
                nextOffset = offset + skew * updateTime + np.expand_dims(clkNoise[:, 0, :], 1) \
                                                  - np.expand_dims(processNoise[:, 0, :], 1)
                nextSkew = skew + np.expand_dims(clkNoise[:, 1, :], 1)
        
        clockCorrection = np.concatenate((deltaTheta,deltaGamma),axis=1)
        
        return nextOffset, nextSkew, clockCorrection

    def computeNoises(self, nAgents, nSamples, duration, updateTime, 
                      sigmaMeasureOffsetVal=0., sigmaProcessOffsetVal=0.,
                      sigmaOffsetVal=0., sigmaSkewVal=0.):
        # clkNoise: nSamples x time x 2 x nAgents
        # measurementNoise: nSamples x time x 2 x nAgents
        # processingNoise: nSamples x time x 2 x nAgents

        time = np.int64(duration/updateTime)

        sigmaOffsetSq = sigmaOffsetVal**2 # unit: 10000 us^2
        sigmaSkewSq = sigmaSkewVal**2 # unit: 100 ppm^2          
                
        sigmaMeasureOffsetSq = sigmaMeasureOffsetVal**2 # unit: 10000 us^2
        sigmaMeasureSkewSq = (sigmaMeasureOffsetVal**2)*100 # unit: 10000 us^2 -> 100 ppm^2   
        sigmaProcessOffsetSq = sigmaProcessOffsetVal**2 # unit: 10000 us^2        
        
        # covariance matrix M for the clock offset and skew noises
        covM = np.array([[sigmaOffsetSq, 0], \
                         [0, sigmaSkewSq]], self.dataType)                     
        
        # covariance matrix R for the packet exchange delay
        covR = np.array([[sigmaMeasureOffsetSq, sigmaMeasureOffsetSq*10], \
                         [sigmaMeasureOffsetSq*10, 2*sigmaMeasureSkewSq]], self.dataType)
            
        # covariance matrix Q for the processing delay, occuring on the offset correction
        covQ = np.array([[sigmaProcessOffsetSq, 0], \
                         [0, 0]], self.dataType)         
            
        # zero mean value for both packet exchange and processing delays
        clockMeanVal = np.array([0, 0], self.dataType)         
        measureMeanVal = np.array([0, 0], self.dataType)
        processingMeanVal = np.array([0, 0], self.dataType)         
                    
        rng = default_rng()
        # for the clock noise configurations, we use the clock A in the following 
        # paper:
        # Giada Giorgi, Claudio Narduzzi, Performance Analysis of Kalman-Filter-Based 
        # Clock Synchronization in IEEE 1588 Networks,
        # the standard deviation of clock offset is 1us, and the standard 
        # deviation of the clock skew is 0.01ppm
        clkNoise = rng.multivariate_normal(clockMeanVal, covM, nSamples*time*nAgents)                
        clkNoise = clkNoise.reshape((nSamples, time, nAgents, 2))     
        clkNoise = clkNoise.transpose(0,1,3,2)
        
        # for the above configurations of packet exchange and processing delay,
        # see the following paper:
        # Yan Zong, Xuewu Dai, Zhiwei Gao, "Proportional-Integral Synchronisation 
        #  for Non-identical Wireless Packet-Coupled Oscillators with Delays"
        # In Table 1, the mean value and standard deviation of processing delay 
        # are 311us and 4us respectively, for the standard deviation of packet 
        # exchange delay is 0.3us, we assume the effects of the mean value of 
        # packe exchange delay can be removed.                
        measurementNoise = rng.multivariate_normal(measureMeanVal, covR, nSamples*time*nAgents)        
        measurementNoise = measurementNoise.reshape((nSamples, time, nAgents, 2))     
        measurementNoise = measurementNoise.transpose(0,1,3,2)        

        processingNoise = rng.multivariate_normal(processingMeanVal, covQ, nSamples*time*nAgents)                
        processingNoise = processingNoise.reshape((nSamples, time, nAgents, 2))     
        processingNoise = processingNoise.transpose(0,1,3,2)    
                
        return clkNoise, measurementNoise, processingNoise
    
    def computeInitialConditions(self, nAgents, nSamples, commRadius, minDist=0.1,
                                initOffsetVal=1., initSkewVal=2.5,
                                meanOffset=0.0, stdOffset=0.5, meanSkew=0.0, stdSkew=2.5,
                                **kwargs):             
        # initPos: nSamples x 2 x nNodes
        # initVel: nSamples x 2 x nNodes
        # initOffset: nSamples x 1 x nNodes
        # initSkew: nSamples x 1 x nNodes                
        
        assert minDist*(1.+zeroTolerance) <= commRadius*(1.-zeroTolerance)
        minDist = minDist * (1. + zeroTolerance)
        commRadius = commRadius * (1. - zeroTolerance)
        
        #### generate the position information of UAVs ####       
        rFixed = (commRadius + minDist)/2.
        rPerturb = (commRadius - minDist)/4.
        fixedRadius = np.arange(0, rFixed*nAgents, step=rFixed) + rFixed        
        
        aFixed = (commRadius/fixedRadius + minDist/fixedRadius)/2.
        for a in range(len(aFixed)):
            nAgentsPerCircle = 2 * np.pi // aFixed[a]
            aFixed[a] = 2 * np.pi / nAgentsPerCircle

        initRadius = np.empty((0))
        initAngles = np.empty((0))
        agentsSoFar = 0 
        n = 0 
        while agentsSoFar < nAgents:
            thisRadius = fixedRadius[n]
            thisAngles = np.arange(0, 2*np.pi, step=aFixed[n])
            agentsSoFar += len(thisAngles)
            initRadius = np.concatenate((initRadius,
                                         np.repeat(thisRadius, len(thisAngles))))
            initAngles = np.concatenate((initAngles, thisAngles))
            n += 1
            assert len(initRadius)==agentsSoFar
            
        initRadius = initRadius[0:nAgents]
        initAngles = initAngles[0:nAgents]            

        initRadius = np.repeat(np.expand_dims(initRadius, 0), nSamples, axis=0)
        initAngles = np.repeat(np.expand_dims(initAngles, 0), nSamples, axis=0)
        
        for n in range(nAgents):
            thisRadius = initRadius[0,n]
            aPerturb = (commRadius/thisRadius - minDist/thisRadius)/4.
            initAngles[:,n] += np.random.uniform(low = -aPerturb,
                                                 high = aPerturb,
                                                 size = (nSamples))
            
        initRadius += np.random.uniform(low = -rPerturb,
                                        high = rPerturb,
                                        size = (nSamples, nAgents))
        
        initPos = np.zeros((nSamples, 2, nAgents)) # nSamples x 2 x nNodes
        initPos[:, 0, :] = initRadius * np.cos(initAngles)
        initPos[:, 1, :] = initRadius * np.sin(initAngles)        
        
        _, distSq = self.computeDifferences(np.expand_dims(initPos, 1))
        distSq = distSq.squeeze(1)
        minDistSq = np.min(distSq + \
                           2 * commRadius\
                             *np.eye(distSq.shape[1]).reshape(1,
                                                              distSq.shape[1],
                                                              distSq.shape[2])
                           )        
        
        assert minDistSq>=(minDist ** 2)
        
        graphMatrix = self.computeCommunicationGraph(np.expand_dims(initPos,1),
                                                     self.commRadius,
                                                     False,
                                                     doPrint = False)
        graphMatrix = graphMatrix.squeeze(1) # nSamples x nAgents x nAgents  
        
        graphMatrix = (np.abs(graphMatrix) > zeroTolerance).astype(initPos.dtype)
        
        for n in range(nSamples):
            assert graph.isConnected(graphMatrix[n,:,:])

        #### then generate the velocity information of UAVs ####                       
        if 'xMaxInitVel' in kwargs.keys():
            xMaxInitVel = kwargs['xMaxInitVel']
        else:
            xMaxInitVel = 3.

        if 'yMaxInitVel' in kwargs.keys():
            yMaxInitVel = kwargs['yMaxInitVel']
        else:
            yMaxInitVel = 3.
        
        xInitVel = np.random.uniform(low = -xMaxInitVel, high = xMaxInitVel,
                                     size = (nSamples, 1, nAgents))
        yInitVel = np.random.uniform(low = -yMaxInitVel, high = yMaxInitVel,
                                     size = (nSamples, 1, nAgents))
        
        xVelBias = np.random.uniform(low = -xMaxInitVel, high = xMaxInitVel,
                                     size = (nSamples))
        yVelBias = np.random.uniform(low = -yMaxInitVel, high = yMaxInitVel,
                                     size = (nSamples))
        
        velBias = np.concatenate((xVelBias, yVelBias)).reshape((nSamples,2,1))
        initVel = np.concatenate((xInitVel, yInitVel), axis = 1) + velBias # nSamples x 2 x nAgents            
        
        #### next generate the time information of UAVs ####
        # ToDo: the distributions of clock offset and skew noises should be determined carefully                                  
        offsetFixed = np.repeat(initOffsetVal, nSamples*nAgents, axis = 0)             
        skewFixed = np.repeat(initSkewVal, nSamples*nAgents, axis = 0)
               
        offsetPerturb = np.random.normal(loc = meanOffset,
                                          scale = stdOffset,
                                          size = nSamples*nAgents)    

        skewPerturb = np.random.normal(loc = meanSkew,
                                        scale = stdSkew,
                                        size = nSamples*nAgents)
        
        initOffset = offsetFixed + offsetPerturb 
        initSkew = skewFixed + skewPerturb
        
        initOffset = initOffset.reshape(nSamples, nAgents) # nSamples x nNodes            
        initSkew = initSkew.reshape(nSamples, nAgents) # nSamples x nNodes         

        initOffset = np.expand_dims(initOffset, 1) # nSamples x 1 x nNodes
        initSkew = np.expand_dims(initSkew, 1) # nSamples x 1 x nNodes       
                          
        return initPos, initVel, initOffset, initSkew        