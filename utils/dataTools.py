import os
import pickle
import hdf5storage # This is required to import old Matlab(R) files.
import urllib.request # To download from the internet
import zipfile # To handle zip files
import gzip # To handle gz files
import shutil # Command line utilities
import matplotlib
import csv
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

import numpy as np
import torch

import utils.graphTools as graph

import networkx as netwk # ver 2.8.4

zeroTolerance = 1e-9 # Values below this number are considered zero.

def changeDataType(x, dataType):
    """
    changeDataType(x, dataType): change the dataType of variable x into dataType
    """
    
    # So this is the thing: To change data type it depends on both, what dtype
    # the variable already is, and what dtype we want to make it.
    # Torch changes type by .type(), but numpy by .astype()
    # If we have already a torch defined, and we apply a torch.tensor() to it,
    # then there will be warnings because of gradient accounting.
    
    # All of these facts make changing types considerably cumbersome. So we
    # create a function that just changes type and handles all this issues
    # inside.
    
    # If we can't recognize the type, we just make everything numpy.
    
    # Check if the variable has an argument called 'dtype' so that we can now
    # what type of data type the variable is
    if 'dtype' in dir(x):
        varType = x.dtype
    
    # So, let's start assuming we want to convert to numpy
    if 'numpy' in repr(dataType):
        # Then, the variable con be torch, in which case we move it to cpu, to
        # numpy, and convert it to the right type.
        if 'torch' in repr(varType):
            x = x.cpu().numpy().astype(dataType)
        # Or it could be numpy, in which case we just use .astype
        elif 'numpy' in repr(type(x)):
            x = x.astype(dataType)
    # Now, we want to convert to torch
    elif 'torch' in repr(dataType):
        # If the variable is torch in itself
        if 'torch' in repr(varType):
            x = x.type(dataType)
        # But, if it's numpy
        elif 'numpy' in repr(type(x)):
            x = torch.tensor(x, dtype = dataType)
            
    # This only converts between numpy and torch. Any other thing is ignored
    return x

def invertTensorEW(x):
    
    # Elementwise inversion of a tensor where the 0 elements are kept as zero.
    # Warning: Creates a copy of the tensor
    xInv = x.copy() # Copy the matrix to invert
    # Replace zeros for ones.
    xInv[x < zeroTolerance] = 1. # Replace zeros for ones
    xInv = 1./xInv # Now we can invert safely
    xInv[x < zeroTolerance] = 0. # Put back the zeros
    
    return xInv

class initClockNetwk():
    def __init__(self, nNodes,
                 nTrain, nValid, nTest,
                 duration, samplingTimeScale,
                 initOffsetValue, initSkewValue,                 
                 normaliseGraph = True,
                 dataType = np.float64, device = 'cpu'):
        
        self.samples = {} # create a dict
        self.samples['train'] = {} # create a dict in the above samples dict 
        self.samples['train']['signals'] = None
        self.samples['train']['targets'] = None
        self.samples['valid'] = {} # create a dict in the above samples dict
        self.samples['valid']['signals'] = None
        self.samples['valid']['targets'] = None
        self.samples['test'] = {} # create a dict in the above samples dict
        self.samples['test']['signals'] = None
        self.samples['test']['targets'] = None        
        """
        我们初始化clock offset skew可以是和网络的拓扑结构分开的，只要注意好维度问题即可。
        我们可以先考虑进去这个noise的问题，但是在刚开始仿真的时候，并不会考虑这个噪声
        的问题，而是等到后期的时候，再考虑这个噪声的问题
        
        需要考虑，如果以后test就是重新创建的时候，也许就是需要考虑不同的节点个数，图的状态了。
        所以到底是使用self版本，还是不需要使用self版本，就是需要重点关注一下，
        同时的呢，就是需要想一下，估计还是需要封装行数，因为后期就是需要test创建的时候
        会使用到
        
        """
        # save the relevant input information
        # number of nodes
        self.nNodes = nNodes
        # number of samples
        self.nTrain = nTrain
        self.nValid = nValid
        self.nTest = nTest
        nSamples = nTrain + nValid + nTest
        # simulation duration
        self.duration = float(duration)
        self.samplingTimeScale = samplingTimeScale        
        # agents
        self.initOffsetValue = initOffsetValue
        self.initSkewValue = initSkewValue
        # data
        self.normaliseGraph = normaliseGraph
        self.dataType = dataType
        self.device = device
        # places to store the data
        self.initOffset = None
        self.initSkew = None
        self.commGraph = None
        
        print("\tComputing initial conditions...", end = ' ', flush = True)
        
        # repeat and reshape to obtain the offsets and skews                                          
        fixedOffsetTmp = np.repeat(self.initOffsetValue, nSamples*self.nNodes, axis = 0)     
        fixedOffset = fixedOffsetTmp.reshape(nSamples, self.nNodes)             
        
        fixedSkewTmp = np.repeat(self.initSkewValue, nSamples*self.nNodes, axis = 0)     
        fixedSkew = fixedSkewTmp.reshape(nSamples, self.nNodes)                             
        
        # generate the noises
        offsetPerturb = 2e+5 # us
        skewPerturb = 25 # ppm        
        perturbOffset = np.random.uniform(low = -offsetPerturb,
                                          high = offsetPerturb,
                                          size = (nSamples, self.nNodes))
        
        perturbSkew = np.random.uniform(low = -skewPerturb,
                                        high = skewPerturb,
                                        size = (nSamples, self.nNodes))        
        
        # compute the initial clock offsets and skews    
        initOffsetAll = fixedOffset + perturbOffset     
        initSkewAll = fixedSkew + perturbSkew
                
        # split all the initial clock offsets and skews in the corresponding 
        # datasets (train, valid and test)
        self.initOffset = {}
        self.initSkew = {}
        
        print("OK", flush = True)
        print("\tComputing the network topologies...",
              end=' ', flush=True)
        
        # compute communication graph
                
        commGraphAll = np.zeros((nSamples, nNodes, nNodes))
        
        for i in range(nSamples):
            networkTopology = netwk.random_tree(nNodes, seed=np.random, create_using=netwk.DiGraph)            
            # networkTopology = netwk.random_tree(nNodes, seed=np.random, create_using=netwk.Graph)
            networkTopologyMatrix = netwk.adjacency_matrix(networkTopology).todense()            
            # # calculate the eigenvalues
            # W = np.linalg.eigvals(networkTopologyMatrix)            
            # maxEigenvalue = np.max(np.real(W))
            # # 需要区分在使用不同的graph，digraph的时候，可能会出现特征值=0的情况
            # # normalise
            # networkTopologyMatrix = networkTopologyMatrix / maxEigenvalue                                    
            commGraphAll[i, :, :] = networkTopologyMatrix
        # end for
                
        self.commNetwk = {}
        
        print("OK", flush = True)   
        print("\tComputing the centralised optimal synchronisation trajectories...",
              end=' ', flush=True)
        
        

        # The optimal trajectory is given by
        # u_{i} = - \sum_{j=1}^{N} (v_{i} - v_{j})
        #         + 2 \sum_{j=1}^{N} (r_{i} - r_{j}) *
        #                                 (1/\|r_{i}\|^{4} + 1/\|r_{j}\|^{2}) *
        #                                 1{\|r_{ij}\| < R}
        # for each agent i=1,...,N, where v_{i} is the velocity and r_{i} the
        # position.
        
        # 我前面应该就是需要修改代码了，因为前面的clock offset，skew都是两个维度的
        # 最好是修改成为3个维度，多加一个feature的维度，虽然feature就是1 ，没有任何de@
        # 的区别。另外的呢，就是这个
        # 对于计算来说，应该是4个维度，就是需要有一个时间的维度
        # time
        time = np.arange(0, duration, samplingTimeScale)
        tSamples = len(time) # number of time samples
        
        # create arrays to store the offset and skew
        offset = np.zeros((nSamples, tSamples, 1, nNodes))
        skew = np.zeros((nSamples, tSamples, 1, nNodes))
        integralOffset = np.zeros((nSamples, tSamples, 1, nNodes))
        integralSkew = np.zeros((nSamples, tSamples, 1, nNodes))    
        correctionOffset = np.zeros((nSamples, tSamples, 1, nNodes))
        correctionSkew = np.zeros((nSamples, tSamples, 1, nNodes))
        
        # initial settings
        offset[:,0,:,:] = initOffsetAll
        skew[:,0,:,:] = initSkewAll
        
        # sample percentage count
        percentageCount = int(100/tSamples)
        # print new value
        print("%3d%%" % percentageCount, end = '', flush = True)
        
        gainOffset1=1
        gainOffset2=1
        gainOffset3=1
        gainOffset4=1
                
        gainSkew1=1
        gainSkew2=1
        gainSkew3=1
        gainSkew4=1
        
        
        
        # for each time instant
        for t in range(1, tSamples):
            
            # compute the clock offset and skew correction input
            #   compute the the clock offset differences between all elements
            ijDiffOffset, _ = self.computeDifferences(offset[:,t-1,:,:]) # 也是可以正常的使用的了，但是注意我们只有一个feature，就是需要做精简
            #       ijDiffOffset: nSamples x 1 x nNodes x nNodes
            #   and also the difference in clock skews
            ijDiffSkew, _ = self.computeDifferences(skew[:,t-1,:,:])
            #       ijDiffSkew: nSamples x 1 x nNodes x nNodes

            # update the clock offset and skew correction input
            #   update clock offset correction value
            integralOffset[:,t,:,:] = gainOffset1 * integralOffset[:,t-1,:,:] + gainOffset2 * np.sum(ijDiffOffset, axis=3)
            correctionOffset[:,t,:,:] = gainOffset3 * integralOffset[:,t-1,:,:] + gainOffset4 * np.sum(ijDiffOffset, axis=3)
            #   Update clock skew correction value
            integralSkew[:,t,:,:] = gainSkew1 * integralSkew[:,t-1,:,:] + gainSkew2 * np.sum(ijDiffSkew, axis=3)
            correctionSkew[:,t,:,:] = gainSkew3 * integralSkew[:,t-1,:,:] + gainSkew4 * np.sum(ijDiffSkew, axis=3)            
            
            # update the values Todo: still need the noises by yan zong
            #   update clock offset 
            offset[:,t,:,:] = offset[:,t-1,:,:] + skew[:,t-1,:,:] * samplingTimeScale + correctionOffset[:,t-1,:,:]
            #   update clock skew 
            skew[:,t,:,:] = skew[:,t-1,:,:] + correctionSkew[:,t-1,:,:]              
            
            # sample percentage count
            percentageCount = int(100*(t+1)/tSamples)
            # remove previous pecentage and print new value
            print('\b \b' * 4 + "%3d%%" % percentageCount, end = '', flush = True)
                
        # erase the percentage
        print('\b \b' * 4, end = '', flush = True)            


            
            #   The last element we need to compute the acceleration is the
            #   gradient. Note that the gradient only counts when the distance 
            #   is smaller than the repel distance
            #       This is the mask to consider each of the differences
            repelMask = (ijDistSq < (repelDist**2)).astype(ijDiffPos.dtype)
            #       Apply the mask to the relevant differences
            ijDiffPos = ijDiffPos * np.expand_dims(repelMask, 1)
            #       Compute the constant (1/||r_ij||^4 + 1/||r_ij||^2)
            ijDistSqInv = invertTensorEW(ijDistSq)
            #       Add the extra dimension
            ijDistSqInv = np.expand_dims(ijDistSqInv, 1)
            #   Compute the acceleration
            accel[:,t-1,:,:] = \
                    -np.sum(ijDiffVel, axis = 3) \
                    +2* np.sum(ijDiffPos * (ijDistSqInv ** 2 + ijDistSqInv),
                               axis = 3)
                    
            # Finally, note that if the agents are too close together, the
            # acceleration will be very big to get them as far apart as
            # possible, and this is physically impossible.
            # So let's add a limitation to the maximum aceleration

            # Find the places where the acceleration is big
            thisAccel = accel[:,t-1,:,:].copy()
            # Values that exceed accelMax, force them to be accelMax
            thisAccel[accel[:,t-1,:,:] > accelMax] = accelMax
            # Values that are smaller than -accelMax, force them to be accelMax
            thisAccel[accel[:,t-1,:,:] < -accelMax] = -accelMax
            # And put it back
            accel[:,t-1,:,:] = thisAccel
            
            # Update the values
            #   Update velocity
            vel[:,t,:,:] = accel[:,t-1,:,:] * samplingTime + vel[:,t-1,:,:]
            #   Update the position
            pos[:,t,:,:] = accel[:,t-1,:,:] * (samplingTime ** 2)/2 + \
                                 vel[:,t-1,:,:] * samplingTime + pos[:,t-1,:,:]
            
            # Sample percentage count
            percentageCount = int(100*(t+1)/tSamples)
            # Erase previous pecentage and print new value
            print('\b \b' * 4 + "%3d%%" % percentageCount,
                  end = '', flush = True)
                
        # Erase the percentage
        print('\b \b' * 4, end = '', flush = True)
            
        return pos, vel, accel


        
        
        
        
        # Compute the optimal trajectory
        posAll, velAll, accelAll = self.computeOptimalTrajectory(
                                        initPosAll, initVelAll, self.duration,
                                        self.samplingTime, self.repelDist,
                                        accelMax = self.accelMax)
        
        self.offset = {}
        self.skew = {}
        
        print("OK", flush = True)
        print("\tComputing the communication network topologies...",
              end=' ', flush=True)
        
        # Compute communication graph
        commGraphAll = self.computeCommunicationGraph(posAll, self.commRadius,
                                                      self.normalizeGraph)
        
        self.commGraph = {}
        
        print("OK", flush = True)
        # Erase the label first, then print it
        print("\tComputing the agent states...", end = ' ', flush = True)
        
        # Compute the states
        stateAll = self.computeStates(posAll, velAll, commGraphAll)
        
        self.state = {}
        
        # Erase the label
        print("OK", flush = True)
        
        # Separate the states into training, validation and testing samples
        # and save them
        #   Training set
        self.samples['train']['signals'] = stateAll[0:self.nTrain].copy()
        self.samples['train']['targets'] = accelAll[0:self.nTrain].copy()
        self.initPos['train'] = initPosAll[0:self.nTrain]
        self.initVel['train'] = initVelAll[0:self.nTrain]
        self.pos['train'] = posAll[0:self.nTrain]
        self.vel['train'] = velAll[0:self.nTrain]
        self.accel['train'] = accelAll[0:self.nTrain]
        self.commGraph['train'] = commGraphAll[0:self.nTrain]
        self.state['train'] = stateAll[0:self.nTrain]
        #   Validation set
        startSample = self.nTrain
        endSample = self.nTrain + self.nValid
        self.samples['valid']['signals']=stateAll[startSample:endSample].copy()
        self.samples['valid']['targets']=accelAll[startSample:endSample].copy()
        self.initPos['valid'] = initPosAll[startSample:endSample]
        self.initVel['valid'] = initVelAll[startSample:endSample]
        self.pos['valid'] = posAll[startSample:endSample]
        self.vel['valid'] = velAll[startSample:endSample]
        self.accel['valid'] = accelAll[startSample:endSample]
        self.commGraph['valid'] = commGraphAll[startSample:endSample]
        self.state['valid'] = stateAll[startSample:endSample]
        #   Testing set
        startSample = self.nTrain + self.nValid
        endSample = self.nTrain + self.nValid + self.nTest
        self.samples['test']['signals']=stateAll[startSample:endSample].copy()
        self.samples['test']['targets']=accelAll[startSample:endSample].copy()
        self.initPos['test'] = initPosAll[startSample:endSample]
        self.initVel['test'] = initVelAll[startSample:endSample]
        self.pos['test'] = posAll[startSample:endSample]
        self.vel['test'] = velAll[startSample:endSample]
        self.accel['test'] = accelAll[startSample:endSample]
        self.commGraph['test'] = commGraphAll[startSample:endSample]
        self.state['test'] = stateAll[startSample:endSample]
        
        # Change data to specified type and device
        self.astype(self.dataType)
        self.to(self.device)
        
    def astype(self, dataType):
        
        # Change all other signals to the correct place
        datasetType = ['train', 'valid', 'test']
        for key in datasetType:
            self.initPos[key] = changeDataType(self.initPos[key], dataType)
            self.initVel[key] = changeDataType(self.initVel[key], dataType)
            self.pos[key] = changeDataType(self.pos[key], dataType)
            self.vel[key] = changeDataType(self.vel[key], dataType)
            self.accel[key] = changeDataType(self.accel[key], dataType)
            self.commGraph[key] = changeDataType(self.commGraph[key], dataType)
            self.state[key] = changeDataType(self.state[key], dataType)
        
        # And call the parent
        super().astype(dataType)
        
    def to(self, device):
        
        # Check the data is actually torch
        if 'torch' in repr(self.dataType):
            datasetType = ['train', 'valid', 'test']
            # Move the data
            for key in datasetType:
                self.initPos[key].to(device)
                self.initVel[key].to(device)
                self.pos[key].to(device)
                self.vel[key].to(device)
                self.accel[key].to(device)
                self.commGraph[key].to(device)
                self.state[key].to(device)
            
            super().to(device)
            
    def computeStates(self, pos, vel, graphMatrix, **kwargs):
        
        # We get the following inputs.
        # positions: nSamples x tSamples x 2 x nNodes
        # velocities: nSamples x tSamples x 2 x nNodes
        # graphMatrix: nSaples x tSamples x nNodes x nNodes
        
        # And we want to build the state, which is a vector of dimension 6 on 
        # each node, that is, the output shape is
        #   nSamples x tSamples x 6 x nNodes
                
        # Check correct dimensions
        assert len(pos.shape) == len(vel.shape) == len(graphMatrix.shape) == 4
        nSamples = pos.shape[0]
        tSamples = pos.shape[1]
        assert pos.shape[2] == 2
        nNodes = pos.shape[3]
        assert vel.shape[0] == graphMatrix.shape[0] == nSamples
        assert vel.shape[1] == graphMatrix.shape[1] == tSamples
        assert vel.shape[2] == 2
        assert vel.shape[3] == graphMatrix.shape[2] == graphMatrix.shape[3] \
                == nNodes
                
        # If we have a lot of batches and a particularly long sequence, this
        # is bound to fail, memory-wise, so let's do it time instant by time
        # instant if we have a large number of time instants, and split the
        # batches
        maxTimeSamples = 200 # Set the maximum number of t.Samples before
            # which to start doing this time by time.
        maxBatchSize = 100 # Maximum number of samples to process at a given
            # time
        
        # Compute the number of samples, and split the indices accordingly
        if nSamples < maxBatchSize:
            nBatches = 1
            batchSize = [nSamples]
        elif nSamples % maxBatchSize != 0:
            # If we know it's not divisible, then we do floor division and
            # add one more batch
            nBatches = nSamples // maxBatchSize + 1
            batchSize = [maxBatchSize] * nBatches
            # But the last batch is actually smaller, so just add the 
            # remaining ones
            batchSize[-1] = nSamples - sum(batchSize[0:-1])
        # If they fit evenly, then just do so.
        else:
            nBatches = int(nSamples/maxBatchSize)
            batchSize = [maxBatchSize] * nBatches
        # batchIndex is used to determine the first and last element of each
        # batch. We need to add the 0 because it's the first index.
        batchIndex = np.cumsum(batchSize).tolist()
        batchIndex = [0] + batchIndex
        
        # Create the output state variable
        state = np.zeros((nSamples, tSamples, 6, nNodes))
        
        for b in range(nBatches):
            
            # Pick the batch elements
            posBatch = pos[batchIndex[b]:batchIndex[b+1]]
            velBatch = vel[batchIndex[b]:batchIndex[b+1]]
            graphMatrixBatch = graphMatrix[batchIndex[b]:batchIndex[b+1]]
        
            if tSamples > maxTimeSamples:
                
                # For each time instant
                for t in range(tSamples):
                    
                    # Now, we need to compute the differences, in velocities and in 
                    # positions, for each agent, for each time instant
                    posDiff, posDistSq = \
                                     self.computeDifferences(posBatch[:,t,:,:])
                    #   posDiff: batchSize[b] x 2 x nNodes x nNodes
                    #   posDistSq: batchSize[b] x nNodes x nNodes
                    velDiff, _ = self.computeDifferences(velBatch[:,t,:,:])
                    #   velDiff: batchSize[b] x 2 x nNodes x nNodes
                    
                    # Next, we need to get ride of all those places where there are
                    # no neighborhoods. That is given by the nonzero elements of the 
                    # graph matrix.
                    graphMatrixTime = (np.abs(graphMatrixBatch[:,t,:,:])\
                                                               >zeroTolerance)\
                                                             .astype(pos.dtype)
                    #   graphMatrix: batchSize[b] x nNodes x nNodes
                    # We also need to invert the squares of the distances
                    posDistSqInv = invertTensorEW(posDistSq)
                    #   posDistSqInv: batchSize[b] x nNodes x nNodes
                    
                    # Now we add the extra dimensions so that all the 
                    # multiplications are adequate
                    graphMatrixTime = np.expand_dims(graphMatrixTime, 1)
                    #   graphMatrix: batchSize[b] x 1 x nNodes x nNodes
                    
                    # Then, we can get rid of non-neighbors
                    posDiff = posDiff * graphMatrixTime
                    posDistSqInv = np.expand_dims(posDistSqInv,1)\
                                                              * graphMatrixTime
                    velDiff = velDiff * graphMatrixTime
                    
                    # Finally, we can compute the states
                    stateVel = np.sum(velDiff, axis = 3)
                    #   stateVel: batchSize[b] x 2 x nNodes
                    statePosFourth = np.sum(posDiff * (posDistSqInv ** 2),
                                            axis = 3)
                    #   statePosFourth: batchSize[b] x 2 x nNodes
                    statePosSq = np.sum(posDiff * posDistSqInv, axis = 3)
                    #   statePosSq: batchSize[b] x 2 x nNodes
                    
                    # Concatentate the states and return the result
                    state[batchIndex[b]:batchIndex[b+1],t,:,:] = \
                                                np.concatenate((stateVel,
                                                                statePosFourth,
                                                                statePosSq),
                                                               axis = 1)
                    #   batchSize[b] x 6 x nNodes
                    
                    # Sample percentage count
                    percentageCount = int(100*(t+1+b*tSamples)\
                                                      /(nBatches*tSamples))
                    
                    if t == 0 and b == 0:
                        # It's the first one, so just print it
                        print("%3d%%" % percentageCount,
                              end = '', flush = True)
                    else:
                        # Erase the previous characters
                        print('\b \b' * 4 + "%3d%%" % percentageCount,
                              end = '', flush = True)
                
            else:
                
                # Now, we need to compute the differences, in velocities and in 
                # positions, for each agent, for each time instante
                posDiff, posDistSq = self.computeDifferences(posBatch)
                #   posDiff: batchSize[b] x tSamples x 2 x nNodes x nNodes
                #   posDistSq: batchSize[b] x tSamples x nNodes x nNodes
                velDiff, _ = self.computeDifferences(velBatch)
                #   velDiff: batchSize[b] x tSamples x 2 x nNodes x nNodes
                
                # Next, we need to get ride of all those places where there are
                # no neighborhoods. That is given by the nonzero elements of the 
                # graph matrix.
                graphMatrixBatch = (np.abs(graphMatrixBatch) > zeroTolerance)\
                                                             .astype(pos.dtype)
                #   graphMatrix: batchSize[b] x tSamples x nNodes x nNodes
                # We also need to invert the squares of the distances
                posDistSqInv = invertTensorEW(posDistSq)
                #   posDistSqInv: batchSize[b] x tSamples x nNodes x nNodes
                
                # Now we add the extra dimensions so that all the multiplications
                # are adequate
                graphMatrixBatch = np.expand_dims(graphMatrixBatch, 2)
                #   graphMatrix:batchSize[b] x tSamples x 1 x nNodes x nNodes
                
                # Then, we can get rid of non-neighbors
                posDiff = posDiff * graphMatrixBatch
                posDistSqInv = np.expand_dims(posDistSqInv, 2)\
                                                             * graphMatrixBatch
                velDiff = velDiff * graphMatrixBatch
                
                # Finally, we can compute the states
                stateVel = np.sum(velDiff, axis = 4)
                #   stateVel: batchSize[b] x tSamples x 2 x nNodes
                statePosFourth = np.sum(posDiff * (posDistSqInv ** 2), axis = 4)
                #   statePosFourth: batchSize[b] x tSamples x 2 x nNodes
                statePosSq = np.sum(posDiff * posDistSqInv, axis = 4)
                #   statePosSq: batchSize[b] x tSamples x 2 x nNodes
                
                # Concatentate the states and return the result
                state[batchIndex[b]:batchIndex[b+1]] = \
                                                np.concatenate((stateVel,
                                                                statePosFourth,
                                                                statePosSq),
                                                               axis = 2)
                #   state: batchSize[b] x tSamples x 6 x nNodes
                                                
                # Sample percentage count
                percentageCount = int(100*(b+1)/nBatches)
                
                if b == 0:
                    # It's the first one, so just print it
                    print("%3d%%" % percentageCount,
                          end = '', flush = True)
                else:
                    # Erase the previous characters
                    print('\b \b' * 4 + "%3d%%" % percentageCount,
                          end = '', flush = True)
                        
        # Erase the percentage
        print('\b \b' * 4, end = '', flush = True)
        
        return state

    def computeCommunicationGraph(self, pos, commRadius, normalizeGraph,
                                  **kwargs):
        
        # Take in the position and the communication radius, and return the
        # trajectory of communication graphs
        # Input will be of shape
        #   nSamples x tSamples x 2 x nNodes
        # Output will be of shape
        #   nSamples x tSamples x nNodes x nNodes
        
        assert commRadius > 0
        assert len(pos.shape) == 4
        nSamples = pos.shape[0]
        tSamples = pos.shape[1]
        assert pos.shape[2] == 2
        nNodes = pos.shape[3]
        
        # Graph type options
        #   Kernel type (only Gaussian implemented so far)
        if 'kernelType' in kwargs.keys():
            kernelType = kwargs['kernelType']
        else:
            kernelType = 'gaussian'
        #   Decide if the graph is weighted or not
        if 'weighted' in kwargs.keys():
            weighted = kwargs['weighted']
        else:
            weighted = False
        
        # If it is a Gaussian kernel, we need to determine the scale
        if kernelType == 'gaussian':
            if 'kernelScale' in kwargs.keys():
                kernelScale = kwargs['kernelScale']
            else:
                kernelScale = 1.
                        
        # If we have a lot of batches and a particularly long sequence, this
        # is bound to fail, memory-wise, so let's do it time instant by time
        # instant if we have a large number of time instants, and split the
        # batches
        maxTimeSamples = 200 # Set the maximum number of t.Samples before
            # which to start doing this time by time.
        maxBatchSize = 100 # Maximum number of samples to process at a given
            # time
        
        # Compute the number of samples, and split the indices accordingly
        if nSamples < maxBatchSize:
            nBatches = 1
            batchSize = [nSamples]
        elif nSamples % maxBatchSize != 0:
            # If we know it's not divisible, then we do floor division and
            # add one more batch
            nBatches = nSamples // maxBatchSize + 1
            batchSize = [maxBatchSize] * nBatches
            # But the last batch is actually smaller, so just add the 
            # remaining ones
            batchSize[-1] = nSamples - sum(batchSize[0:-1])
        # If they fit evenly, then just do so.
        else:
            nBatches = int(nSamples/maxBatchSize)
            batchSize = [maxBatchSize] * nBatches
        # batchIndex is used to determine the first and last element of each
        # batch. We need to add the 0 because it's the first index.
        batchIndex = np.cumsum(batchSize).tolist()
        batchIndex = [0] + batchIndex
        
        # Create the output state variable
        graphMatrix = np.zeros((nSamples, tSamples, nNodes, nNodes))
        
        for b in range(nBatches):
            
            # Pick the batch elements
            posBatch = pos[batchIndex[b]:batchIndex[b+1]]
                
            if tSamples > maxTimeSamples:
                # If the trajectories are longer than 200 points, then do it 
                # time by time.
                
                # For each time instant
                for t in range(tSamples):
                    
                    # Let's start by computing the distance squared
                    _, distSq = self.computeDifferences(posBatch[:,t,:,:])
                    # Apply the Kernel
                    if kernelType == 'gaussian':
                        graphMatrixTime = np.exp(-kernelScale * distSq)
                    else:
                        graphMatrixTime = distSq
                    # Now let's place zeros in all places whose distance is greater
                    # than the radius
                    graphMatrixTime[distSq > (commRadius ** 2)] = 0.
                    # Set the diagonal elements to zero
                    graphMatrixTime[:,\
                                    np.arange(0,nNodes),np.arange(0,nNodes)]\
                                                                           = 0.
                    # If it is unweighted, force all nonzero values to be 1
                    if not weighted:
                        graphMatrixTime = (graphMatrixTime > zeroTolerance)\
                                                          .astype(distSq.dtype)
                                                              
                    if normalizeGraph:
                        isSymmetric = np.allclose(graphMatrixTime,
                                                  np.transpose(graphMatrixTime,
                                                               axes = [0,2,1]))
                        # Tries to make the computation faster, only the 
                        # eigenvalues (while there is a cost involved in 
                        # computing whether the matrix is symmetric, 
                        # experiments found that it is still faster to use the
                        # symmetric algorithm for the eigenvalues)
                        if isSymmetric:
                            W = np.linalg.eigvalsh(graphMatrixTime)
                        else:
                            W = np.linalg.eigvals(graphMatrixTime)
                        maxEigenvalue = np.max(np.real(W), axis = 1)
                        #   batchSize[b]
                        # Reshape to be able to divide by the graph matrix
                        maxEigenvalue=maxEigenvalue.reshape((batchSize[b],1,1))
                        # Normalize
                        graphMatrixTime = graphMatrixTime / maxEigenvalue
                                                              
                    # And put it in the corresponding time instant
                    graphMatrix[batchIndex[b]:batchIndex[b+1],t,:,:] = \
                                                                graphMatrixTime
                    
                    # Sample percentage count
                    percentageCount = int(100*(t+1+b*tSamples)\
                                                      /(nBatches*tSamples))
                    
                    if t == 0 and b == 0:
                        # It's the first one, so just print it
                        print("%3d%%" % percentageCount,
                              end = '', flush = True)
                    else:
                        # Erase the previous characters
                        print('\b \b' * 4 + "%3d%%" % percentageCount,
                              end = '', flush = True)
                
            else:
                # Let's start by computing the distance squared
                _, distSq = self.computeDifferences(posBatch)
                # Apply the Kernel
                if kernelType == 'gaussian':
                    graphMatrixBatch = np.exp(-kernelScale * distSq)
                else:
                    graphMatrixBatch = distSq
                # Now let's place zeros in all places whose distance is greater
                # than the radius
                graphMatrixBatch[distSq > (commRadius ** 2)] = 0.
                # Set the diagonal elements to zero
                graphMatrixBatch[:,:,
                                 np.arange(0,nNodes),np.arange(0,nNodes)] =0.
                # If it is unweighted, force all nonzero values to be 1
                if not weighted:
                    graphMatrixBatch = (graphMatrixBatch > zeroTolerance)\
                                                          .astype(distSq.dtype)
                    
                if normalizeGraph:
                    isSymmetric = np.allclose(graphMatrixBatch,
                                              np.transpose(graphMatrixBatch,
                                                            axes = [0,1,3,2]))
                    # Tries to make the computation faster
                    if isSymmetric:
                        W = np.linalg.eigvalsh(graphMatrixBatch)
                    else:
                        W = np.linalg.eigvals(graphMatrixBatch)
                    maxEigenvalue = np.max(np.real(W), axis = 2)
                    #   batchSize[b] x tSamples
                    # Reshape to be able to divide by the graph matrix
                    maxEigenvalue = maxEigenvalue.reshape((batchSize[b],
                                                           tSamples,
                                                           1, 1))
                    # Normalize
                    graphMatrixBatch = graphMatrixBatch / maxEigenvalue
                    
                # Store
                graphMatrix[batchIndex[b]:batchIndex[b+1]] = graphMatrixBatch
                
                # Sample percentage count
                percentageCount = int(100*(b+1)/nBatches)
                
                if b == 0:
                    # It's the first one, so just print it
                    print("%3d%%" % percentageCount,
                          end = '', flush = True)
                else:
                    # Erase the previous characters
                    print('\b \b' * 4 + "%3d%%" % percentageCount,
                          end = '', flush = True)
                    
        # Erase the percentage
        print('\b \b' * 4, end = '', flush = True)
            
        return graphMatrix

    def getData(self, name, samplesType, *args):
        
        # samplesType: train, valid, test
        # args: 0 args, give back all
        # args: 1 arg: if int, give that number of samples, chosen at random
        # args: 1 arg: if list, give those samples precisely.
        
        # Check that the type is one of the possible ones
        assert samplesType == 'train' or samplesType == 'valid' \
                    or samplesType == 'test'
        # Check that the number of extra arguments fits
        assert len(args) <= 1
                    
        # Check that the name is actually an attribute
        assert name in dir(self)
        
        # Get the desired attribute
        thisDataDict = getattr(self, name)
        
        # Check it's a dictionary and that it has the corresponding key
        assert type(thisDataDict) is dict
        assert samplesType in thisDataDict.keys()
        
        # Get the data now
        thisData = thisDataDict[samplesType]
        # Get the dimension length
        thisDataDims = len(thisData.shape)
        
        # Check that it has at least two dimension, where the first one is
        # always the number of samples
        assert thisDataDims > 1
        
        if len(args) == 1:
            # If it is an int, just return that number of randomly chosen
            # samples.
            if type(args[0]) == int:
                nSamples = thisData.shape[0] # total number of samples
                # We can't return more samples than there are available
                assert args[0] <= nSamples
                # Randomly choose args[0] indices
                selectedIndices = np.random.choice(nSamples, size = args[0],
                                                   replace = False)
                # Select the corresponding samples
                thisData = thisData[selectedIndices]
            else:
                # The fact that we put else here instead of elif type()==list
                # allows for np.array to be used as indices as well. In general,
                # any variable with the ability to index.
                thisData = thisData[args[0]]
                
            # If we only selected a single element, then the nDataPoints dim
            # has been left out. So if we have less dimensions, we have to
            # put it back
            if len(thisData.shape) < thisDataDims:
                if 'torch' in repr(thisData.dtype):
                    thisData =thisData.unsqueeze(0)
                else:
                    thisData = np.expand_dims(thisData, axis = 0)

        return thisData
        
    def evaluate(self, vel = None, accel = None, initVel = None,
                 samplingTime = None):
        
        # It is optional to add a different sampling time, if not, it uses
        # the internal one
        if samplingTime is None:
            # If there's no argument use the internal sampling time
            samplingTime = self.samplingTime
        
        # Check whether we have vel, or accel and initVel (i.e. we are either
        # given the velocities, or we are given the elements to compute them)
        if vel is not None:
            assert len(vel.shape) == 4
            nSamples = vel.shape[0]
            tSamples = vel.shape[1]
            assert vel.shape[2] == 2
            nNodes = vel.shape[3]
        elif accel is not None and initVel is not None:
            assert len(accel.shape) == 4 and len(initVel.shape) == 3
            nSamples = accel.shape[0]
            tSamples = accel.shape[1]
            assert accel.shape[2] == 2
            nNodes = accel.shape[3]
            assert initVel.shape[0] == nSamples
            assert initVel.shape[1] == 2
            assert initVel.shape[2] == nNodes
            
            # Now that we know we have a accel and init velocity, compute the
            # velocity trajectory
            # Compute the velocity trajectory
            if 'torch' in repr(accel.dtype):
                # Check that initVel is also torch
                assert 'torch' in repr(initVel.dtype)
                # Create the tensor to save the velocity trajectory
                vel = torch.zeros(nSamples,tSamples,2,nNodes,
                                  dtype = accel.dtype, device = accel.device)
                # Add the initial velocity
                vel[:,0,:,:] = initVel.clone().detach()
            else:
                # Create the space
                vel = np.zeros((nSamples, tSamples, 2, nNodes),
                               dtype=accel.dtype)
                # Add the initial velocity
                vel[:,0,:,:] = initVel.copy()
                
            # Go over time
            for t in range(1,tSamples):
                # Compute velocity
                vel[:,t,:,:] = accel[:,t-1,:,:] * samplingTime + vel[:,t-1,:,:]
            
        # Check that I did enter one of the if clauses
        assert vel is not None
            
        # And now that we have the velocities, we can compute the cost
        if 'torch' in repr(vel.dtype):
            # Average velocity for time t, averaged across agents
            avgVel = torch.mean(vel, dim = 3) # nSamples x tSamples x 2
            # Compute the difference in velocity between each agent and the
            # mean velocity
            diffVel = vel - avgVel.unsqueeze(3) 
            #   nSamples x tSamples x 2 x nNodes
            # Compute the MSE velocity
            diffVelNorm = torch.sum(diffVel ** 2, dim = 2) 
            #   nSamples x tSamples x nNodes
            # Average over agents
            diffVelAvg = torch.mean(diffVelNorm, dim = 2) # nSamples x tSamples
            # Sum over time
            costPerSample = torch.sum(diffVelAvg, dim = 1) # nSamples
            # Final average cost
            cost = torch.mean(costPerSample)
        else:
            # Repeat for numpy
            avgVel = np.mean(vel, axis = 3) # nSamples x tSamples x 2
            diffVel = vel - np.tile(np.expand_dims(avgVel, 3),
                                    (1, 1, 1, nNodes))
            #   nSamples x tSamples x 2 x nNodes
            diffVelNorm = np.sum(diffVel ** 2, axis = 2)
            #   nSamples x tSamples x nNodes
            diffVelAvg = np.mean(diffVelNorm, axis = 2) # nSamples x tSamples
            costPerSample = np.sum(diffVelAvg, axis = 1) # nSamples
            cost = np.mean(costPerSample) # scalar
        
        return cost
    
    def computeTrajectory(self, initPos, initVel, duration, **kwargs):
        
        # Check initPos is of shape batchSize x 2 x nNodes
        assert len(initPos.shape) == 3
        batchSize = initPos.shape[0]
        assert initPos.shape[1]
        nNodes = initPos.shape[2]
        
        # Check initVel is of shape batchSize x 2 x nNodes
        assert len(initVel.shape) == 3
        assert initVel.shape[0] == batchSize
        assert initVel.shape[1] == 2
        assert initVel.shape[2] == nNodes
        
        # Check what kind of data it is
        #   This is because all the functions are numpy, but if this was
        #   torch, we need to return torch, to make it consistent
        if 'torch' in repr(initPos.dtype):
            assert 'torch' in repr(initVel.dtype)
            useTorch = True
            device = initPos.device
            assert initVel.device == device
        else:
            useTorch = False
        
        # Create time line
        time = np.arange(0, duration, self.samplingTime)
        tSamples = len(time)
        
        # Here, we have two options, or we're given the acceleration or the
        # architecture
        assert 'archit' in kwargs.keys() or 'accel' in kwargs.keys()
        # Flags to determine which method to use
        useArchit = False
        useAccel = False
        
        if 'archit' in kwargs.keys():
            archit = kwargs['archit'] # This is a torch.nn.Module architecture
            architDevice = list(archit.parameters())[0].device
            useArchit = True
        elif 'accel' in kwargs.keys():
            accel = kwargs['accel']
            # accel has to be of shape batchSize x tSamples x 2 x nNodes
            assert len(accel.shape) == 4
            assert accel.shape[0] == batchSize
            assert accel.shape[1] == tSamples
            assert accel.shape[2] == 2
            assert accel.shape[3] == nNodes
            if useTorch:
                assert 'torch' in repr(accel.dtype)
            useAccel = True
        
        # Now create the outputs that will be filled afterwards
        pos = np.zeros((batchSize, tSamples, 2, nNodes), dtype = np.float)
        vel = np.zeros((batchSize, tSamples, 2, nNodes), dtype = np.float)
        if useArchit:
            accel = np.zeros((batchSize, tSamples, 2, nNodes), dtype=np.float)
            state = np.zeros((batchSize, tSamples, 6, nNodes), dtype=np.float)
            graph = np.zeros((batchSize, tSamples, nNodes, nNodes),
                             dtype = np.float)
            
        # Assign the initial positions and velocities
        if useTorch:
            pos[:,0,:,:] = initPos.cpu().numpy()
            vel[:,0,:,:] = initVel.cpu().numpy()
            if useAccel:
                accel = accel.cpu().numpy()
        else:
            pos[:,0,:,:] = initPos.copy()
            vel[:,0,:,:] = initVel.copy()
            
        # Sample percentage count
        percentageCount = int(100/tSamples)
        # Print new value
        print("%3d%%" % percentageCount, end = '', flush = True)
            
        # Now, let's get started:
        for t in range(1, tSamples):
            
            # If it is architecture-based, we need to compute the state, and
            # for that, we need to compute the graph
            if useArchit:
                # Adjust pos value for graph computation
                thisPos = np.expand_dims(pos[:,t-1,:,:], 1)
                # Compute graph
                thisGraph = self.computeCommunicationGraph(thisPos,
                                                           self.commRadius,
                                                           True,
                                                           doPrint = False)
                # Save graph
                graph[:,t-1,:,:] = thisGraph.squeeze(1)
                # Adjust vel value for state computation
                thisVel = np.expand_dims(vel[:,t-1,:,:], 1)
                # Compute state
                thisState = self.computeStates(thisPos, thisVel, thisGraph,
                                               doPrint = False)
                # Save state
                state[:,t-1,:,:] = thisState.squeeze(1)
                
                # Compute the output of the architecture
                #   Note that we need the collection of all time instants up
                #   to now, because when we do the communication exchanges,
                #   it involves past times.
                x = torch.tensor(state[:,0:t,:,:], device = architDevice)
                S = torch.tensor(graph[:,0:t,:,:], device = architDevice)
                with torch.no_grad():
                    thisAccel = archit(x, S)
                # Now that we have computed the acceleration, we only care 
                # about the last element in time
                thisAccel = thisAccel.cpu().numpy()[:,-1,:,:]
                thisAccel[thisAccel > self.accelMax] = self.accelMax
                thisAccel[thisAccel < -self.accelMax] = self.accelMax
                # And save it
                accel[:,t-1,:,:] = thisAccel
                
            # Now that we have the acceleration, we can update position and
            # velocity
            vel[:,t,:,:] = accel[:,t-1,:,:] * self.samplingTime +vel[:,t-1,:,:]
            pos[:,t,:,:] = accel[:,t-1,:,:] * (self.samplingTime ** 2)/2 + \
                            vel[:,t-1,:,:] * self.samplingTime + pos[:,t-1,:,:]
            
            # Sample percentage count
            percentageCount = int(100*(t+1)/tSamples)
            # Erase previous value and print new value
            print('\b \b' * 4 + "%3d%%" % percentageCount,
                  end = '', flush = True)
                
        # And we're missing the last values of graph, state and accel, so
        # let's compute them for completeness
        #   Graph
        thisPos = np.expand_dims(pos[:,-1,:,:], 1)
        thisGraph = self.computeCommunicationGraph(thisPos, self.commRadius,
                                                   True, doPrint = False)
        graph[:,-1,:,:] = thisGraph.squeeze(1)
        #   State
        thisVel = np.expand_dims(vel[:,-1,:,:], 1)
        thisState = self.computeStates(thisPos, thisVel, thisGraph,
                                       doPrint = False)
        state[:,-1,:,:] = thisState.squeeze(1)
        #   Accel
        x = torch.tensor(state).to(architDevice)
        S = torch.tensor(graph).to(architDevice)
        with torch.no_grad():
            thisAccel = archit(x, S)
        thisAccel = thisAccel.cpu().numpy()[:,-1,:,:]
        thisAccel[thisAccel > self.accelMax] = self.accelMax
        thisAccel[thisAccel < -self.accelMax] = self.accelMax
        # And save it
        accel[:,-1,:,:] = thisAccel
                
        # Erase the percentage
        print('\b \b' * 4, end = '', flush = True)
            
        # After we have finished, turn it back into tensor, if required
        if useTorch:
            pos = torch.tensor(pos).to(device)
            vel = torch.tensor(vel).to(device)
            accel = torch.tensor(accel).to(device)
            
        # And return it
        if useArchit:
            return pos, vel, accel, state, graph
        elif useAccel:
            return pos, vel
    
    def computeDifferences(self, u):
        
        # Takes as input a tensor of shape
        #   nSamples x tSamples x 2 x nNodes
        # or of shape
        #   nSamples x 2 x nNodes
        # And returns the elementwise difference u_i - u_j of shape
        #   nSamples (x tSamples) x 2 x nNodes x nNodes
        # And the distance squared ||u_i - u_j||^2 of shape
        #   nSamples (x tSamples) x nNodes x nNodes
        
        # Check dimensions
        assert len(u.shape) == 3 or len(u.shape) == 4
        # If it has shape 3, which means it's only a single time instant, then
        # add the extra dimension so we move along assuming we have multiple
        # time instants
        if len(u.shape) == 3:
            u = np.expand_dims(u, 1)
            hasTimeDim = False
        else:
            hasTimeDim = True
        
        # Now we have that pos always has shape
        #   nSamples x tSamples x 2 x nNodes
        nSamples = u.shape[0]
        tSamples = u.shape[1]
        assert u.shape[2] == 2
        nNodes = u.shape[3]
        
        # Compute the difference along each axis. For this, we subtract a
        # column vector from a row vector. The difference tensor on each
        # position will have shape nSamples x tSamples x nNodes x nNodes
        # and then we add the extra dimension to concatenate and obtain a final
        # tensor of shape nSamples x tSamples x 2 x nNodes x nNodes
        # First, axis x
        #   Reshape as column and row vector, respectively
        uCol_x = u[:,:,0,:].reshape((nSamples, tSamples, nNodes, 1))
        uRow_x = u[:,:,0,:].reshape((nSamples, tSamples, 1, nNodes))
        #   Subtract them
        uDiff_x = uCol_x - uRow_x # nSamples x tSamples x nNodes x nNodes
        # Second, for axis y
        uCol_y = u[:,:,1,:].reshape((nSamples, tSamples, nNodes, 1))
        uRow_y = u[:,:,1,:].reshape((nSamples, tSamples, 1, nNodes))
        uDiff_y = uCol_y - uRow_y # nSamples x tSamples x nNodes x nNodes
        # Third, compute the distance tensor of shape
        #   nSamples x tSamples x nNodes x nNodes
        uDistSq = uDiff_x ** 2 + uDiff_y ** 2
        # Finally, concatenate to obtain the tensor of differences
        #   Add the extra dimension in the position
        uDiff_x = np.expand_dims(uDiff_x, 2)
        uDiff_y = np.expand_dims(uDiff_y, 2)
        #   And concatenate them
        uDiff = np.concatenate((uDiff_x, uDiff_y), 2)
        #   nSamples x tSamples x 2 x nNodes x nNodes
            
        # Get rid of the time dimension if we don't need it
        if not hasTimeDim:
            # (This fails if tSamples > 1)
            uDistSq = uDistSq.squeeze(1)
            #   nSamples x nNodes x nNodes
            uDiff = uDiff.squeeze(1)
            #   nSamples x 2 x nNodes x nNodes
            
        return uDiff, uDistSq
 
    def computeOptimalTrajectory(self, initPos, initVel, duration, 
                                 samplingTime, repelDist,
                                 accelMax = 100.):
        
        # The optimal trajectory is given by
        # u_{i} = - \sum_{j=1}^{N} (v_{i} - v_{j})
        #         + 2 \sum_{j=1}^{N} (r_{i} - r_{j}) *
        #                                 (1/\|r_{i}\|^{4} + 1/\|r_{j}\|^{2}) *
        #                                 1{\|r_{ij}\| < R}
        # for each agent i=1,...,N, where v_{i} is the velocity and r_{i} the
        # position.
        
        # Check that initPos and initVel as nSamples x 2 x nNodes arrays
        assert len(initPos.shape) == len(initVel.shape) == 3
        nSamples = initPos.shape[0]
        assert initPos.shape[1] == initVel.shape[1] == 2
        nNodes = initPos.shape[2]
        assert initVel.shape[0] == nSamples
        assert initVel.shape[2] == nNodes
        
        # time
        time = np.arange(0, duration, samplingTime)
        tSamples = len(time) # number of time samples
        
        # Create arrays to store the trajectory
        pos = np.zeros((nSamples, tSamples, 2, nNodes))
        vel = np.zeros((nSamples, tSamples, 2, nNodes))
        accel = np.zeros((nSamples, tSamples, 2, nNodes))
        
        # Initial settings
        pos[:,0,:,:] = initPos
        vel[:,0,:,:] = initVel
        
        # Sample percentage count
        percentageCount = int(100/tSamples)
        # Print new value
        print("%3d%%" % percentageCount, end = '', flush = True)
        
        # For each time instant
        for t in range(1,tSamples):
            
            # Compute the optimal acceleration
            #   Compute the distance between all elements (positions)
            ijDiffPos, ijDistSq = self.computeDifferences(pos[:,t-1,:,:])
            #       ijDiffPos: nSamples x 2 x nNodes x nNodes
            #       ijDistSq:  nSamples x nNodes x nNodes
            #   And also the difference in velocities
            ijDiffVel, _ = self.computeDifferences(vel[:,t-1,:,:])
            #       ijDiffVel: nSamples x 2 x nNodes x nNodes
            #   The last element we need to compute the acceleration is the
            #   gradient. Note that the gradient only counts when the distance 
            #   is smaller than the repel distance
            #       This is the mask to consider each of the differences
            repelMask = (ijDistSq < (repelDist**2)).astype(ijDiffPos.dtype)
            #       Apply the mask to the relevant differences
            ijDiffPos = ijDiffPos * np.expand_dims(repelMask, 1)
            #       Compute the constant (1/||r_ij||^4 + 1/||r_ij||^2)
            ijDistSqInv = invertTensorEW(ijDistSq)
            #       Add the extra dimension
            ijDistSqInv = np.expand_dims(ijDistSqInv, 1)
            #   Compute the acceleration
            accel[:,t-1,:,:] = \
                    -np.sum(ijDiffVel, axis = 3) \
                    +2* np.sum(ijDiffPos * (ijDistSqInv ** 2 + ijDistSqInv),
                               axis = 3)
                    
            # Finally, note that if the agents are too close together, the
            # acceleration will be very big to get them as far apart as
            # possible, and this is physically impossible.
            # So let's add a limitation to the maximum aceleration

            # Find the places where the acceleration is big
            thisAccel = accel[:,t-1,:,:].copy()
            # Values that exceed accelMax, force them to be accelMax
            thisAccel[accel[:,t-1,:,:] > accelMax] = accelMax
            # Values that are smaller than -accelMax, force them to be accelMax
            thisAccel[accel[:,t-1,:,:] < -accelMax] = -accelMax
            # And put it back
            accel[:,t-1,:,:] = thisAccel
            
            # Update the values
            #   Update velocity
            vel[:,t,:,:] = accel[:,t-1,:,:] * samplingTime + vel[:,t-1,:,:]
            #   Update the position
            pos[:,t,:,:] = accel[:,t-1,:,:] * (samplingTime ** 2)/2 + \
                                 vel[:,t-1,:,:] * samplingTime + pos[:,t-1,:,:]
            
            # Sample percentage count
            percentageCount = int(100*(t+1)/tSamples)
            # Erase previous pecentage and print new value
            print('\b \b' * 4 + "%3d%%" % percentageCount,
                  end = '', flush = True)
                
        # Erase the percentage
        print('\b \b' * 4, end = '', flush = True)
            
        return pos, vel, accel

    def computeInitialPositions(self, nNodes, nSamples, commRadius,
                                minDist = 0.1, geometry = 'rectangular',
                                **kwargs):
        
        # It will always be uniform. We can select whether it is rectangular
        # or circular (or some other shape) and the parameters respecting
        # that
        assert geometry == 'rectangular' or geometry == 'circular'
        assert minDist * (1.+zeroTolerance) <= commRadius * (1.-zeroTolerance)
        # We use a zeroTolerance buffer zone, just in case
        minDist = minDist * (1. + zeroTolerance)
        commRadius = commRadius * (1. - zeroTolerance)
        
        # If there are other keys in the kwargs argument, they will just be
        # ignored
        
        # We will first create the grid, whether it is rectangular or
        # circular.
        
        # Let's start by setting the fixed position
        if geometry == 'rectangular':
            
            # This grid has a distance that depends on the desired minDist and
            # the commRadius
            distFixed = (commRadius + minDist)/(2.*np.sqrt(2))
            #   This is the fixed distance between points in the grid
            distPerturb = (commRadius - minDist)/(4.*np.sqrt(2))
            #   This is the standard deviation of a uniform perturbation around
            #   the fixed point.
            # This should guarantee that, even after the perturbations, there
            # are no agents below minDist, and that all agents have at least
            # one other agent within commRadius.
            
            # How many agents per axis
            nNodesPerAxis = int(np.ceil(np.sqrt(nNodes)))
            
            axisFixedPos = np.arange(-(nNodesPerAxis * distFixed)/2,
                                       (nNodesPerAxis * distFixed)/2,
                                      step = distFixed)
            
            # Repeat the positions in the same order (x coordinate)
            xFixedPos = np.tile(axisFixedPos, nNodesPerAxis)
            # Repeat each element (y coordinate)
            yFixedPos = np.repeat(axisFixedPos, nNodesPerAxis)
            
            # Concatenate this to obtain the positions
            fixedPos = np.concatenate((np.expand_dims(xFixedPos, 0),
                                       np.expand_dims(yFixedPos, 0)),
                                      axis = 0)
            
            # Get rid of unnecessary agents
            fixedPos = fixedPos[:, 0:nNodes]
            # And repeat for the number of samples we want to generate
            fixedPos = np.repeat(np.expand_dims(fixedPos, 0), nSamples,
                                 axis = 0)
            #   nSamples x 2 x nNodes
            
            # Now generate the noise
            perturbPos = np.random.uniform(low = -distPerturb,
                                           high = distPerturb,
                                           size = (nSamples, 2, nNodes))
            
            # Initial positions
            initPos = fixedPos + perturbPos
                
        elif geometry == 'circular':
            
            # Radius for the grid
            rFixed = (commRadius + minDist)/2.
            rPerturb = (commRadius - minDist)/4.
            fixedRadius = np.arange(0, rFixed * nNodes, step = rFixed)+rFixed
            
            # Angles for the grid
            aFixed = (commRadius/fixedRadius + minDist/fixedRadius)/2.
            for a in range(len(aFixed)):
                # How many times does aFixed[a] fits within 2pi?
                nNodesPerCircle = 2 * np.pi // aFixed[a]
                # And now divide 2*np.pi by this number
                aFixed[a] = 2 * np.pi / nNodesPerCircle
            #   Fixed angle difference for each value of fixedRadius
            
            # Now, let's get the radius, angle coordinates for each agents
            initRadius = np.empty((0))
            initAngles = np.empty((0))
            agentsSoFar = 0 # Number of agents located so far
            n = 0 # Index for radius
            while agentsSoFar < nNodes:
                thisRadius = fixedRadius[n]
                thisAngles = np.arange(0, 2*np.pi, step = aFixed[n])
                agentsSoFar += len(thisAngles)
                initRadius = np.concatenate((initRadius,
                                             np.repeat(thisRadius,
                                                       len(thisAngles))))
                initAngles = np.concatenate((initAngles, thisAngles))
                n += 1
                assert len(initRadius) == agentsSoFar
                
            # Restrict to the number of agents we need
            initRadius = initRadius[0:nNodes]
            initAngles = initAngles[0:nNodes]
            
            # Add the number of samples
            initRadius = np.repeat(np.expand_dims(initRadius, 0), nSamples,
                                   axis = 0)
            initAngles = np.repeat(np.expand_dims(initAngles, 0), nSamples,
                                   axis = 0)
            
            # Add the noise
            #   First, to the angles
            for n in range(nNodes):
                # Get the radius (the angle noise depends on the radius); so
                # far the radius is the same for all samples
                thisRadius = initRadius[0,n]
                aPerturb = (commRadius/thisRadius - minDist/thisRadius)/4.
                # Add the noise to the angles
                initAngles[:,n] += np.random.uniform(low = -aPerturb,
                                                     high = aPerturb,
                                                     size = (nSamples))
            #   Then, to the radius
            initRadius += np.random.uniform(low = -rPerturb,
                                            high = rPerturb,
                                            size = (nSamples, nNodes))
            
            # And finally, get the positions in the cartesian coordinates
            initPos = np.zeros((nSamples, 2, nNodes))
            initPos[:, 0, :] = initRadius * np.cos(initAngles)
            initPos[:, 1, :] = initRadius * np.sin(initAngles)
            
        # Now, check that the conditions are met:
        #   Compute square distances
        _, distSq = self.computeDifferences(np.expand_dims(initPos, 1))
        #   Get rid of the "time" dimension that arises from using the 
        #   method to compute distances
        distSq = distSq.squeeze(1)
        #   Compute the minimum distance (don't forget to add something in
        #   the diagonal, which otherwise is zero)
        minDistSq = np.min(distSq + \
                           2 * commRadius\
                             *np.eye(distSq.shape[1]).reshape(1,
                                                              distSq.shape[1],
                                                              distSq.shape[2])
                           )
        
        assert minDistSq >= minDist ** 2
        
        #   Now the number of neighbors -- still confused is this necesary in this function
        graphMatrix = self.computeCommunicationGraph(np.expand_dims(initPos,1),
                                                     self.commRadius,
                                                     False,
                                                     doPrint = False)
        graphMatrix = graphMatrix.squeeze(1) # nSamples x nNodes x nNodes  
        
        #   Binarize the matrix
        graphMatrix = (np.abs(graphMatrix) > zeroTolerance)\
                                                         .astype(initPos.dtype)
        
        #   And check that we always have initially connected graphs
        for n in range(nSamples):
            assert graph.isConnected(graphMatrix[n,:,:])
        
        # We move to compute the initial velocities. Velocities can be
        # either positive or negative, so we do not need to determine
        # the lower and higher, just around zero
        if 'xMaxInitVel' in kwargs.keys():
            xMaxInitVel = kwargs['xMaxInitVel']
        else:
            xMaxInitVel = 3.
            #   Takes five seconds to traverse half the map
        # Same for the other axis
        if 'yMaxInitVel' in kwargs.keys():
            yMaxInitVel = kwargs['yMaxInitVel']
        else:
            yMaxInitVel = 3.
        
        # And sample the velocities
        xInitVel = np.random.uniform(low = -xMaxInitVel, high = xMaxInitVel,
                                     size = (nSamples, 1, nNodes))
        yInitVel = np.random.uniform(low = -yMaxInitVel, high = yMaxInitVel,
                                     size = (nSamples, 1, nNodes))
        # Add bias
        xVelBias = np.random.uniform(low = -xMaxInitVel, high = xMaxInitVel,
                                     size = (nSamples))
        yVelBias = np.random.uniform(low = -yMaxInitVel, high = yMaxInitVel,
                                     size = (nSamples))
        
        # And concatenate them
        velBias = np.concatenate((xVelBias, yVelBias)).reshape((nSamples,2,1))
        initVel = np.concatenate((xInitVel, yInitVel), axis = 1) + velBias
        #   nSamples x 2 x nNodes
        
        return initPos, initVel # initPos=(421, 2, 50), initVel=(421, 2, 50)
    
    def getSamples(self, samplesType, *args):
        # samplesType: train, valid, test
        # args: 0 args, give back all
        # args: 1 arg: if int, give that number of samples, chosen at random
        # args: 1 arg: if list, give those samples precisely.
        # Check that the type is one of the possible ones
        assert samplesType == 'train' or samplesType == 'valid' \
                    or samplesType == 'test'
        # Check that the number of extra arguments fits
        assert len(args) <= 1
        # If there are no arguments, just return all the desired samples
        x = self.samples[samplesType]['signals']
        y = self.samples[samplesType]['targets']
        # If there's an argument, we have to check whether it is an int or a
        # list
        if len(args) == 1:
            # If it is an int, just return that number of randomly chosen
            # samples.
            if type(args[0]) == int:
                nSamples = x.shape[0] # total number of samples
                # We can't return more samples than there are available
                assert args[0] <= nSamples
                # Randomly choose args[0] indices
                selectedIndices = np.random.choice(nSamples, size = args[0],
                                                   replace = False)
                # Select the corresponding samples
                xSelected = x[selectedIndices]
                y = y[selectedIndices]
            else:
                # The fact that we put else here instead of elif type()==list
                # allows for np.array to be used as indices as well. In general,
                # any variable with the ability to index.
                xSelected = x[args[0]]
                # And assign the labels
                y = y[args[0]]
                
            # If we only selected a single element, then the nDataPoints dim
            # has been left out. So if we have less dimensions, we have to
            # put it back
            if len(xSelected.shape) < len(x.shape):
                if 'torch' in self.dataType:
                    x = xSelected.unsqueeze(0)
                else:
                    x = np.expand_dims(xSelected, axis = 0)
            else:
                x = xSelected

        return x, y
        
    def astype(self, dataType):
        # This changes the type for the minimal attributes (samples). This 
        # methods should still be initialized within the data classes, if more
        # attributes are used.
        
        # The labels could be integers as created from the dataset, so if they
        # are, we need to be sure they are integers also after conversion. 
        # To do this we need to match the desired dataType to its int 
        # counterpart. Typical examples are:
        #   numpy.float64 -> numpy.int64
        #   numpy.float32 -> numpy.int32
        #   torch.float64 -> torch.int64
        #   torch.float32 -> torch.int32
        
        targetType = str(self.samples['train']['targets'].dtype)
        if 'int' in targetType:
            if 'numpy' in repr(dataType):
                if '64' in targetType:
                    targetType = np.int64
                elif '32' in targetType:
                    targetType = np.int32
            elif 'torch' in repr(dataType):
                if '64' in targetType:
                    targetType = torch.int64
                elif '32' in targetType:
                    targetType = torch.int32
        else: # If there is no int, just stick with the given dataType
            targetType = dataType
        
        # Now that we have selected the dataType, and the corresponding
        # labelType, we can proceed to convert the data into the corresponding
        # type
        for key in self.samples.keys():
            self.samples[key]['signals'] = changeDataType(
                                                   self.samples[key]['signals'],
                                                   dataType)
            self.samples[key]['targets'] = changeDataType(
                                                   self.samples[key]['targets'],
                                                   targetType)

        # Update attribute
        if dataType is not self.dataType:
            self.dataType = dataType

    def to(self, device):
        # This changes the type for the minimal attributes (samples). This 
        # methods should still be initialized within the data classes, if more
        # attributes are used.
        # This can only be done if they are torch tensors
        if 'torch' in repr(self.dataType):
            for key in self.samples.keys():
                for secondKey in self.samples[key].keys():
                    self.samples[key][secondKey] \
                                      = self.samples[key][secondKey].to(device)

            # If the device changed, save it.
            if device is not self.device:
                self.device = device    
            
