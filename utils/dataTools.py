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
                 gainOffset, gainSkew,                 
                 netwkType='digraph', normaliseGraph=False,
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
        self.gainOffset = gainOffset 
        self.gainSkew = gainSkew
        # data
        self.netwkType = netwkType        
        self.normaliseGraph = normaliseGraph
        self.dataType = dataType
        self.device = device
        # places to store the data
        self.initOffset = None
        self.initSkew = None
        self.commGraph = None
        
        print("\tComputing initial conditions...", end = ' ', flush = True)
        
        # compute the initial clock offsets and skews
        initOffsetAll, initSkewAll = self.computeInitialOffsetsSkews(self.nNodes, \
                                                                nSamples, \
                                                                self.initOffsetValue, \
                                                                self.initSkewValue)     
                                
        # split all the initial clock offsets and skews in the corresponding 
        # datasets (train, valid and test)
        self.initOffset = {}
        self.initSkew = {}
        
        print("OK", flush = True)
        print("\tComputing the network topologies...",
              end=' ', flush=True)

        # compute communication graph           
        # when network topology is directed graph, 'normaliseGraph' does NOT work
        commGraphAll = self.computeNetworkTopologies(self.nNodes, nSamples, \
                                                     self.duration, self.samplingTimeScale, \
                                                     self.netwkType, \
                                                     self.normaliseGraph)      
                                        
        self.commNetwk = {}
        
        print("OK", flush = True)   
        print("\tComputing the centralised optimal synchronisation trajectories...",
              end=' ', flush=True)
        
        # realise time synchronisation via the centralised dynamic controller
        offsetAll, skewAll, offsetCorrectionAll, skewCorrectionAll = self.computeViaCentralisedDynamicController(
                                                                     self.nNodes, nSamples, 
                                                                     initOffsetAll, initSkewAll,
                                                                     self.gainOffset, self.gainSkew,
                                                                     self.duration, self.samplingTimeScale)   

        inputOffsetSkewAll = np.concatenate((offsetAll, skewAll), axis = 2)
        outputOffsetSkewAll = np.concatenate((offsetCorrectionAll, skewCorrectionAll), axis = 2)        
                
        self.offset = {}
        self.skew = {}
        self.offsetCorrection = {}
        self.skewCorrection = {}        
        
        print("OK", flush = True)
        
        # separate the states into training, validation and testing samples
        # and save them
        #   Training set
        self.samples['train']['signals'] = inputOffsetSkewAll[0:self.nTrain].copy()
        self.samples['train']['targets'] = outputOffsetSkewAll[0:self.nTrain].copy()
        self.initOffset['train'] = initOffsetAll[0:self.nTrain]
        self.initSkew['train'] = initSkewAll[0:self.nTrain]
        self.offset['train'] = offsetAll[0:self.nTrain]
        self.skew['train'] = skewAll[0:self.nTrain]
        self.offsetCorrection['train'] = offsetCorrectionAll[0:self.nTrain]
        self.skewCorrection['train'] = skewCorrectionAll[0:self.nTrain]        
        self.commNetwk['train'] = commGraphAll[0:self.nTrain]
        #   Validation set
        startSample = self.nTrain
        endSample = self.nTrain + self.nValid
        self.samples['valid']['signals']=inputOffsetSkewAll[startSample:endSample].copy()
        self.samples['valid']['targets']=outputOffsetSkewAll[startSample:endSample].copy()
        self.initOffset['valid'] = initOffsetAll[startSample:endSample]
        self.initSkew['valid'] = initSkewAll[startSample:endSample]
        self.offset['valid'] = offsetAll[startSample:endSample]
        self.skew['valid'] = skewAll[startSample:endSample]
        self.offsetCorrection['valid'] = offsetCorrectionAll[startSample:endSample]
        self.skewCorrection['valid'] = skewCorrectionAll[startSample:endSample]
        self.commNetwk['valid'] = commGraphAll[startSample:endSample]
        #   Testing set
        startSample = self.nTrain + self.nValid
        endSample = self.nTrain + self.nValid + self.nTest
        self.samples['test']['signals']=inputOffsetSkewAll[startSample:endSample].copy()
        self.samples['test']['targets']=outputOffsetSkewAll[startSample:endSample].copy()
        self.initOffset['test'] = initOffsetAll[startSample:endSample]
        self.initSkew['test'] = initSkewAll[startSample:endSample]
        self.offset['test'] = offsetAll[startSample:endSample]
        self.skew['test'] = skewAll[startSample:endSample]
        self.offsetCorrection['test'] = offsetCorrectionAll[startSample:endSample]
        self.skewCorrection['test'] = skewCorrectionAll[startSample:endSample]
        self.commNetwk['test'] = commGraphAll[startSample:endSample]
        
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
            
    def computeNetworkTopologies(self, nNodes, nSamples,
                                 duration, samplingTimeScale=0.01,
                                 netwkType='digraph', normaliseGraph=False):

        assert netwkType == 'digraph' or netwkType == 'bigraph'        
             
        commGraph = np.zeros((nSamples, nNodes, nNodes))
        
        if netwkType=='digraph':
            for i in range(nSamples):
                networkTopology = netwk.random_tree(nNodes, seed=np.random, create_using=netwk.DiGraph)            
                # calculate the adjacency matrix
                networkTopologyMatrix = netwk.adjacency_matrix(networkTopology).todense()            
                # the maximum eigenvalue is zero, no normalisation
                commGraph[i, :, :] = networkTopologyMatrix
            # end for                    
        elif netwkType=='bigraph':
            for i in range(nSamples):
                networkTopology = netwk.random_tree(nNodes, seed=np.random, create_using=netwk.Graph)            
                # calculate the adjacency matrix
                networkTopologyMatrix = netwk.adjacency_matrix(networkTopology).todense()            
                if normaliseGraph==True:
                    # calculate the eigenvalues
                    W = np.linalg.eigvals(networkTopologyMatrix)            
                    maxEigenvalue = np.max(np.real(W))
                    # normalise
                    networkTopologyMatrix = networkTopologyMatrix / maxEigenvalue                                                    
                    commGraph[i, :, :] = networkTopologyMatrix
                # end if 
            # end for                    
        # end if    
        
        # add the time dimension, but the network is constant during the experiments
        commGraph = np.expand_dims(commGraph, 1) 
        commGraph = np.repeat(commGraph, int(duration/samplingTimeScale), axis = 1) # nSamples x time x feature x nNodes     
        
        return commGraph

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
        
    def evaluate(self, offset = None, skew = None,
                 correctionOffset = None, correctionSkew= None, 
                 initOffset = None, initSkew = None,
                 samplingTimeScale = None):
        
        # It is optional to add a different sampling time, if not, it uses
        # the internal one
        if samplingTimeScale is None:
            # If there's no argument use the internal sampling time
            samplingTimeScale = self.samplingTimeScale
        # end if 
        
        # Check whether we have offset, skew, or correctionOffset, correctionSkew, 
        # initOffset, initSkew (i.e. we are either given the offsets and skews, 
        # or we are given the elements to compute them)
        if offset is not None and skew is not None:
            assert len(offset.shape) == 4
            nSamples = offset.shape[0]
            tSamples = offset.shape[1]
            assert offset.shape[2] == 1
            nNodes = offset.shape[3]            
            assert len(skew.shape) == 4
            assert nSamples == skew.shape[0]
            assert tSamples == skew.shape[1]
            assert offset.shape[2] == 1
            assert nNodes == skew.shape[3]                        
        elif correctionOffset is not None and correctionSkew is not None and\
            initOffset is not None and initSkew is not None:
            assert len(correctionOffset.shape) == 4 and len(correctionSkew.shape) == 4                
            assert len(initOffset.shape) == 3 and len(initSkew.shape) == 3                            
            nSamples = correctionOffset.shape[0]
            tSamples = correctionOffset.shape[1]  
            assert correctionOffset.shape[2] == 1
            nNodes = correctionOffset.shape[3]                 
            assert correctionSkew.shape[0] == nSamples
            assert correctionSkew.shape[1] == tSamples  
            assert correctionSkew.shape[2] == 1
            assert correctionSkew.shape[3] == nNodes
            assert initOffset.shape[0] == nSamples
            assert initOffset.shape[1] == 1  
            assert initOffset.shape[2] == nNodes  
            assert initSkew.shape[0] == nSamples
            assert initSkew.shape[1] == 1  
            assert initSkew.shape[2] == nNodes              
                        
            # # Now that we know we have a accel and init velocity, compute the
            # # velocity trajectory
            # # Compute the velocity trajectory
            # if 'torch' in repr(accel.dtype):
            #     # Check that initVel is also torch
            #     assert 'torch' in repr(initVel.dtype)
            #     # Create the tensor to save the velocity trajectory
            #     vel = torch.zeros(nSamples,tSamples,2,nNodes,
            #                       dtype = accel.dtype, device = accel.device)
            #     # Add the initial velocity
            #     vel[:,0,:,:] = initVel.clone().detach()
            # else:
            #     # Create the space
            #     vel = np.zeros((nSamples, tSamples, 2, nNodes),
            #                    dtype=accel.dtype)
            #     # Add the initial velocity
            #     vel[:,0,:,:] = initVel.copy()
                
            # # Go over time
            # for t in range(1,tSamples):
            #     # Compute velocity
            #     vel[:,t,:,:] = accel[:,t-1,:,:] * samplingTime + vel[:,t-1,:,:]
            
        # Check that I did enter one of the if clauses
        assert offset is not None
        assert skew is not None        
            
        # And now that we have the offsets and skew, we can compute the cost
        if 'torch' in repr(offset.dtype):
            # Average offset for time t, averaged across agents
            avgOffset = torch.mean(offset, dim = 3) # nSamples x tSamples x 1
            # Compute the difference in velocity between each agent and the
            # mean velocity
            diffOffset = offset - avgOffset.unsqueeze(3) 
            #   nSamples x tSamples x 1 x nNodes
            # Compute the MSE velocity
            diffOffsetNorm = torch.sum(diffOffset ** 2, dim = 2) 
            #   nSamples x tSamples x nNodes
            # Average over agents
            diffOffsetAvg = torch.mean(diffOffsetNorm, dim = 2) # nSamples x tSamples
            # Sum over time
            costPerSample = torch.sum(diffOffsetAvg, dim = 1) # nSamples
            # Final average cost
            cost = torch.mean(costPerSample)
        else:
            # Repeat for numpy
            avgOffset = np.mean(offset, axis = 3) # nSamples x tSamples x 2
            diffOffset = offset - np.tile(np.expand_dims(avgOffset, 3),
                                    (1, 1, 1, nNodes))
            #   nSamples x tSamples x 2 x nNodes
            diffOffsetNorm = np.sum(diffOffset ** 2, axis = 2)
            #   nSamples x tSamples x nNodes
            diffOffsetAvg = np.mean(diffOffsetNorm, axis = 2) # nSamples x tSamples
            costPerSample = np.sum(diffOffsetAvg, axis = 1) # nSamples
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
                
    def computeTimeSynchronisation(self, initOffset, initSkew, netwkTopology, 
                                   duration, archit, displayProgress=True):
        
        # Check initOffset is of shape batchSize x 1 x nNodes
        assert len(initOffset.shape) == 3
        batchSize = initOffset.shape[0]
        assert initOffset.shape[1]
        nNodes = initOffset.shape[2]
        
        # Check initSkew is of shape batchSize x 1 x nNodes
        assert len(initSkew.shape) == 3
        assert initSkew.shape[0] == batchSize
        assert initSkew.shape[1] == 1
        assert initSkew.shape[2] == nNodes
        
        # Check netwkTopology is of shape batchSize x time x nNodes x nNodes
        assert len(netwkTopology.shape) == 4
        assert netwkTopology.shape[0] == batchSize
        assert netwkTopology.shape[1] == int(duration/self.samplingTimeScale)
        assert netwkTopology.shape[2] == nNodes        
        assert netwkTopology.shape[3] == nNodes                
        
        # Check what kind of data it is
        #   This is because all the functions are numpy, but if this was
        #   torch, we need to return torch, to make it consistent
        if 'torch' in repr(initOffset.dtype):
            assert 'torch' in repr(initSkew.dtype)
            useTorch = True
            device = initOffset.device
            assert initSkew.device == device
        else:
            useTorch = False
        
        # Create time line
        time = np.arange(0, duration, self.samplingTimeScale)
        tSamples = len(time)
                
        architDevice = list(archit.parameters())[0].device
                
        # Now create the outputs that will be filled afterwards
        offset = np.zeros((batchSize, tSamples, 1, nNodes), dtype = np.float)
        skew = np.zeros((batchSize, tSamples, 1, nNodes), dtype = np.float)
        offsetCorrection = np.zeros((batchSize, tSamples, 1, nNodes), dtype=np.float)
        skewCorrection = np.zeros((batchSize, tSamples, 1, nNodes), dtype=np.float)
        thisOffsetSkew = np.zeros((batchSize, tSamples, 2, nNodes), dtype=np.float)
            
        # Assign the initial positions and velocities
        if useTorch:
            offset[:,0,:,:] = initOffset.cpu().numpy()
            skew[:,0,:,:] = initSkew.cpu().numpy()
            offsetCorrection = offsetCorrection.cpu().numpy()
            skewCorrection = skewCorrection.cpu().numpy()                
        else:
            offset[:,0,:,:] = initOffset.copy()
            skew[:,0,:,:] = initSkew.copy()

        if displayProgress == True:            
            # Sample percentage count
            percentageCount = int(100/tSamples)
            # Print new value
            print("%3d%%" % percentageCount, end = '', flush = True)
        # end if
        
        # Now, let's get started:
        for t in range(1, tSamples):
            
            # Adjust pos value for input computation
            thisOffset = offset[:,t-1,:,:]
            thisSkew = skew[:,t-1,:,:]                
            thisOffsetSkew[:,t-1,:,:] = np.concatenate((thisOffset, thisSkew), axis = 1)
           
            # Compute the output of the architecture
            #   Note that we need the collection of all time instants up
            #   to now, because when we do the communication exchanges,
            #   it involves past times.
            x = torch.tensor(thisOffsetSkew[:,0:t,:,:], device = architDevice)
            S = torch.tensor(netwkTopology[:,0:t,:,:], device = architDevice)
            with torch.no_grad():
                thisOffsetSkewCorrection = archit(x, S)
            # Now that we have computed the correction input, we only care 
            # about the last element in time
            thisOffsetSkewCorrection = thisOffsetSkewCorrection.cpu().numpy()[:,-1,:,:]
            # And save it
            offsetCorrection[:,t-1,:,:] = np.expand_dims(thisOffsetSkewCorrection[:,0,:], 1)
            skewCorrection[:,t-1,:,:] = np.expand_dims(thisOffsetSkewCorrection[:,1,:], 1)            
                
            # Now that we have the offset and skew correction inputs, we can 
            # clock offset and skew 
            offset[:,t,:,:] =  offset[:,t-1,:,:] + skew[:,t-1,:,:] * self.samplingTimeScale + offsetCorrection[:,t-1,:,:]
            skew[:,t,:,:] = skew[:,t-1,:,:] + skewCorrection[:,t-1,:,:]  
            
            if displayProgress == True:
                # Sample percentage count
                percentageCount = int(100*(t+1)/tSamples)
                # Erase previous value and print new value
                print('\b \b' * 4 + "%3d%%" % percentageCount,
                      end = '', flush = True)
            # end if 
            
        # end for
                
        # And we're missing the last values of clock offset and skew, so
        # let's compute them for completeness
        #   Clock offset and skew
        thisOffset = offset[:,-1,:,:]
        thisSkew = skew[:,-1,:,:]                
        thisOffsetSkew[:,-1,:,:] = np.concatenate((thisOffset, thisSkew), axis = 1)

        #   Accel
        x = torch.tensor(thisOffsetSkew).to(architDevice)
        S = torch.tensor(netwkTopology).to(architDevice)
        with torch.no_grad():
            thisOffsetSkewCorrection = archit(x, S)
        thisOffsetSkewCorrection = thisOffsetSkewCorrection.cpu().numpy()[:,-1,:,:]
        # And save it
        offsetCorrection[:,-1,:,:] = np.expand_dims(thisOffsetSkewCorrection[:,0,:], 1)
        skewCorrection[:,-1,:,:] = np.expand_dims(thisOffsetSkewCorrection[:,1,:], 1)            
        
        if displayProgress == True:
            # Erase the percentage
            print('\b \b' * 4, end = '', flush = True)
        # end if
        
        # After we have finished, turn it back into tensor, if required
        if useTorch:
            offset = torch.tensor(offset).to(device)
            skew = torch.tensor(skew).to(device)
            offsetCorrection = torch.tensor(offsetCorrection).to(device)
            skewCorrection = torch.tensor(skewCorrection).to(device)            
            
        # And return it
        return offset, skew, offsetCorrection, skewCorrection
    
    def computeDifferences(self, u):
        
        # take as input a tensor of shape
        #   nSamples x tSamples x 1 x nNodes
        # and return the elementwise difference u_i - u_j of shape
        #   nSamples x tSamples x 1 x nNodes x nNodes
        
        # check dimensions
        assert len(u.shape) == 3 or len(u.shape) == 4
        # If it has shape 3, which means it's only a single time instant, then
        # add the extra dimension so we move along assuming we have multiple
        # time instants
        if len(u.shape) == 3:
            u = np.expand_dims(u, 1)
            hasTimeDim = False
        else:
            hasTimeDim = True
        
        # now we have that offset or skew always has shape
        #   nSamples x tSamples x 1 x nNodes
        nSamples = u.shape[0]
        tSamples = u.shape[1]
        assert u.shape[2] == 1
        nNodes = u.shape[3]
        
        # compute the difference along each axis. For this, we subtract a
        # column vector from a row vector. The difference tensor on offset 
        # will have shape nSamples x tSamples x nNodes x nNodes
        # and then we add the extra dimension to obtain a final
        # tensor of shape nSamples x tSamples x 1 x nNodes x nNodes
        #   reshape as column and row vector, respectively
        uCol = u[:,:,0,:].reshape((nSamples, tSamples, nNodes, 1))
        uRow = u[:,:,0,:].reshape((nSamples, tSamples, 1, nNodes))
        #   subtract them
        uDiff = uCol - uRow # nSamples x tSamples x nNodes x nNodes
        # finally, concatenate to obtain the tensor of differences
        #   Add the extra dimension in the position
        uDiff = np.expand_dims(uDiff, 2)
        #   nSamples x tSamples x 1 x nNodes x nNodes
        
        if not hasTimeDim:
            # (This fails if tSamples > 1)
            uDiff = uDiff.squeeze(1)
            #   nSamples x 1 x nNodes x nNodes        
                        
        return uDiff

    def computeViaCentralisedDynamicController(self, nNodes, nSamples, 
                                               initOffset, initSkew,
                                               gainOffset, gainSkew,
                                               duration, samplingTimeScale):    
        
        # the centralised dynamic controller is given accoring to Zong2022b_TIE,
        # see equ (14) in the above paper (http://yzong.com/pub/Zong2022b.pdf)

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
        offset[:,0,:,:] = initOffset
        skew[:,0,:,:] = initSkew
        
        # sample percentage count
        percentageCount = int(100/tSamples)
        # print new value
        print("%3d%%" % percentageCount, end = '', flush = True)
        
        # for each time instant
        for t in range(1, tSamples):
            
            # compute the clock offset and skew correction input
            #   compute the the clock offset differences between all elements
            ijDiffOffset = self.computeDifferences(offset[:,t-1,:,:])
            #       ijDiffOffset: nSamples x 1 x 1 x nNodes x nNodes
            #   and also the difference in clock skews
            ijDiffSkew = self.computeDifferences(skew[:,t-1,:,:])
            #       ijDiffSkew: nSamples x 1 x 1 x nNodes x nNodes

            # update the clock offset and skew correction input
            #   update clock offset correction value
            integralOffset[:,t,:,:] = gainOffset[0] * integralOffset[:,t-1,:,:] + gainOffset[1] * np.sum(ijDiffOffset, axis=3)
            correctionOffset[:,t,:,:] = gainOffset[3] * integralOffset[:,t-1,:,:] + gainOffset[3] * np.sum(ijDiffOffset, axis=3)
            #   Update clock skew correction value
            integralSkew[:,t,:,:] = gainSkew[0] * integralSkew[:,t-1,:,:] + gainSkew[1] * np.sum(ijDiffSkew, axis=3)
            correctionSkew[:,t,:,:] = gainSkew[2] * integralSkew[:,t-1,:,:] + gainSkew[3] * np.sum(ijDiffSkew, axis=3)            
            
            # update the values Todo: noises will be required in the future work
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
        
        return offset, skew, correctionOffset, correctionSkew

    def computeInitialOffsetsSkews(self, nNodes, nSamples, 
                                   initOffsetValue, initSkewValue,
                                   offsetPerturb=2e+5, # us
                                   skewPerturb=25 # ppm
                                   ):

        # repeat and reshape to obtain the offsets and skews                                          
        fixedOffset = np.repeat(initOffsetValue, nSamples*nNodes, axis = 0)     
        
        fixedSkew = np.repeat(initSkewValue, nSamples*nNodes, axis = 0)     
        
        # generate the noises uniformly
        perturbOffset = np.random.uniform(low = -offsetPerturb,
                                          high = offsetPerturb,
                                          size = nSamples*nNodes)

        perturbSkew = np.random.uniform(low = -skewPerturb,
                                        high = skewPerturb,
                                        size = nSamples*nNodes)
        
        # compute the initial clock offsets and skews            
        initOffset = fixedOffset + perturbOffset # nSamples x nNodes     
        initSkew = fixedSkew + perturbSkew # nSamples x nNodes
               
        # reshape
        initOffset = initOffset.reshape(nSamples, nNodes)        
        initSkew = initSkew.reshape(nSamples, nNodes)                
                
        # add the extra feature=1 dimensions
        initOffset = np.expand_dims(initOffset, 1) # nSamples x 1 x nNodes
        initSkew = np.expand_dims(initSkew, 1) # nSamples x 1 x nNodes
        
        return initOffset, initSkew
    
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
            
