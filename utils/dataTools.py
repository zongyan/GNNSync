import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'

import numpy as np
import torch

import utils.graphTools as graph

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

class _data:
    # Internal supraclass from which all data sets will inherit.
    # There are certain methods that all Data classes must have:
    #   getSamples(), expandDims(), to() and astype().
    # To avoid coding this methods over and over again, we create a class from
    # which the data can inherit this basic methods.
    
    # All the signals are always assumed to be graph signals that are written
    #   nDataPoints (x nFeatures) x nNodes
    # If we have one feature, we have the expandDims() that adds a x1 so that
    # it can be readily processed by architectures/functions that always assume
    # a 3-dimensional signal.
    
    def __init__(self):
        # Minimal set of attributes that all data classes should have
        self.dataType = None
        self.device = None
        self.nTrain = None
        self.nValid = None
        self.nTest = None
        self.samples = {}
        self.samples['train'] = {}
        self.samples['train']['signals'] = None
        self.samples['train']['targets'] = None
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
    
    def expandDims(self):
        
        # For each data set partition
        for key in self.samples.keys():
            # If there's something in them
            if self.samples[key]['signals'] is not None:
                # And if it has only two dimensions
                #   (shape: nDataPoints x nNodes)
                if len(self.samples[key]['signals'].shape) == 2:
                    # Then add a third dimension in between so that it ends
                    # up with shape
                    #   nDataPoints x 1 x nNodes
                    # and it respects the 3-dimensional format that is taken
                    # by many of the processing functions
                    if 'torch' in repr(self.dataType):
                        self.samples[key]['signals'] = \
                                       self.samples[key]['signals'].unsqueeze(1)
                    else:
                        self.samples[key]['signals'] = np.expand_dims(
                                                   self.samples[key]['signals'],
                                                   axis = 1)
                elif len(self.samples[key]['signals'].shape) == 3:
                    if 'torch' in repr(self.dataType):
                        self.samples[key]['signals'] = \
                                       self.samples[key]['signals'].unsqueeze(2)
                    else:
                        self.samples[key]['signals'] = np.expand_dims(
                                                   self.samples[key]['signals'],
                                                   axis = 2)
        
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



            
class Flocking(_data):
    """
    Flocking: Creates synthetic trajectories for the problem of coordinating
        a team of robots to fly together while avoiding collisions. See the
        following  paper for details
        
        E. Tolstaya, F. Gama, J. Paulos, G. Pappas, V. Kumar, and A. Ribeiro, 
        "Learning Decentralized Controllers for Robot Swarms with Graph Neural
        Networks," in Conf. Robot Learning 2019. Osaka, Japan: Int. Found.
        Robotics Res., 30 Oct.-1 Nov. 2019.
    
    Initialization:
        
    Input:
        nAgents (int): Number of agents
        commRadius (float): communication radius (in meters)
        repelDist (float): minimum target separation of agents (in meters)
        nTrain (int): number of training trajectories
        nValid (int): number of validation trajectories
        nTest (int): number of testing trajectories
        duration (float): duration of each trajectory (in seconds)
        samplingTime (float): time between consecutive time instants (in sec)
        initGeometry ('circular', 'rectangular'): initial positioning geometry
            (default: 'circular')
        initVelValue (float): maximum initial velocity (in meters/seconds,
            default: 3.)
        initMinDist (float): minimum initial distance between agents (in
            meters, default: 0.1)
        accelMax (float): maximum possible acceleration (in meters/seconds^2,
            default: 10.)
        normalizeGraph (bool): if True normalizes the communication graph
            adjacency matrix by the maximum eigenvalue (default: True)
        doPrint (bool): If True prints messages (default: True)
        dataType (dtype): datatype for the samples created (default: np.float64)
        device (device): if torch.Tensor datatype is selected, this is on what
            device the data is saved (default: 'cpu')
            
    Methods:
        
    signals, targets = .getSamples(samplesType[, optionalArguments])
        Input:
            samplesType (string): 'train', 'valid' or 'test' to determine from
                which dataset to get the samples from
            optionalArguments:
                0 optional arguments: get all the samples from the specified set
                1 optional argument (int): number of samples to get (at random)
                1 optional argument (list): specific indices of samples to get
        Output:
            signals (dtype.array): numberSamples x 6 x numberNodes
            targets (dtype.array): numberSamples x 2 x numberNodes
            'signals' are the state variables as described in the corresponding
            paper; 'targets' is the 2-D acceleration for each node
            
    cost = .evaluate(vel = None, accel = None, initVel = None,
                     samplingTime = None)
        Input:
            vel (array): velocities; nSamples x tSamples x 2 x nAgents
            accel (array): accelerations; nSamples x tSamples x 2 x nAgents
            initVel (array): initial velocities; nSamples x 2 x nAgents
            samplingTime (float): sampling time
            >> Obs.: Either vel or (accel and initVel) have to be specified
            for the cost to be computed, if all of them are specified, only
            vel is used
        Output:
            cost (float): flocking cost as specified in eq. (13)

    .astype(dataType): change the type of the data matrix arrays.
        Input:
            dataType (dtype): target type of the variables (e.g. torch.float64,
                numpy.float64, etc.)

    .to(device): if dtype is torch.tensor, move them to the specified device.
        Input:
            device (string): target device to move the variables to (e.g. 
                'cpu', 'cuda:0', etc.)

    state = .computeStates(pos, vel, graphMatrix, ['doPrint'])
        Input:
            pos (array): positions; nSamples x tSamples x 2 x nAgents
            vel (array): velocities; nSamples x tSamples x 2 x nAgents
            graphMatrix (array): matrix description of communication graph;
                nSamples x tSamples x nAgents x nAgents
            'doPrint' (bool): optional argument to print outputs; if not used
                uses the same status set for the entire class in the
                initialization
        Output:
            state (array): states; nSamples x tSamples x 6 x nAgents
    
    graphMatrix = .computeCommunicationGraph(pos, commRadius, normalizeGraph,
                    ['kernelType' = 'gaussian', 'weighted' = False, 'doPrint'])
        Input:
            pos (array): positions; nSamples x tSamples x 2 x nAgents
            commRadius (float): communication radius (in meters)
            normalizeGraph (bool): if True normalize adjacency matrix by 
                largest eigenvalue
            'kernelType' ('gaussian'): kernel to apply to the distance in order
                to compute the weights of the adjacency matrix, default is
                the 'gaussian' kernel; other kernels have to be coded, and also
                the parameters of the kernel have to be included as well, in
                the case of the gaussian kernel, 'kernelScale' determines the
                scale (default: 1.)
            'weighted' (bool): if True the graph is weighted according to the
                kernel type; if False, it's just a binary adjacency matrix
            'doPrint' (bool): optional argument to print outputs; if not used
                uses the same status set for the entire class in the
                initialization
        Output:
            graphMatrix (array): adjacency matrix of the communication graph;
                nSamples x tSamples x nAgents x nAgents
    
    thisData = .getData(name, samplesType[, optionalArguments])
        Input:
            name (string): variable name to get (for example, 'pos', 'vel', 
                etc.)
            samplesType ('train', 'test' or 'valid')
            optionalArguments:
                0 optional arguments: get all the samples from the specified set
                1 optional argument (int): number of samples to get (at random)
                1 optional argument (list): specific indices of samples to get
        Output:
            thisData (array): specific type of data requested
    
    pos, vel[, accel, state, graph] = computeTrajectory(initPos, initVel,
                                            duration[, 'archit', 'accel',
                                            'doPrint'])
        Input:
            initPos (array): initial positions; nSamples x 2 x nAgents
            initVel (array): initial velocities; nSamples x 2 x nAgents
            duration (float): duration of trajectory (in seconds)
            Optional arguments: (either 'accel' or 'archit' have to be there)
            'archit' (nn.Module): torch architecture that computes the output
                from the states
            'accel' (array): accelerations; nSamples x tSamples x 2 x nAgents
            'doPrint' (bool): optional argument to print outputs; if not used
                uses the same status set for the entire class in the
                initialization
        Output:
            pos (array): positions; nSamples x tSamples x 2 x nAgents
            vel (array): velocities; nSamples x tSamples x 2 x nAgents
            Optional outputs (only if 'archit' was used)
            accel (array): accelerations; nSamples x tSamples x 2 x nAgents
            state (array): state; nSamples x tSamples x 6 x nAgents
            graph (array): adjacency matrix of communication graph;
                nSamples x tSamples x nAgents x nAgents
            
    uDiff, uDistSq = .computeDifferences (u):
        Input:
            u (array): nSamples (x tSamples) x 2 x nAgents
        Output:
            uDiff (array): pairwise differences between the agent entries of u;
                nSamples (x tSamples) x 2 x nAgents x nAgents
            uDistSq (array): squared distances between agent entries of u;
                nSamples (x tSamples) x nAgents x nAgents
    
    pos, vel, accel = .computeOptimalTrajectory(initPos, initVel, duration, 
                                                samplingTime, repelDist,
                                                accelMax = 100.)
        Input:
            initPos (array): initial positions; nSamples x 2 x nAgents
            initVel (array): initial velocities; nSamples x 2 x nAgents
            duration (float): duration of trajectory (in seconds)
            samplingTime (float): time elapsed between consecutive time 
                instants (in seconds)
            repelDist (float): minimum desired distance between agents (in m)
            accelMax (float, default = 100.): maximum possible acceleration
        Output:
            pos (array): positions; nSamples x tSamples x 2 x nAgents
            vel (array): velocities; nSamples x tSamples x 2 x nAgents
            accel (array): accelerations; nSamples x tSamples x 2 x nAgents
            
    initPos, initVel = .computeInitialPositions(nAgents, nSamples, commRadius,
                                                minDist = 0.1,
                                                geometry = 'rectangular',
                                                xMaxInitVel = 3.,
                                                yMaxInitVel = 3.)
        Input:
            nAgents (int): number of agents
            nSamples (int): number of sample trajectories
            commRadius (float): communication radius (in meters)
            minDist (float): minimum initial distance between agents (in m)
            geometry ('rectangular', 'circular'): initial geometry
            xMaxInitVel (float): maximum velocity in the x-axis
            yMaxInitVel (float): maximum velocity in the y-axis
        Output:
            initPos (array): initial positions; nSamples x 2 x nAgents
            initVel (array): initial velocities; nSamples x 2 x nAgents
    
    .saveVideo(saveDir, pos, [, optionalArguments], commGraph = None,
               [optionalKeyArguments])
        Input:
            saveDir (os.path, string): directory where to save the trajectory
                videos
            pos (array): positions; nSamples x tSamples x 2 x nAgents
            optionalArguments:
                0 optional arguments: get all the samples from the specified set
                1 optional argument (int): number of samples to get (at random)
                1 optional argument (list): specific indices of samples to get
            commGraph (array): adjacency matrix of communication graph;
                nSamples x tSamples x nAgents x nAgents
                if not None, then this array is used to produce snapshots of
                the video that include the communication graph at that time
                instant
            'doPrint' (bool): optional argument to print outputs; if not used
                uses the same status set for the entire class in the
                initialization
            'videoSpeed' (float): how faster or slower the video is reproduced
                (default: 1.)
            'showVideoSpeed' (bool): if True shows the legend with the video
                speed in the video; by default it will show it whenever the
                video speed is different from 1.
            'vel' (array): velocities; nSamples x tSamples x 2 x nAgents
            'showCost' (bool): if True and velocities are set, the snapshots
                will show the instantaneous cost (default: True)
            'showArrows' (bool): if True and velocities are set, the snapshots
                will show the arrows of the velocities (default: True)
            
            
    """
    
    def __init__(self, nAgents, commRadius, repelDist,
                 nTrain, nValid, nTest,
                 duration, samplingTime,
                 initGeometry = 'circular',initVelValue = 3.,initMinDist = 0.1,
                 accelMax = 10.,
                 normalizeGraph = True, doPrint = True,
                 dataType = np.float64, device = 'cpu'):
        
        # Initialize parent class
        super().__init__()
        # Save the relevant input information
        #   Number of nodes
        self.nAgents = nAgents
        self.commRadius = commRadius
        self.repelDist = repelDist
        #   Number of samples
        self.nTrain = nTrain
        self.nValid = nValid
        self.nTest = nTest
        nSamples = nTrain + nValid + nTest
        #   Geometry
        self.mapWidth = None
        self.mapHeight = None
        #   Agents
        self.initGeometry = initGeometry
        self.initVelValue = initVelValue
        self.initMinDist = initMinDist
        self.accelMax = accelMax
        #   Duration of the trajectory
        self.duration = float(duration)
        self.samplingTime = samplingTime
        #   Data
        self.normalizeGraph = normalizeGraph
        self.dataType = dataType
        self.device = device
        #   Options
        self.doPrint = doPrint
        
        #   Places to store the data
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
         
        if self.doPrint:
            print("\tComputing initial conditions...", end = ' ', flush = True)
        
        # Compute the initial conditions        
        initPosAll, initVelAll, \
            initOffsetAll, initSkewAll = self.computeInitialPositions(
                                          self.nAgents, nSamples, self.commRadius,
                                          minDist = self.initMinDist,
                                          geometry = self.initGeometry,
                                          initOffsetVal=1, # x100 us
                                          initSkewVal=0, # x10 ppm
                                          maxOffset=0.5, # x100 us
                                          maxSkew=5, # x10 ppm                                                                                           
                                          xMaxInitVel = self.initVelValue,
                                          yMaxInitVel = self.initVelValue)
            
        #   Once we have all positions and velocities, we will need to split 
        #   them in the corresponding datasets (train, valid and test)
        self.initOffset = {}
        self.initSkew = {}

        if self.doPrint:
            print("OK", flush = True)
            # Erase the label first, then print it            
            print("\tComputing delays...", end = ' ', flush = True)
            
       # Compute the packet exchange and processing delays
        clockNoiseAll, packetExchangeDelayAll, processingDelayAll = self.computeNoises(
                                                        self.nAgents, nSamples, 
                                                        self.duration, self.samplingTime,
                                                        sigmaMeasureOffsetVal=0, # x100us 
                                                        sigmaProcessOffsetVal=0, # x100us
                                                        # sigmaMeasureOffsetVal=0.003, # x100us 
                                                        # sigmaProcessOffsetVal=0.04, # x100us
                                                        sigmaOffsetVal=0, 
                                                        sigmaSkewVal=0)
                      
        self.clockNoise = {}   
        self.packetExchangeDelay = {}
        self.processingDelay = {}                      
        
        if self.doPrint:
            print("OK", flush = True)
            # Erase the label first, then print it
            print("\tComputing the optimal trajectories...",
                  end=' ', flush=True)
        
        # Compute the optimal trajectory
        posAll, velAll, accelAll, \
            offsetAll, skewAll, adjAll = self.computeOptimalTrajectory(
                                        initPosAll, initVelAll, initOffsetAll, initSkewAll, 
                                        packetExchangeDelayAll, processingDelayAll, clockNoiseAll,
                                        self.duration, self.samplingTime, self.repelDist,
                                        accelMax = self.accelMax)
        
        self.offset = {}
        self.skew = {}
        self.adj = {}
        
        if self.doPrint:
            print("OK", flush = True)
            # Erase the label first, then print it
            print("\tComputing the communication graphs...",
                  end=' ', flush=True)
        
        # Compute communication graph
        commGraphAll = self.computeCommunicationGraph(posAll, self.commRadius,
                                                      self.normalizeGraph)
        
        self.commGraph = {}
        
        if self.doPrint:
            print("OK", flush = True)
            # Erase the label first, then print it
            print("\tComputing the agent states...", end = ' ', flush = True)
        
        # Compute the states
        stateAll = self.computeStates(offsetAll, skewAll, commGraphAll)
        
        self.state = {}
        
        if self.doPrint:
            # Erase the label
            print("OK", flush = True)
        
        # Separate the states into training, validation and testing samples
        # and save them
        #   Training set
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
        #   Validation set
        startSample = self.nTrain
        endSample = self.nTrain + self.nValid
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
        #   Testing set
        startSample = self.nTrain + self.nValid
        endSample = self.nTrain + self.nValid + self.nTest
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
                        
        # Change data to specified type and device
        self.astype(self.dataType)
        self.to(self.device)
        
    def astype(self, dataType):
        
        # Change all other signals to the correct place
        datasetType = ['train', 'valid', 'test']
        for key in datasetType:
            self.initOffset[key] = changeDataType(self.initOffset[key], dataType)
            self.initSkew[key] = changeDataType(self.initSkew[key], dataType)
            self.offset[key] = changeDataType(self.offset[key], dataType)
            self.skew[key] = changeDataType(self.skew[key], dataType)
            self.adj[key] = changeDataType(self.adj[key], dataType)
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
            
    def expandDims(self):
        # Just avoid the 'expandDims' method in the parent class
        pass
        
    def computeStates(self, theta, gamma, graphMatrix, **kwargs):

        # We get the following inputs.
        # offsets: nSamples x tSamples x 1 x nAgents
        # skews: nSamples x tSamples x 1 x nAgents
        # graphMatrix: nSaples x tSamples x nAgents x nAgents
        
        # And we want to build the state, which is a vector of dimension 2 on 
        # each node, that is, the output shape is
        #   nSamples x tSamples x 2 x nAgents
        
        # The print for this one can be settled independently, if not, use the
        # default of the data object
        if 'doPrint' in kwargs.keys():
            doPrint = kwargs['doPrint']
        else:
            doPrint = self.doPrint
                 
        # Check correct dimensions
        assert len(theta.shape) == len(gamma.shape) == len(graphMatrix.shape) == 4
        nSamples = theta.shape[0]
        tSamples = theta.shape[1]
        assert theta.shape[2] == 1
        nAgents = theta.shape[3]
        assert gamma.shape[0] == graphMatrix.shape[0] == nSamples
        assert gamma.shape[1] == graphMatrix.shape[1] == tSamples
        assert gamma.shape[2] == 1
        assert gamma.shape[3] == graphMatrix.shape[2] == graphMatrix.shape[3] == nAgents
                
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
        state = np.zeros((nSamples, tSamples, 2, nAgents))
        
        for b in range(nBatches):
            
            # Pick the batch elements
            thetaBatch = theta[batchIndex[b]:batchIndex[b+1]]
            gammaBatch = gamma[batchIndex[b]:batchIndex[b+1]]            
            graphMatrixBatch = graphMatrix[batchIndex[b]:batchIndex[b+1]]
        
            if tSamples > maxTimeSamples:
                
                # For each time instant
                for t in range(tSamples):
                    
                    # Now, we need to compute the differences, in offsets and in 
                    # skews, for each agent, for each time instant
                    offsetDiff, _ = self.computeDifferences(thetaBatch[:,t,:,:])
                    #   offsetDiff: batchSize[b] x 1 x nAgents x nAgents
                    skewDiff, _ = self.computeDifferences(gammaBatch[:,t,:,:])
                    #   skewDiff: batchSize[b] x 1 x nAgents x nAgents
                    
                    # Next, we need to get ride of all those places where there are
                    # no neighborhoods. That is given by the nonzero elements of the 
                    # graph matrix.
                    graphMatrixTime = (np.abs(graphMatrixBatch[:,t,:,:])>zeroTolerance).astype(theta.dtype)
                    #   graphMatrix: batchSize[b] x nAgents x nAgents
                    
                    # Now we add the extra dimensions so that all the 
                    # multiplications are adequate
                    graphMatrixTime = np.expand_dims(graphMatrixTime, 1)
                    #   graphMatrix: batchSize[b] x 1 x nAgents x nAgents
                    
                    # Then, we can get rid of non-neighbors
                    offsetDiff = offsetDiff * graphMatrixTime # element-wise multiplication 
                    skewDiff = skewDiff * graphMatrixTime # element-wise multiplication
                    
                    # Finally, we can compute the states
                    stateOffset = np.sum(offsetDiff, axis = 3)
                    #   stateOffset : batchSize[b] x 1 x nAgents
                    stateSkew = np.sum(skewDiff, axis = 3)
                    #   stateOffset: batchSize[b] x 1 x nAgents
                    
                    # Concatentate the states and return the result
                    state[batchIndex[b]:batchIndex[b+1],t,:,:] = np.concatenate((stateOffset, stateSkew), axis = 1)
                    #   batchSize[b] x 2 x nAgents
                    
                    if doPrint:
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
                # Now, we need to compute the differences, in offsets and in 
                # skews, for each agent, for each time instante
                offsetDiff, _ = self.computeDifferences(thetaBatch)
                #   posDiff: batchSize[b] x tSamples x 1 x nAgents x nAgents
                skewDiff, _ = self.computeDifferences(gammaBatch)
                #   velDiff: batchSize[b] x tSamples x 1 x nAgents x nAgents
                
                # Next, we need to get ride of all those places where there are
                # no neighborhoods. That is given by the nonzero elements of the 
                # graph matrix.
                graphMatrixBatch = (np.abs(graphMatrixBatch) > zeroTolerance).astype(theta.dtype)
                #   graphMatrix: batchSize[b] x tSamples x nAgents x nAgents
                
                # Now we add the extra dimensions so that all the multiplications
                # are adequate
                graphMatrixBatch = np.expand_dims(graphMatrixBatch, 2)
                #   graphMatrix:batchSize[b] x tSamples x 1 x nAgents x nAgents
                
                # Then, we can get rid of non-neighbors
                offsetDiff = offsetDiff * graphMatrixBatch # element-wise multiplication
                skewDiff = skewDiff * graphMatrixBatch # element-wise multiplication
                
                # Finally, we can compute the states
                stateOffset = np.sum(offsetDiff, axis = 4)
                #   stateOffset: batchSize[b] x tSamples x 1 x nAgents
                stateSkew = np.sum(skewDiff, axis = 4)
                #   stateSkew: batchSize[b] x tSamples x 1 x nAgents
                
                # Concatentate the states and return the result
                state[batchIndex[b]:batchIndex[b+1]] = np.concatenate((stateOffset, stateSkew), axis = 2)
                #   state: batchSize[b] x tSamples x 2 x nAgents

                if doPrint:
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
                                                                                      
        # Print
        if doPrint:
            # Erase the percentage
            print('\b \b' * 4, end = '', flush = True)        
        
        return state
        
    def computeCommunicationGraph(self, pos, commRadius, normalizeGraph,
                                  **kwargs):
        
        # Take in the position and the communication radius, and return the
        # trajectory of communication graphs
        # Input will be of shape
        #   nSamples x tSamples x 2 x nAgents
        # Output will be of shape
        #   nSamples x tSamples x nAgents x nAgents
        
        assert commRadius > 0
        assert len(pos.shape) == 4
        nSamples = pos.shape[0]
        tSamples = pos.shape[1]
        assert pos.shape[2] == 2
        nAgents = pos.shape[3]
        
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
        
        # The print for this one can be settled independently, if not, use the
        # default of the data object
        if 'doPrint' in kwargs.keys():
            doPrint = kwargs['doPrint']
        else:
            doPrint = self.doPrint
                
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
        graphMatrix = np.zeros((nSamples, tSamples, nAgents, nAgents))
        
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
                                    np.arange(0,nAgents),np.arange(0,nAgents)]\
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
                    
                    if doPrint:
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
                                 np.arange(0,nAgents),np.arange(0,nAgents)] =0.
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
                
                if doPrint:
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
                    
        # Print
        if doPrint:
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
        
    def evaluate(self, thetaOffset=None, gammaSkew=None, samplingTime=None):
        
        # It is optional to add a different sampling time, if not, it uses
        # the internal one
        if samplingTime is None:
            # If there's no argument use the internal sampling time
            samplingTime = self.samplingTime    
        
        # Check whether we have thetaOffset and gammaSkew 

        assert thetaOffset is not None
        assert gammaSkew is not None        

        assert len(thetaOffset.shape) == len(gammaSkew.shape) == 4
        nSamples = thetaOffset.shape[0]
        tSamples = thetaOffset.shape[1]
        assert thetaOffset.shape[2] == 1
        nAgents = thetaOffset.shape[3]
       
        assert nSamples == gammaSkew.shape[0]
        assert tSamples == gammaSkew.shape[1]
        assert gammaSkew.shape[2] == 1
        assert nAgents == gammaSkew.shape[3]        
            
        # And now that we have the velocities, we can compute the cost
        if 'torch' in repr(thetaOffset.dtype):            
            # Average clock offset and skew for time t, averaged across nodes
            avgOffset = torch.mean(thetaOffset, dim = 3) # nSamples x tSamples x 1
            avgSkew = torch.mean(gammaSkew*0.1, dim = 3) # nSamples x tSamples x 1            
            # Compute the difference in offset(skew) between each node and the
            # mean offset(skew)
            diffOffset = thetaOffset - avgOffset.unsqueeze(3) 
            diffSkew = gammaSkew*0.1 - avgSkew.unsqueeze(3)             
            #   nSamples x tSamples x 1 x nAgents
            # Compute the summation of clock offset and skew
            diffOffset = torch.sum(diffOffset**2, dim = 2) 
            diffSkew = torch.sum(diffSkew**2, dim = 2)             
            #   nSamples x tSamples x nAgents                        
            # Average over agents
            diffOffsetAvg = torch.mean(diffOffset, dim = 2) # nSamples x tSamples
            diffSkewAvg = torch.mean(diffSkew, dim = 2) # nSamples x tSamples            
            # Sum over time
            costPerSample = torch.sum(diffOffsetAvg, dim = 1) + torch.sum(diffSkewAvg, dim = 1)*samplingTime # nSamples            
            # Final average cost
            cost = torch.mean(costPerSample)            
        else:            
            # Repeat for numpy
            avgOffset = np.mean(thetaOffset, axis = 3) # nSamples x tSamples x 1
            avgSkew = np.mean(gammaSkew*0.1, axis= 3) # nSamples x tSamples x 1               
            diffOffset = thetaOffset - np.tile(np.expand_dims(avgOffset, 3), (1, 1, 1, nAgents))
            diffSkew = gammaSkew*0.1 - np.tile(np.expand_dims(avgSkew, 3), (1, 1, 1, nAgents))
            #   nSamples x tSamples x 1 x nAgents
            diffOffset = np.sum(diffOffset**2, 2)
            diffSkew = np.sum(diffSkew**2, 2)
            #   nSamples x tSamples x nAgents                        
            diffOffsetAvg = np.mean(diffOffset, axis = 2) # nSamples x tSamples
            diffSkewAvg = np.mean(diffSkew, axis = 2) # nSamples x tSamples
            costPerSample = np.sum(diffOffsetAvg, axis = 1) + np.sum(diffSkewAvg, axis = 1)*samplingTime # nSamples
            cost = np.mean(costPerSample) # scalar            
        
        return cost
    
    def computeTrajectory(self, initTheta, initGamma, measureNoise, processNoise, clkNoise, graph, duration, **kwargs):
        
        # Check initOffset is of shape batchSize x 1 x nAgents
        assert len(initTheta.shape) == 3
        batchSize = initTheta.shape[0]
        assert initTheta.shape[1] == 1
        nAgents = initTheta.shape[2]
        
        # Check initSkew is of shape batchSize x 1 x nAgents
        assert len(initGamma.shape) == 3
        assert initGamma.shape[0] == batchSize
        assert initGamma.shape[1] == 1
        assert initGamma.shape[2] == nAgents
        
        # Check graph is of shape batchSize x time x nNodes x nNodes
        assert len(graph.shape) == 4
        assert graph.shape[0] == batchSize
        assert graph.shape[1] == int(duration/self.samplingTime)
        assert graph.shape[2] == nAgents
        assert graph.shape[3] == nAgents       

        # Check what kind of data it is
        #   This is because all the functions are numpy, but if this was
        #   torch, we need to return torch, to make it consistent
        if 'torch' in repr(initTheta.dtype):
            assert 'torch' in repr(initGamma.dtype)
            useTorch = True
            device = initTheta.device
            assert initGamma.device == device
        else:
            useTorch = False
        
        # Create time line
        time = np.arange(0, duration, self.samplingTime)
        tSamples = len(time)
        
        # Here, we have two options, or we're given the acceleration or the
        # architecture
        assert 'archit' in kwargs.keys()
        # Flags to determine which method to use
        useArchit = False
        
        if 'archit' in kwargs.keys():
            archit = kwargs['archit'] # This is a torch.nn.Module architecture
            architDevice = list(archit.parameters())[0].device
            useArchit = True
            
        # Decide on printing or not:
        if 'doPrint' in kwargs.keys():
            doPrint = kwargs['doPrint']
        else:
            doPrint = self.doPrint # Use default      
            
        # Now create the outputs that will be filled afterwards
        theta = np.zeros((batchSize, tSamples, 1, nAgents), dtype = np.float)    
        gamma = np.zeros((batchSize, tSamples, 1, nAgents), dtype = np.float)    
        clock = np.zeros((batchSize, tSamples, 2, nAgents), dtype = np.float)        
        if useArchit:
            adjust = np.zeros((batchSize, tSamples, 2, nAgents), dtype=np.float)
            state = np.zeros((batchSize, tSamples, 2, nAgents), dtype=np.float)
            
        # Assign the initial positions and velocities
        if useTorch:
            clock[:,0,0,:] = initTheta.cpu().numpy().squeeze(1)
            clock[:,0,1,:] = initGamma.cpu().numpy().squeeze(1)            
        else:
            clock[:,0,0,:] = initTheta.copy().squeeze(1)
            clock[:,0,1,:] = initGamma.copy().squeeze(1)            
        
        aMatrix=np.array([[1, self.samplingTime], [0, 1]])
        aMatrix = np.expand_dims(aMatrix, 0)
        aMatrix = np.repeat(aMatrix, batchSize, axis=0)
            
        if doPrint:
            # Sample percentage count
            percentageCount = int(100/tSamples)
            # Print new value
            print("%3d%%" % percentageCount, end = '', flush = True)            

        # Now, let's get started:
        for t in range(1, tSamples):
            
            # If it is architecture-based, we need to compute the state
            if useArchit:                
                # Adjust offset value
                thisOffset = np.expand_dims((clock[:,t-1,0,:]+measureNoise[:,t-1,0,:]), axis=(1, 2))                                                 
                # Obtain graph
                thisGraph = np.expand_dims(graph[:,t-1,:,:], 1)
                # Adjust skew value for state computation
                thisSkew = np.expand_dims((clock[:,t-1,1,:]+measureNoise[:,t-1,1,:]), axis=(1, 2))                
                # Compute state
                thisState = self.computeStates(thisOffset, thisSkew, thisGraph,
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
                    thisAdjust = archit(x, S)
                # Now that we have computed the acceleration, we only care 
                # about the last element in time
                thisAdjust = thisAdjust.cpu().numpy()[:,-1,:,:]
                # And save it
                adjust[:,t-1,:,:] = thisAdjust
                
            # Now that we have the acceleration, we can update position and
            # velocity
            clock[:,t,:,:] = np.matmul(aMatrix, clock[:,t-1,:,:]) + (1/self.nAgents) * adjust[:,t-1,:,:] #+ clkNoise[:,t-1,:,:] - processNoise[:,t-1,:,:] 
            
            if doPrint:
                # Sample percentage count
                percentageCount = int(100*(t+1)/tSamples)
                # Erase previous value and print new value
                print('\b \b' * 4 + "%3d%%" % percentageCount,
                      end = '', flush = True)

        # And we're missing the last values of state and adjust, so
        # let's compute them for completeness
        #   Graph
        thisOffset = np.expand_dims(clock[:,-1,0,:], axis=(1, 2))
        thisGraph = np.expand_dims(graph[:,-1,:,:], 1)
        #   State
        thisSkew = np.expand_dims(clock[:,-1,1,:], axis=(1, 2)) 
        thisState = self.computeStates(thisOffset, thisSkew, thisGraph,
                                       doPrint = False)
        state[:,-1,:,:] = thisState.squeeze(1)
        #   Accel
        x = torch.tensor(state).to(architDevice)
        S = torch.tensor(graph).to(architDevice)
        with torch.no_grad():
            thisAdjust = archit(x, S)
        thisAdjust = thisAdjust.cpu().numpy()[:,-1,:,:]
        # And save it
        adjust[:,-1,:,:] = thisAdjust
               
        theta = np.expand_dims(clock[:,:,0,:], 2).copy()
        gamma = np.expand_dims(clock[:,:,1,:], 2).copy()
        
        # Print
        if doPrint:
            # Erase the percentage
            print('\b \b' * 4, end = '', flush = True)
            
        # After we have finished, turn it back into tensor, if required
        if useTorch:
            theta = torch.tensor(theta).to(device)
            gamma = torch.tensor(gamma).to(device)
            adjust = torch.tensor(adjust).to(device)                                    
            
        # And return it
        return theta, gamma, adjust, state, graph
    
    def computeTrajectory2(self, initTheta, initGamma, graph, duration, **kwargs):
        
        # Check initOffset is of shape batchSize x 1 x nAgents
        assert len(initTheta.shape) == 3
        batchSize = initTheta.shape[0]
        assert initTheta.shape[1] == 1
        nAgents = initTheta.shape[2]
        
        # Check initSkew is of shape batchSize x 1 x nAgents
        assert len(initGamma.shape) == 3
        assert initGamma.shape[0] == batchSize
        assert initGamma.shape[1] == 1
        assert initGamma.shape[2] == nAgents
        
        # Check graph is of shape batchSize x time x nNodes x nNodes
        assert len(graph.shape) == 4
        assert graph.shape[0] == batchSize
        assert graph.shape[1] == int(duration/self.samplingTime)
        assert graph.shape[2] == nAgents
        assert graph.shape[3] == nAgents       

        # Check what kind of data it is
        #   This is because all the functions are numpy, but if this was
        #   torch, we need to return torch, to make it consistent
        if 'torch' in repr(initTheta.dtype):
            assert 'torch' in repr(initGamma.dtype)
            useTorch = True
            device = initTheta.device
            assert initGamma.device == device
        else:
            useTorch = False
        
        # Create time line
        time = np.arange(0, duration, self.samplingTime)
        tSamples = len(time)
        
        # Here, we have two options, or we're given the acceleration or the
        # architecture
        assert 'archit' in kwargs.keys() or 'adjust' in kwargs.keys()
        # Flags to determine which method to use
        useArchit = False
        useAdjust = False         
        
        if 'archit' in kwargs.keys():
            archit = kwargs['archit'] # This is a torch.nn.Module architecture
            architDevice = list(archit.parameters())[0].device
            useArchit = True
        elif 'adjust' in kwargs.keys():
            adjust = kwargs['adjust']
            # accel has to be of shape batchSize x tSamples x 2 x nAgents
            assert len(adjust.shape) == 4
            assert adjust.shape[0] == batchSize
            assert adjust.shape[1] == tSamples
            assert adjust.shape[2] == 2
            assert adjust.shape[3] == nAgents
            if useTorch:
                assert 'torch' in repr(adjust.dtype)
            useAdjust = True
            
        # Decide on printing or not:
        if 'doPrint' in kwargs.keys():
            doPrint = kwargs['doPrint']
        else:
            doPrint = self.doPrint # Use default      
            
        # Now create the outputs that will be filled afterwards
        theta = np.zeros((batchSize, tSamples, 1, nAgents), dtype = np.float)
        gamma = np.zeros((batchSize, tSamples, 1, nAgents), dtype = np.float)
        if useArchit:
            adjust = np.zeros((batchSize, tSamples, 2, nAgents), dtype=np.float)
            state = np.zeros((batchSize, tSamples, 2, nAgents), dtype=np.float)
            
        # Assign the initial positions and velocities
        if useTorch:
            theta[:,0,:,:] = initTheta.cpu().numpy()
            gamma[:,0,:,:] = initGamma.cpu().numpy()
            if useAdjust:
                adjust = adjust.cpu().numpy()
        else:
            theta[:,0,:,:] = initTheta.copy()
            gamma[:,0,:,:] = initGamma.copy()
            
        if doPrint:
            # Sample percentage count
            percentageCount = int(100/tSamples)
            # Print new value
            print("%3d%%" % percentageCount, end = '', flush = True)            

        # Now, let's get started:
        for t in range(1, tSamples):
            
            # If it is architecture-based, we need to compute the state
            if useArchit:
                # Adjust offset value
                thisOffset = np.expand_dims(theta[:,t-1,:,:], 1)
                # Obtain graph
                thisGraph = np.expand_dims(graph[:,t-1,:,:], 1)
                # Adjust skew value for state computation
                thisSkew = np.expand_dims(gamma[:,t-1,:,:], 1)
                # Compute state
                thisState = self.computeStates(thisOffset, thisSkew, thisGraph,
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
                    thisAdjust = archit(x, S)
                # Now that we have computed the acceleration, we only care 
                # about the last element in time
                thisAdjust = thisAdjust.cpu().numpy()[:,-1,:,:]
                # And save it
                adjust[:,t-1,:,:] = thisAdjust
                
            # Now that we have the acceleration, we can update position and
            # velocity
            theta[:,t,:,:] = theta[:,t-1,:,:] + (1/nAgents) * np.expand_dims(adjust[:,t-1,0,:], 1) + gamma[:,t-1,:,:] * self.samplingTime                       
            gamma[:,t,:,:] = gamma[:,t-1,:,:] + (1/nAgents) * np.expand_dims(adjust[:,t-1,1,:], 1)
            
            if doPrint:
                # Sample percentage count
                percentageCount = int(100*(t+1)/tSamples)
                # Erase previous value and print new value
                print('\b \b' * 4 + "%3d%%" % percentageCount,
                      end = '', flush = True)

        # And we're missing the last values of state and adjust, so
        # let's compute them for completeness
        #   Graph
        thisOffset = np.expand_dims(theta[:,-1,:,:], 1)
        thisGraph = np.expand_dims(graph[:,-1,:,:], 1)
        #   State
        thisSkew = np.expand_dims(gamma[:,-1,:,:], 1)
        thisState = self.computeStates(thisOffset, thisSkew, thisGraph,
                                       doPrint = False)
        state[:,-1,:,:] = thisState.squeeze(1)
        #   Accel
        x = torch.tensor(state).to(architDevice)
        S = torch.tensor(graph).to(architDevice)
        with torch.no_grad():
            thisAdjust = archit(x, S)
        thisAdjust = thisAdjust.cpu().numpy()[:,-1,:,:]
        # And save it
        adjust[:,-1,:,:] = thisAdjust
                
        # Print
        if doPrint:
            # Erase the percentage
            print('\b \b' * 4, end = '', flush = True)
            
        # After we have finished, turn it back into tensor, if required
        if useTorch:
            theta = torch.tensor(theta).to(device)
            gamma = torch.tensor(gamma).to(device)
            adjust = torch.tensor(adjust).to(device)
            
        # And return it
        if useArchit:
            return theta, gamma, adjust, state, graph
        elif useAdjust:
            return theta, gamma    
    
    def computeDifferences(self, u):
        
        # Takes as input a tensor of shape
        #   nSamples x tSamples x 2 x nAgents
        # or of shape
        #   nSamples x tSamples x 1 x nAgents
        # or of shape
        #   nSamples x 2 x nAgents
        # or of shape
        #   nSamples x 1 x nAgents                
        # And returns the elementwise difference u_i - u_j of shape
        #   nSamples (x tSamples) x 2 x nAgents x nAgents
        # or of shape
        #   nSamples (x tSamples) x 1 x nAgents x nAgents                
        # And the distance squared ||u_i - u_j||^2 of shape
        #   nSamples (x tSamples) x nAgents x nAgents
        
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
        #   nSamples x tSamples x 2 x nAgents
        nSamples = u.shape[0]
        tSamples = u.shape[1]
        uFeatureDim = u.shape[2]
        nAgents = u.shape[3]
        
        if uFeatureDim == 2:
            # Compute the difference along each axis. For this, we subtract a
            # column vector from a row vector. The difference tensor on each
            # position will have shape nSamples x tSamples x nAgents x nAgents
            # and then we add the extra dimension to concatenate and obtain a final
            # tensor of shape nSamples x tSamples x 2 x nAgents x nAgents
            # First, axis x
            #   Reshape as column and row vector, respectively
            uCol_x = u[:,:,0,:].reshape((nSamples, tSamples, nAgents, 1))
            uRow_x = u[:,:,0,:].reshape((nSamples, tSamples, 1, nAgents))
            #   Subtract them
            uDiff_x = uCol_x - uRow_x # nSamples x tSamples x nAgents x nAgents
            # Second, for axis y
            uCol_y = u[:,:,1,:].reshape((nSamples, tSamples, nAgents, 1))
            uRow_y = u[:,:,1,:].reshape((nSamples, tSamples, 1, nAgents))
            uDiff_y = uCol_y - uRow_y # nSamples x tSamples x nAgents x nAgents
            # Third, compute the distance tensor of shape
            #   nSamples x tSamples x nAgents x nAgents
            uDistSq = uDiff_x ** 2 + uDiff_y ** 2
            # Finally, concatenate to obtain the tensor of differences
            #   Add the extra dimension in the position
            uDiff_x = np.expand_dims(uDiff_x, 2)
            uDiff_y = np.expand_dims(uDiff_y, 2)
            #   And concatenate them
            uDiff = np.concatenate((uDiff_x, uDiff_y), 2)
            #   nSamples x tSamples x 2 x nAgents x nAgents                                    
        elif uFeatureDim == 1: 
            # Compute the difference along each axis. For this, we subtract a
            # column vector from a row vector. The difference tensor on each
            # position will have shape nSamples x tSamples x nAgents x nAgents
            # and then we add the extra dimension to concatenate and obtain a final
            # tensor of shape nSamples x tSamples x 1 x nAgents x nAgents
            # First,
            #   Reshape as column and row vector, respectively
            uCol = u[:,:,0,:].reshape((nSamples, tSamples, nAgents, 1))
            uRow = u[:,:,0,:].reshape((nSamples, tSamples, 1, nAgents))
            #   Subtract them
            uDiff = uCol - uRow # nSamples x tSamples x nAgents x nAgents
            # Second, compute the distance tensor of shape
            #   nSamples x tSamples x nAgents x nAgents
            uDistSq = uDiff ** 2 
            # Finally, add the extra dimension in the position
            uDiff = np.expand_dims(uDiff, 2)
            #   nSamples x tSamples x 1 x nAgents x nAgents                                    
        else:
            raise Exception("unexpected feature dimension is found!")
                
        # Get rid of the time dimension if we don't need it
        if not hasTimeDim:
            # (This fails if tSamples > 1)
            uDistSq = uDistSq.squeeze(1)
            #   nSamples x nAgents x nAgents
            uDiff = uDiff.squeeze(1)
            #   nSamples x 2 x nAgents x nAgents            

        return uDiff, uDistSq
        
    def computeOptimalTrajectory(self, initPos, initVel, initOffset, initSkew, 
                                 measureNoise, processNoise, clkNoise, duration, samplingTime, 
                                 repelDist, accelMax = 100.):
        
        # The optimal trajectory is given by
        # u_{i} = - \sum_{j=1}^{N} (v_{i} - v_{j})
        #         + 2 \sum_{j=1}^{N} (r_{i} - r_{j}) *
        #                                 (1/\|r_{i}\|^{4} + 1/\|r_{j}\|^{2}) *
        #                                 1{\|r_{ij}\| < R}
        # for each agent i=1,...,N, where v_{i} is the velocity and r_{i} the
        # position.

        # Check that initPos and initVel as nSamples x 2 x nAgents arrays
        assert len(initPos.shape) == len(initVel.shape) == 3
        nSamples = initPos.shape[0]
        assert initPos.shape[1] == initVel.shape[1] == 2
        nAgents = initPos.shape[2]
        assert initVel.shape[0] == nSamples
        assert initVel.shape[2] == nAgents

        # Check that initOffset and initSkew as nSamples x 1 x nAgents arrays
        assert len(initOffset.shape) == len(initSkew.shape) == 3
        assert nSamples == initOffset.shape[0]
        assert initOffset.shape[1] == initSkew.shape[1] == 1
        assert nAgents == initOffset.shape[2]
        assert initSkew.shape[0] == nSamples
        assert initSkew.shape[2] == nAgents
       
        # time
        time = np.arange(0, duration, samplingTime)
        tSamples = len(time) # number of time samples
        
        # Create arrays to store the trajectory
        pos = np.zeros((nSamples, tSamples, 2, nAgents))
        vel = np.zeros((nSamples, tSamples, 2, nAgents))
        accel = np.zeros((nSamples, tSamples, 2, nAgents))        
        theta = np.zeros((nSamples, tSamples, 1, nAgents)) # offset 
        gamma = np.zeros((nSamples, tSamples, 1, nAgents)) # skew 
        deltaTheta = np.zeros((nSamples, tSamples, 1, nAgents)) # offset adjustment        
        deltaGamma = np.zeros((nSamples, tSamples, 1, nAgents)) # skew adjustment               
        
        # Initial settings
        pos[:,0,:,:] = initPos
        vel[:,0,:,:] = initVel        
        theta[:,0,:,:] = initOffset
        gamma[:,0,:,:] = initSkew        
        
        if self.doPrint:
            # Sample percentage count
            percentageCount = int(100/tSamples)
            # Print new value
            print("%3d%%" % percentageCount, end = '', flush = True)        

        # For each time instant
        for t in range(1,tSamples):
            
            ### Compute the optimal acceleration ###
            #   Compute the distance between all elements (positions)
            ijDiffPos, ijDistSq = self.computeDifferences(pos[:,t-1,:,:])
            #       ijDiffPos: nSamples x 2 x nAgents x nAgents
            #       ijDistSq:  nSamples x nAgents x nAgents
            #   And also the difference in velocities
            ijDiffVel, _ = self.computeDifferences(vel[:,t-1,:,:])
            #       ijDiffVel: nSamples x 2 x nAgents x nAgents
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
            
            ### Compute the optimal clock offset and skew correction values ###
            #   Compute the distance between all elements (offsets)
            ijDiffOffset, _ = self.computeDifferences(theta[:,t-1,:,:] + np.expand_dims(measureNoise[:,t-1,0,:], 1))
            #       ijDiffOffset: nSamples x 1 x nAgents x nAgents
            #   And also the difference in skews
            ijDiffSkew, _ = self.computeDifferences(gamma[:,t-1,:,:] + np.expand_dims(measureNoise[:,t-1,1,:], 1))
            #       ijDiffVel: nSamples x 1 x nAgents x nAgents
            #   Compute the clock offset and skew correction
            deltaTheta[:,t-1,:,:] = -0.5*np.sum(ijDiffOffset, axis = 3)                                
            deltaGamma[:,t-1,:,:] = -0.5*np.sum(ijDiffSkew, axis = 3)                  

            ### Update the values ###
            #   Update velocity
            vel[:,t,:,:] = vel[:,t-1,:,:] + accel[:,t-1,:,:] * samplingTime
            #   Update the position
            pos[:,t,:,:] = pos[:,t-1,:,:] + vel[:,t-1,:,:] * samplingTime + accel[:,t-1,:,:] * (samplingTime ** 2)/2 

            #   Update the clock offset
            theta[:,t,:,:] = theta[:,t-1,:,:] + gamma[:,t-1,:,:] * self.samplingTime \
                                              + (1/self.nAgents) * deltaTheta[:,t-1,:,:] #\
                                              #+ np.expand_dims(clkNoise[:,t-1,0,:], 1) \
                                              #- np.expand_dims(processNoise[:,t-1,0,:], 1)
            #   Update the clock skew
            gamma[:,t,:,:] = gamma[:,t-1,:,:] + (1/self.nAgents) * deltaGamma[:,t-1,:,:] #\
                                              #+ np.expand_dims(clkNoise[:,t-1,1,:], 1)
            
            if self.doPrint:
                # Sample percentage count
                percentageCount = int(100*(t+1)/tSamples)
                # Erase previous pecentage and print new value
                print('\b \b' * 4 + "%3d%%" % percentageCount,
                      end = '', flush = True)
        
        clockCorrection = np.concatenate((deltaTheta,deltaGamma),axis=2)
        
        # Print
        if self.doPrint:
            # Erase the percentage
            print('\b \b' * 4, end = '', flush = True)

        return pos, vel, accel, theta, gamma, clockCorrection

    def computeNoises(self, nAgents, nSamples, duration, samplingTime, 
                      sigmaMeasureOffsetVal=0.003, sigmaProcessOffsetVal=0.04,
                      sigmaOffsetVal=0, sigmaSkewVal=0):

        time = np.int64(duration/samplingTime)

        sigmaOffsetSq = sigmaOffsetVal**2 # 10000 us^2
        sigmaSkewSq = sigmaSkewVal**2 # 100 ppm^2          
                
        sigmaMeasureOffsetSq = sigmaMeasureOffsetVal**2 # 10000 us^2
        sigmaMeasureSkewSq = ((sigmaMeasureOffsetVal)**2)*100 # 100 ppm^2   
        sigmaProcessOffsetSq = sigmaProcessOffsetVal**2 # 10000 us^2        
        
        # covariance matrix M for the clock offse and skew noises
        covM = np.array([[sigmaOffsetSq, 0], \
                         [0, sigmaSkewSq]], self.dataType)                     
        
        # covariance matrix R for the packet exchange delay
        covR = np.array([[sigmaMeasureOffsetSq, sigmaMeasureOffsetSq*10], \
                         [sigmaMeasureOffsetSq*10, 2*sigmaMeasureSkewSq]], self.dataType)
            
        # covariance matrix Q for the processing delay, occuring on the offset
        covQ = np.array([[sigmaProcessOffsetSq, 0], \
                         [0, 0]], self.dataType)         
            
        # zero mean value for both packet exchange and processing delays
        clockMeanVal = np.array([0, 0], self.dataType)         
        measureMeanVal = np.array([0, 0], self.dataType)
        processingMeanVal = np.array([0, 0], self.dataType)         
            
        from numpy.random import default_rng
        rng = default_rng()
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
    
    # def computeInitialConditions
    def computeInitialPositions(self, nAgents, nSamples, commRadius,
                                minDist=0.1, geometry='circular',
                                initOffsetVal=1., initSkewVal=2.5,
                                maxOffset=0.5, maxSkew=2.5,                                
                                **kwargs):        
                
        assert geometry == 'circular'
        assert minDist * (1.+zeroTolerance) <= commRadius * (1.-zeroTolerance)
        # We use a zeroTolerance buffer zone, just in case
        minDist = minDist * (1. + zeroTolerance)
        commRadius = commRadius * (1. - zeroTolerance)
        
        #### We first generate the position information of UAVs ####       
        # Radius for the grid
        rFixed = (commRadius + minDist)/2.
        rPerturb = (commRadius - minDist)/4.
        fixedRadius = np.arange(0, rFixed*nAgents, step=rFixed) + rFixed        
        
        # Angles for the grid
        aFixed = (commRadius/fixedRadius + minDist/fixedRadius)/2.
        for a in range(len(aFixed)):
            # How many times does aFixed[a] fits within 2pi?
            nAgentsPerCircle = 2 * np.pi // aFixed[a]
            # And now divide 2*np.pi by this number
            aFixed[a] = 2 * np.pi / nAgentsPerCircle
        #   Fixed angle difference for each value of fixedRadius

        # Now, let's get the radius, angle coordinates for each agents
        initRadius = np.empty((0))
        initAngles = np.empty((0))
        agentsSoFar = 0 # Number of agents located so far
        n = 0 # Index for radius
        while agentsSoFar < nAgents:
            thisRadius = fixedRadius[n]
            thisAngles = np.arange(0, 2*np.pi, step=aFixed[n])
            agentsSoFar += len(thisAngles)
            initRadius = np.concatenate((initRadius,
                                         np.repeat(thisRadius, len(thisAngles))))
            initAngles = np.concatenate((initAngles, thisAngles))
            n += 1
            assert len(initRadius) == agentsSoFar
            
        # Restrict to the number of agents we need
        initRadius = initRadius[0:nAgents]
        initAngles = initAngles[0:nAgents]            

        # Add the number of samples
        initRadius = np.repeat(np.expand_dims(initRadius, 0), nSamples, axis=0)
        initAngles = np.repeat(np.expand_dims(initAngles, 0), nSamples, axis=0)
        
        # Add the noise
        #   First, to the angles
        for n in range(nAgents):
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
                                        size = (nSamples, nAgents))
        
        # And finally, get the positions in the cartesian coordinates
        initPos = np.zeros((nSamples, 2, nAgents))
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
        
        #   Now the number of neighbors
        graphMatrix = self.computeCommunicationGraph(np.expand_dims(initPos,1),
                                                     self.commRadius,
                                                     False,
                                                     doPrint = False)
        graphMatrix = graphMatrix.squeeze(1) # nSamples x nAgents x nAgents  
        
        #   Binarize the matrix
        graphMatrix = (np.abs(graphMatrix) > zeroTolerance)\
                                                         .astype(initPos.dtype)
        
        #   And check that we always have initially connected graphs
        for n in range(nSamples):
            assert graph.isConnected(graphMatrix[n,:,:])

        #### We then generate the velocity information of UAVs ####                       
        # Velocities can be either positive or negative, so we do not need 
        # to determine the lower and higher, just around zero
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
                                     size = (nSamples, 1, nAgents))
        yInitVel = np.random.uniform(low = -yMaxInitVel, high = yMaxInitVel,
                                     size = (nSamples, 1, nAgents))
        # Add bias
        xVelBias = np.random.uniform(low = -xMaxInitVel, high = xMaxInitVel,
                                     size = (nSamples))
        yVelBias = np.random.uniform(low = -yMaxInitVel, high = yMaxInitVel,
                                     size = (nSamples))
        
        # And concatenate them
        velBias = np.concatenate((xVelBias, yVelBias)).reshape((nSamples,2,1))
        initVel = np.concatenate((xInitVel, yInitVel), axis = 1) + velBias
        #   nSamples x 2 x nAgents            
        
        #### We next generate the time information of UAVs ####                                
        # Let's start by setting the fixed offset and skew 
        offsetFixed = np.repeat(initOffsetVal, nSamples*nAgents, axis = 0)             
        skewFixed = np.repeat(initSkewVal, nSamples*nAgents, axis = 0)     
                
        # Add the noise
        offsetPerturb = np.random.uniform(low = -maxOffset,
                                          high = maxOffset,
                                          size = nSamples*nAgents)

        skewPerturb = np.random.uniform(low = -maxSkew,
                                        high = maxSkew,
                                        size = nSamples*nAgents)
        
        # Finally, get the initial offsets and skews 
        initOffset = offsetFixed + offsetPerturb # nSamples x nNodes     
        initSkew = skewFixed + skewPerturb # nSamples x nNodes        
        
        # And reshape them 
        initOffset = initOffset.reshape(nSamples, nAgents)        
        initSkew = initSkew.reshape(nSamples, nAgents)    

        # Add the extra feature=1 dimensions
        initOffset = np.expand_dims(initOffset, 1) # nSamples x 1 x nNodes
        initSkew = np.expand_dims(initSkew, 1) # nSamples x 1 x nNodes       
                          
        return initPos, initVel, initOffset, initSkew        