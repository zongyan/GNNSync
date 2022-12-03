import numpy as np
import torch
import torch.nn as nn

import utils.graphML as gml

zeroTolerance = 1e-9 # Absolute values below this number are considered zero.

class LocalGNN_DB(nn.Module):
    """
    LocalGNN_DB: implement the local GNN architecture where all operations are
        implemented locally, i.e. by means of neighboring exchanges only. More
        specifically, it has graph convolutional layers, but the readout layer,
        instead of being an MLP for the entire graph signal, it is a linear
        combination of the features at each node. It considers signals
        that change in time with batch GSOs.

    Initialization:

        LocalGNN_DB(dimNodeSignals, nFilterTaps, bias, # Graph Filtering
                    nonlinearity, # Nonlinearity
                    dimReadout, # Local readout layer
                    dimEdgeFeatures) # Structure

        Input:
            /** Graph convolutional layers **/
            dimNodeSignals (list of int): dimension of the signals at each layer
                (i.e. number of features at each node, or size of the vector
                 supported at each node)
            nFilterTaps (list of int): number of filter taps on each layer
                (i.e. nFilterTaps-1 is the extent of neighborhoods that are
                 reached, for example K=2 is info from the 1-hop neighbors)
            bias (bool): include bias after graph filter on every layer
            >> Obs.: dimNodeSignals[0] is the number of features (the dimension
                of the node signals) of the data, where dimNodeSignals[l] is the
                dimension obtained at the output of layer l, l=1,...,L.
                Therefore, for L layers, len(dimNodeSignals) = L+1. Slightly
                different, nFilterTaps[l] is the number of filter taps for the
                filters implemented at layer l+1, thus len(nFilterTaps) = L.
                
            /** Activation function **/
            nonlinearity (torch.nn): module from torch.nn non-linear activations
            
            /** Readout layers **/
            dimReadout (list of int): number of output hidden units of a
                sequence of fully connected layers applied locally at each node
                (i.e. no exchange of information involved).
                
            /** Graph structure **/
            dimEdgeFeatures (int): number of edge features

        Output:
            nn.Module with a Local GNN architecture with the above specified
            characteristics that considers time-varying batch GSO and delayed
            signals

    Forward call:

        LocalGNN_DB(x, S)

        Input:
            x (torch.tensor): input data of shape
                batchSize x timeSamples x dimFeatures x numberNodes
            GSO (torch.tensor): graph shift operator; shape
                batchSize x timeSamples (x dimEdgeFeatures)
                                                    x numberNodes x numberNodes

        Output:
            y (torch.tensor): output data after being processed by the GNN; 
                batchSize x timeSamples x dimReadout[-1] x numberNodes
                
    Other methods:
            
        y, yGNN = .splitForward(x, S): gives the output of the entire GNN y,
        which has shape batchSize x timeSamples x dimReadout[-1] x numberNodes,
        as well as the output of all the GNN layers (i.e. before the readout
        layers), yGNN of shape batchSize x timeSamples x dimFeatures[-1]
        x numberNodes. This can be used to isolate the effect of the graph
        convolutions from the effect of the readout layer.
        
        y = .singleNodeForward(x, S, nodes): outputs the value of the last
        layer at a single node. x is the usual input of shape batchSize 
        x timeSamples x dimFeatures x numberNodes. nodes is either a single
        node (int) or a collection of nodes (list or numpy.array) of length
        batchSize, where for each element in the batch, we get the output at
        the single specified node. The output y is of shape batchSize 
        x timeSamples x dimReadout[-1].
    """

    def __init__(self,
                 # Graph filtering
                 dimNodeSignals, nFilterTaps, bias,
                 # Nonlinearity
                 nonlinearity,
                 # MLP in the end
                 dimReadout,
                 # Structure
                 dimEdgeFeatures):
        # Initialize parent:
        super().__init__()
        # dimNodeSignals should be a list and of size 1 more than nFilter taps.
        assert len(dimNodeSignals) == len(nFilterTaps) + 1
        
        # Store the values (using the notation in the paper):
        self.L = len(nFilterTaps) # Number of graph filtering layers
        self.F = dimNodeSignals # Features
        self.K = nFilterTaps # Filter taps
        self.E = dimEdgeFeatures # Number of edge features
        self.bias = bias # Boolean
        # Store the rest of the variables
        self.sigma = nonlinearity
        self.dimReadout = dimReadout
        # And now, we're finally ready to create the architecture:
        #\\\ Graph filtering layers \\\
        # OBS.: We could join this for with the one before, but we keep separate
        # for clarity of code.
        gfl = [] # Graph Filtering Layers
        for l in range(self.L):
            #\\ Graph filtering stage:
            gfl.append(gml.GraphFilter_DB(self.F[l], self.F[l+1], self.K[l],
                                              self.E, self.bias))
            #\\ Nonlinearity
            gfl.append(self.sigma())
        # And now feed them into the sequential
        self.GFL = nn.Sequential(*gfl) # Graph Filtering Layers
        #\\\ MLP (Fully Connected Layers) \\\
        fc = []
        if len(self.dimReadout) > 0: # Maybe we don't want to readout anything
            # The first layer has to connect whatever was left of the graph 
            # filtering stage to create the number of features required by
            # the readout layer
            fc.append(nn.Linear(self.F[-1], dimReadout[0], bias = self.bias))
            # The last linear layer cannot be followed by nonlinearity, because
            # usually, this nonlinearity depends on the loss function (for
            # instance, if we have a classification problem, this nonlinearity
            # is already handled by the cross entropy loss or we add a softmax.)
            for l in range(len(dimReadout)-1):
                # Add the nonlinearity because there's another linear layer
                # coming
                fc.append(self.sigma())
                # And add the linear layer
                fc.append(nn.Linear(dimReadout[l], dimReadout[l+1],
                                    bias = self.bias))
        # And we're done
        self.Readout = nn.Sequential(*fc)
        # so we finally have the architecture.

    def splitForward(self, x, S):

        # Check the dimensions of the input
        #   S: B x T (x E) x N x N
        #   x: B x T x F[0] x N
        assert len(S.shape) == 4 or len(S.shape) == 5
        if len(S.shape) == 4:
            S = S.unsqueeze(2)
        B = S.shape[0]
        T = S.shape[1]
        assert S.shape[2] == self.E
        N = S.shape[3]
        assert S.shape[4] == N
        
        assert len(x.shape) == 4
        assert x.shape[0] == B
        assert x.shape[1] == T
        assert x.shape[2] == self.F[0]
        assert x.shape[3] == N
        
        # Add the GSO at each layer
        for l in range(self.L):
            self.GFL[2*l].addGSO(S)
        # Let's call the graph filtering layer
        yGFL = self.GFL(x)
        # Change the order, for the readout
        y = yGFL.permute(0, 1, 3, 2) # B x T x N x F[-1]
        # And, feed it into the Readout layer
        y = self.Readout(y) # B x T x N x dimReadout[-1]
        # Reshape and return
        return y.permute(0, 1, 3, 2), yGFL
        # B x T x dimReadout[-1] x N, B x T x dimFeatures[-1] x N
    
    def forward(self, x, S):
        
        # Most of the times, we just need the actual, last output. But, since in
        # this case, we also want to compare with the output of the GNN itself,
        # we need to create this other forward funciton that takes both outputs
        # (the GNN and the MLP) and returns only the MLP output in the proper
        # forward function.
        output, _ = self.splitForward(x, S)
        
        return output
    
    def singleNodeForward(self, x, S, nodes):
        
        # x is of shape B x T x F[0] x N
        batchSize = x.shape[0]
        N = x.shape[3]
        
        # nodes is either an int, or a list/np.array of ints of size B
        assert type(nodes) is int \
                or type(nodes) is list \
                or type(nodes) is np.ndarray
        
        # Let us start by building the selection matrix
        # This selection matrix has to be a matrix of shape
        #   B x 1 x N[-1] x 1
        # so that when multiplying with the output of the forward, we get a
        #   B x T x dimRedout[-1] x 1
        # and we just squeeze the last dimension
        
        # TODO: The big question here is if multiplying by a matrix is faster
        # than doing torch.index_select
        
        # Let's always work with numpy arrays to make it easier.
        if type(nodes) is int:
            # Change the node number to accommodate the new order
            nodes = self.order.index(nodes)
            # If it's int, make it a list and an array
            nodes = np.array([nodes], dtype=np.int)
            # And repeat for the number of batches
            nodes = np.tile(nodes, batchSize)
        if type(nodes) is list:
            newNodes = [self.order.index(n) for n in nodes]
            nodes = np.array(newNodes, dtype = np.int)
        elif type(nodes) is np.ndarray:
            newNodes = np.array([np.where(np.array(self.order) == n)[0][0] \
                                                                for n in nodes])
            nodes = newNodes.astype(np.int)
        # Now, nodes is an np.int np.ndarray with shape batchSize
        
        # Build the selection matrix
        selectionMatrix = np.zeros([batchSize, 1, N, 1])
        selectionMatrix[np.arange(batchSize), nodes, 0] = 1.
        # And convert it to a tensor
        selectionMatrix = torch.tensor(selectionMatrix,
                                       dtype = x.dtype,
                                       device = x.device)
        
        # Now compute the output
        y = self.forward(x, S)
        # This output is of size B x T x dimReadout[-1] x N
        
        # Multiply the output
        y = torch.matmul(y, selectionMatrix)
        #   B x T x dimReadout[-1] x 1
        
        # Squeeze the last dimension and return
        return y.squeeze(3)
    
