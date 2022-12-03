import numpy as np
import torch
import torch.nn as nn

import utils.graphML as gml

zeroTolerance = 1e-9 # Absolute values below this number are considered zero.

class LocalGNN_DB(nn.Module):
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