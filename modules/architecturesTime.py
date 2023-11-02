import torch.nn as nn
import numpy as np
import utils.graphML as gml

zeroTolerance = 1e-9 # values below this number are zero

class LocalGNN_DB(nn.Module):
    def __init__(self,
                 dimNodeSignals, nFilterTaps, bias,
                 nonlinearity,
                 dimReadout,
                 dimEdgeFeatures):
        
        super().__init__() # initialise parent:

        assert len(dimNodeSignals) == len(nFilterTaps) + 1
        
        self.L = len(nFilterTaps) # number of graph filtering layers
        self.F = dimNodeSignals # features
        self.K = nFilterTaps # filter taps
        self.E = dimEdgeFeatures # number of edge features
        self.bias = bias # boolean
        self.sigma = nonlinearity
        self.dimReadout = dimReadout
        
        gfl = []
        gfl.append(gml.GraphFilter_DB(self.F[0], self.F[1], self.K[0], self.E, self.bias))
            
        for l in range(1, self.L):
            gfl.append(self.sigma())
            gfl.append(gml.GraphFilter_DB(self.F[l], self.F[l+1], self.K[l], self.E, self.bias))                

        self.GFL = nn.Sequential(*gfl) # graph filtering layers

        fc = []
        if len(self.dimReadout) > 0:
            fc.append(self.sigma())            
            fc.append(nn.Linear(self.F[-1], dimReadout[0], bias = self.bias))
            for l in range(len(dimReadout)-1):
                fc.append(self.sigma())
                fc.append(nn.Linear(dimReadout[l], dimReadout[l+1], bias = self.bias))

            self.Readout = nn.Sequential(*fc) # readout layers
            
            for i in range(len(self.dimReadout)):
                nn.init.xavier_uniform_(self.Readout[np.int64(2*i+1)].weight)
                nn.init.zeros_(self.Readout[np.int64(2*i+1)].bias)
            
    def gflLayerWiseInit(self, layerWiseStructure):        
        self.GFL = nn.Sequential(*layerWiseStructure)
        
    def readoutLayerWiseInit(self, readoutStructure):        
        self.Readout = nn.Sequential(*readoutStructure)        

    def splitForward(self, x, S):
        # input: ### 
        # x: B x T x F[0] x N        
        # S: B x T (x E) x N x N
        
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
        
        for l in range(self.L):
            self.GFL[2*l].addGSO(S)

        yGFL = self.GFL(x)
        y = yGFL.permute(0, 1, 3, 2) # B x T x N x F[-1]
        
        if 'self.Readout' in globals():
            y = self.Readout(y) # B x T x N x dimReadout[-1]
        
        return y.permute(0, 1, 3, 2), yGFL 
                      # B x T x dimReadout[-1] x N, B x T x dimFeatures[-1] x N
    
    def forward(self, x, S):        
        output, _ = self.splitForward(x, S)
        
        return output