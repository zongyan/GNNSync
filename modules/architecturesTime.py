import numpy as np
import torch
import torch.nn as nn

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

        gfl = [] # graph filtering layers
        for l in range(self.L):
            gfl.append(gml.GraphFilter_DB(self.F[l], self.F[l+1], self.K[l],
                                              self.E, self.bias))
            gfl.append(self.sigma())
        self.GFL = nn.Sequential(*gfl) # graph filtering layers

        fc = []
        if len(self.dimReadout) > 0: 
            fc.append(nn.Linear(self.F[-1], dimReadout[0], bias = self.bias))
            for l in range(len(dimReadout)-1):
                fc.append(self.sigma())
                fc.append(nn.Linear(dimReadout[l], dimReadout[l+1],
                                    bias = self.bias))
        self.Readout = nn.Sequential(*fc)

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
        y = self.Readout(y) # B x T x N x dimReadout[-1]
        
        return y.permute(0, 1, 3, 2), yGFL 
                      # B x T x dimReadout[-1] x N, B x T x dimFeatures[-1] x N
    
    def forward(self, x, S):        
        output, _ = self.splitForward(x, S)
        
        return output
    
class GraphRecurrentNN_DB(nn.Module):
    def __init__(self, dimInputSignals, dimOutputSignals, dimHiddenSignals,
                 nFilterTaps, bias,
                 nonlinearityHidden, nonlinearityOutput, nonlinearityReadout,
                 dimReadout, dimEdgeFeatures):
        super().__init__()

        # hParamsGraphRNN['dimInputSignals'] = 2 
        # hParamsGraphRNN['dimOutputSignals'] = 2
        # hParamsGraphRNN['dimHiddenSignals'] = 2
        # hParamsGraphRNN['nFilterTaps'] = [1] * 2 
        # hParamsGraphRNN['bias'] = True 
        # hParamsGraphRNN['nonlinearityHidden'] = nonlinearityHidden
        # hParamsGraphRNN['nonlinearityOutput'] = nonlinearity
        # hParamsGraphRNN['nonlinearityReadout'] = nonlinearity
    
        # hParamsGraphRNN['dimReadout'] = [2]
        # hParamsGraphRNN['dimEdgeFeatures'] = 1 # Scalar edge weights
        
        assert len(nFilterTaps) == 2
        
        self.F = dimInputSignals # number of input features = 2
        self.G = dimOutputSignals # number of output features = 2
        self.H = dimHiddenSignals # number of hidden features = 2
        self.K = nFilterTaps # filter taps = 1
        self.E = dimEdgeFeatures # number of edge features = 1
        self.bias = bias # boolean = yes

        self.sigma = nonlinearityHidden # yes
        self.rho = nonlinearityOutput # no
        self.nonlinearityReadout = nonlinearityReadout # no
        self.dimReadout = dimReadout # = 2

        # hidden state
        self.hiddenState = gml.HiddenState_DB(self.F, self.H, self.K[0],
                                       nonlinearity = self.sigma, E = self.E,
                                       bias = self.bias)
        # output state
        self.outputState = gml.OutputState_DB(self.F, self.H, self.G, self.K[1],
                                              E = self.E, bias = self.bias)

        fc = []
        if len(self.dimReadout) > 0: 
            fc.append(nn.Linear(self.G, dimReadout[0], bias = self.bias))
            for l in range(len(dimReadout)-1):
                fc.append(self.nonlinearityReadout())
                fc.append(nn.Linear(dimReadout[l], dimReadout[l+1],
                                    bias = self.bias))

        self.Readout = nn.Sequential(*fc)
        
    def splitForward(self, x, S):
        # input: ### 
        # x: B x T x F x N        
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
        assert x.shape[2] == self.F
        assert x.shape[3] == N
        
        # This can be generated here or generated outside of here, not clear yet
        # what's the most coherent option
        # yan: from the control perspective, this w0 shoud not be initialised to 
        # be zero
        # w0 = torch.randn((B, self.H, N), device = x.device)
        w0 = torch.zeros((B, self.H, N), device = x.device)        
        
        # Add the GSO for each graph filter
        self.hiddenState.addGSO(S)
        self.outputState.addGSO(S)
        
        # Compute the trajectory of hidden states
        z, _ = self.hiddenState(w0, x)
        # Compute the output trajectory from the hidden states and input signals
        yOut = self.outputState(z, x)
        # yOut = self.rho(yOut) # Don't forget the nonlinearity!
        #   B x T x G x N
        # Change the order, for the readout
        y = yOut.permute(0, 1, 3, 2) # B x T x N x G
        # And, feed it into the Readout layer
        y = self.Readout(y) # B x T x N x dimReadout[-1]
        # Reshape and return
        return y.permute(0, 1, 3, 2), yOut
        # B x T x dimReadout[-1] x N, B x T x dimFeatures[-1] x N
    
    def forward(self, x, S):
        output, _ = self.splitForward(x, S)
        
        return output