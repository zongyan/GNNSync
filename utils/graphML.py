import math
import torch
import torch.nn as nn

zeroTolerance = 1e-9 # values below this number are zero
infiniteNumber = 1e12 # infinity not less than this number

# WARNING: Only scalar bias.

def LSIGF_DB(h, S, x, b=None):    
    # input: ###
    # h: F x E x K x G
    # S: B x T x E x N x N
    # x: B x T x G x N
    # b: F x N
    
    # output: ###
    # y: B x T x F x N
    
    assert len(h.shape) == 4
    F = h.shape[0]
    E = h.shape[1]
    K = h.shape[2]
    G = h.shape[3]
    assert len(S.shape) == 5
    B = S.shape[0]
    T = S.shape[1]
    assert S.shape[2] == E
    N = S.shape[3]
    assert S.shape[4] == N
    assert len(x.shape) == 4
    assert x.shape[0] == B
    assert x.shape[1] == T
    assert x.shape[2] == G
    assert x.shape[3] == N
        
    x = x.reshape([B, T, 1, G, N]).repeat(1, 1, E, 1, 1)
    z = x.reshape([B, T, 1, E, G, N]) # k=0, k is counted in dim = 2
    
    for k in range(1,K):
        x, _ = torch.split(x, [T-1, 1], dim = 1)        
        zeroRow = torch.zeros(B, 1, E, G, N, dtype=x.dtype, device=x.device)
        x = torch.cat((zeroRow, x), dim = 1)
        
        x = torch.matmul(x, S)
        xS = x.reshape(B, T, 1, E, G, N)

        z = torch.cat((z, xS), dim = 2) # B x T x K x E x G x N
        
    z = z.permute(0, 1, 5, 3, 2, 4) # B x T x N x E x K x G
    z = z.reshape(B, T, N, E*K*G)

    h = h.reshape(F, E*K*G)
    h = h.permute(1, 0) # E*K*G x F
    
    y = torch.matmul(z, h) # B x T x N x F
    y = y.permute(0, 1, 3, 2) # B x T x F x N

    if b is not None:
        y = y + b
    
    return y
    
class GraphFilter_DB(nn.Module):
    def __init__(self, G, F, K, E = 1, bias = True):
        # K: number of filter taps
        # GSOs will be added later.
        # This combines both weight scalars and weight vectors.
        # Bias will always be shared and scalar.

        super().__init__() # initialize parent

        self.G = G
        self.F = F
        self.K = K
        self.E = E
        self.S = None # No GSO assigned yet

        self.weight = nn.parameter.Parameter(torch.Tensor(F, E, K, G))
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(F, 1))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # taken from _ConvNd initialization of parameters
        stdv = 1. / math.sqrt(self.G * self.K)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def addGSO(self, S):
        # S: B x T x E x N x N        
        assert len(S.shape) == 5
        assert S.shape[2] == self.E
        self.N = S.shape[3]
        assert S.shape[4] == self.N
        self.S = S

    def forward(self, x):
        # input: ###
        # x: batchSize x time x dimInFeatures x numberNodesIn
        # output: ###
        # u: batchSize x time x dimOutFeatures x numberNodes
            
        assert len(x.shape) == 4
        B = x.shape[0]
        assert self.S.shape[0] == B
        T = x.shape[1]
        assert self.S.shape[1] == T
        F = x.shape[2]
        assert x.shape[3] == self.N

        u = LSIGF_DB(self.weight, self.S, x, self.bias)
        
        return u
    
def GRNN_DB(K1, K2, S, x, sigma, xBias=None, wBias = None):
    # input: ###
    # K1: H x H (hidden to hidden filters)    
    # K2: H x E x K x F (input to hidden filters)
    # S: B x T x E x N x N (GSO)
    # x: B x T x F x N (input signal)
    # xBias: 1 x 1 x H x 1 (bias on the input to hidden features)
    # wBias: 1 x 1 x H x 1 (bias on the hidden to hidden features)    
    
    # output: ###
    # z: B x T x H x N (Hidden state signal)    
    
    # Check dimensions
    H = K2.shape[0] # number of hidden state features
    E = K2.shape[1] # number of edge features
    K = K2.shape[2] # number of filter taps
    F = K2.shape[3] # number of input features
    
    assert K1.shape[0] == H
    assert K1.shape[1] == H

    B = S.shape[0]
    T = S.shape[1]
    assert S.shape[2] == E
    N = S.shape[3]
    assert S.shape[4] == N

    assert x.shape[0] == B
    assert x.shape[1] == T
    assert x.shape[2] == F
    assert x.shape[3] == N
            
    '''
    w[t+1] = sigma(K1 w[t] + K2 x[t])
    '''
    XK2 = LSIGF_DB(K2, S, x, b = xBias) # B x T x H x N 

    # now compute the first time instant
    wt = torch.index_select(XK2, 1, torch.tensor(0, device = XK2.device)).reshape(B, H, N)
    w = wt.unsqueeze(1) # B x 1 x H x N
    
    # starting now, we need to multiply this by K1 every time
    for t in range(1,T):
        WK1t = torch.matmul(K1.unsqueeze(0), wt) # [1 x H x H] * [B x H x N]

        if wBias is not None:
            WK1t = WK1t + wBias
        
        # Get the corresponding value of XK2 
        XK2t = torch.index_select(XK2, 1, torch.tensor(0, device = XK2.device)).reshape(B, H, N)        
        wt = WK1t + XK2t # [B x H x N] + [B x H x N]
        
        wt = sigma(wt).unsqueeze(1) # B x 1 x H x N        
        w = torch.cat((w, wt), dim = 1) # B x (t+1) x H x N        
    
    return w # B x T x H x N

class HiddenState_DB(nn.Module):
    def __init__(self, F, H, K, nonlinearity = torch.tanh, E = 1, bias = True):
        # Initialize parent:
        super().__init__()
        
        # Store the values (using the notation in the paper):
        self.F = F # Input Features
        self.H = H # Hidden Features
        self.K = K # Filter taps
        self.E = E # Number of edge features
        self.S = None
        self.bias = bias # Boolean
        self.sigma = nonlinearity # torch.nn.functional
        
        # Create parameters:
        self.weightsK1 = nn.parameter.Parameter(torch.Tensor(H, H))
        self.weightsK2 = nn.parameter.Parameter(torch.Tensor(H, E, K, F))
        if self.bias:
            self.xBias = nn.parameter.Parameter(torch.Tensor(H, 1))
            self.wBias = nn.parameter.Parameter(torch.Tensor(H, 1))
        else:
            self.register_parameter('xBias', None)
            self.register_parameter('wBias', None)
        # Initialize parameters
        self.reset_parameters()        
               
    def reset_parameters(self):
        # Taken from _ConvNd initialization of parameters:
        stdv = 1. / math.sqrt(self.F * self.K)
        self.weightsK1.data.uniform_(-stdv, stdv)
        self.weightsK2.data.uniform_(-stdv, stdv)
        if self.bias:
            self.xBias.data.uniform_(-stdv, stdv)
            self.wBias.data.uniform_(-stdv, stdv)

    def forward(self, x, z0):
        
        assert self.S is not None

        # Input
        #   S: B x T (x E) x N x N
        #   x: B x T x F x N
        #   z0: B x H x N
        # Output
        #   z: B x T x H x N
        
        # Check dimensions
        assert len(x.shape) == 4
        B = x.shape[0]
        assert self.S.shape[0] == B
        T = x.shape[1]
        assert self.S.shape[1] == T
        assert x.shape[2] == self.F
        N = x.shape[3]
        
        assert len(z0.shape) == 3
        assert z0.shape[0] == B
        assert z0.shape[1] == self.H
        assert z0.shape[2] == N
        
        z = GRNN_DB(self.weightsK1, self.weightsK2,
                    self.S, x, self.sigma,
                    xBias = self.xBias, wBias = self.wBias)
        
        zT = torch.index_select(z, 1, torch.tensor(T-1, device = z.device)) 
        # Give out the last one, to be used as starting point if used in
        # succession
        
        return z, zT.unsqueeze(1)
    
    def addGSO(self, S):
        # Every S has 5 dimensions.
        assert len(S.shape) == 5
        # S is of shape B x T x E x N x N
        assert S.shape[2] == self.E
        self.N = S.shape[3]
        assert S.shape[4] == self.N
        self.S = S
    
    def extra_repr(self):
        reprString = "in_features=%d, hidden_features=%d, " % (
                        self.F, self.H) + "filter_taps=%d, " % (
                        self.K) + "edge_features=%d, " % (self.E) +\
                        "bias=%s, " % (self.bias) +\
                        "nonlinearity=%s" % (self.sigma)
        if self.S is not None:
            reprString += "GSO stored"
        else:
            reprString += "no GSO stored"
        return reprString
    
class OutoutState_DB(nn.Module):
    def __init__(self, G, F, K, E = 1, bias = True):
        # K: number of filter taps
        # GSOs will be added later.
        # This combines both weight scalars and weight vectors.
        # Bias will always be shared and scalar.

        super().__init__() # initialize parent

        self.G = G
        self.F = F
        self.K = K
        self.E = E
        self.S = None # No GSO assigned yet

        self.weight = nn.parameter.Parameter(torch.Tensor(F, E, K, G))
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(F, 1))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # taken from _ConvNd initialization of parameters
        stdv = 1. / math.sqrt(self.G * self.K)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def addGSO(self, S):
        # S: B x T x E x N x N        
        assert len(S.shape) == 5
        assert S.shape[2] == self.E
        self.N = S.shape[3]
        assert S.shape[4] == self.N
        self.S = S

    def forward(self, x):
        # input: ###
        # x: batchSize x time x dimInFeatures x numberNodesIn
        # output: ###
        # u: batchSize x time x dimOutFeatures x numberNodes
            
        assert len(x.shape) == 4
        B = x.shape[0]
        assert self.S.shape[0] == B
        T = x.shape[1]
        assert self.S.shape[1] == T
        F = x.shape[2]
        assert x.shape[3] == self.N

        u = LSIGF_DB(self.weight, self.S, x, self.bias)
        
        return u    