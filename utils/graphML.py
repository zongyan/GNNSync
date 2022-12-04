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


