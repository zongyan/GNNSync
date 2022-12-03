# 2021/03/04~
# Fernando Gama, fgama@seas.upenn.edu.
# Luana Ruiz, rubruiz@seas.upenn.edu.
# Kate Tolstaya, eig@seas.upenn.edu
"""
graphML.py Module for basic GSP and graph machine learning functions.

Functionals

LSIGF: Applies a linear shift-invariant graph filter
spectralGF: Applies a linear shift-invariant graph filter in spectral form
NVGF: Applies a node-variant graph filter
EVGF: Applies an edge-variant graph filter
jARMA: Applies an ARMA filter using Jacobi iterations
learnAttentionGSO: Computes the GSO following the attention mechanism
graphAttention: Applies a graph attention layer
graphAttentionLSIGF: Applies a LSIGF over the learned graph
graphAttentionEVGF: Applies a EVGF over the learned graph

LSIGF_DB: Applies a delayed linear shift-invariant graph filter for batch GSO
GRNN_DB: Computes the sequence of hidden states for batch GSO

GatedGRNN: Computes the sequence of hidden states for static GSO

Filtering Layers (nn.Module)

GraphFilter: Creates a graph convolutional layer using LSI graph filters
SpectralGF: Creates a graph convolutional layer using LSI graph filters in
    spectral form
NodeVariantGF: Creates a graph filtering layer using node-variant graph filters
EdgeVariantGF: Creates a graph filtering layer using edge-variant graph filters
GraphFilterARMA: Creates a (linear) layer that applies a ARMA graph filter
    using Jacobi's method
GraphAttentional: Creates a layer using graph attention mechanisms
GraphFilterAttentional: Creates a layer using a graph convolution on a GSO 
    learned through attention
EdgeVariantAttentional: Creates a layer using an edge variant graph filter
    parameterized by several attention mechanisms
    
GraphFilter_DB: Creates a graph convolutional layer using LSI graph filters
    that are applied to a delayed sequence of shift operators
HiddenState_DB: Creates the layer for computing the hidden state of a GRNN

HiddenState: Creates the layer for computing the hidden state of a GRNN (with 
    static GSO)
TimeGatedHiddenState: Creates the layer for computing the time gated hidden 
    state of a GRNN
NodeGatedHiddenState: Creates the layer for computing the node gated hidden 
    state of a GRNN
EdgeGatedHiddenState: Creates the layer for computing the edge gated hidden 
    state of a GRNN

Activation Functions - Nonlinearities (nn.Module)

MaxLocalActivation: Creates a localized max activation function layer
MedianLocalActivation: Creates a localized median activation function layer
NoActivation: Creates a layer for no activation function

Summarizing Functions - Pooling (nn.Module)

NoPool: No summarizing function.
MaxPoolLocal: Max-summarizing function
"""

import math
import numpy as np
import torch
import torch.nn as nn

import utils.graphTools as graphTools

zeroTolerance = 1e-9 # Values below this number are considered zero.
infiniteNumber = 1e12 # infinity equals this number

# WARNING: Only scalar bias.

#############################################################################
#                                                                           #
#                           FUNCTIONALS                                     #
#                                                                           #
#############################################################################

def LSIGF(h, S, x, b=None):
    """
    LSIGF(filter_taps, GSO, input, bias=None) Computes the output of a linear
        shift-invariant graph filter on input and then adds bias.

    Denote as G the number of input features, F the number of output features,
    E the number of edge features, K the number of filter taps, N the number of
    nodes, S_{e} in R^{N x N} the GSO for edge feature e, x in R^{G x N} the
    input data where x_{g} in R^{N} is the graph signal representing feature
    g, and b in R^{F x N} the bias vector, with b_{f} in R^{N} representing the
    bias for feature f.

    Then, the LSI-GF is computed as
        y_{f} = \sum_{e=1}^{E}
                    \sum_{k=0}^{K-1}
                    \sum_{g=1}^{G}
                        [h_{f,g,e}]_{k} S_{e}^{k} x_{g}
                + b_{f}
    for f = 1, ..., F.

    Inputs:
        filter_taps (torch.tensor): array of filter taps; shape:
            output_features x edge_features x filter_taps x input_features
        GSO (torch.tensor): graph shift operator; shape:
            edge_features x number_nodes x number_nodes
        input (torch.tensor): input signal; shape:
            batch_size x input_features x number_nodes
        bias (torch.tensor): shape: output_features x number_nodes
            if the same bias is to be applied to all nodes, set number_nodes = 1
            so that b_{f} vector becomes b_{f} \mathbf{1}_{N}

    Outputs:
        output: filtered signals; shape:
            batch_size x output_features x number_nodes
    """
    # The basic idea of what follows is to start reshaping the input and the
    # GSO so the filter coefficients go just as a very plain and simple
    # linear operation, so that all the derivatives and stuff on them can be
    # easily computed.

    # h is output_features x edge_weights x filter_taps x input_features
    # S is edge_weighs x number_nodes x number_nodes
    # x is batch_size x input_features x number_nodes
    # b is output_features x number_nodes
    # Output:
    # y is batch_size x output_features x number_nodes

    # Get the parameter numbers:
    F = h.shape[0]
    E = h.shape[1]
    K = h.shape[2]
    G = h.shape[3]
    assert S.shape[0] == E
    N = S.shape[1]
    assert S.shape[2] == N
    B = x.shape[0]
    assert x.shape[1] == G
    assert x.shape[2] == N
    # Or, in the notation we've been using:
    # h in F x E x K x G
    # S in E x N x N
    # x in B x G x N
    # b in F x N
    # y in B x F x N

    # Now, we have x in B x G x N and S in E x N x N, and we want to come up
    # with matrix multiplication that yields z = x * S with shape
    # B x E x K x G x N.
    # For this, we first add the corresponding dimensions
    x = x.reshape([B, 1, G, N])
    S = S.reshape([1, E, N, N])
    z = x.reshape([B, 1, 1, G, N]).repeat(1, E, 1, 1, 1) # This is for k = 0
    # We need to repeat along the E dimension, because for k=0, S_{e} = I for
    # all e, and therefore, the same signal values have to be used along all
    # edge feature dimensions.
    for k in range(1,K):
        x = torch.matmul(x, S) # B x E x G x N
        xS = x.reshape([B, E, 1, G, N]) # B x E x 1 x G x N
        z = torch.cat((z, xS), dim = 2) # B x E x k x G x N
    # This output z is of size B x E x K x G x N
    # Now we have the x*S_{e}^{k} product, and we need to multiply with the
    # filter taps.
    # We multiply z on the left, and h on the right, the output is to be
    # B x N x F (the multiplication is not along the N dimension), so we reshape
    # z to be B x N x E x K x G and reshape it to B x N x EKG (remember we
    # always reshape the last dimensions), and then make h be E x K x G x F and
    # reshape it to EKG x F, and then multiply
    y = torch.matmul(z.permute(0, 4, 1, 2, 3).reshape([B, N, E*K*G]),
                     h.reshape([F, E*K*G]).permute(1, 0)).permute(0, 2, 1)
    # And permute againt to bring it from B x N x F to B x F x N.
    # Finally, add the bias
    if b is not None:
        y = y + b
    return y

#############################################################################
#                                                                           #
#                  FUNCTIONALS (Batch, and time-varying)                    #
#                                                                           #
#############################################################################

def LSIGF_DB(h, S, x, b=None):
    """
    LSIGF_DB(filter_taps, GSO, input, bias=None) Computes the output of a 
        linear shift-invariant graph filter (graph convolution) on delayed 
        input and then adds bias.

    Denote as G the number of input features, F the number of output features,
    E the number of edge features, K the number of filter taps, N the number of
    nodes, S_{e}(t) in R^{N x N} the GSO for edge feature e at time t, 
    x(t) in R^{G x N} the input data at time t where x_{g}(t) in R^{N} is the
    graph signal representing feature g, and b in R^{F x N} the bias vector,
    with b_{f} in R^{N} representing the bias for feature f.

    Then, the LSI-GF is computed as
        y_{f} = \sum_{e=1}^{E}
                    \sum_{k=0}^{K-1}
                    \sum_{g=1}^{G}
                        [h_{f,g,e}]_{k} S_{e}(t)S_{e}(t-1)...S_{e}(t-(k-1))
                                        x_{g}(t-k)
                + b_{f}
    for f = 1, ..., F.

    Inputs:
        filter_taps (torch.tensor): array of filter taps; shape:
            output_features x edge_features x filter_taps x input_features
        GSO (torch.tensor): graph shift operator; shape:
            batch_size x time_samples x edge_features 
                                                  x number_nodes x number_nodes
        input (torch.tensor): input signal; shape:
            batch_size x time_samples x input_features x number_nodes
        bias (torch.tensor): shape: output_features x number_nodes
            if the same bias is to be applied to all nodes, set number_nodes = 1
            so that b_{f} vector becomes b_{f} \mathbf{1}_{N}

    Outputs:
        output: filtered signals; shape:
            batch_size x time_samples x output_features x number_nodes
    """
    # This is the LSIGF with (B)atch and (D)elay capabilities (i.e. there is
    # a different GSO for each element in the batch, and it handles time
    # sequences, both in the GSO as in the input signal. The GSO should be
    # transparent).
    
    # So, the input 
    #   h: F x E x K x G
    #   S: B x T x E x N x N
    #   x: B x T x G x N
    #   b: F x N
    # And the output has to be
    #   y: B x T x F x N
    
    # Check dimensions
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
    
    # We would like a z of shape B x T x K x E x G x N that represents, for
    # each t, x_t, S_t x_{t-1}, S_t S_{t-1} x_{t-2}, ...,
    #         S_{t} ... S_{t-(k-1)} x_{t-k}, ..., S_{t} S_{t-1} ... x_{t-(K-1)}
    # But we don't want to do "for each t". We just want to do "for each k".
    
    # Let's start by reshaping x so it can be multiplied by S
    x = x.reshape([B, T, 1, G, N]).repeat(1, 1, E, 1, 1)
    
    # Now, for the first value of k, we just have the same signal
    z = x.reshape([B, T, 1, E, G, N])
    # For k = 0, k is counted in dim = 2
    
    # Now we need to start multiplying with S, but displacing the entire thing
    # once across time
    for k in range(1,K):
        # Across dim = 1 we need to "displace the dimension down", i.e. where
        # it used to be t = 1 we now need it to be t=0 and so on. For t=0
        # we add a "row" of zeros.
        x, _ = torch.split(x, [T-1, 1], dim = 1)
        #   The second part is the most recent time instant which we do not need
        #   anymore (it's used only once for the first value of K)
        # Now, we need to add a "row" of zeros at the beginning (for t = 0)
        zeroRow = torch.zeros(B, 1, E, G, N, dtype=x.dtype,device=x.device)
        x = torch.cat((zeroRow, x), dim = 1)
        # And now we multiply with S
        x = torch.matmul(x, S)
        # Add the dimension along K
        xS = x.reshape(B, T, 1, E, G, N)
        # And concatenate it with z
        z = torch.cat((z, xS), dim = 2)
        
    # Now, we finally made it to a vector z of shape B x T x K x E x G x N
    # To finally multiply with the filter taps, we need to swap the sizes
    # and reshape
    z = z.permute(0, 1, 5, 3, 2, 4) # B x T x N x E x K x G
    z = z.reshape(B, T, N, E*K*G)
    # And the same with the filter taps
    h = h.reshape(F, E*K*G)
    h = h.permute(1, 0) # E*K*G x F
    
    # Multiply
    y = torch.matmul(z, h) # B x T x N x F
    # And permute
    y = y.permute(0, 1, 3, 2) # B x T x F x N
    # Finally, add the bias
    if b is not None:
        y = y + b
    
    return y
    
#############################################################################
#                                                                           #
#                           LAYERS (Filtering)                              #
#                                                                           #
#############################################################################

class GraphFilter(nn.Module):
    """
    GraphFilter Creates a (linear) layer that applies a graph filter

    Initialization:

        GraphFilter(in_features, out_features, filter_taps,
                    edge_features=1, bias=True)

        Inputs:
            in_features (int): number of input features (each feature is a graph
                signal)
            out_features (int): number of output features (each feature is a
                graph signal)
            filter_taps (int): number of filter taps
            edge_features (int): number of features over each edge
            bias (bool): add bias vector (one bias per feature) after graph
                filtering

        Output:
            torch.nn.Module for a graph filtering layer (also known as graph
            convolutional layer).

        Observation: Filter taps have shape
            out_features x edge_features x filter_taps x in_features

    Add graph shift operator:

        GraphFilter.addGSO(GSO) Before applying the filter, we need to define
        the GSO that we are going to use. This allows to change the GSO while
        using the same filtering coefficients (as long as the number of edge
        features is the same; but the number of nodes can change).

        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                edge_features x number_nodes x number_nodes

    Forward call:

        y = GraphFilter(x)

        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x in_features x number_nodes

        Outputs:
            y (torch.tensor): output; shape:
                batch_size x out_features x number_nodes
    """

    def __init__(self, G, F, K, E = 1, bias = True):
        # K: Number of filter taps
        # GSOs will be added later.
        # This combines both weight scalars and weight vectors.
        # Bias will always be shared and scalar.

        # Initialize parent
        super().__init__()
        # Save parameters:
        self.G = G
        self.F = F
        self.K = K
        self.E = E
        self.S = None # No GSO assigned yet
        # Create parameters:
        self.weight = nn.parameter.Parameter(torch.Tensor(F, E, K, G))
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(F, 1))
        else:
            self.register_parameter('bias', None)
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Taken from _ConvNd initialization of parameters:
        stdv = 1. / math.sqrt(self.G * self.K)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def addGSO(self, S):
        # Every S has 3 dimensions.
        assert len(S.shape) == 3
        # S is of shape E x N x N
        assert S.shape[0] == self.E
        self.N = S.shape[1]
        assert S.shape[2] == self.N
        self.S = S

    def forward(self, x):
        # x is of shape: batchSize x dimInFeatures x numberNodesIn
        B = x.shape[0]
        F = x.shape[1]
        Nin = x.shape[2]
        # And now we add the zero padding
        if Nin < self.N:
            x = torch.cat((x,
                           torch.zeros(B, F, self.N-Nin)\
                                   .type(x.dtype).to(x.device)
                          ), dim = 2)
        # Compute the filter output
        u = LSIGF(self.weight, self.S, x, self.bias)
        # So far, u is of shape batchSize x dimOutFeatures x numberNodes
        # And we want to return a tensor of shape
        # batchSize x dimOutFeatures x numberNodesIn
        # since the nodes between numberNodesIn and numberNodes are not required
        if Nin < self.N:
            u = torch.index_select(u, 2, torch.arange(Nin).to(u.device))
        return u

    def extra_repr(self):
        reprString = "in_features=%d, out_features=%d, " % (
                        self.G, self.F) + "filter_taps=%d, " % (
                        self.K) + "edge_features=%d, " % (self.E) +\
                        "bias=%s, " % (self.bias is not None)
        if self.S is not None:
            reprString += "GSO stored"
        else:
            reprString += "no GSO stored"
        return reprString

#############################################################################
#                                                                           #
#              LAYERS (Filtering batch-dependent time-varying)              #
#                                                                           #
#############################################################################

class GraphFilter_DB(nn.Module):
    """
    GraphFilter_DB Creates a (linear) layer that applies a graph filter (i.e. a
        graph convolution) considering batches of GSO and the corresponding 
        time delays

    Initialization:

        GraphFilter_DB(in_features, out_features, filter_taps,
                       edge_features=1, bias=True)

        Inputs:
            in_features (int): number of input features (each feature is a 
                graph signal)
            out_features (int): number of output features (each feature is a
                graph signal)
            filter_taps (int): number of filter taps
            edge_features (int): number of features over each edge
            bias (bool): add bias vector (one bias per feature) after graph
                filtering

        Output:
            torch.nn.Module for a graph filtering layer (also known as graph
            convolutional layer).

        Observation: Filter taps have shape
            out_features x edge_features x filter_taps x in_features

    Add graph shift operator:

        GraphFilter_DB.addGSO(GSO) Before applying the filter, we need to define
        the GSO that we are going to use. This allows to change the GSO while
        using the same filtering coefficients (as long as the number of edge
        features is the same; but the number of nodes can change).

        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                batch_size x time_samples x edge_features 
                                                  x number_nodes x number_nodes

    Forward call:

        y = GraphFilter_DB(x)

        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x time_samples x in_features x number_nodes

        Outputs:
            y (torch.tensor): output; shape:
                batch_size x time_samples x out_features x number_nodes
    """

    def __init__(self, G, F, K, E = 1, bias = True):
        # K: Number of filter taps
        # GSOs will be added later.
        # This combines both weight scalars and weight vectors.
        # Bias will always be shared and scalar.

        # Initialize parent
        super().__init__()
        # Save parameters:
        self.G = G
        self.F = F
        self.K = K
        self.E = E
        self.S = None # No GSO assigned yet
        # Create parameters:
        self.weight = nn.parameter.Parameter(torch.Tensor(F, E, K, G))
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(F, 1))
        else:
            self.register_parameter('bias', None)
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Taken from _ConvNd initialization of parameters:
        stdv = 1. / math.sqrt(self.G * self.K)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def addGSO(self, S):
        # Every S has 5 dimensions.
        assert len(S.shape) == 5
        # S is of shape B x T x E x N x N
        assert S.shape[2] == self.E
        self.N = S.shape[3]
        assert S.shape[4] == self.N
        self.S = S

    def forward(self, x):
        # x is of shape: batchSize x time x dimInFeatures x numberNodesIn
        assert len(x.shape) == 4
        B = x.shape[0]
        assert self.S.shape[0] == B
        T = x.shape[1]
        assert self.S.shape[1] == T
        #F = x.shape[2]
        assert x.shape[3] == self.N
        # Compute the filter output
        u = LSIGF_DB(self.weight, self.S, x, self.bias)
        # u is of shape batchSize x time x dimOutFeatures x numberNodes
        return u

    def extra_repr(self):
        reprString = "in_features=%d, out_features=%d, " % (
                        self.G, self.F) + "filter_taps=%d, " % (
                        self.K) + "edge_features=%d, " % (self.E) +\
                        "bias=%s, " % (self.bias is not None)
        if self.S is not None:
            reprString += "GSO stored"
        else:
            reprString += "no GSO stored"
        return reprString
    
