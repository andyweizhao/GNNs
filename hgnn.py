
# Adapted from https://raw.githubusercontent.com/HazyResearch/hgcn/master/models/encoders.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import manifolds
from layers.att_layers import GraphAttentionLayer
import layers.hyp_layers as hyp_layers
from layers.layers import GraphConvolution, Linear, get_dim_act
import utils.math_utils as pmath

from torch_geometric.nn import GCNConv, GATConv, ChebConv, SGConv, GINConv
from torch import Tensor
import layers.hyp_layers as hyp_layers
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor
from torch_geometric.typing import Adj, OptTensor, Size
import torch.nn.init as init
from typing import Optional
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot

def addition_id(manifold, c, bias, x):
    bias = manifold.proj_tan0(bias.view(1, -1), c)
    hyp_bias = manifold.expmap0(bias, c)
    hyp_bias = manifold.proj(hyp_bias, c)
    x = manifold.mobius_add(x, hyp_bias, c=c)
    x = manifold.proj(x, c)
    return x

class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    def __init__(self, c):
        super(Encoder, self).__init__()
        self.c = c

    def encode(self, x, adj):
        input = (x, adj) # packing two inputs as a tuple is a necessity for customized layers to work.
        output, _ = self.layers.forward(input)
        return output

class HGNN(Encoder):

    def __init__(self, c, args):
        super(HGNN, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        self.curvatures.append(self.c)
        hgc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            
            print(args.model)
            if args.model == 'HGCN':
                layer = HGCNConv(self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias, args.use_att, args.local_agg)
                
            if args.model == 'HGAT':
                layer = HGATConv(self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias, args.use_att, args.local_agg)

            if args.model == 'HSGC':
                layer = HGATConv(self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias, args.use_att, args.local_agg)

            if args.model == 'HGIN':
                layer = HGINConv(self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias, args.use_att, args.local_agg)

            if args.model == 'HCheb':
                layer = HChebConv(self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias, args.use_att, args.local_agg)

            hgc_layers.append(layer)
        
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True

    def encode(self, x, adj):
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])
        return super(HGNN, self).encode(x_hyp, adj)

class HGATConv(GATConv):
    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias, use_att, local_agg):
        GATConv.__init__(self, in_features, out_features, heads=1, dropout=0, concat=False, 
                         add_self_loops=False, edge_dim=None, bias=False)             
        
        self.att_src = Parameter(torch.Tensor(1, self.heads, out_features))
        self.att_dst = Parameter(torch.Tensor(1, self.heads, out_features))        
        
        glorot(self.att_src)
        glorot(self.att_dst)
        
        self.linear = hyp_layers.HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.manifold = manifold
        self.c = c_in
        self.out_features = out_features
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        init.constant_(self.bias, 0)
    
    def forward(self, input) -> Tensor:
        """
        :params: input in H^n
        :return: output in H^n
        """
        x, edge_index = input 
        num_nodes, n = x.size()
        
        x = self.linear.forward(x) # a map: H^n -> H^n

        x_src = x_dst = self.manifold.logmap0(x, c=self.c).view(-1, self.heads, self.out_features)

        x = (x_src, x_dst)
            
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)
        out = self.propagate(edge_index, x=x, alpha=alpha, edge_attr=None, size=None)

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        
        out = self.manifold.proj(self.manifold.expmap0(out, c=self.c), c=self.c)

        out = addition_id(self.manifold, self.c, self.bias, out)
                
        return out, edge_index

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge
        
        alpha = F.leaky_relu(alpha, self.negative_slope)
        
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha  # Save for later use.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        return x_j * alpha.unsqueeze(-1)

class SPDSGConv(SGConv):

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias, use_att, local_agg):

        SGConv.__init__(self, in_features, out_features, K=2, cached = False, add_self_loops=True, bias=False)

        self.linear = hyp_layers.HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.manifold = manifold
        self.c = c_in
        
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        init.constant_(self.bias, 0)
        
    def forward(self, input) -> Tensor:
        """"""
        """
        :params: input in H^n
        :return: output in H^n
        """
        
        x, edge_index = input
        
        num_nodes, n = x.size()
        
        x = self.manifold.logmap0(x, c=self.c)
        
        cache = self._cached_x
        if cache is None:
            if isinstance(edge_index, Tensor):
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, num_nodes, False,
                    self.add_self_loops, dtype=x.dtype)
            elif isinstance(edge_index, SparseTensor):
                edge_index = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, num_nodes, False,
                    self.add_self_loops, dtype=x.dtype)


            for k in range(self.K):
                # propagate_type: (x: Tensor, edge_weight: OptTensor)
                x = self.propagate(edge_index, x=x, edge_weight=None,
                                   size=None)
                if self.cached:
                    self._cached_x = x
        else:
            x = cache.detach()
        
        x = self.manifold.proj(self.manifold.expmap0(x, c=self.c), c=self.c)
        
        out = self.linear.forward(x)
        
        out = addition_id(self.manifold, self.c, self.bias, out)
        
        return out, edge_index

    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.K)

class HGINConv(GINConv):

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias, use_att, local_agg):
        

        GINConv.__init__(self, nn=None, eps = 0, train_eps = False)
        
        self.eps.data.fill_(self.initial_eps)
        
        self.linear = hyp_layers.HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.manifold = manifold
        self.c = c_in
        
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        init.constant_(self.bias, 0)
        
    def forward(self, input) -> Tensor:
        """"""
        """
        :params: input in H^n
        :return: output in H^n
        """
        
        x, edge_index = input
        
        num_nodes, n = x.size() 
        
        x = self.manifold.logmap0(x, c=self.c)
        
        x = (x, x)        
        
        out = self.propagate(edge_index, x=x, size=None)
                   
        x_r = x[1]
        
        if x_r is not None:
            out += (1 + self.eps) * x_r

        out = self.manifold.proj(self.manifold.expmap0(out, c=self.c), c=self.c)
        
        out = self.linear.forward(out)
        
        out = addition_id(self.manifold, self.c, self.bias, out)
            
        return out, edge_index

class HChebConv(ChebConv):

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias, use_att, local_agg):
                
        ChebConv.__init__(self, in_features, out_features, K=2, normalization = 'sym', bias=False)
        
        self.linear = hyp_layers.HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.manifold = manifold
        self.c = c_in
        
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        init.constant_(self.bias, 0)
        
    def forward(self, input):
        """"""
        """
        :params: input in H^n
        :return: output in H^n
        """
        
        x, edge_index = input
        lambda_max = None
        
        if self.normalization != 'sym' and lambda_max is None:
            raise ValueError('You need to pass `lambda_max` to `forward() in`'
                             'case the normalization is non-symmetric.')

        if lambda_max is None:
            lambda_max = torch.tensor(2.0, dtype=x.dtype, device=x.device)
        if not isinstance(lambda_max, torch.Tensor):
            lambda_max = torch.tensor(lambda_max, dtype=x.dtype,
                                      device=x.device)
        assert lambda_max is not None

        num_nodes, n = x.size()

        edge_index, norm = self.__norm__(edge_index, num_nodes,
                                         edge_weight=None, normalization=self.normalization,
                                         lambda_max=lambda_max, dtype=x.dtype,
                                         batch=None)

        Tx_0 = x
        Tx_1 = x  # Dummy.
        
        out = self.linear.forward(Tx_0) # a map: H^n -> H^n
        
        x = self.manifold.logmap0(x, c=self.c)
        
        Tx_1 = self.propagate(edge_index, x=x, norm=norm, size=None)
      
        Tx_1 = self.manifold.proj(self.manifold.expmap0(Tx_1, c=self.c), c=self.c) # a map: R^n -> H^n
                        
        Tx_1 = self.linear.forward(Tx_1) 
        
        out = self.manifold.mobius_add(out, Tx_1, c=self.c)                
        out = self.manifold.proj(out, self.c)
        
        out = addition_id(self.manifold, self.c, self.bias, out)
        
        return out, edge_index

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={}, normalization={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            len(self.lins), self.normalization)            

        
class HGCNConv(GCNConv):
    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias, use_att, local_agg):

        GCNConv.__init__(self, in_features, out_features, add_self_loops=True,
                         normalize=True, bias=True, improved=False, cached=False)
        
        self.linear = hyp_layers.HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.manifold = manifold
        self.c = c_in
        
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        init.constant_(self.bias, 0)
        
 
    def forward(self, input) -> Tensor:
        """"""
        """
        :params: input in H^n
        :return: output in H^n
        """
        x, edge_index = input
        
        num_nodes, n = x.size()
        
        edge_index, edge_weight = self.get_edge_index_and_weights(edge_index, edge_weight=None, num_nodes=num_nodes)
        
        x = self.linear.forward(x)
        
        x = self.manifold.logmap0(x, c=self.c)

        out = self.propagate(edge_index, x=x, edge_weight=None, size=None)
        
        out = self.manifold.proj(self.manifold.expmap0(out, c=self.c), c=self.c)
        
        out = addition_id(self.manifold, self.c, self.bias, out)
            
        return out, edge_index

    def get_edge_index_and_weights(self, edge_index, edge_weight, num_nodes):
        """adapted from GCNConv.forward()"""
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, num_nodes,
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, num_nodes,
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache
        return edge_index, edge_weight        
    
