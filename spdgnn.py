import linalg as lalg
import torch
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor, Size
from torch_geometric.nn import GCNConv, GATConv, ChebConv, SGConv, GINConv
from typing import Optional
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot

from typing import Union
from torch_geometric.typing import OptPairTensor

from torch_geometric.utils import add_remaining_self_loops
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import get_laplacian
from torch_geometric.utils import remove_self_loops, add_self_loops

def expmap_id(x):
    x = lalg.sym(x)    

    x = lalg.sym_funcm(x, torch.exp)
    return x
        
def logmap_id(y): 
    y = lalg.sym(y)

    y = lalg.sym_funcm(y, torch.log)
    return y 

def addition_id(a, b):
    sqrt_a = lalg.sym_sqrtm(a)
    return sqrt_a @ b @ sqrt_a

def normalizating(x, dims, mode='matrix_norm'):
    
    x = lalg.sym(x)     
    num_nodes, n = x.size()[:2]

    if mode == 'matrix_norm':
        x = x / torch.linalg.matrix_norm(x).reshape(-1, 1, 1)
        
    if mode == 'det_norm':
        dets = torch.linalg.det(x)
        x = torch.sign(x) * x / torch.pow(abs(dets), 1/n).reshape(-1, 1, 1)
        
    if mode == 'vector_norm':
        x = x.reshape(num_nodes, -1)
        x = F.normalize(x, p=2., dim=-1)
        x = x.reshape(num_nodes, n, n)

    if mode == 'rescaling':                 
        norm = torch.linalg.matrix_norm(x, keepdim=True)
        maxnorm = 10        
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        x = torch.where(cond, projected, x)
    return x


class SPDGATConv(GATConv):
    def __init__(self, input_dims, args):
        GATConv.__init__(self, input_dims, input_dims, heads=1, dropout=0, concat=False, 
                         add_self_loops=False, edge_dim=None, bias=False)
        
        self.args = args
        self.transform = SPDTransform.get(args.transform, input_dims)
        
        self.att_src = Parameter(torch.Tensor(1, self.heads, input_dims * input_dims))
        self.att_dst = Parameter(torch.Tensor(1, self.heads, input_dims * input_dims))        
        
        glorot(self.att_src)
        glorot(self.att_dst)
        
        self.bias = Parameter(torch.Tensor(1, input_dims, input_dims))

        glorot(self.bias)
    
    def forward(self, mat_feats: Tensor, edge_index: Adj, edge_attr: OptTensor = None, size: Size = None,
                return_attention_weights=None) -> Tensor:
        # input and output are in SPD
        
        num_nodes, n, _ = mat_feats.size()
        
        mat_feats = self.transform(mat_feats)
        
        symmat_feats = logmap_id(mat_feats)        
        
        x_src = x_dst = symmat_feats.reshape(num_nodes, self.heads, -1)      # b x n*n only suppors heads = 1
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
                
        out = out.reshape(num_nodes, n, n)
        
        out = expmap_id(out)         
        
        if self.args.has_bias:
            out = addition_id(out, expmap_id(self.bias))
            
        return out     

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

    def __init__(self, input_dims, args):

        self.args = args
        SGConv.__init__(self, input_dims, input_dims, K=2, cached = False, add_self_loops=True, bias=False)

        self.transform = SPDTransform.get(args.transform, input_dims)

        self.bias = Parameter(torch.Tensor(1, input_dims, input_dims))

        glorot(self.bias)
        
    def forward(self, mat_feats: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        # input and output are in SPD

        num_nodes, n, _ = mat_feats.size()

        cache = self._cached_x
        if cache is None:
            if isinstance(edge_index, Tensor):
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, num_nodes, False,
                    self.add_self_loops, dtype=mat_feats.dtype)
            elif isinstance(edge_index, SparseTensor):
                edge_index = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, num_nodes, False,
                    self.add_self_loops, dtype=mat_feats.dtype)

            symmat_feats = logmap_id(mat_feats)
            symmat_feats = symmat_feats.reshape(num_nodes, -1)
            
            for k in range(self.K):
                symmat_feats = self.propagate(edge_index, x=symmat_feats, edge_weight=edge_weight,
                                   size=None)
                
                
                if self.cached:
                    self._cached_x = symmat_feats
        else:
            symmat_feats = cache.detach()

        symmat_feats = symmat_feats.reshape(num_nodes, n, n)                
        
        mat_feats = expmap_id(symmat_feats)
        out = self.transform(mat_feats)

        if self.args.has_bias:
            out = addition_id(out, expmap_id(self.bias))
            
        return out

    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.K)

class SPDGINConv(GINConv):

    def __init__(self, input_dims, args):
        
        self.args = args

        GINConv.__init__(self, nn=None, eps = 0, train_eps = False)
        
        self.eps.data.fill_(self.initial_eps)
        
        self.transform = SPDTransform.get(args.transform, input_dims)
        
        self.bias = Parameter(torch.Tensor(1, input_dims, input_dims))

        glorot(self.bias)
        
    def forward(self, mat_feats: Tensor, edge_index: Adj, edge_weight: OptTensor = None,
                size: Size = None, lambda_max: OptTensor = None, batch: OptTensor = None) -> Tensor:

        """"""
        # input and output are in SPD

        num_nodes, n, _ = mat_feats.size() 
                  
        symmat_feats = logmap_id(mat_feats)
        symmat_feats = symmat_feats.reshape(num_nodes, -1)
            
        x = (symmat_feats, symmat_feats)        
        
        out = self.propagate(edge_index, x=x, size=size)

        x_r = x[1]
        
        if x_r is not None:
            out += (1 + self.eps) * x_r        
        
        out = out.reshape(num_nodes, n, n)
        
        out = normalizating(out, mode=self.args.spd_norm)
        
        out = expmap_id(out) 
        
        out = self.transform(out)
        
        if self.args.has_bias:
            out = addition_id(out, expmap_id(self.bias))
            
        return out

class SPDChebConv(ChebConv):

    def __init__(self, input_dims, args):
        
        self.args = args
        ChebConv.__init__(self, input_dims, input_dims, K=2, normalization = 'sym', bias=False)

        self.transform1 = SPDTransform.get(args.transform, input_dims)

        self.transform2 = SPDTransform.get(args.transform, input_dims)
        
        self.bias = Parameter(torch.Tensor(1, input_dims, input_dims))        
        
        self.scaling_matrix = Parameter(torch.Tensor(1, input_dims, input_dims))
        
        glorot(self.bias)
        glorot(self.scaling_matrix)
        
    def forward(self, mat_feats: Tensor, edge_index: Adj, edge_weight: OptTensor = None,
                batch: OptTensor = None, lambda_max: OptTensor = None):
        """"""

        # input and output are in SPD

        if self.normalization != 'sym' and lambda_max is None:
            raise ValueError('You need to pass `lambda_max` to `forward() in`'
                             'case the normalization is non-symmetric.')

        if lambda_max is None:
            lambda_max = torch.tensor(2.0, dtype=mat_feats.dtype, device=mat_feats.device)
        if not isinstance(lambda_max, torch.Tensor):
            lambda_max = torch.tensor(lambda_max, dtype=mat_feats.dtype,
                                      device=mat_feats.device)
        assert lambda_max is not None

        num_nodes, n, _ = mat_feats.size()

        edge_index, norm = self.__norm__(edge_index, num_nodes,
                                         edge_weight, self.normalization,
                                         lambda_max, dtype=mat_feats.dtype,
                                         batch=batch)

        Tx_0 = mat_feats
        Tx_1 = mat_feats  # Dummy.
        
        Tx_0 = self.transform1(Tx_0)
        
        mat_feats = logmap_id(mat_feats)
        mat_feats = mat_feats.reshape(num_nodes, -1)
            
        Tx_1 = self.propagate(edge_index, x=mat_feats, norm=norm, size=None)
        Tx_1 = Tx_1.reshape(num_nodes, n, n)        
                          
        Tx_1 = expmap_id(Tx_1)
        out = addition_id(self.transform2(Tx_1), expmap_id(self.scaling_matrix * logmap_id(Tx_0)))                                     
        
        if self.args.has_bias:
            out = addition_id(out, expmap_id(self.bias))
            
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={}, normalization={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            len(self.lins), self.normalization)

        
class SPDGCNConv(GCNConv):
    def __init__(self, input_dims, args):

        self.args = args
        GCNConv.__init__(self, input_dims, input_dims, add_self_loops=True,
                         normalize=True, bias=True, improved=False, cached=False)
        
        self.transform = SPDTransform.get(args.transform, input_dims)
        
        self.bias = Parameter(torch.Tensor(1, input_dims, input_dims))
        
        glorot(self.bias)
    
    def forward(self, mat_feats: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:

        # input and output are in SPD
        
        num_nodes, n, _ = mat_feats.size()
        edge_index, edge_weight = self.get_edge_index_and_weights(edge_index, edge_weight, num_nodes)

        mat_feats = self.transform(mat_feats)
        
        symmat_feats = logmap_id(mat_feats)        

        symmat_feats = symmat_feats.reshape(num_nodes, -1)

        out = self.propagate(edge_index, x=symmat_feats, edge_weight=edge_weight, size=None)
                
        out = out.reshape(num_nodes, n, n)        
        
        out = expmap_id(out) 
        
        if self.args.has_bias:
            out = addition_id(out, expmap_id(self.bias))
        
        return out

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
