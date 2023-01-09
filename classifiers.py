import linalg as lalg
import torch
from torch import nn

from spdgnn import expmap_id, logmap_id
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot


class VectorNodeClassifier(nn.Module):
     def __init__(self, args):
         super().__init__()
     
         self.rows, self.cols = torch.triu_indices(args.hidden_dims, args.hidden_dims, device=args.device)
         self.proj = nn.Linear(int(args.hidden_dims * (args.hidden_dims + 1) / 2), args.num_classes)
         self.dropout = nn.Dropout(args.dropout)

     def forward(self, mat_feats):
         # input: SPD_n
         log_mats = logmap_id(mat_feats)
         node_feats = log_mats[:, self.rows, self.cols]
        
         node_feats = self.dropout(node_feats)
         node_feats = self.proj(node_feats)
         return node_feats    
     

class SVMClassifier(nn.Module):
    def __init__(self, args):
        super().__init__()
    
        self.args = args
        
        self.W = Parameter(torch.Tensor(args.num_classes, args.hidden_dims, args.hidden_dims))
        
        self.dropout = nn.Dropout(args.dropout)
     
        glorot(self.W)
        
    def forward(self, mat_feats):
        
        mat_feats = logmap_id(mat_feats)
        
        mat_feats = lalg.sym(self.dropout(mat_feats))
     
        proj = lalg.sym(self.W) @ mat_feats.unsqueeze(dim=1)
        
        output = self.b_trace(proj)
        
        center = mat_feats.mean(dim=0)
        proj = lalg.sym(self.W) @ center
        
        g_invariant = self.b_trace(proj @ proj)
        
        return output, g_invariant.mean(-1)
    
    def b_trace(self, a):    
        return a.diagonal(offset=0, dim1=-2, dim2=-1).sum(dim=-1)
    
     
from torch_geometric.nn.inits import glorot

def logmap(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
#    https://github.com/geoopt/geoopt/blob/master/geoopt/manifolds/symmetric_positive_definite.py#L247 
#    https://geoopt.readthedocs.io/en/latest/manifolds.html
#    log_x(u)
    inv_sqrt_x, sqrt_x = lalg.sym_inv_sqrtm2(x)
    return sqrt_x @ lalg.sym_logm(inv_sqrt_x @ u @ inv_sqrt_x) @ sqrt_x
    
import torch.nn.functional as F
class SPDLogisticClassifier(nn.Module):

    def __init__(self, args):
        super().__init__()
        
        self.dims = args.hidden_dims
        self.n_centroids = args.n_centroids
        
        self.means = Parameter(torch.Tensor(self.n_centroids, self.dims, self.dims))
        
        self.upper_diag_n = int(self.dims * (self.dims + 1) / 2)         

        self.sigma = Parameter(torch.Tensor(self.n_centroids, self.upper_diag_n, self.upper_diag_n))
        
        self.bias = Parameter(torch.ones(self.n_centroids))
                                
        self.rows, self.cols = torch.triu_indices(self.dims, self.dims, device=args.device)
     
        glorot(self.means)
        glorot(self.sigma)
        
    def forward(self, node_feats):                
        
        # input: SPD_n
        
        node_feats = logmap(expmap_id(lalg.sym(self.means)), node_feats.unsqueeze(1))        
        
        v = node_feats[:,:,self.rows, self.cols]
               
        v = v.unsqueeze(-2)    
        
        distances = (v @ expmap_id(lalg.sym(self.sigma)) @ v.transpose(-1, -2)).squeeze()
                
        distances = -0.5 * distances + self.bias 
                         
        return distances
    

class SPDLogisticClassifier_Euclidean(nn.Module):

    def __init__(self, args):
        super().__init__()
        
        self.dims = args.hidden_dims
        self.n_centroids = args.n_centroids
                
        self.upper_diag_n = int(self.dims * (self.dims + 1) / 2)         

        self.means = Parameter(torch.Tensor(self.n_centroids, self.upper_diag_n))
        
        self.sigma = Parameter(torch.Tensor(self.n_centroids, self.upper_diag_n, self.upper_diag_n))
        
        self.bias = Parameter(torch.ones(self.n_centroids))
                                
        self.rows, self.cols = torch.triu_indices(self.dims, self.dims, device=args.device)
        
        self.dropout = nn.Dropout(args.dropout)
               
        glorot(self.means)
        glorot(self.sigma)
        
    def forward(self, node_feats):                
        
        # input: SPD_n
        log_mats = logmap_id(node_feats)
        node_feats = log_mats[:, self.rows, self.cols]
        
        node_feats = self.dropout(node_feats)
     
        v = node_feats.unsqueeze(1) - self.means 
        
        v = v.unsqueeze(-2)    
        
        distances = (v @ expmap_id(lalg.sym(self.sigma)) @ v.transpose(-1, -2)).squeeze()                
        
        distances = -0.5 * distances + self.bias
        
        return distances   
        
