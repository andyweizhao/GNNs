from abc import ABC, abstractmethod
import torch
from torch import nn
import linalg as lalg
import logging
import sys
from torch.nn import Parameter
# Adapted from https://github.com/fedelopez77/gyrospd/blob/main/gyrospd/models/tgsymspace.py 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SPDTransform(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, mat_feats):
        pass

    @classmethod
    def get(cls, type_str: str, input_dims: int):
        
        return SPDIsometry(type_str, input_dims)

def productory(factors: torch.Tensor, dim=1) -> torch.Tensor:
    """
    Computes the matrix product of the sequence
    :param factors: * x n x n
    :param dim: acc
    :return: * x n x n: the result of multiplying all matrices on the dim dimension, according
    to the given order. The result will have one less dimension
    """
    m = factors.size(dim)
    acum = factors.select(dim=dim, index=0)
    for i in range(1, m):
        current = factors.select(dim=dim, index=i)
        acum = acum @ current
    return acum

class SPDIsometry(SPDTransform):
    def __init__(self, type_str, input_dims):
        super().__init__()
        
        self.dims = input_dims
        self.n_isom = input_dims * (input_dims - 1) // 2

        self.type_str = type_str

        self.isom_params = Parameter(torch.Tensor(self.dims, self.dims))
        
        self.isom_params.data.copy_(torch.eye(self.dims))
                
        if self.type_str in ['rot', 'ref']:
            # U[-0.25, 0.25] radians ~ U[-15°, 15°]
            self.isom_params_angles = torch.nn.Parameter(torch.rand((1, self.n_isom)) * 0.5 - 0.25)
            self.embed_index = self.get_isometry_embed_index(input_dims)

    def get_isometry_embed_index(self, dims):
        """
        Build the index to embed the respective isometries into an n x n identity.

        We store a flattened version of the index. This is, for (i, j) the position of an entry in an
        n x n matrix, we reshape the matrix to a single row of len == n*n, and store the equivalent of
        (i, j) in the flattened version of the matrix

        For n dims we build m isometries, where m = n * (n - 1) / 2

        :param dims: int with number of dimensions
        :return: 1 x m x 4: the initial 1 dimension is just to make the repetion of this index faster
        """
        # indexes := 1 <= i < j < n. Using 1-based notation to make it equivalent to matrix math notation
        indexes = [(i, j) for i in range(1, dims + 1) for j in range(i + 1, dims + 1)]

        embed_index = []
        for i, j in indexes:
            row = []
            for c_i, c_j in [(i, i), (i, j), (j, i), (j, j)]:  # 4 combinations that we care for each (i, j) pair
                flatten_index = dims * (c_i - 1) + c_j - 1
                row.append(flatten_index)
            embed_index.append(row)
        return torch.LongTensor(embed_index).unsqueeze(0).to(device)  # 1 x m x 4


    def get_isometry_params(self) -> torch.Tensor:
        """
        This method must be implemented by concrete clases where the isometry parameters
        for each relation are computed  and returned as a tensor of r x m x 4 where
            r: num of relations
            m: num of isometries
        :return: tensor of r x m x 4
        """
        raise NotImplementedError()

    def build_relation_isometry_matrices(self, isom_params):
        """
        Builds the rotation isometries as matrices for all available relations
        :param isom_params: r x m x 4
        :return: r x n x n
        """
        embeded_isoms = self.embed_params(isom_params, self.dims)  # r x m x n x n
        isom_matrix = productory(embeded_isoms)  # r x n x n
        return isom_matrix

    def embed_params(self, isom_params: torch.Tensor, dims: int) -> torch.Tensor:
        """
        Embeds the isometry params.
        For each isometric operation there are m isometries with 4 params each.
        This method embeds the 4 params into a dims x dims identity, in positions given by self.embed_index

        :param iso_params: r x m x 4, where m = dims * (dims - 1) / 2, which is the amount of isometries
        :param dims: (also called n) dimension of output identities, with params embedded
        :return: r x m x n x n
        """
        r, m, _ = isom_params.size()
        target = torch.eye(dims, requires_grad=True, device=isom_params.device)
        target = target.reshape(1, 1, dims * dims).repeat(r, m, 1)  # b x m x n * n
        scatter_index = self.embed_index.repeat(r, 1, 1)  # b x m x 4
        embed_isometries = target.scatter(dim=-1, index=scatter_index, src=isom_params)  # b x m x n * n
        embed_isometries = embed_isometries.reshape(r, m, dims, dims)  # b x m x n x n
        return embed_isometries

    def forward(self, mat_feats):
        
        # input and output are in SPD.

        W = self.isom_params.data
        
#       use the orthogonal basis of a square matrix
        if self.type_str == 'qr':
            q, r = torch.linalg.qr(W)        
            d = torch.diag(r, 0)
            ph = d.sign()
            q *= ph
            W.copy_(q)
            W = W.unsqueeze(0)
            
        if self.type_str == 'cayley':
            W = W - W.t() # skew-symmetric A^T = -A
            Id = torch.eye(W.size(0)).cuda()
    #       the following approach is faster than (Id - A) @ torch.linalg.inv(Id + A)
    #       See https://pytorch.org/docs/stable/generated/torch.linalg.inv.html#torch.linalg.inv        
            W.copy_(torch.linalg.solve(Id - W, Id + W).t()) 
            W = W.unsqueeze(0)
        
        if self.type_str == 'rot':
            cos_x = torch.cos(self.isom_params_angles)
            sin_x = torch.sin(self.isom_params_angles)
            isometry_params = torch.stack([cos_x, -sin_x, sin_x, cos_x], dim=-1)

            W = self.build_relation_isometry_matrices(isometry_params)  # r x n x n

        if self.type_str == 'ref':

            cos_x = torch.cos(self.isom_params_angles)
            sin_x = torch.sin(self.isom_params_angles)
            isometry_params = torch.stack([cos_x, sin_x, sin_x, -cos_x], dim=-1)

            W = self.build_relation_isometry_matrices(isometry_params)  # r x n x n

        mat_feats = W @ mat_feats @ W.transpose(-1, -2)

        return mat_feats
