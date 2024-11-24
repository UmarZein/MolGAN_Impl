import torch
from torch import nn
from constants import *
from utils import *
import mlp

class Generator(nn.Module):
    'deepchem ref: https://github.com/deepchem/deepchem/blob/master/deepchem/models/torch_models/molgan.py#L256'
    def __init__(self, z_dim, dims, A_dims, X_dims, N, dropout_rate=DEFAULT_DO_RATE):
        '''
        z_dim: int
        dims: List[int]
        A_dims: List[int]
        X_dims: List[int]
        N: int
            max # of atoms in the output
        dropout_rate: float
                     A_dims --> R^{N, N, len(BONDS})}
        z --> dims <
                     X_dims --> R^{N, len(MOLS)}
        '''
        super().__init__()
        self.z_dim=z_dim
        self.do_rate=dropout_rate
        self.N=N
        main_dims=[z_dim]+dims
        a_dims=main_dims[-1:]+A_dims+[len(BONDS)*(N**2)]
        x_dims=main_dims[-1:]+X_dims+[N*len(MOLS)]
        self.main_mlp = mlp.MLP(main_dims[0],main_dims[1:-1],main_dims[-1], dropout_rate=self.do_rate)
        self.a_mlp = mlp.MLP(a_dims[0],a_dims[1:-1],a_dims[-1],final_activation=None, dropout_rate=self.do_rate)
        self.x_mlp = mlp.MLP(x_dims[0],x_dims[1:-1],x_dims[-1],final_activation=None, dropout_rate=self.do_rate)
    def forward(self, z):
        zshape=z.shape
        base=self.main_mlp(z)
        a=self.a_mlp(base).view(-1,len(BONDS), self.N, self.N)
        
        a=(a+a.transpose(-1,-2))/2 # the adjacency matrix should be symmetrical, I think
        

        
        a*=1-torch.eye(a.shape[-1]) # remove self-loop adjacency. 
                                    # Why? Because in RGCN paper (https://arxiv.org/pdf/1703.06103) eq. (2),
                                    # We see on the right-side of the equation, therein on the right side 
                                    # is W_0^{(l)}h_i^{(l)} This makes me think that if we dont remove it in this stage
                                    # self-loop will be done twice
                                    # 
                                    # Also, think about it, in molecules, atoms are not connected to themselves 
        
        x=self.x_mlp(base).view(-1,self.N,len(MOLS))
        a+=a.transpose(-1,-2).clone()
        a=a.softmax(-3)#softmax over the relation-type dimension
        x=x.softmax(-1)#softmax over the atom_type dimension
        if len(zshape)==1:
            assert x.shape[0]==1 and a.shape[0]==1, "if z is 1-dim, therefor unbatched, x and a should be shaped (1, ...)"
            x=x[0]
            a=a[0]
        return data_from_graph_and_try_batch(x,a)