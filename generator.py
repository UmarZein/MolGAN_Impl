import torch
from torch import nn
from constants import *


class Generator(nn.Module):
    def __init__(self, z_dim, dims, A_dims, X_dims, N, dropout_rate=0.1):
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
                     X_dims --> R^{N, len(MOLS)+1}
        '''
        super().__init__()
        self.z_dim=z_dim
        self.do_rate=dropout_rate
        self.N=N
        main_dims=[z_dim]+dims
        a_dims=main_dims[-1:]+A_dims+[len(BONDS)*(N**2)]
        x_dims=main_dims[-1:]+X_dims+[N*(len(MOLS)+1)] # +1 because apparently empty attoms are included
        self.main_mlp = nn.Sequential(
            *[
                x
                for xs in [(
                    nn.Linear(main_dims[i],main_dims[i+1]),
                    nn.Tanh(),
                    nn.Dropout(self.do_rate),
                ) for i in range(len(main_dims)-1)]
                for x in xs
            ]
        )
        self.a_mlp = nn.Sequential(
            *[
                x
                for xs in [(
                    nn.Linear(a_dims[i],a_dims[i+1]),
                    nn.Tanh(),
                    nn.Dropout(self.do_rate),
                ) if i+1<len(a_dims)-1 else (
                    nn.Linear(a_dims[i],a_dims[i+1]),
                    nn.Dropout(self.do_rate),
                ) for i in range(len(a_dims)-1)]
                for x in xs
            ]
        )
        self.x_mlp = nn.Sequential(
            *[
                x
                for xs in [(
                    nn.Linear(x_dims[i],x_dims[i+1]),
                    nn.Tanh(),
                    nn.Dropout(self.do_rate),
                ) if i+1<len(x_dims)-1 else (
                    nn.Linear(x_dims[i],x_dims[i+1]),
                    nn.Dropout(self.do_rate),
                ) for i in range(len(x_dims)-1)]
                for x in xs
            ]
        )
    def generate(self, z):
        base=self.main_mlp(z)
        a=self.a_mlp(base).view(-1,len(BONDS), self.N, self.N)
        x=self.x_mlp(base).view(-1,self.N,len(MOLS)+1) # +1 because apparently empty attoms are included
        return {
            'A':a,
            'X':x,
        }
        