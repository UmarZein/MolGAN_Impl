import torch
from torch import nn
from constants import *
from rgcn import RGCNLayer
import mlp

# Rewarder and discriminator are largely the same thing

class Discriminator(nn.Module):
    'Implementation of eq. (6) of Molgan Paper https://arxiv.org/abs/1805.11973v2'
    def __init__(self, rgcn_dims, i_dims, j_dims, final_mlp_dims, do_rate=DEFAULT_DO_RATE):
        super().__init__()
        self.input_dim=len(MOLS)
        assert len(rgcn_dims)>0
        assert i_dims[-1]==j_dims[-1]
        self.dims0=[self.input_dim]+rgcn_dims
        self.idims=[self.input_dim+rgcn_dims[-1]]+i_dims
        self.jdims=[self.input_dim+rgcn_dims[-1]]+j_dims
        self.final_mlp_dims=[i_dims[-1]]+final_mlp_dims+[1]
        self.do_rate=do_rate
        self.layers = nn.Sequential(
            *[
                x
                for xs in [(
                    RGCNLayer(self.dims0[i],self.dims0[i+1], do_rate=self.do_rate),#RGCN already includes an activation function (tanh)
                ) if i+1<len(self.dims0)-1 else (
                    RGCNLayer(self.dims0[i],self.dims0[i+1], do_rate=self.do_rate),
                ) for i in range(len(self.dims0)-1)]
                for x in xs
            ]
        )
        self.i = nn.Sequential(
            *[
                x
                for xs in [(
                    RGCNLayer(self.idims[i],self.idims[i+1], do_rate=self.do_rate),#RGCN already includes an activation function (tanh)
                ) if i+1<len(self.idims)-1 else (
                    RGCNLayer(self.idims[i],self.idims[i+1], do_rate=self.do_rate, activation_function=nn.functional.sigmoid),
                ) for i in range(len(self.idims)-1)]
                for x in xs
            ]
        )
        self.j = nn.Sequential(
            *[
                x
                for xs in [(
                    RGCNLayer(self.jdims[i],self.jdims[i+1], do_rate=self.do_rate),#RGCN already includes an activation function (tanh)
                ) if i+1<len(self.jdims)-1 else (
                    RGCNLayer(self.jdims[i],self.jdims[i+1], do_rate=0.0, activation_function=nn.functional.sigmoid),
                ) for i in range(len(self.jdims)-1)]
                for x in xs
            ]
        )
        self.final_mlp = mlp.MLP(self.final_mlp_dims[0],self.final_mlp_dims[1:-1],self.final_mlp_dims[-1],
                                 final_activation=None, dropout_rate=self.do_rate)
    def forward(self, inputs):
        x0,a=inputs
        h,_= self.layers(inputs)
        h=torch.cat([x0,h],-1)
        (i_out,_) = self.i((h,a))
        (j_out,_) = self.j((h,a))
        h=(i_out*j_out).sum(-2).tanh()
        h=self.final_mlp(h)
        return h,a