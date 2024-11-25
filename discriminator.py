import torch
from torch import nn
from constants import *
from rgcn import RGCN
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
        self.layers = RGCN(self.dims0[0],self.dims0[1:-1],self.dims0[-1], 
                           dropout_rate=self.do_rate)
        self.i = mlp.MLP(self.idims[0],self.idims[1:-1],self.idims[-1],
                                 final_activation=nn.Sigmoid, dropout_rate=self.do_rate)
        self.j = mlp.MLP(self.jdims[0],self.jdims[1:-1],self.jdims[-1],
                                 final_activation=nn.Tanh, dropout_rate=self.do_rate)
        self.final_mlp = mlp.MLP(self.final_mlp_dims[0],self.final_mlp_dims[1:-1],self.final_mlp_dims[-1],
                                 final_activation=None, dropout_rate=self.do_rate)
    def forward(self, inputs, minibatch_sizes=None, use_old=False):
        x0,a=inputs
        if use_old:
            h=x0
            for l in self.layers:
                h,_=l((h,a),use_old=True)
        else:
            h,_= self.layers(inputs)#(B, N, H)
        h=torch.cat([x0,h],-1)
        i_out = self.i(h)
        j_out = self.j(h)
        h=(i_out*j_out)
        if minibatch_sizes is None:
            h=h.sum(-2).tanh()
        else:
            assert isinstance(minibatch_sizes,list) or isinstance(minibatch_sizes,tuple) or isinstance(minibatch_sizes,torch.Tensor), "minibatch_sizes must be a list of ints, representing the vertex count of each graph within the inputs"
            new_h=torch.zeros_like(h[...,:len(minibatch_sizes),:])#...,len(minibatch_sizes),h.shape[-1])
            cumsum=0
            for i,size in enumerate(minibatch_sizes):
                new_h[...,i,:]=h[...,cumsum:cumsum+size,:].sum(-2).tanh()
                cumsum+=size
            h=new_h
        #print("h:",h.shape)
        h=self.final_mlp(h)
        return h