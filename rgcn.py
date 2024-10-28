import torch
from torch import nn
from constants import *

class RGCNLayer(nn.Module):
    'https://arxiv.org/pdf/1703.06103'
    def __init__(self, in_dim, out_dim, do_rate, activation_function=nn.functional.tanh):
        super().__init__()
        self.num_relations=len(BONDS)
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.do_rate=do_rate
        self.activation_function=activation_function
        self.wr=nn.Parameter(torch.empty(self.num_relations, in_dim, out_dim))
        self.w0=nn.Parameter(torch.empty(in_dim, out_dim))
        self.init_weights()
    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.wr)
        torch.nn.init.xavier_uniform_(self.w0)
    def forward(self, 
                inputs, 
                #inputs = (X,A) 
                #X: (B*, N, in_dim) ; x_v^{(l)} synonymous with h_v^{(l)}
                #A: (B*, num_relations, N, N)
               ):
        X,A=inputs
        # h_i^{l+1} = act(left_side^{l} + right_side^{l})
        # see eq. (2) in the paper
        # we decide to use neither basis-decomposition nor block-diagonal-decomposition
        # because molecules can have at most 4 edge types, including disonnected edge type
        right_side=X@self.w0

        tmp1 = torch.einsum(X, [..., 0, 1], self.wr, [2, 1, 3], [..., 0, 2, 3]).transpose(-3,-2)
        tmp2 = (A@tmp1)
        cir = A.sum(-1).unsqueeze(-1)
        left_side = tmp2.div(cir).sum(-3)
        new_x=nn.functional.dropout(self.activation_function(left_side+right_side),p=self.do_rate)
        return new_x,A
    def forward_old(self, 
                X, #X: (B*, N, in_dim) 
                A, #A: (B*, num_relations, N, N)
               ):
        # h_i^{l+1} = left_side^{l} + right_side^{l}
        # see eq. (2) in the paper
        # we decide to use neither basis-decomposition nor block-diagonal-decomposition
        # because molecules can have at most 4 edge types, including disonnected edge type
        right_side=X@self.w0

        # GCN: X@A@W
        
        left_side = torch.zeros_like(right_side)
        tmp1s=[]
        tmp2s=[]
        # A: (B*, num_relations, N, N)
        # A.sum(-1): (B*, num_relations, N)
        # X: (B*, N, in_dim) 
        # W: (num_relations, in_dim, out_dim)
        # tmp1: (B*, N, num_relations, out_dim)
        for i in range(A.shape[-3]):
            A_ = A[...,i,:,:]
            cir = A_.sum(-1).unsqueeze(-1) # either sum(-1) or sum(-2)
            tmp1=(X@self.wr[i])
            tmp1s.append(tmp1.detach())
            tmp2=A_@tmp1
            tmp2s.append(tmp2.detach())
            #print(A.shape, A_.shape, X.shape, self.wr.shape, cir.shape, tmp1.shape, tmp2.shape)
            left_side += tmp2/cir
            
        tmp1_alt = torch.einsum(X, [..., 0, 1], self.wr, [2, 1, 3], [..., 0, 2, 3]).transpose(-3,-2)
        #tmp1_alt == torch.stack(tmp1s)
        tmp2_alt = (A@tmp1_alt) #: (B*, num_relations, N, out_dim)
        cir_alt = A.sum(-1).unsqueeze(-1) #: (B*, num_relations, N, 1)
        left_side_alt = tmp2_alt.div(cir_alt).sum(-3)
        
        return left_side,right_side,tmp1s,tmp2s,tmp1_alt,tmp2_alt,left_side_alt

    #def forward2(self, 
    #            X, #X: (B*, N, in_dim) 
    #            A, #A: (B*, num_relations, N, N)
    #           ):
    #    # h_i^{l+1} = left_side^{l} + right_side^{l}
    #    # see eq. (2) in the paper
    #    # we decide to use neither basis-decomposition nor block-diagonal-decomposition
    #    # because molecules can have at most 4 edge types, including disonnected edge type
    #    right_side=X@self.w0
    #
    #    # GCN: X@A@W
    #    
    #    left_side = torch.zeros_like(right_side)
    #    # A: (B*, num_relations, N, N)
    #    # A.sum(-1): (B*, num_relations, N)
    #    # X: (B*, N, in_dim) 
    #    # W: (num_relations, in_dim, out_dim)
    #    # tmp1: (B*, N, num_relations, out_dim)
    #    
    #    for i in range(A.shape[-3]):
    #        A_ = A[...,i,:,:]
    #        cir = A_.sum(-1).unsqueeze(-1) # either sum(-1) or sum(-2)
    #        tmp1=(X@self.wr[i])
    #        tmp2=A_@tmp1
    #        #print(A.shape, A_.shape, X.shape, self.wr.shape, cir.shape, tmp1.shape, tmp2.shape)
    #        left_side += tmp2/cir
    #        
    #    return right_side,left_side