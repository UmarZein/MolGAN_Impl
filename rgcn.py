import torch
from torch import nn
from constants import *
from torch_geometric.nn import FastRGCNConv
from torch_geometric.data import Data

class RGCNLayer(nn.Module):
    'https://arxiv.org/pdf/1703.06103'
    def __init__(self, in_dim, out_dim, do_rate, activation=nn.Tanh):
        super().__init__()
        self.inner = FastRGCNConv(in_dim, out_dim, len(BONDS)-1)
        self.do = nn.Dropout(p=do_rate)
        self.act = activation() if activation is not None else None
    def forward(self, 
                data, 
                use_old=False
                #inputs = (X,A) 
                #X: (B*, N, in_dim) ; x_v^{(l)} synonymous with h_v^{(l)}
                #A: (B*, num_relations, N, N)
               ):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_attr
        x=self.inner(x,edge_index,edge_type)
        if self.act is not None:
            data = self.act(x)
        x=self.do(x)
        return Data(x=x,edge_index=edge_index, edge_attr=edge_type)
        
class RGCN(nn.Module):
    def __init__(self, input_dim, dims, output_dim, activation=nn.Tanh, final_activation=nn.Tanh, dropout_rate=0.1):
        super().__init__()
        self.dims=[input_dim]+dims+[output_dim]
        self.do_rate=dropout_rate
        self.layers = nn.Sequential(
            *[
                x
                for xs in [(
                    RGCNLayer(self.dims[i],self.dims[i+1], activation=activation, do_rate=self.do_rate),
                ) if i+1<len(self.dims)-1 else (
                    RGCNLayer(self.dims[i],self.dims[i+1], activation=final_activation, do_rate=self.do_rate),
                ) if final_activation is not None else (
                    RGCNLayer(self.dims[i],self.dims[i+1], activation=None, do_rate=self.do_rate),
                )  for i in range(len(self.dims)-1)]
                for x in xs
            ]
        )
    def forward(self, x):
        return self.layers(x)





# class RGCNL(Function):
#     @staticmethod
#     def forward(ctx, input, adj, W_r, W_0):
#         # Save inputs and parameters for backward
#         ctx.save_for_backward(input, adj, W_r, W_0)
#         
#         # Apply relational transformation
#         output = torch.zeros_like(input)
#         for r in range(len(W_r)):
#             output += adj[r] @ (input @ W_r[r])  # Relation-based transformations
#         output += input @ W_0  # Self-loop transformation
#         return output
# 
#     @staticmethod
#     def backward(ctx, grad_output):
#         input, adj, W_r, W_0 = ctx.saved_tensors
#         grad_input = grad_W_r = grad_W_0 = None
# 
#         # Calculate gradients for each saved tensor
#         if ctx.needs_input_grad[0]:
#             grad_input = torch.zeros_like(input)
#             for r in range(len(W_r)):
#                 grad_input += adj[r].t() @ (grad_output @ W_r[r].t())
#             grad_input += grad_output @ W_0.t()
# 
#         if ctx.needs_input_grad[2]:  # W_r gradients
#             grad_W_r = [adj[r].t() @ (input.t() @ grad_output) for r in range(len(W_r))]
# 
#         if ctx.needs_input_grad[3]:  # W_0 gradient
#             grad_W_0 = input.t() @ grad_output
# 
#         return grad_input, None, grad_W_r, grad_W_0






