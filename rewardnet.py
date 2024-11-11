import torch
from torch import nn
from constants import *
from rgcn import RGCN
import discriminator
import mlp

# Rewarder and discriminator are largely the same thing

class Rewarder(nn.Module):
    'Implementation of eq. (6) of Molgan Paper https://arxiv.org/abs/1805.11973v2'
    def __init__(self, rgcn_dims, i_dims, j_dims, final_mlp_dims, do_rate=DEFAULT_DO_RATE):
        super().__init__()
        self.inner = discriminator.Discriminator(rgcn_dims, i_dims, j_dims, final_mlp_dims, do_rate=do_rate)
    def forward(self, inputs, use_old=False):
        return self.inner(inputs, use_old=use_old).sigmoid()