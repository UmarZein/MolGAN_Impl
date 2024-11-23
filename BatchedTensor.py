import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
#TODO / INCOMPLETE
class BatchedGraph(torch.Tensor):
    _extra_data = ''
    
    @staticmethod 
    def __new__(cls, x, extra_data, *args, **kwargs): 
        return super().__new__(cls, x, *args, **kwargs) 
      
    def __init__(self, x, extra_data): 
        self._extra_data = extra_data

    def clone(self, *args, **kwargs): 
        return MyObject(super().clone(*args, **kwargs), self._extra_data)

    def to(self, *args, **kwargs):
        new_obj = MyObject([], self._extra_data)
        tempTensor=super().to(*args, **kwargs)
        new_obj.data=tempTensor.data
        new_obj.requires_grad=tempTensor.requires_grad
        return(new_obj)

    @property
    def extra_data(self):
        return self._extra_data
        
    @extra_data.setter
    def extra_data(self, d):
        self._extra_data = d

