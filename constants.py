import torch
device = torch.cuda.device(0)
BONDS = [1,2,3]
MOLS = {
    5:"B",
    6:"C",
    7:"N",
#    8:"O",
#    9:"F",
#    11:"Na",
#    12:"Mg",
#    13:"Al",
#    14:"Si",
}

def sample(*shape):
    return torch.randn(*shape)