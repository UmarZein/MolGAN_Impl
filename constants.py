import torch
import rdkit
from torch_geometric.data import Data

device = torch.cuda.device(0)

BONDS = {
    rdkit.Chem.rdchem.BondType.ZERO:0, #IMPORTANT THAT THIS IS INCLUDED. IT IS ACCOUNTED IN, FOR EXAMPLE, RGCN CODE
    rdkit.Chem.rdchem.BondType.SINGLE:1,
    rdkit.Chem.rdchem.BondType.DOUBLE:2,
    rdkit.Chem.rdchem.BondType.TRIPLE:3,
}

DEFAULT_DO_RATE = 0.1
#FOR GDB9
MOLS = {
    6: 'C',
    7: 'N',
    8: 'O',
    9: 'F',
}
MAX_N=9#9 for gdb9, 31 for ZINC
# FOR ZINC250k
#MOLS = {
#    6: 'C',
#    7: 'N',
#    8: 'O',
#    9: 'F',
#    15: 'P',
#    16: 'S',
#    17: 'Cl',
#    35: 'Br',
#    53: 'I',
#}

LR=0.001
BATCH_SIZE=32

GP_LAMBDA = 10.0#10.0

WGAN_BIAS_LAMBDA = 0.01#0.01