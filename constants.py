import torch
import rdkit

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
MAX_N=9#9 for gdb9, 31 for zinc250k
BATCH_SIZE=32
GP_LAMBDA = 10.0#10.0

WGAN_BIAS_LAMBDA = 0.01#0.01

def sample(*shape):
    return torch.randn(*shape)