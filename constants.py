import torch
import rdkit

device = torch.cuda.device(0)

BONDS = {
    rdkit.Chem.rdchem.BondType.ZERO:0,
    rdkit.Chem.rdchem.BondType.SINGLE:1,
    rdkit.Chem.rdchem.BondType.DOUBLE:2,
    rdkit.Chem.rdchem.BondType.TRIPLE:3,
}

DEFAULT_DO_RATE = 0.01

MOLS = {
    6: 'C',
    7: 'N',
    8: 'O',
    9: 'F',
    15: 'P',
    16: 'S',
    17: 'Cl',
    35: 'Br',
    53: 'I',
}

def sample(*shape):
    return torch.randn(*shape)