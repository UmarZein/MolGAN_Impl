import datetime
import torch
import rdkit
from torch_geometric.data import Data, Batch
import torch
from torch import nn
import rdkit
from tqdm import tqdm
from rdkit import Chem
import pandas as pd
from importlib import reload
import matplotlib.pyplot as plt
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.conv import RGCNConv, FastRGCNConv
from torch_geometric.datasets import QM7b

from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')
import sascorer
from molecularmetrics import MolecularMetrics
from constants import *



def sample(*shape):
    return torch.randn(*shape)

def sparse_pyg_graph_from_graph(x,a,mol=None):
    edge_index=a.nonzero(as_tuple=False).t()#(type, row, col)
    data = Data(x=x, edge_index=edge_index[1:], edge_attr=edge_index[0])
    if mol is not None:
        data.molecule = mol
    return data

def graph_from_smiles(smiles,return_mol=False,typ=None):
    rdkit_mol=Chem.MolFromSmiles(smiles)
    X=[list(MOLS.keys()).index(i.GetAtomicNum()) for i in rdkit_mol.GetAtoms() if i.GetAtomicNum()!=1]
    N=len(X)
    A=torch.zeros(N,N)
    for i in range(N):
        for j in range(N):
            try:
                A[i,j]=BONDS[rdkit_mol.GetBondBetweenAtoms(i,j).GetBondType()]
            except:
                pass
    X=nn.functional.one_hot(torch.tensor(X), num_classes=len(MOLS))
    A=nn.functional.one_hot(A.to(int), num_classes=len(BONDS)).permute(-1,0,1)
    if typ is not None:
        X = X.to(typ)
        A = A.to(typ)
    if return_mol: 
        return X,A,rdkit_mol
    return X,A

def graph_remove_zero_bonds(x,a,mol = None):
    if a.shape[-3]==len(BONDS):
        a=a[...,1:,:,:]
    if mol is not None:
        return x,a,mol
    return x,a

def smiles_to_dataloader(smiles,batch_size=BATCH_SIZE):
    data=[]
    iterator = tqdm(smiles)
    for s in iterator:
        data.append(sparse_pyg_graph_from_graph(*graph_remove_zero_bonds(*graph_from_smiles(s, return_mol=True, typ=torch.float32))))
    return DataLoader(data,batch_size=batch_size)

def sample_gumbel(x,a,temperature=1.0,hard=False,method=None,make_symmetrical_adj=True,remove_self_loop=True):
    x = nn.functional.gumbel_softmax(x,tau=temperature,hard=hard,dim=-1)
    a = nn.functional.gumbel_softmax(a,tau=temperature,hard=hard,dim=-3)
    if make_symmetrical_adj:
        a=torch.minimum(torch.tensor(1).to(a.dtype),a+a.transpose(-1,-2).clone())
    if remove_self_loop:
        a*=1-torch.eye(a.shape[-1])
    return (x,a)

def reward(mol,norm=True):
    rr=1
    for m in ('logp','sas','qed'):
        if m == 'logp':
            rr *= MolecularMetrics.water_octanol_partition_coefficient_scores([mol], norm=norm)
        elif m == 'sas':
            rr *= MolecularMetrics.synthetic_accessibility_score_scores([mol], norm=norm)
        elif m == 'qed':
            rr *= MolecularMetrics.quantitative_estimation_druglikeness_scores([mol], norm=norm)
    return rr[0]

def data_from_graph_and_try_batch(x,a):
    # if you ask, why no `mol=None`? This function should only be used
    # for generator output, which do not output molecules
    if len(x.shape)==3:
        assert len(a.shape)==4, "if x is batched, a should also be batched"
        assert a.shape[0]==x.shape[0], "batch size between x and a should be equal"
        data=[]
        for x_,a_ in zip(x,a):
            data.append(sparse_pyg_graph_from_graph(*graph_remove_zero_bonds(*sample_gumbel(x_,a_,hard=True))))
        return Batch.from_data_list(data)
    return sparse_pyg_graph_from_graph(*graph_remove_zero_bonds(*sample_gumbel(x,a,hard=True)))

def gradient_penalty(y, x):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size())
    dydx = torch.autograd.grad(outputs=y,
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]
    #print('dydx',dydx.sum().detach().item())
    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.linalg.norm(dydx, dim=1, ord=2)
    res = ((dydx_l2norm - 1) ** 2).mean()
    #print('res',res.detach().item())
    return res

def write_to_file(filename,content,mode='w'):
    try:
        dts=datetime.datetime.now().replace(microsecond=0).isoformat().replace(":","-")[:19]
        with open(f"log-{dts}",mode) as f:
            f.write(content)
        return True
    except:
        return False

def mol_from_graph(x,a,strict=True,print_errors=False):
    assert a.shape[-3]==len(BONDS), "a: [N_BONDS, N_ATOMS, N_ATOMS]"
    assert len(a.shape)==3 or (len(a.shape)==4 and a.shape[0]==1), "batched operation not supported"
    if len(a.shape)==4:
        a=a[0]
        x=x[0]
    X_map=[*MOLS.keys()]
    iBOND = {BONDS[k]:k for k in BONDS}
    mol = Chem.RWMol()
    bonds=frozenset([])
    for node_label in x:
        mol.AddAtom(Chem.Atom(X_map[node_label.argmax().detach().cpu().numpy()]))

    for t,r,c in a[1:,:,:].argwhere().detach().cpu().numpy().tolist():
        if frozenset([r,c]) in bonds: continue
        if r==c: continue
        mol.AddBond(r, c, iBOND[t+1])#t+1 because t=0 now maps to SINGLEBOND because of a[1:,:,:]
        bonds |= frozenset([frozenset([r,c])])
        
    if strict:
        try:
            Chem.SanitizeMol(mol)
        except:
            mol = None
            if print_errors:
                print("error in Chem.SanitizeMol")
    return mol



def graph_from_data(data):
    assert not isinstance(data, Batch), "data must be a single unbatched graph"
    x=data.x
    n_edge_types=data.edge_attr.unique().shape[0]
    edge_types = data.edge_attr
    if n_edge_types==len(BONDS)-1:
        edge_types+=1
    edge_index = data.edge_index
    n=x.shape[-2]
    a=torch.zeros(len(BONDS),n,n)
    for t,t2 in zip(edge_types, edge_index.t()):
        r,c=t2
        a[int(t),int(r),int(c)]=1
    a[...,0,:,:]=1-a[...,1:,:,:].sum(-3)
    return x,a