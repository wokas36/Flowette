from typing import Optional
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdchem
import torch.nn.functional as F

from fm_graphex.config import (
    Z_TO_TYPE,
    ATOM_TYPES, 
    NUM_ATOM_TYPES,
    BOND_TYPES,
    NUM_BOND_TYPES,
)


def sanitize_mol_safe(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """Return sanitized mol or None; avoids throwing."""
    if mol is None:
        return None
    mol = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None
    
    
def kekulize_preserve_hs(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """
    Kekulize + sanitize but preserve H (implicit/explicit).
    This is for MOSES/GuacaMol.
    """
    mol = sanitize_mol_safe(mol)
    if mol is None:
        return None
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception:
        pass
    return sanitize_mol_safe(mol)


def kekulize_and_remove_hs(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """
    Only use for QM9/ZINC evaluation if you really want heavy-only.
    """
    mol = sanitize_mol_safe(mol)
    if mol is None:
        return None
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception:
        pass
    try:
        mol_noH = Chem.RemoveHs(mol)
    except Exception:
        return None
    return sanitize_mol_safe(mol_noH)


# RDKit conversions & ring features
def data_to_rdkit_mol(data: Data) -> Optional[Chem.Mol]:
    """
    Generic conversion for processed Data objects:
      - data.x: (N, NUM_ATOM_TYPES) one-hot
      - data.edge_index: (2, E)
      - data.edge_attr: (E, NUM_BOND_TYPES) one-hot
      - optional data.z: (N,)
    """
    if data.edge_index is None:
        return None
    N = int(data.num_nodes)
    if N == 0:
        return None

    # Prefer z if present; otherwise infer from x
    if hasattr(data, "z") and data.z is not None and data.z.numel() == N:
        z_list = data.z.detach().cpu().tolist()
        atom_syms = []
        for z in z_list:
            z = int(z)
            if z in Z_TO_TYPE:
                atom_syms.append(ATOM_TYPES[Z_TO_TYPE[z]])
            else:
                atom_syms.append("C")
    else:
        x = data.x
        if x is None or x.size(-1) != NUM_ATOM_TYPES:
            return None
        idx = x.argmax(dim=-1).detach().cpu().tolist()
        atom_syms = [TYPE_TO_SYM.get(int(i), "C") for i in idx]

    mol = rdchem.RWMol()
    for sym in atom_syms:
        mol.AddAtom(rdchem.Atom(sym))

    edge_index = data.edge_index.detach().cpu().numpy()
    edge_attr = data.edge_attr.detach().cpu().numpy() if data.edge_attr is not None else None

    seen = set()
    E = edge_index.shape[1]
    for e in range(E):
        i = int(edge_index[0, e])
        j = int(edge_index[1, e])
        if i == j:
            continue
        a, b = sorted((i, j))
        if (a, b) in seen:
            continue
        seen.add((a, b))

        bt_idx = 0
        if edge_attr is not None and edge_attr.shape[0] == E:
            bt_idx = int(edge_attr[e, :NUM_BOND_TYPES].argmax())
            bt_idx = min(bt_idx, NUM_BOND_TYPES - 1)
        btype = BOND_TYPES[bt_idx]
        try:
            mol.AddBond(a, b, btype)
        except Exception:
            continue

    rmol = mol.GetMol()
    try:
        Chem.SanitizeMol(rmol)
    except Exception:
        return None
    return rmol