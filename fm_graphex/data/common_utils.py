import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdchem
from typing import Optional, List

from fm_graphex.data.rdkit_conversions import sanitize_mol_safe
from fm_graphex.config import get_spec


def truncate_mol_keep_first_atoms(mol: Chem.Mol, max_nodes: int) -> Optional[Chem.Mol]:
    """Truncate by keeping first max_nodes atoms; preserve atomic nums and bond types."""
    if mol is None:
        return None
    N = mol.GetNumAtoms()
    if N <= max_nodes:
        return mol

    keep = list(range(max_nodes))
    emol = rdchem.RWMol()
    old_to_new = {}
    for old_idx in keep:
        a = mol.GetAtomWithIdx(old_idx)
        new_atom = rdchem.Atom(a.GetAtomicNum())
        new_atom.SetFormalCharge(a.GetFormalCharge())
        new_atom.SetIsAromatic(a.GetIsAromatic())
        new_idx = emol.AddAtom(new_atom)
        old_to_new[old_idx] = new_idx

    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in old_to_new and j in old_to_new:
            try:
                emol.AddBond(old_to_new[i], old_to_new[j], b.GetBondType())
            except Exception:
                pass

    out = emol.GetMol()
    return sanitize_mol_safe(out)


# Common: RDKit Mol -> PyG Data with extra features
def mol_to_pyg_with_extras(mol: Chem.Mol, max_nodes: Optional[int] = None) -> Optional[Data]:
    """
    Returns Data with:
      x: (N, NUM_ATOM_TYPES + EXTRA_NODE_FEATS)
         - first NUM_ATOM_TYPES: one-hot atom identity (by vocab)
         - extras: [formal_charge_norm, is_aromatic]
      z: (N,) atomic numbers
      edge_index: (2, E) bidirectional
      edge_attr: (E, NUM_BOND_TYPES) one-hot
    """
    from fm_graphex.config import Z_TO_TYPE, NUM_ATOM_TYPES, NUM_BOND_TYPES
    
    if mol is None:
        return None
    mol = sanitize_mol_safe(mol)
    if mol is None:
        return None

    if max_nodes is not None and mol.GetNumAtoms() > max_nodes:
        # For MOSES you were dropping > max_nodes; for GuacaMol you were not truncating.
        # Truncation is safer than dropping (dropping biases distribution).
        mol = _truncate_mol_keep_first_atoms(mol, max_nodes)
        if mol is None or mol.GetNumAtoms() == 0:
            return None

    N = mol.GetNumAtoms()
    if N == 0:
        return None

    # Node identity + extras
    Z: List[int] = []
    type_idx: List[int] = []
    charge_feat: List[float] = []
    arom_feat: List[float] = []

    for a in mol.GetAtoms():
        z = int(a.GetAtomicNum())
        Z.append(z)
        type_idx.append(Z_TO_TYPE.get(z, Z_TO_TYPE.get(6, 0)))

        # formal charge in [-3..+3] typically; clamp then normalize
        fc = float(a.GetFormalCharge())
        fc = max(-3.0, min(3.0, fc)) / 3.0
        charge_feat.append(fc)

        arom_feat.append(1.0 if a.GetIsAromatic() else 0.0)

    z_t = torch.tensor(Z, dtype=torch.long)
    onehot = F.one_hot(torch.tensor(type_idx, dtype=torch.long), num_classes=NUM_ATOM_TYPES).float()
    extra = torch.tensor(list(zip(charge_feat, arom_feat)), dtype=torch.float32)
    
    task = get_spec().name
    
    # x = onehot
    if task == "qm9" or task == "zinc": # or task == "moses"
        x = onehot
    else:
        x = torch.cat([onehot, extra], dim=-1) # (N, NUM_ATOM_TYPES+2)

    # Edges
    edges = []
    bond_types = []
    for b in mol.GetBonds():
        i, j = int(b.GetBeginAtomIdx()), int(b.GetEndAtomIdx())
        bt = b.GetBondType()
        edges.append([i, j]); edges.append([j, i])

        if bt == rdchem.BondType.SINGLE:
            idx = 0
        elif bt == rdchem.BondType.DOUBLE:
            idx = 1
        elif bt == rdchem.BondType.TRIPLE:
            idx = 2
        elif bt == rdchem.BondType.AROMATIC:
            idx = 3
        else:
            idx = 0

        bond_types.append(idx); bond_types.append(idx)

    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, NUM_BOND_TYPES), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = F.one_hot(torch.tensor(bond_types, dtype=torch.long), num_classes=NUM_BOND_TYPES).float()

    return Data(x=x, z=z_t, edge_index=edge_index, edge_attr=edge_attr)