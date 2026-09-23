from __future__ import annotations

from collections import deque
from typing import List, Optional

import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import rdchem
from torch_geometric.data import Data

import fm_graphex.config as cfg
from fm_graphex.data.rdkit_conversions import sanitize_mol_safe


def truncate_mol_connected(mol: Chem.Mol, max_nodes: int) -> Optional[Chem.Mol]:
    """Keep a deterministic connected BFS subgraph of at most ``max_nodes`` atoms.

    The old implementation called a misspelled helper and could crash whenever a
    GuacaMol molecule exceeded the node cap. Keeping a connected subgraph also
    avoids training on arbitrary disconnected atom prefixes.
    """
    if mol is None:
        return None
    if max_nodes <= 0:
        return None
    n = mol.GetNumAtoms()
    if n <= max_nodes:
        return Chem.Mol(mol)

    # Start from the highest-degree atom; this retains a chemically central
    # connected region more reliably than simply taking atoms 0:max_nodes.
    start = max(range(n), key=lambda i: mol.GetAtomWithIdx(i).GetDegree())
    keep: List[int] = []
    seen = {start}
    queue = deque([start])
    while queue and len(keep) < max_nodes:
        i = queue.popleft()
        keep.append(i)
        nbrs = sorted(a.GetIdx() for a in mol.GetAtomWithIdx(i).GetNeighbors())
        for j in nbrs:
            if j not in seen:
                seen.add(j)
                queue.append(j)

    # Molecules should be connected, but fill deterministically if needed.
    if len(keep) < max_nodes:
        for i in range(n):
            if i not in seen:
                keep.append(i)
                if len(keep) == max_nodes:
                    break

    keep_set = set(keep)
    rw = rdchem.RWMol()
    old_to_new = {}
    for old_idx in keep:
        src = mol.GetAtomWithIdx(old_idx)
        atom = rdchem.Atom(src.GetAtomicNum())
        atom.SetFormalCharge(src.GetFormalCharge())
        atom.SetIsAromatic(src.GetIsAromatic())
        atom.SetNoImplicit(src.GetNoImplicit())
        old_to_new[old_idx] = rw.AddAtom(atom)

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i not in keep_set or j not in keep_set:
            continue
        try:
            rw.AddBond(old_to_new[i], old_to_new[j], bond.GetBondType())
            new_bond = rw.GetBondBetweenAtoms(old_to_new[i], old_to_new[j])
            if new_bond is not None:
                new_bond.SetIsAromatic(bond.GetIsAromatic())
        except Exception:
            continue

    out = rw.GetMol()
    try:
        out.UpdatePropertyCache(strict=False)
    except Exception:
        pass
    return sanitize_mol_safe(out)


# Backward-compatible alias used by older notebooks.
def truncate_mol_keep_first_atoms(mol: Chem.Mol, max_nodes: int) -> Optional[Chem.Mol]:
    return truncate_mol_connected(mol, max_nodes)


def mol_to_pyg_with_extras(mol: Chem.Mol, max_nodes: Optional[int] = None) -> Optional[Data]:
    """Convert an RDKit molecule to the active dataset's PyG representation.

    QM9/ZINC
        x = atom one-hot.

    GuacaMol/MOSES
        x = [atom one-hot | formal-charge one-hot over {-1,0,+1}].

    Aromaticity is represented by the aromatic bond channel. It is deliberately
    not duplicated as an independently predicted node scalar, matching the
    working OptionA notebook and avoiding atom/bond aromaticity contradictions.
    """
    if mol is None:
        return None
    mol = sanitize_mol_safe(mol)
    if mol is None:
        return None

    if max_nodes is not None and mol.GetNumAtoms() > max_nodes:
        mol = truncate_mol_connected(mol, max_nodes)
        if mol is None or mol.GetNumAtoms() == 0:
            return None

    n = mol.GetNumAtoms()
    if n == 0:
        return None

    atomic_numbers: List[int] = []
    atom_type_indices: List[int] = []
    charge_values: List[int] = []
    charge_indices: List[int] = []

    for atom in mol.GetAtoms():
        z = int(atom.GetAtomicNum())
        q = int(atom.GetFormalCharge())
        # Do not silently relabel unsupported elements as carbon. That creates
        # contradictory x/z targets and badly distorts FGW feature distances.
        if z not in cfg.Z_TO_TYPE:
            return None
        atomic_numbers.append(z)
        atom_type_indices.append(cfg.Z_TO_TYPE[z])
        charge_values.append(cfg.formal_charge_from_index(cfg.formal_charge_to_index(q)))
        charge_indices.append(cfg.formal_charge_to_index(q))

    z_t = torch.tensor(atomic_numbers, dtype=torch.long)
    atom_x = F.one_hot(
        torch.tensor(atom_type_indices, dtype=torch.long),
        num_classes=cfg.NUM_ATOM_TYPES,
    ).float()

    if cfg.use_formal_charge():
        charge_x = F.one_hot(
            torch.tensor(charge_indices, dtype=torch.long),
            num_classes=len(cfg.FORMAL_CHARGES),
        ).float()
        x = torch.cat([atom_x, charge_x], dim=-1)
    else:
        x = atom_x

    edges: List[List[int]] = []
    bond_type_indices: List[int] = []
    for bond in mol.GetBonds():
        i, j = int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())
        bt = bond.GetBondType()
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
        edges.extend(([i, j], [j, i]))
        bond_type_indices.extend((idx, idx))

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = F.one_hot(
            torch.tensor(bond_type_indices, dtype=torch.long),
            num_classes=cfg.NUM_BOND_TYPES,
        ).float()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, cfg.NUM_BOND_TYPES), dtype=torch.float32)

    data = Data(x=x, z=z_t, edge_index=edge_index, edge_attr=edge_attr)
    # Keep an explicit integer attribute so conversions remain robust even if a
    # downstream transform changes x.
    data.formal_charge = torch.tensor(charge_values, dtype=torch.long)
    return data
