from __future__ import annotations

from typing import Optional

from rdkit import Chem
from rdkit.Chem import rdchem
from torch_geometric.data import Data

import fm_graphex.config as cfg


def sanitize_mol_safe(mol: Chem.Mol) -> Optional[Chem.Mol]:
    if mol is None:
        return None
    out = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(out)
        return out
    except Exception:
        return None


def sanitize_mol_with_fallback(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """Full RDKit sanitization with a conservative aromatic fallback."""
    out = sanitize_mol_safe(mol)
    if out is not None:
        return out
    if mol is None:
        return None
    try:
        rw = Chem.RWMol(mol)
        for bond in rw.GetBonds():
            if bond.GetBondType() == rdchem.BondType.AROMATIC:
                bond.SetBondType(rdchem.BondType.SINGLE)
                bond.SetIsAromatic(False)
        for atom in rw.GetAtoms():
            atom.SetIsAromatic(False)
        return sanitize_mol_safe(rw.GetMol())
    except Exception:
        return None


def kekulize_preserve_hs(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """Validate a GuacaMol/MOSES molecule without removing hydrogens.

    The previous function always cleared aromatic flags despite its name. For
    benchmark canonical SMILES it is preferable to preserve RDKit aromaticity.
    """
    return sanitize_mol_safe(mol)


def kekulize_and_remove_hs(mol: Chem.Mol) -> Optional[Chem.Mol]:
    mol = sanitize_mol_safe(mol)
    if mol is None:
        return None
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception:
        pass
    try:
        mol = Chem.RemoveHs(mol)
    except Exception:
        return None
    return sanitize_mol_safe(mol)


def _charge_from_data(data: Data, atom_idx: int) -> int:
    charges = getattr(data, "formal_charge", None)
    if charges is not None and int(charges.numel()) > atom_idx:
        return int(charges[atom_idx].detach().cpu().item())

    x = getattr(data, "x", None)
    q0 = cfg.NUM_ATOM_TYPES
    qdim = len(cfg.FORMAL_CHARGES)
    if cfg.use_formal_charge() and x is not None and x.dim() == 2 and x.size(1) >= q0 + qdim:
        qidx = int(x[atom_idx, q0:q0 + qdim].argmax().detach().cpu().item())
        return cfg.formal_charge_from_index(qidx)
    return 0


def data_to_rdkit_mol(data: Data) -> Optional[Chem.Mol]:
    """Convert processed PyG data back to an RDKit molecule.

    This version preserves formal charges and aromatic bonds for GuacaMol and
    MOSES. The old converter rebuilt every atom as neutral, which made charge
    statistics and the OptionA charge fallback silently incorrect.
    """
    if getattr(data, "edge_index", None) is None:
        return None
    n = int(data.num_nodes)
    if n == 0:
        return None

    atomic_numbers = None
    atom_symbols = None
    if hasattr(data, "z") and data.z is not None and int(data.z.numel()) == n:
        # Use atomic numbers directly. This is essential for raw QM9, whose
        # PyG graphs still contain explicit hydrogens before heavy-atom
        # preprocessing; mapping an unsupported H to carbon corrupts the graph.
        atomic_numbers = [int(z) for z in data.z.detach().cpu().tolist()]
    else:
        x = getattr(data, "x", None)
        if x is None or x.dim() != 2 or x.size(1) < cfg.NUM_ATOM_TYPES:
            return None
        atom_idx = x[:, :cfg.NUM_ATOM_TYPES].argmax(dim=-1).detach().cpu().tolist()
        atom_symbols = [cfg.TYPE_TO_SYM.get(int(i), "C") for i in atom_idx]

    rw = rdchem.RWMol()
    for i in range(n):
        atom = (
            rdchem.Atom(int(atomic_numbers[i]))
            if atomic_numbers is not None
            else rdchem.Atom(atom_symbols[i])
        )
        q = _charge_from_data(data, i)
        if q:
            atom.SetFormalCharge(q)
        rw.AddAtom(atom)

    edge_index = data.edge_index.detach().cpu()
    edge_attr = data.edge_attr.detach().cpu() if getattr(data, "edge_attr", None) is not None else None
    seen = set()
    for e in range(edge_index.size(1)):
        i = int(edge_index[0, e])
        j = int(edge_index[1, e])
        if i == j:
            continue
        a, b = sorted((i, j))
        if (a, b) in seen:
            continue
        seen.add((a, b))

        bt_idx = 0
        if edge_attr is not None and edge_attr.size(0) == edge_index.size(1):
            bt_idx = int(edge_attr[e, :cfg.NUM_BOND_TYPES].argmax().item())
        bt_idx = max(0, min(bt_idx, cfg.NUM_BOND_TYPES - 1))
        bt = cfg.BOND_TYPES[bt_idx]
        try:
            rw.AddBond(a, b, bt)
            if bt == rdchem.BondType.AROMATIC:
                rw.GetAtomWithIdx(a).SetIsAromatic(True)
                rw.GetAtomWithIdx(b).SetIsAromatic(True)
                bond = rw.GetBondBetweenAtoms(a, b)
                if bond is not None:
                    bond.SetIsAromatic(True)
        except Exception:
            continue

    return sanitize_mol_safe(rw.GetMol())