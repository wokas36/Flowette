from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

import fm_graphex.config as cfg


def _normalise_prob(prob: Optional[np.ndarray], size: int, neutral_idx: Optional[int] = None) -> np.ndarray:
    if prob is None:
        out = np.zeros(size, dtype=np.float64)
        out[neutral_idx if neutral_idx is not None else 0] = 1.0
        return out
    out = np.asarray(prob, dtype=np.float64).reshape(-1)
    if out.size != size or not np.isfinite(out).all() or out.sum() <= 0:
        fallback = np.zeros(size, dtype=np.float64)
        fallback[neutral_idx if neutral_idx is not None else 0] = 1.0
        return fallback
    out = np.maximum(out, 0.0)
    return out / out.sum()


def sample_random_noise_features(
    n_nodes: int,
    edge_index: torch.Tensor,
    atom_prob: np.ndarray,
    atom_z: np.ndarray,
    bond_prob: np.ndarray,
    formal_charge_prob: Optional[np.ndarray] = None,
):
    """Sample graphex source features in the same space as target data."""
    device = edge_index.device
    atom_prob = _normalise_prob(atom_prob, cfg.NUM_ATOM_TYPES)
    bond_prob = _normalise_prob(bond_prob, cfg.NUM_BOND_TYPES)

    type_idx = np.random.choice(cfg.NUM_ATOM_TYPES, size=n_nodes, p=atom_prob)
    type_idx_t = torch.tensor(type_idx, dtype=torch.long, device=device)
    x_atom = F.one_hot(type_idx_t, num_classes=cfg.NUM_ATOM_TYPES).float()
    z = torch.tensor(atom_z[type_idx], dtype=torch.long, device=device)

    if cfg.use_formal_charge():
        charge_prob = _normalise_prob(
            formal_charge_prob,
            len(cfg.FORMAL_CHARGES),
            neutral_idx=cfg.NEUTRAL_CHARGE_IDX,
        )
        charge_idx = np.random.choice(len(cfg.FORMAL_CHARGES), size=n_nodes, p=charge_prob)
        charge_idx_t = torch.tensor(charge_idx, dtype=torch.long, device=device)
        charge_x = F.one_hot(charge_idx_t, num_classes=len(cfg.FORMAL_CHARGES)).float()
        x = torch.cat([x_atom, charge_x], dim=-1)
        formal_charge = torch.tensor(
            [cfg.formal_charge_from_index(int(i)) for i in charge_idx],
            dtype=torch.long,
            device=device,
        )
    else:
        x = x_atom
        formal_charge = torch.zeros(n_nodes, dtype=torch.long, device=device)

    e = int(edge_index.size(1))
    if e:
        bond_idx = np.random.choice(cfg.NUM_BOND_TYPES, size=e, p=bond_prob)
        edge_attr = F.one_hot(
            torch.tensor(bond_idx, dtype=torch.long, device=device),
            num_classes=cfg.NUM_BOND_TYPES,
        ).float()
    else:
        edge_attr = torch.empty((0, cfg.NUM_BOND_TYPES), dtype=torch.float32, device=device)

    pos = torch.randn(n_nodes, 3, device=device)
    if x.size(1) != cfg.node_feat_dim():
        raise RuntimeError(
            f"Noise x has dim {x.size(1)}, expected {cfg.node_feat_dim()} for {cfg.get_spec().name}."
        )
    return x, z, edge_attr, pos, formal_charge


def igraph_to_pyg(
    gr,
    atom_prob: np.ndarray,
    atom_z: np.ndarray,
    bond_prob: np.ndarray,
    formal_charge_prob: Optional[np.ndarray] = None,
) -> Data:
    """Convert an igraph source graph to bidirectional PyG format."""
    undirected = gr.get_edgelist() if gr.ecount() > 0 else []
    directed = []
    for u, v in undirected:
        directed.append((u, v))
        directed.append((v, u))
    if directed:
        edge_index = torch.tensor(directed, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    n_nodes = int(gr.vcount())
    x, z, edge_attr, pos, formal_charge = sample_random_noise_features(
        n_nodes=n_nodes,
        edge_index=edge_index,
        atom_prob=atom_prob,
        atom_z=atom_z,
        bond_prob=bond_prob,
        formal_charge_prob=formal_charge_prob,
    )
    return Data(
        x=x,
        z=z,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos,
        formal_charge=formal_charge,
    )


def igraph_to_rdkit_mol(
    gr,
    z: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
):
    """Convert sampled igraph/PyG tensors to an RDKit molecule.

    This compatibility helper handles bidirectional ``edge_index`` safely and
    reads the active dataset's bond vocabulary at runtime.
    """
    from rdkit import Chem
    from rdkit.Chem import rdchem

    rw = Chem.RWMol()
    for i in range(int(gr.vcount())):
        rw.AddAtom(Chem.Atom(int(z[i].detach().cpu())))

    seen = set()
    for e in range(int(edge_index.size(1))):
        u, v = int(edge_index[0, e]), int(edge_index[1, e])
        if u == v:
            continue
        a, b = sorted((u, v))
        if (a, b) in seen:
            continue
        seen.add((a, b))
        bidx = int(edge_attr[e, :cfg.NUM_BOND_TYPES].argmax().detach().cpu())
        bt = cfg.BOND_TYPES[min(max(bidx, 0), cfg.NUM_BOND_TYPES - 1)]
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

    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
        return out
    except Exception:
        return None