from typing import Optional, List, Sequence, Tuple
import torch
import torch.nn.functional as F
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from rdkit.Chem import rdchem


from fm_graphex.data.rdkit_conversions import (
    kekulize_and_remove_hs,
    data_to_rdkit_mol,
)

# 28 raw ZINC atom classes -> 9 atom types (C,N,O,F,Br,Cl,I,P,S)
ZINC_INDS_28_TO_9 = torch.tensor(
    [0, 2, 1, 3, 0, 8, 5, 2, 1, 4, 1, 1, 1, 1, 8, 6, 7, 2, 1, 2, 8, 7, 7, 0, 7, 8, 0, 7],
    dtype=torch.long
)

# 9 types -> atomic number z (C,N,O,F,Br,Cl,I,P,S)
TYPE_TO_Z_9 = torch.tensor([6, 7, 8, 9, 35, 17, 53, 15, 16], dtype=torch.long)

# NUM_ATOM_TYPES = 9
# NUM_BOND_TYPES = 4 # single/double/triple/aromatic


def process_graph_zinc(data: Data, max_nodes: int = 38) -> Data:
    """
    Convert PyG ZINC graph (x integer atom classes, edge_attr integer bond classes)
    into your codebase format:
      x: (N,9) one-hot
      z: (N,) atomic numbers
      edge_attr: (E,4) one-hot
    Never returns None.
    """
    from fm_graphex.config import (
        Z_TO_TYPE,
        NUM_ATOM_TYPES,
        NUM_BOND_TYPES,
    )

    # nodes: x is [N,1] or [N] with values in 0..27
    x_raw = data.x
    if x_raw is None:
        x_raw = torch.zeros((0,), dtype=torch.long)

    if x_raw.dim() == 2 and x_raw.size(-1) == 1:
        x_raw = x_raw.view(-1)
    x_raw = x_raw.long()

    # truncate nodes
    if x_raw.numel() > max_nodes:
        x_raw = x_raw[:max_nodes]

    N = int(x_raw.numel())

    # map 0..27 -> 0..8
    x_raw = x_raw.clamp(0, ZINC_INDS_28_TO_9.numel() - 1)
    type_idx = ZINC_INDS_28_TO_9[x_raw] # (N,)
    x_onehot = F.one_hot(type_idx, num_classes=NUM_ATOM_TYPES).float()
    z = TYPE_TO_Z_9[type_idx] # (N,)

    # edges
    edge_index = data.edge_index
    edge_attr = data.edge_attr

    if edge_index is None or edge_index.numel() == 0 or N == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr_oh = torch.empty((0, NUM_BOND_TYPES), dtype=torch.float)
        return Data(x=x_onehot, z=z, edge_index=edge_index, edge_attr=edge_attr_oh, y=getattr(data, "y", None))

    # keep only edges inside truncated node set
    mask = (edge_index[0] < N) & (edge_index[1] < N)
    edge_index = edge_index[:, mask]

    # edge_attr can be [E] or [E,1]
    if edge_attr is None:
        e_idx = torch.zeros((edge_index.size(1),), dtype=torch.long)
    else:
        if edge_attr.dim() == 2 and edge_attr.size(-1) == 1:
            e_idx = edge_attr.view(-1).long()[mask]
        else:
            e_idx = edge_attr.long()[mask]

        # handle 1-indexed bonds (1..4) vs 0-indexed (0..3)
        if e_idx.numel() > 0:
            mn, mx = int(e_idx.min().item()), int(e_idx.max().item())
            if mn >= 1 and mx <= 4:
                e_idx = e_idx - 1

        e_idx = e_idx.clamp(0, NUM_BOND_TYPES - 1)

    edge_attr_oh = F.one_hot(e_idx, num_classes=NUM_BOND_TYPES).float()

    out = Data(
        x=x_onehot,
        z=z,
        edge_index=edge_index.long(),
        edge_attr=edge_attr_oh,
        y=getattr(data, "y", None),
    )
    return out


def load_zinc_dataset(root: str, max_num_graphs: Optional[int] = None):
    # full ZINC (subset=False) splits: train/val/test
    train_raw = ZINC(root=root, subset=False, split="train")
    val_raw   = ZINC(root=root, subset=False, split="val")
    test_raw  = ZINC(root=root, subset=False, split="test")

    # typical upper bound ~ 38 atoms
    max_atoms = 38

    def _process(split):
        out = []
        for d in split:
            pd_ = process_graph_zinc(d, max_nodes=max_atoms)
            if pd_ is not None and pd_.num_nodes > 0:
                out.append(pd_)
        return out

    train = _process(train_raw)
    val   = _process(val_raw)
    test  = _process(test_raw)

    if max_num_graphs is not None:
        train = train[:max_num_graphs]
        val_num_graphs = int(max_num_graphs * 0.2)
        val = val[:val_num_graphs]

    return train, val, test, max_atoms