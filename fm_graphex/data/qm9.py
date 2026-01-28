from typing import Optional, List, Sequence, Tuple
import torch
import torch.nn.functional as F
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from rdkit.Chem import rdchem

from fm_graphex.data.rdkit_conversions import kekulize_and_remove_hs, data_to_rdkit_mol
from fm_graphex.data.common_utils import truncate_mol_keep_first_atoms, mol_to_pyg_with_extras


def process_graph_qm9(data: Data, max_nodes: int) -> Optional[Data]:
    mol = data_to_rdkit_mol(data)
    mol = kekulize_and_remove_hs(mol) # heavy-only here
    if mol is None:
        return None
    mol = truncate_mol_keep_first_atoms(mol, max_nodes)
    if mol is None:
        return None
    return mol_to_pyg_with_extras(mol, max_nodes=max_nodes)


def load_qm9_heavy_dataset(
    root: str,
    max_num_graphs: Optional[int] = None,
) -> Tuple[Sequence[Data], Sequence[Data], Sequence[Data], int]:
    raw = QM9(root)
    raw = raw.shuffle()
    raw.shuffle()

    train_raw = raw[:100000]
    val_raw   = raw[100000:120000]
    test_raw  = raw[120000:]

    max_atoms = max(d.num_nodes for d in raw)

    def _process_split(split_raw):
        out = []
        for d in split_raw:
            pd = process_graph_qm9(d, max_nodes=max_atoms)
            if pd is not None and pd.num_nodes > 0:
                out.append(pd)
        return out

    train_dataset = _process_split(train_raw)
    val_dataset   = _process_split(val_raw)
    test_dataset  = _process_split(test_raw)

    if max_num_graphs is not None:
        train_dataset = train_dataset[:max_num_graphs]
        val_num_graphs = int(max_num_graphs * 0.2)
        val_dataset = val_dataset[:val_num_graphs]

    return train_dataset, val_dataset, test_dataset, max_atoms