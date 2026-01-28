from typing import Optional, List, Sequence, Tuple
import os
import hashlib
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, download_url
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdchem
from pathlib import Path
from datasets import load_dataset

from fm_graphex.data.rdkit_conversions import sanitize_mol_safe
from fm_graphex.data.common_utils import mol_to_pyg_with_extras


def _ensure_guacamol_download(root):
    # download with datasets and export to .smiles
    root = Path(root)
    root = root / "raw"
    root.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("jxie/guacamol") # splits: train/validation/test

    mapping = {
        "train": "guacamol_v1_train.smiles",
        "validation": "guacamol_v1_valid.smiles",
        "test": "guacamol_v1_test.smiles",
    }

    for split, fname in mapping.items():
        path = root / fname
        with path.open("w", encoding="utf-8") as f:
            # each row has ds[split][ "text" ] = SMILES
            for smi in ds[split]["text"]:
                if smi is None:
                    continue
                smi = smi.strip()
                if smi:
                    f.write(smi + "\n")
        print(f"Wrote {split} -> {path} ({len(ds[split])} lines)")

    return (root / mapping["train"], root / mapping["validation"], root / mapping["test"])


def _read_smiles_file(path: str) -> List[str]:
    with open(path, "r") as f:
        return [ln.strip() for ln in f.readlines() if ln.strip()]

    
def process_graph_guacamol(mol: Chem.Mol, max_nodes: int) -> Optional[Data]:
    # For GuacaMol: keep H implicit; keep charges/aromatic
    mol = sanitize_mol_safe(mol)
    if mol is None:
        return None
    return mol_to_pyg_with_extras(mol, max_nodes=max_nodes)


def load_guacamol_dataset(
    root: str,
    max_nodes: int = 64,
    max_num_graphs: Optional[int] = None,
) -> Tuple[Sequence[Data], Sequence[Data], Sequence[Data], int]:
    """
    Load Guacamol dataset from SMILES splits and convert to canonical Data format.

    Args:
      root: dataset directory (will use {root}/raw)
      max_nodes: truncation cap for heavy atoms
      max_num_graphs: if set, cap train to max_num_graphs and val to 0.2 * max_num_graphs

    Returns:
      train_dataset, val_dataset, test_dataset, max_atoms
      where max_atoms == max_nodes (cap used for dense FM)
    """
    train_path, valid_path, test_path = _ensure_guacamol_download(root)

    train_smiles = _read_smiles_file(train_path)
    val_smiles   = _read_smiles_file(valid_path)
    test_smiles  = _read_smiles_file(test_path)

    def _process_split(smiles_list: List[str]) -> List[Data]:
        out: List[Data] = []
        for s in smiles_list:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue
            d = process_graph_guacamol(mol, max_nodes=max_nodes)
            # keep only usable graphs
            if d is not None and d.num_nodes > 0 and d.edge_index is not None:
                out.append(d)
            # early stop for speed
            if max_num_graphs is not None and len(out) >= max_num_graphs:
                break
        return out

    train_dataset = _process_split(train_smiles)
    val_dataset   = _process_split(val_smiles)
    test_dataset  = _process_split(test_smiles)

    if max_num_graphs is not None:
        # val is 20% of train cap
        train_dataset = train_dataset[:max_num_graphs]
        val_num_graphs = int(max_num_graphs * 0.2)
        val_dataset = val_dataset[:val_num_graphs]

    max_atoms = max_nodes
    return train_dataset, val_dataset, test_dataset, max_atoms