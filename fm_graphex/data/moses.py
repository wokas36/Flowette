import os
from pathlib import Path
from typing import Tuple, List, Optional, Sequence
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, download_url
from rdkit import Chem
from rdkit.Chem import rdchem
import pandas as pd

from fm_graphex.data.rdkit_conversions import sanitize_mol_safe
from fm_graphex.data.common_utils import mol_to_pyg_with_extras


MOSES_TRAIN_URL = "https://media.githubusercontent.com/media/molecularsets/moses/master/data/train.csv"
MOSES_VALID_URL = "https://media.githubusercontent.com/media/molecularsets/moses/master/data/test.csv"
MOSES_TEST_URL  = "https://media.githubusercontent.com/media/molecularsets/moses/master/data/test_scaffolds.csv"


def _ensure_moses_download(root: str) -> Tuple[str, str, str]:
    """
    Ensures MOSES official split files exist under:
      {root}/raw/train_moses.csv
      {root}/raw/val_moses.csv
      {root}/raw/test_moses.csv

    Mapping:
      train -> train.csv
      val   -> test.csv
      test  -> test_scaffolds.csv
    """
    raw = Path(root) / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    train_csv = raw / "train_moses.csv"
    val_csv   = raw / "val_moses.csv"
    test_csv  = raw / "test_moses.csv"

    if not train_csv.exists():
        tmp = download_url(MOSES_TRAIN_URL, str(raw))
        os.replace(tmp, train_csv)

    if not val_csv.exists():
        tmp = download_url(MOSES_VALID_URL, str(raw))
        os.replace(tmp, val_csv)

    if not test_csv.exists():
        tmp = download_url(MOSES_TEST_URL, str(raw))
        os.replace(tmp, test_csv)

    return str(train_csv), str(val_csv), str(test_csv)


def _read_smiles_from_csv(csv_path: str, smiles_col: str = "SMILES") -> List[str]:
    df = pd.read_csv(csv_path)
    if smiles_col not in df.columns:
        raise KeyError(f"Expected column '{smiles_col}' in {csv_path}, got {list(df.columns)}")
    out = []
    for s in df[smiles_col].astype(str).tolist():
        s = s.strip()
        if s:
            out.append(s)
    return out


def process_graph_moses(mol: Chem.Mol, max_nodes: int) -> Optional[Data]:
    mol = sanitize_mol_safe(mol)
    if mol is None:
        return None
    # Prefer truncation rather than dropping, but MOSES papers often use <= 27 heavy atoms.
    return mol_to_pyg_with_extras(mol, max_nodes=max_nodes)


def load_moses_dataset(
    root: str,
    max_nodes: int = 27,
    max_num_graphs: Optional[int] = None,
) -> Tuple[Sequence[Data], Sequence[Data], Sequence[Data], int]:
    """
    Returns:
      train_dataset, val_dataset, test_dataset, max_atoms
    using MOSES official split mapping:
      train <- train.csv
      val   <- test.csv
      test  <- test_scaffolds.csv
    """
    train_csv, val_csv, test_csv = _ensure_moses_download(root)

    train_smiles = _read_smiles_from_csv(train_csv, smiles_col="SMILES")
    val_smiles   = _read_smiles_from_csv(val_csv, smiles_col="SMILES")
    test_smiles  = _read_smiles_from_csv(test_csv, smiles_col="SMILES")

    def _process(smiles_list: List[str]) -> List[Data]:
        out = []
        for s in smiles_list:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue
            d = process_graph_moses(mol, max_nodes=max_nodes)
            if d is not None:
                out.append(d)
            if max_num_graphs is not None and len(out) >= max_num_graphs:
                break
        return out

    train_dataset = _process(train_smiles)
    val_dataset   = _process(val_smiles)
    test_dataset  = _process(test_smiles)
    
    if max_num_graphs is not None:
        # val is 20% of train cap
        train_dataset = train_dataset[:max_num_graphs]
        val_num_graphs = int(max_num_graphs * 0.2)
        val_dataset = val_dataset[:val_num_graphs]

    return train_dataset, val_dataset, test_dataset, max_nodes