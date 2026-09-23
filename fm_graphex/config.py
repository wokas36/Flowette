from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import random

import numpy as np
import torch
from rdkit.Chem import rdchem
from torch_geometric.data import Data


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Formal-charge representation used only for GuacaMol and MOSES.
# The three channels are categorical logits/one-hot entries, not a scalar.
FORMAL_CHARGES: Tuple[int, ...] = (-1, 0, 1)
NEUTRAL_CHARGE_IDX: int = FORMAL_CHARGES.index(0)
FEATURE_VERSION = "flowette_amortized_v"

@dataclass(frozen=True)
class MolGraphSpec:
    name: str
    atom_types: List[str]
    sym_to_z: Dict[str, int]
    z_to_type: Dict[int, int]
    bond_types: List[rdchem.BondType]
    max_valence_per_type: torch.Tensor
    bond_orders: torch.Tensor
    extra_node_feat_dim: int
    root: str
    cycles_count_path: str
    nodes_count_path: str
    cycles_count_train_path: str
    cycles_count_test_path: str

    @property
    def num_atom_types(self) -> int:
        return len(self.atom_types)

    @property
    def num_bond_types(self) -> int:
        return len(self.bond_types)

    @property
    def type_to_sym(self) -> Dict[int, str]:
        return {i: s for i, s in enumerate(self.atom_types)}

    @property
    def type_to_z(self) -> Dict[int, int]:
        return {i: self.sym_to_z[s] for i, s in enumerate(self.atom_types)}


_COMMON_BONDS = [
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC,
]
_COMMON_BOND_ORDERS = torch.tensor([1.0, 2.0, 3.0, 1.5])


QM9_SPEC = MolGraphSpec(
    name="qm9",
    atom_types=["C", "N", "O", "F"],
    sym_to_z={"C": 6, "N": 7, "O": 8, "F": 9},
    z_to_type={6: 0, 7: 1, 8: 2, 9: 3},
    bond_types=_COMMON_BONDS,
    max_valence_per_type=torch.tensor([4.0, 3.0, 2.0, 1.0]),
    bond_orders=_COMMON_BOND_ORDERS,
    extra_node_feat_dim=0,
    root="./dataset/QM9",
    cycles_count_path="./fm_graphex/files/cycles_count_qm9.txt",
    nodes_count_path="./fm_graphex/files/nodes_count_qm9.txt",
    cycles_count_train_path="./fm_graphex/files/cycle_count_full_train_qm9.txt",
    cycles_count_test_path="./fm_graphex/files/cycle_count_full_test_qm9.txt",
)

ZINC_SPEC = MolGraphSpec(
    name="zinc",
    atom_types=["C", "N", "O", "F", "Br", "Cl", "I", "P", "S"],
    sym_to_z={
        "C": 6, "N": 7, "O": 8, "F": 9, "Br": 35,
        "Cl": 17, "I": 53, "P": 15, "S": 16,
    },
    z_to_type={6: 0, 7: 1, 8: 2, 9: 3, 35: 4, 17: 5, 53: 6, 15: 7, 16: 8},
    bond_types=_COMMON_BONDS,
    max_valence_per_type=torch.tensor([4.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0, 5.0, 6.0]),
    bond_orders=_COMMON_BOND_ORDERS,
    extra_node_feat_dim=0,
    root="./dataset/ZINC",
    cycles_count_path="./fm_graphex/files/cycles_count_zinc.txt",
    nodes_count_path="./fm_graphex/files/nodes_count_zinc.txt",
    cycles_count_train_path="./fm_graphex/files/cycle_count_full_train_zinc.txt",
    cycles_count_test_path="./fm_graphex/files/cycle_count_full_test_zinc.txt",
)

GUACAMOL_SPEC = MolGraphSpec(
    name="guacamol",
    atom_types=["C", "N", "O", "F", "B", "Br", "Cl", "I", "P", "S", "Se", "Si"],
    sym_to_z={
        "C": 6, "N": 7, "O": 8, "F": 9, "B": 5, "Br": 35,
        "Cl": 17, "I": 53, "P": 15, "S": 16, "Se": 34, "Si": 14,
    },
    z_to_type={6: 0, 7: 1, 8: 2, 9: 3, 5: 4, 35: 5, 17: 6, 53: 7, 15: 8, 16: 9, 34: 10, 14: 11},
    bond_types=_COMMON_BONDS,
    max_valence_per_type=torch.tensor([4.0, 3.0, 2.0, 1.0, 3.0, 1.0, 1.0, 1.0, 5.0, 6.0, 6.0, 4.0]),
    bond_orders=_COMMON_BOND_ORDERS,
    # atom identity + categorical formal charge {-1, 0, +1}.
    extra_node_feat_dim=len(FORMAL_CHARGES),
    root="./dataset/GUACAMOL",
    cycles_count_path="./fm_graphex/files/cycles_count_guacamol.txt",
    nodes_count_path="./fm_graphex/files/nodes_count_guacamol.txt",
    cycles_count_train_path="./fm_graphex/files/cycle_count_full_train_guacamol.txt",
    cycles_count_test_path="./fm_graphex/files/cycle_count_full_test_guacamol.txt",
)

MOSES_SPEC = MolGraphSpec(
    name="moses",
    atom_types=["C", "N", "S", "O", "F", "Cl", "Br"],
    sym_to_z={"C": 6, "N": 7, "S": 16, "O": 8, "F": 9, "Cl": 17, "Br": 35},
    z_to_type={6: 0, 7: 1, 16: 2, 8: 3, 9: 4, 17: 5, 35: 6},
    bond_types=_COMMON_BONDS,
    max_valence_per_type=torch.tensor([4.0, 3.0, 6.0, 2.0, 1.0, 1.0, 1.0]),
    bond_orders=_COMMON_BOND_ORDERS,
    extra_node_feat_dim=len(FORMAL_CHARGES),
    root="./dataset/MOSES",
    cycles_count_path="./fm_graphex/files/cycles_count_moses.txt",
    nodes_count_path="./fm_graphex/files/nodes_count_moses.txt",
    cycles_count_train_path="./fm_graphex/files/cycle_count_full_train_moses.txt",
    cycles_count_test_path="./fm_graphex/files/cycle_count_full_test_moses.txt",
)


_ACTIVE_SPEC = QM9_SPEC

# Backward-compatible globals. They are refreshed by set_dataset(). Modules
# that can be imported before set_dataset() should import this module as cfg
# and read cfg.NUM_ATOM_TYPES at runtime rather than copying the value.
ATOM_TYPES: List[str]
Z_TO_TYPE: Dict[int, int]
TYPE_TO_SYM: Dict[int, str]
TYPE_TO_Z: Dict[int, int]
BOND_TYPES: List[rdchem.BondType]
NUM_ATOM_TYPES: int
NUM_BOND_TYPES: int
MAX_VALENCE_PER_TYPE: torch.Tensor
BOND_ORDERS: torch.Tensor
EXTRA_NODE_FEATS: int
ROOT: str
CYCLES_COUNT_PATH: str
NODES_COUNT_PATH: str
CYCLES_COUNT_TRAIN_PATH: str
CYCLES_COUNT_TEST_PATH: str


def _refresh_globals() -> None:
    global ATOM_TYPES, Z_TO_TYPE, TYPE_TO_SYM, TYPE_TO_Z
    global BOND_TYPES, NUM_ATOM_TYPES, NUM_BOND_TYPES
    global MAX_VALENCE_PER_TYPE, BOND_ORDERS, EXTRA_NODE_FEATS
    global ROOT, CYCLES_COUNT_PATH, NODES_COUNT_PATH
    global CYCLES_COUNT_TRAIN_PATH, CYCLES_COUNT_TEST_PATH

    s = _ACTIVE_SPEC
    ATOM_TYPES = list(s.atom_types)
    Z_TO_TYPE = dict(s.z_to_type)
    TYPE_TO_SYM = dict(s.type_to_sym)
    TYPE_TO_Z = dict(s.type_to_z)
    BOND_TYPES = list(s.bond_types)
    NUM_ATOM_TYPES = s.num_atom_types
    NUM_BOND_TYPES = s.num_bond_types
    MAX_VALENCE_PER_TYPE = s.max_valence_per_type.clone()
    BOND_ORDERS = s.bond_orders.clone()
    EXTRA_NODE_FEATS = int(s.extra_node_feat_dim)
    ROOT = s.root
    CYCLES_COUNT_PATH = s.cycles_count_path
    NODES_COUNT_PATH = s.nodes_count_path
    CYCLES_COUNT_TRAIN_PATH = s.cycles_count_train_path
    CYCLES_COUNT_TEST_PATH = s.cycles_count_test_path


def set_dataset(task: str) -> None:
    global _ACTIVE_SPEC
    t = str(task).lower()
    if t.startswith("qm9"):
        _ACTIVE_SPEC = QM9_SPEC
    elif t.startswith("zinc"):
        _ACTIVE_SPEC = ZINC_SPEC
    elif t.startswith("guacamol"):
        _ACTIVE_SPEC = GUACAMOL_SPEC
    elif t.startswith("moses"):
        _ACTIVE_SPEC = MOSES_SPEC
    else:
        raise ValueError(f"Unknown task={task!r}. Use qm9, zinc, guacamol, or moses.")
    _refresh_globals()


def get_spec() -> MolGraphSpec:
    return _ACTIVE_SPEC


def use_formal_charge(task: Optional[str] = None) -> bool:
    name = get_spec().name if task is None else str(task).lower()
    return name in {"guacamol", "moses"}


def formal_charge_dim(task: Optional[str] = None) -> int:
    return len(FORMAL_CHARGES) if use_formal_charge(task) else 0


def formal_charge_to_index(charge: int) -> int:
    q = int(charge)
    if q in FORMAL_CHARGES:
        return FORMAL_CHARGES.index(q)
    if q < FORMAL_CHARGES[0]:
        return 0
    if q > FORMAL_CHARGES[-1]:
        return len(FORMAL_CHARGES) - 1
    return NEUTRAL_CHARGE_IDX


def formal_charge_from_index(index: int) -> int:
    i = max(0, min(int(index), len(FORMAL_CHARGES) - 1))
    return int(FORMAL_CHARGES[i])


def node_feat_dim() -> int:
    return NUM_ATOM_TYPES + EXTRA_NODE_FEATS


def charge_slice(task: Optional[str] = None) -> slice:
    if not use_formal_charge(task):
        return slice(NUM_ATOM_TYPES, NUM_ATOM_TYPES)
    return slice(NUM_ATOM_TYPES, NUM_ATOM_TYPES + len(FORMAL_CHARGES))


def get_root() -> str:
    return ROOT


def get_cycles_count_path() -> str:
    return CYCLES_COUNT_PATH


def get_nodes_count_path() -> str:
    return NODES_COUNT_PATH


def get_cycles_count_train_path() -> str:
    return CYCLES_COUNT_TRAIN_PATH


def get_cycles_count_test_path() -> str:
    return CYCLES_COUNT_TEST_PATH


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_node_count_list(dataset: Sequence[Data]) -> List[int]:
    return [int(d.num_nodes) for d in dataset if getattr(d, "num_nodes", None) and int(d.num_nodes) > 0]


def estimate_molecular_feature_probs(
    dataset: Sequence[Data],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Estimate atom, bond and (when applicable) formal-charge priors.

    Returns atom_prob, atom_z, bond_prob, charge_prob. Charge probabilities are
    None for QM9/ZINC. A small pseudocount prevents zero-probability categories.
    """
    atom_counts = np.full(NUM_ATOM_TYPES, 1e-3, dtype=np.float64)
    bond_counts = np.full(NUM_BOND_TYPES, 1e-3, dtype=np.float64)
    charge_counts = (
        np.full(len(FORMAL_CHARGES), 1e-3, dtype=np.float64)
        if use_formal_charge() else None
    )

    for data in dataset:
        x = getattr(data, "x", None)
        if x is not None and x.dim() == 2 and x.size(1) >= NUM_ATOM_TYPES:
            atom_counts += x[:, :NUM_ATOM_TYPES].sum(dim=0).detach().cpu().numpy()
            if charge_counts is not None and x.size(1) >= NUM_ATOM_TYPES + len(FORMAL_CHARGES):
                charge_counts += (
                    x[:, NUM_ATOM_TYPES:NUM_ATOM_TYPES + len(FORMAL_CHARGES)]
                    .sum(dim=0).detach().cpu().numpy()
                )

        edge_attr = getattr(data, "edge_attr", None)
        if edge_attr is not None and edge_attr.dim() == 2 and edge_attr.size(1) >= NUM_BOND_TYPES:
            bond_counts += edge_attr[:, :NUM_BOND_TYPES].sum(dim=0).detach().cpu().numpy()

    atom_prob = atom_counts / atom_counts.sum()
    bond_prob = bond_counts / bond_counts.sum()
    atom_z = np.array([TYPE_TO_Z[i] for i in range(NUM_ATOM_TYPES)], dtype=int)
    charge_prob = None if charge_counts is None else charge_counts / charge_counts.sum()
    return atom_prob, atom_z, bond_prob, charge_prob


def estimate_atom_and_bond_probs(dataset: Sequence[Data]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward-compatible wrapper returning the original three values."""
    atom_prob, atom_z, bond_prob, _ = estimate_molecular_feature_probs(dataset)
    return atom_prob, atom_z, bond_prob


def estimate_edge_density(dataset: Sequence[Data]) -> float:
    num_edges = 0
    num_possible = 0
    for d in dataset:
        n = int(d.num_nodes)
        if n <= 1:
            continue
        e = int(d.edge_index.shape[1])
        # Processed molecular graphs are bidirectional. If a legacy graph is
        # not, counting unique pairs still gives the correct undirected value.
        if e:
            pairs = {
                tuple(sorted((int(d.edge_index[0, k]), int(d.edge_index[1, k]))))
                for k in range(e)
                if int(d.edge_index[0, k]) != int(d.edge_index[1, k])
            }
            num_edges += len(pairs)
        num_possible += n * (n - 1) // 2
    if num_possible == 0:
        return 0.1
    return float(num_edges / num_possible)


_refresh_globals()