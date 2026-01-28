from typing import List, Dict
from rdkit.Chem import rdchem
from dataclasses import dataclass
import numpy as np
import random
import torch
from torch_geometric.data import Data


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset spec + globals
@dataclass(frozen=True)
class MolGraphSpec:
    name: str
    atom_types: List[str] # vocab
    sym_to_z: Dict[str, int] # symbol -> atomic number
    z_to_type: Dict[int, int] # atomic number -> type index
    bond_types: List[rdchem.BondType] # 4 bond types
    max_valence_per_type: torch.Tensor # (num_atom_types,)
    bond_orders: torch.Tensor # (num_bond_types,)
    extra_node_feat_dim: int # extra node featires for Guacamol and Moses datasets
    root: str # dataset path
    cycles_count_path: str # cycle count file path
    nodes_count_path: str # node count file path
    cycles_count_train_path: str # cycle count train file path
    cycles_count_test_path: str # cycle count test file path
    

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


QM9_SPEC = MolGraphSpec(
    name="qm9",
    atom_types=["C", "N", "O", "F"],
    sym_to_z={"C": 6, "N": 7, "O": 8, "F": 9},
    z_to_type={6: 0, 7: 1, 8: 2, 9: 3},
    bond_types=[
        rdchem.BondType.SINGLE,
        rdchem.BondType.DOUBLE,
        rdchem.BondType.TRIPLE,
        rdchem.BondType.AROMATIC,
    ],
    max_valence_per_type=torch.tensor([4.0, 3.0, 2.0, 1.0]),
    bond_orders=torch.tensor([1.0, 2.0, 3.0, 1.5]),
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
    sym_to_z={"C": 6, "N": 7, "O": 8, "F": 9, "Br": 35, "Cl": 17, "I": 53, "P": 15, "S": 16},
    z_to_type={6: 0, 7: 1, 8: 2, 9: 3, 35: 4, 17: 5, 53: 6, 15: 7, 16: 8},
    bond_types=[
        rdchem.BondType.SINGLE,
        rdchem.BondType.DOUBLE,
        rdchem.BondType.TRIPLE,
        rdchem.BondType.AROMATIC,
    ],
    # conservative caps; tune if needed
    max_valence_per_type=torch.tensor([4.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0, 5.0, 6.0]),
    bond_orders=torch.tensor([1.0, 2.0, 3.0, 1.5]),
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
    sym_to_z={"C": 6, "N": 7, "O": 8, "F": 9, "B": 5, "Br": 35, "Cl": 17, "I": 53, "P": 15, "S": 16, "Se": 34, "Si": 14},
    z_to_type={6:0, 7:1, 8:2, 9:3, 5:4, 35:5, 17:6, 53:7, 15:8, 16:9, 34:10, 14:11},
    bond_types=[
        rdchem.BondType.SINGLE,
        rdchem.BondType.DOUBLE,
        rdchem.BondType.TRIPLE,
        rdchem.BondType.AROMATIC,
    ],
    # conservative typical max valences
    max_valence_per_type=torch.tensor([4.0, 3.0, 2.0, 1.0, 3.0, 1.0, 1.0, 1.0, 5.0, 6.0, 2.0, 4.0]),
    # max_valence_per_type=torch.tensor([4.0, 3.0, 2.0, 1.0, 3.0, 1.0, 1.0, 1.0, 3.0, 2.0, 2.0, 4.0]),
    bond_orders=torch.tensor([1.0, 2.0, 3.0, 1.5]),
    extra_node_feat_dim=2, # [formal_charge_norm, is_aromatic]
    root="./dataset/GUACAMOL",
    cycles_count_path="./fm_graphex/files/cycles_count_guacamol.txt",
    nodes_count_path="./fm_graphex/files/nodes_count_guacamol.txt",
    cycles_count_train_path="./fm_graphex/files/cycle_count_full_train_guacamol.txt",
    cycles_count_test_path="./fm_graphex/files/cycle_count_full_test_guacamol.txt",
)

MOSES_SPEC = MolGraphSpec(
    name="moses",
    # MOSES atom vocabulary (heavy atoms + explicit H)
    atom_types=["C", "N", "S", "O", "F", "Cl", "Br"],
    sym_to_z={
        "C": 6,
        "N": 7,
        "S": 16,
        "O": 8,
        "F": 9,
        "Cl": 17,
        "Br": 35,
        # "H": 1,
    },
    z_to_type={
        6: 0,   # C
        7: 1,   # N
        16: 2,  # S
        8: 3,   # O
        9: 4,   # F
        17: 5,  # Cl
        35: 6,  # Br
        # 1: 7,   # H
    },
    bond_types=[
        rdchem.BondType.SINGLE,
        rdchem.BondType.DOUBLE,
        rdchem.BondType.TRIPLE,
        rdchem.BondType.AROMATIC,
    ],
    # Conservative chemistry-safe caps (for valence-aware pruning)
    max_valence_per_type=torch.tensor([
        4.0,  # C  (sp3)
        3.0,  # N  (neutral; quaternary via charge repair if needed)
        6.0,  # S  (allows sulfonyl, sulfones etc.) -> 4.0
        2.0,  # O
        1.0,  # F
        1.0,  # Cl
        1.0,  # Br
        # 1.0,  # H
    ]),
    bond_orders=torch.tensor([1.0, 2.0, 3.0, 1.5]),
    extra_node_feat_dim=2,
    root="./dataset/MOSES",
    cycles_count_path="./fm_graphex/files/cycles_count_moses.txt",
    nodes_count_path="./fm_graphex/files/nodes_count_moses.txt",
    cycles_count_train_path="./fm_graphex/files/cycle_count_full_train_moses.txt",
    cycles_count_test_path="./fm_graphex/files/cycle_count_full_test_moses.txt",
)


_ACTIVE_SPEC = QM9_SPEC

# Backward-compatible globals
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

def _refresh_globals():
    global ATOM_TYPES, Z_TO_TYPE, TYPE_TO_SYM, TYPE_TO_Z
    global BOND_TYPES, NUM_ATOM_TYPES, NUM_BOND_TYPES
    global MAX_VALENCE_PER_TYPE, BOND_ORDERS, EXTRA_NODE_FEATS
    global ROOT, CYCLES_COUNT_PATH, NODES_COUNT_PATH
    global CYCLES_COUNT_TRAIN_PATH, CYCLES_COUNT_TEST_PATH

    s = _ACTIVE_SPEC
    ATOM_TYPES = s.atom_types
    Z_TO_TYPE = s.z_to_type
    TYPE_TO_SYM = s.type_to_sym
    TYPE_TO_Z = s.type_to_z
    BOND_TYPES = s.bond_types
    NUM_ATOM_TYPES = s.num_atom_types
    NUM_BOND_TYPES = s.num_bond_types
    MAX_VALENCE_PER_TYPE = s.max_valence_per_type.clone()
    BOND_ORDERS = s.bond_orders.clone()
    EXTRA_NODE_FEATS = s.extra_node_feat_dim # [formal_charge_norm, is_aromatic]
    ROOT = s.root
    CYCLES_COUNT_PATH = s.cycles_count_path
    NODES_COUNT_PATH = s.nodes_count_path
    CYCLES_COUNT_TRAIN_PATH = s.cycles_count_train_path
    CYCLES_COUNT_TEST_PATH = s.cycles_count_test_path

def set_dataset(task: str):
    global _ACTIVE_SPEC
    t = task.lower()
    if t.startswith("qm9"):
        _ACTIVE_SPEC = QM9_SPEC
    elif t.startswith("zinc"):
        _ACTIVE_SPEC = ZINC_SPEC
    elif t.startswith("guacamol"):
        _ACTIVE_SPEC = GUACAMOL_SPEC
    elif t.startswith("moses"):
        _ACTIVE_SPEC = MOSES_SPEC
    else:
        raise ValueError(f"Unknown task={task}. Use qm9 or zinc.")
    _refresh_globals()
    
def get_spec() -> MolGraphSpec:
    return _ACTIVE_SPEC

def node_feat_dim() -> int:
    return NUM_ATOM_TYPES + EXTRA_NODE_FEATS

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

_refresh_globals()


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
        
def get_node_count_list(dataset: List[Data]) -> List[int]:
    return [int(d.num_nodes) for d in dataset if d.num_nodes is not None and int(d.num_nodes) > 0]
        

def estimate_atom_and_bond_probs(dataset: List[Data]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    atom_counts = np.zeros(NUM_ATOM_TYPES, dtype=np.float64)
    bond_counts = np.zeros(NUM_BOND_TYPES, dtype=np.float64)

    for data in dataset:
        if data.x is not None and data.x.dim() == 2 and data.x.size(1) >= NUM_ATOM_TYPES:
            atom_counts += data.x[:, :NUM_ATOM_TYPES].sum(dim=0).cpu().numpy()
        if data.edge_attr is not None and data.edge_attr.dim() == 2 and data.edge_attr.size(1) >= NUM_BOND_TYPES:
            bond_counts += data.edge_attr[:, :NUM_BOND_TYPES].sum(dim=0).cpu().numpy()

    atom_prob = atom_counts / max(atom_counts.sum(), 1.0)
    bond_prob = bond_counts / max(bond_counts.sum(), 1.0)
    atom_z = np.array([TYPE_TO_Z[i] for i in range(NUM_ATOM_TYPES)], dtype=int)
    return atom_prob, atom_z, bond_prob
        
        
def estimate_edge_density(dataset: list[Data]) -> float:
    num_edges = 0
    num_possible = 0
    for d in dataset:
        n = int(d.num_nodes)
        if n <= 1:
            continue
        # undirected edges: count unique (i<j)
        E = d.edge_index.shape[1]
        # in PyG graph stored both directions, so divide by 2
        undirected_edges = E // 2
        num_edges += undirected_edges
        num_possible += n * (n - 1) // 2
    if num_possible == 0:
        return 0.1
    return num_edges / num_possible