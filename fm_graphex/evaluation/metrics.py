from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Crippen, rdMolDescriptors, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.datasets import QM9
from collections import Counter
from scipy.linalg import sqrtm

from fm_graphex.data.rdkit_conversions import data_to_rdkit_mol, kekulize_and_remove_hs, kekulize_preserve_hs
from fm_graphex.config import BOND_TYPES


def mol_to_canonical_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def dataset_to_smiles_list(dataset, task: str) -> List[str]:
    smiles_list = []
    for data in dataset:
        mol = data_to_rdkit_mol(data)
        if mol is None:
            continue
        if task in ("qm9", "zinc"):
            mol = kekulize_and_remove_hs(mol)
        else:
            mol = kekulize_preserve_hs(mol)
        if mol is None:
            continue
        smiles_list.append(Chem.MolToSmiles(mol, isomericSmiles=True))
    return smiles_list


def compute_validity(mols: List[Optional[Chem.Mol]]) -> float:
    valid = [m for m in mols if m is not None]
    return len(valid) / max(1, len(mols))


def compute_uniqueness(mols: List[Optional[Chem.Mol]]) -> float:
    valid = [m for m in mols if m is not None]
    if not valid:
        return 0.0
    smiles = [mol_to_canonical_smiles(m) for m in valid]
    return len(set(smiles)) / len(smiles)


def compute_training_smiles_set(dataset: QM9) -> Set[str]:
    smiles_set: Set[str] = set()
    for data in dataset:
        mol = data_to_rdkit_mol(data)
        if mol is None:
            continue
        smiles_set.add(mol_to_canonical_smiles(mol))
    return smiles_set


def compute_novelty(
    mols: List[Optional[Chem.Mol]], train_smiles: Set[str]
) -> float:
    valid = [m for m in mols if m is not None]
    if not valid:
        return 0.0
    gen_smiles = [mol_to_canonical_smiles(m) for m in valid]
    novel = [s for s in gen_smiles if s not in train_smiles]
    return len(novel) / len(gen_smiles)


def mol_to_nx_graph(mol: Chem.Mol) -> nx.Graph:
    G = nx.Graph()
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        G.add_node(i, atomic_num=atom.GetAtomicNum())
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        G.add_edge(i, j, bond_type=bond.GetBondType())
    return G


def basic_graph_stats(mols: List[Optional[Chem.Mol]]) -> Dict[str, float]:
    degrees = []
    clusterings = []

    for mol in mols:
        if mol is None:
            continue
        G = mol_to_nx_graph(mol)
        if G.number_of_nodes() == 0:
            continue
        degs = [d for _, d in G.degree()]
        degrees.extend(degs)
        clusterings.append(nx.average_clustering(G))

    stats = {}
    if degrees:
        stats["avg_degree"] = float(np.mean(degrees))
        stats["std_degree"] = float(np.std(degrees))
    else:
        stats["avg_degree"] = 0.0
        stats["std_degree"] = 0.0

    if clusterings:
        stats["avg_clustering"] = float(np.mean(clusterings))
    else:
        stats["avg_clustering"] = 0.0
    return stats


def graph_stats_from_mols(mols: list[Chem.Mol]) -> np.ndarray:
    """
    Compute a vector of graph statistics averaged over molecules.
    """
    if len(mols) == 0:
        return np.zeros(5, dtype=float)

    degrees = []
    clusterings = []
    ring_counts = []
    node_counts = []

    for mol in mols:
        G = mol_to_nx_graph(mol)
        if G.number_of_nodes() == 0:
            continue

        degs = [d for _, d in G.degree()]
        degrees.extend(degs)

        clusterings.append(nx.average_clustering(G))
        ring_counts.append(len(nx.cycle_basis(G)))
        node_counts.append(G.number_of_nodes())

    stats = np.array([
        np.mean(degrees) if degrees else 0.0, # avg degree
        np.std(degrees) if degrees else 0.0, # std degree
        np.mean(clusterings) if clusterings else 0.0, # clustering
        np.mean(ring_counts) if ring_counts else 0.0, # rings
        np.mean(node_counts) if node_counts else 0.0, # nodes
    ], dtype=float)

    return stats


def compute_ratio_statistic(
    train_mols: list[Chem.Mol],
    test_mols: list[Chem.Mol],
    gen_mols: list[Chem.Mol],
    eps: float = 1e-8,
) -> float:
    """
    Ratio = || s(gen) - s(test) || / (|| s(train) - s(test) || + eps)
    """
    s_train = graph_stats_from_mols(train_mols)
    s_test  = graph_stats_from_mols(test_mols)
    s_gen   = graph_stats_from_mols(gen_mols)

    num = np.linalg.norm(s_gen - s_test)
    den = np.linalg.norm(s_train - s_test) + eps
    return float(num / den)


def eval_and_log(
    mols: dict,
    smiles_pair: tuple[list[str], list[str]], # (train_smiles, test_smiles)
    log: bool = False,
):
    """
    CatFlow-style evaluation for our context.

    Args
    ----
    mols         : dict[k -> list[RDKit Mol or None]]
    smiles_pair  : (train_smiles_list, test_smiles_list)
    log          : if True, logs metrics to wandb (same keys as CatFlow)

    Returns
    -------
    all_valid_mols : dict[k -> list[valid RDKit Mol (largest fragments)]]
    metrics        : dict[k -> dict with validity, uniqueness, novelty]
    """
    from fcd import get_fcd, load_ref_model, canonical_smiles, get_predictions, calculate_frechet_distance
    
    if isinstance(mols, list):
        mols = {"default": mols}

    all_valid_mols = defaultdict(list)
    all_metrics = {}

    train_smiles, test_smiles = smiles_pair

    # Pre-canonicalize train/test once
    can_test  = [w for w in canonical_smiles(test_smiles)  if w is not None]
    can_train = [w for w in canonical_smiles(train_smiles) if w is not None]

    # Load ChemNet
    model = load_ref_model()
    test_fc = get_predictions(model, can_test)
    mu_real, sigma_real = np.mean(test_fc, axis=0), np.cov(test_fc, rowvar=False)
    
    ref_test_mols = [Chem.MolFromSmiles(s) for s in can_test if s is not None]
    ref_test_mols = [m for m in ref_test_mols if m is not None]

    for k, k_mols in mols.items():
        valid_mols = []

        # Validity via sanitize + kekulize + largest fragment
        for mol in k_mols:
            if mol is None:
                continue
            try:
                # In our setup, these are already heavy-atom graphs;
                # we just re-sanitize + kekulize for robustness.
                mol_copy = Chem.Mol(mol)
                Chem.SanitizeMol(mol_copy)
                Chem.Kekulize(mol_copy, clearAromaticFlags=True)

                frags = Chem.rdmolops.GetMolFrags(
                    mol_copy, asMols=True, sanitizeFrags=True
                )
                largest_mol = max(frags, default=mol_copy, key=lambda m: m.GetNumAtoms())
                valid_mols.append(largest_mol)
            except Exception:
                continue

        n_total = len(k_mols)
        n_valid = len(valid_mols)

        if n_total == 0:
            percentage_valid = 0.0
            percentage_unique = 0.0
            novelty = 0.0
            all_valid_mols[k] = []
            all_metrics[k] = dict(
                validity=percentage_valid,
                uniqueness=percentage_unique,
                novelty=novelty,
            )
            continue

        percentage_valid = n_valid / n_total

        # Uniqueness in valid set (SMILES-based)
        if n_valid > 0:
            valid_smiles = [Chem.MolToSmiles(m) for m in valid_mols]
            unique_set = set(valid_smiles)
            percentage_unique = len(unique_set) / n_valid
            unique_list = list(unique_set)
        else:
            percentage_unique = 0.0
            unique_list = []

        # Novelty using ChemNet features
        if len(unique_list) > 0:
            can_gen = [w for w in canonical_smiles(unique_list) if w is not None]

            # Novelty wrt training set
            can_novel = set(can_gen) - set(can_train)
            novelty = len(can_novel) / len(can_gen) if len(can_gen) > 0 else 0.0

            gen_fc = get_predictions(model, can_gen)
            mu_gen, sigma_gen = np.mean(gen_fc, axis=0), np.cov(gen_fc, rowvar=False)
        else:
            novelty = 0.0
            
        # VUN Score
        vun_score = percentage_valid * percentage_unique * novelty
    
        print(
            f"[k={k}] valid={percentage_valid:.3f}, "
            f"unique={percentage_unique:.3f}, "
            f"novel={novelty:.3f}, "
            f"vun={vun_score:.3f}"
        )

        all_valid_mols[k] = valid_mols
        
        all_metrics[k] = dict(
            validity=percentage_valid,
            uniqueness=percentage_unique,
            novelty=novelty,
        )

    return all_valid_mols, all_metrics