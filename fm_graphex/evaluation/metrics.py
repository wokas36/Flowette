from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict, Counter

import numpy as np
import networkx as nx

from scipy.linalg import sqrtm

from rdkit import Chem, DataStructs
from rdkit.Chem import (
    AllChem,
    Descriptors,
    Crippen,
    rdMolDescriptors,
    Lipinski,
)
from rdkit.Chem.Scaffolds import MurckoScaffold

from torch_geometric.datasets import QM9

from fm_graphex.data.rdkit_conversions import (
    data_to_rdkit_mol,
    kekulize_and_remove_hs,
    kekulize_preserve_hs,
)
from fm_graphex.config import BOND_TYPES

try:
    from fm_graphex.evaluation.eaden import vectorize as eden_vectorize
except ImportError:
    eden_vectorize = None

def mol_to_canonical_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)


def dataset_to_smiles_list(dataset, task: str) -> List[str]:
    """
    Convert a PyG molecular dataset to canonical SMILES.

    QM9/ZINC:
        remove H and use kekulized representation.

    MOSES/Guacamol:
        preserve H according to the existing project convention.
    """
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

        try:
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            smiles_list.append(smiles)

        except Exception:
            continue

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

        try:
            smiles_set.add(mol_to_canonical_smiles(mol))
        except Exception:
            continue

    return smiles_set


def compute_novelty(mols: List[Optional[Chem.Mol]], train_smiles: Set[str]) -> float:
    valid = [m for m in mols if m is not None]

    if not valid:
        return 0.0

    gen_smiles = [mol_to_canonical_smiles(m) for m in valid]

    novel = [s for s in gen_smiles if s not in train_smiles]

    return len(novel) / len(gen_smiles)


def mol_to_nx_graph(mol: Chem.Mol) -> nx.Graph:
    """
    Convert an RDKit molecule to a NetworkX graph.

    The original attributes used by the existing graph-statistics code are
    retained. We additionally provide EDeN-compatible `label` attributes.

    Node label:
        atomic number

    Edge label:
        RDKit bond type
    """
    G = nx.Graph()

    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        atomic_num = atom.GetAtomicNum()

        G.add_node(i, atomic_num=atomic_num, label=str(atomic_num))

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bond_type = bond.GetBondType()

        G.add_edge(i, j, bond_type=bond_type, label=str(bond_type))

    return G


def mol_to_nspdk_graph(mol: Optional[Chem.Mol]) -> Optional[nx.Graph]:
    """
    Convert an RDKit molecule into a labeled NetworkX graph suitable for
    EDeN NSPDK vectorization.

    A copy is sanitized and kekulized so that generated and reference
    molecules use the same discrete bond representation.
    """
    if mol is None:
        return None

    if mol.GetNumAtoms() == 0:
        return None

    try:
        m = Chem.Mol(mol)

        Chem.SanitizeMol(m)

        # Use a consistent discrete bond representation.
        try:
            Chem.Kekulize(m, clearAromaticFlags=True)
        except Exception:
            # Some molecules may already be in an appropriate kekulized
            # representation. Sanitized molecules are still usable.
            pass

        return mol_to_nx_graph(m)

    except Exception:
        return None


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


def graph_stats_from_mols(mols: List[Chem.Mol]) -> np.ndarray:
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
        if mol is None:
            continue

        G = mol_to_nx_graph(mol)

        if G.number_of_nodes() == 0:
            continue

        degs = [d for _, d in G.degree()]
        degrees.extend(degs)

        clusterings.append(nx.average_clustering(G))

        ring_counts.append(len(nx.cycle_basis(G)))

        node_counts.append(G.number_of_nodes())

    stats = np.array(
        [
            np.mean(degrees)
            if degrees else 0.0,

            np.std(degrees)
            if degrees else 0.0,

            np.mean(clusterings)
            if clusterings else 0.0,

            np.mean(ring_counts)
            if ring_counts else 0.0,

            np.mean(node_counts)
            if node_counts else 0.0,
        ],
        dtype=float,
    )

    return stats


def compute_ratio_statistic(train_mols: List[Chem.Mol], test_mols: List[Chem.Mol], 
                            gen_mols: List[Chem.Mol], eps: float = 1e-8) -> float:
    """
    Ratio =
        ||s(gen) - s(test)||
        --------------------
        ||s(train) - s(test)|| + eps
    """

    s_train = graph_stats_from_mols(train_mols)
    s_test = graph_stats_from_mols(test_mols)
    s_gen = graph_stats_from_mols(gen_mols)

    num = np.linalg.norm(s_gen - s_test)

    den = (np.linalg.norm(s_train - s_test) + eps)

    return float(num / den)


def _check_nspdk_dependency() -> None:
    """
    Raise an informative error if EDeN is not available.
    """
    if eden_vectorize is None:
        raise ImportError(
            "NSPDK evaluation requires the EDeN graph package. "
            "The expected import is:\n"
            "from eden.graph import vectorize"
        )


def vectorize_nspdk_graphs(graphs: List[nx.Graph], complexity: int = 4):
    """
    NSPDK feature representation used by the GraphGen evaluation code.

    Equivalent to:
        vectorize(graphs, complexity=4, discrete=True)
    """
    _check_nspdk_dependency()

    graphs = [
        G
        for G in graphs
        if (
            G is not None
            and G.number_of_nodes() > 0
        )
    ]

    if len(graphs) == 0:
        return None

    return eden_vectorize(
        graphs,
        complexity=complexity,
        discrete=True,
    )


def compute_nspdk_mmd(graph_ref_list: List[nx.Graph], graph_pred_list: List[nx.Graph], complexity: int = 4) -> float:
    """
    Compute NSPDK MMD between reference and generated graph sets.

    GraphGen computes

        mean(K_ref_ref)
        + mean(K_gen_gen)
        - 2 * mean(K_ref_gen),

    where K is a linear kernel over EDeN NSPDK features.

    For a linear kernel this is algebraically equivalent to

        || mean(phi_ref) - mean(phi_gen) ||_2^2.

    We use the latter expression because it avoids explicitly constructing
    dense N x N kernel matrices, which is important when evaluating
    thousands of molecules.

    Lower is better.
    """
    graph_ref_list = [
        G
        for G in graph_ref_list
        if (
            G is not None
            and G.number_of_nodes() > 0
        )
    ]

    graph_pred_list = [
        G
        for G in graph_pred_list
        if (
            G is not None
            and G.number_of_nodes() > 0
        )
    ]

    if (len(graph_ref_list) == 0 or len(graph_pred_list) == 0):
        return float("nan")

    X_ref = vectorize_nspdk_graphs(graph_ref_list, complexity=complexity)

    X_pred = vectorize_nspdk_graphs(graph_pred_list, complexity=complexity)

    if X_ref is None or X_pred is None:
        return float("nan")

    # Mean NSPDK feature embedding.
    mu_ref = np.asarray(X_ref.mean(axis=0)).ravel()

    mu_pred = np.asarray(X_pred.mean(axis=0)).ravel()

    # Same quantity as linear-kernel MMD used by GraphGen.
    diff = mu_ref - mu_pred

    mmd = float(np.dot(diff, diff))

    # Numerical guard.
    return max(0.0, mmd)


def compute_nspdk_from_mols(reference_mols: List[Chem.Mol], generated_mols: List[Chem.Mol], complexity: int = 4) -> float:
    """
    Convenience wrapper for computing NSPDK directly from RDKit molecules.
    """
    reference_graphs = [mol_to_nspdk_graph(m) for m in reference_mols]

    reference_graphs = [
        G
        for G in reference_graphs
        if (
            G is not None
            and G.number_of_nodes() > 0
        )
    ]

    generated_graphs = [mol_to_nspdk_graph(m) for m in generated_mols]

    generated_graphs = [
        G
        for G in generated_graphs
        if (
            G is not None
            and G.number_of_nodes() > 0
        )
    ]

    return compute_nspdk_mmd(reference_graphs, generated_graphs, complexity=complexity)


def eval_and_log(mols: dict, smiles_pair: Tuple[List[str], List[str]], log: bool = False, nspdk_complexity: int = 4):
    """
    CatFlow-style molecular evaluation augmented with NSPDK.

    Args
    ----
    mols:
        dict[k -> list[RDKit Mol or None]]

        A simple list is also accepted and converted internally to
        {"default": mols}.

    smiles_pair:
        (train_smiles_list, test_smiles_list)

        Training SMILES are used for novelty.
        Test SMILES are used as the NSPDK reference distribution.

    log:
        Retained for compatibility with the existing interface.

    nspdk_complexity:
        EDeN NSPDK complexity parameter.
        We use 4 to match the standard GraphGen implementation.

    Returns
    -------
    all_valid_mols:
        dict[k -> list[valid RDKit molecules]]

    all_metrics:
        dict[k -> dict containing:
            validity
            uniqueness
            novelty
            vun
            nspdk
    """

    from fcd import (
        get_fcd,
        load_ref_model,
        canonical_smiles,
        get_predictions,
        calculate_frechet_distance,
    )

    if isinstance(mols, list):
        mols = {
            "default": mols
        }

    all_valid_mols = defaultdict(list)
    all_metrics = {}

    train_smiles, test_smiles = smiles_pair

    can_test = [
        w
        for w in canonical_smiles(
            test_smiles
        )
        if w is not None
    ]

    can_train = [
        w
        for w in canonical_smiles(
            train_smiles
        )
        if w is not None
    ]

    can_train_set = set(can_train)

    model = load_ref_model()

    test_fc = get_predictions(model, can_test)

    mu_real = np.mean(test_fc, axis=0)

    sigma_real = np.cov(test_fc, rowvar=False)

    ref_test_mols = []

    for s in can_test:
        try:
            mol = Chem.MolFromSmiles(s)

            if mol is None:
                continue

            Chem.SanitizeMol(mol)

            ref_test_mols.append(mol)

        except Exception:
            continue

    ref_nspdk_graphs = [mol_to_nspdk_graph(m) for m in ref_test_mols]

    ref_nspdk_graphs = [
        G
        for G in ref_nspdk_graphs
        if (
            G is not None
            and G.number_of_nodes() > 0
        )
    ]

    for k, k_mols in mols.items():

        valid_mols = []

        for mol in k_mols:

            if mol is None:
                continue

            try:
                mol_copy = Chem.Mol(mol)

                Chem.SanitizeMol(mol_copy)

                Chem.Kekulize(mol_copy, clearAromaticFlags=True)

                frags = Chem.rdmolops.GetMolFrags(mol_copy, asMols=True, sanitizeFrags=True)

                largest_mol = max(frags, default=mol_copy, key=lambda m: m.GetNumAtoms())

                valid_mols.append(largest_mol)

            except Exception:
                continue

        n_total = len(k_mols)

        n_valid = len(valid_mols)
        
        # Valid
        if n_total == 0:
            percentage_valid = 0.0
            percentage_unique = 0.0
            novelty = 0.0
            vun_score = 0.0
            nspdk = float("nan")

            all_valid_mols[k] = []

            all_metrics[k] = dict(
                validity=percentage_valid,
                uniqueness=percentage_unique,
                novelty=novelty,
                vun=vun_score,
                nspdk=nspdk,
            )

            print(
                f"[k={k}] "
                f"valid={percentage_valid:.3f}, "
                f"unique={percentage_unique:.3f}, "
                f"novel={novelty:.3f}, "
                f"vun={vun_score:.3f}, "
                f"nspdk={nspdk}"
            )

            continue

        percentage_valid = (n_valid / n_total)

        # Uniqueness
        if n_valid > 0:

            valid_smiles = []

            for m in valid_mols:
                try:
                    s = Chem.MolToSmiles(m, isomericSmiles=True, canonical=True)
                    valid_smiles.append(s)

                except Exception:
                    continue

            unique_set = set(valid_smiles)

            percentage_unique = (len(unique_set) / n_valid if n_valid > 0 else 0.0)

            unique_list = list(unique_set)
        else:
            percentage_unique = 0.0
            unique_list = []

        # Novelty
        if len(unique_list) > 0:

            can_gen = [
                w
                for w in canonical_smiles(
                    unique_list
                )
                if w is not None
            ]

            can_novel = (set(can_gen) - can_train_set)

            novelty = (
                len(can_novel)
                / len(can_gen)
                if len(can_gen) > 0
                else 0.0
            )

            # Existing ChemNet feature computation retained.
            if len(can_gen) > 0:

                gen_fc = get_predictions(model, can_gen)

                mu_gen = np.mean(gen_fc, axis=0)

                sigma_gen = np.cov(gen_fc, rowvar=False)
        else:
            novelty = 0.0

        # V.U.N.
        vun_score = (percentage_valid * percentage_unique * novelty)

        # NSPDK
        gen_nspdk_graphs = [mol_to_nspdk_graph(m) for m in valid_mols]

        gen_nspdk_graphs = [
            G
            for G in gen_nspdk_graphs
            if (
                G is not None
                and G.number_of_nodes() > 0
            )
        ]

        if (len(gen_nspdk_graphs) > 0 and len(ref_nspdk_graphs) > 0):
            nspdk = compute_nspdk_mmd(graph_ref_list=ref_nspdk_graphs, graph_pred_list=gen_nspdk_graphs, complexity=nspdk_complexity)
        else:
            nspdk = float("nan")

        # Logging
        print(
            f"[k={k}] "
            f"valid={percentage_valid:.3f}, "
            f"unique={percentage_unique:.3f}, "
            f"novel={novelty:.3f}, "
            f"vun={vun_score:.3f}, "
            f"nspdk={nspdk:.6f}"
        )

        all_valid_mols[k] = (valid_mols)

        all_metrics[k] = dict(
            validity=percentage_valid,
            uniqueness=percentage_unique,
            novelty=novelty,
            vun=vun_score,
            nspdk=nspdk,
        )

    return (all_valid_mols, all_metrics)