from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict
import numpy as np
import networkx as nx
from rdkit import Chem
from torch_geometric.datasets import QM9
from scipy.linalg import sqrtm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

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

def _morgan_fp(mol: Chem.Mol, radius: int = 2, nBits: int = 2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)

# SNN (Tanimoto similarity to nearest neighbor)
def compute_snn_tanimoto(
    gen_mols: list[Chem.Mol],
    ref_mols: list[Chem.Mol],
    radius: int = 2,
    nBits: int = 2048,
    chunk: int = 256,
) -> float:
    """
    Average max Tanimoto similarity of each generated mol to its nearest neighbor in ref.
    Returns SNN in [0,1] (higher = more similar to reference).
    """
    if len(gen_mols) == 0 or len(ref_mols) == 0:
        return 0.0

    ref_fps = [_morgan_fp(m, radius=radius, nBits=nBits) for m in ref_mols]

    snn_vals = []
    for i in range(0, len(gen_mols), chunk):
        batch = gen_mols[i:i+chunk]
        batch_fps = [_morgan_fp(m, radius=radius, nBits=nBits) for m in batch]
        # bulk similarity: for each gen fp, compute similarity to all ref fps
        for fp in batch_fps:
            sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
            snn_vals.append(max(sims) if len(sims) else 0.0)

    return float(sum(snn_vals) / max(1, len(snn_vals)))


from collections import Counter
from rdkit.Chem.Scaffolds import MurckoScaffold

def _murcko_scaffold_smiles(mol: Chem.Mol) -> str | None:
    """
    Return Bemis–Murcko scaffold as canonical SMILES.
    If scaffold is empty (e.g., acyclic), return None.
    """
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is None or scaf.GetNumAtoms() == 0:
            return None
        return Chem.MolToSmiles(scaf, isomericSmiles=False)
    except Exception:
        return None

# Scaffold similarity (Bemis–Murcko frequency comparison)
def compute_scaffold_l1_distance(
    gen_mols: list[Chem.Mol],
    ref_mols: list[Chem.Mol],
    include_none: bool = True,
) -> float:
    """
    L1 distance between scaffold frequency distributions (lower is better).
    If include_none=True, we include an 'NONE' bucket for molecules with no scaffold.
    """
    if len(gen_mols) == 0 or len(ref_mols) == 0:
        return 1e8

    def hist(mols):
        c = Counter()
        for m in mols:
            s = _murcko_scaffold_smiles(m)
            if s is None:
                if include_none:
                    c["__NONE__"] += 1
            else:
                c[s] += 1
        total = sum(c.values())
        if total == 0:
            return {}, 0
        return {k: v / total for k, v in c.items()}, total

    p_gen, _ = hist(gen_mols)
    p_ref, _ = hist(ref_mols)

    keys = set(p_gen.keys()) | set(p_ref.keys())
    l1 = 0.0
    for k in keys:
        l1 += abs(p_gen.get(k, 0.0) - p_ref.get(k, 0.0))
    return float(l1)

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, Lipinski

def physchem_vector(m: Chem.Mol) -> np.ndarray:
    """
    A commonly-used set of physicochemical descriptors (GuacaMol/MOSES-style).
    Returns a 1D numpy array (float64).
    """
    # Defensive: RDKit can throw on weird molecules
    try:
        mw   = Descriptors.MolWt(m)
        logp = Crippen.MolLogP(m)
        tpsa = rdMolDescriptors.CalcTPSA(m)

        hbd  = Lipinski.NumHDonors(m)
        hba  = Lipinski.NumHAcceptors(m)

        rot  = Lipinski.NumRotatableBonds(m)
        rings = rdMolDescriptors.CalcNumRings(m)
        arom_rings = rdMolDescriptors.CalcNumAromaticRings(m)

        frac_csp3 = rdMolDescriptors.CalcFractionCSP3(m)

        return np.array(
            [mw, logp, tpsa, hbd, hba, rot, rings, arom_rings, frac_csp3],
            dtype=np.float64
        )
    except Exception:
        return None


PHYSchem_NAMES = [
    "MolWt", "LogP", "TPSA",
    "HBD", "HBA",
    "RotB", "Rings", "AromRings",
    "FracCSP3"
]


def kl_divergence_test_vs_gen(test_counts: np.ndarray,
                             gen_counts: np.ndarray,
                             eps: float = 1e-12) -> float:
    """
    KL(test || gen) on discrete distributions with Laplace smoothing.
    """
    p = test_counts.astype(np.float64) + eps
    q = gen_counts.astype(np.float64) + eps
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * (np.log(p) - np.log(q))))


def _descriptor_matrix(mols) -> np.ndarray:
    rows = []
    for m in mols:
        if m is None:
            continue
        v = physchem_vector(m)
        if v is None:
            continue
        rows.append(v)
    if len(rows) == 0:
        return np.zeros((0, len(PHYSchem_NAMES)), dtype=np.float64)
    return np.stack(rows, axis=0)


def physchem_kl_test_vs_gen(
    test_mols,
    gen_mols,
    n_bins: int = 32,
    clip_quantiles=(0.01, 0.99),
    eps: float = 1e-12,
):
    """
    Compute KL(test || gen) per physchem descriptor + mean KL.

    Bins are derived from test set (after quantile clipping).
    """
    X_test = _descriptor_matrix(test_mols) # (Nt, D)
    X_gen  = _descriptor_matrix(gen_mols) # (Ng, D)

    D = len(PHYSchem_NAMES)

    # If one side is empty, KL is ill-defined -> return big number
    if X_test.shape[0] == 0 or X_gen.shape[0] == 0:
        per = {name: float("inf") for name in PHYSchem_NAMES}
        return per, float("inf")

    per_kl = {}

    q_lo, q_hi = clip_quantiles

    for d in range(D):
        name = PHYSchem_NAMES[d]
        xt = X_test[:, d]
        xg = X_gen[:, d]

        # robust bin range based on test distribution
        lo = np.quantile(xt, q_lo)
        hi = np.quantile(xt, q_hi)

        # handle degenerate cases
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            # fall back to min/max
            lo = float(np.min(xt))
            hi = float(np.max(xt))
            if hi <= lo:
                # completely constant descriptor -> KL = 0 if gen matches, else 0-ish under smoothing
                per_kl[name] = 0.0
                continue

        # build bins
        edges = np.linspace(lo, hi, n_bins + 1)

        # clip to bin range so outliers go into edge bins
        xtc = np.clip(xt, lo, hi)
        xgc = np.clip(xg, lo, hi)

        test_counts, _ = np.histogram(xtc, bins=edges)
        gen_counts,  _ = np.histogram(xgc, bins=edges)

        per_kl[name] = kl_divergence_test_vs_gen(test_counts, gen_counts, eps=eps)

    mean_kl = float(np.mean(list(per_kl.values())))
    return per_kl, mean_kl


def eval_and_log_ours(
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
    metrics        : dict[k -> dict with validity, uniqueness, novelty, fcd]
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

    # Load ChemNet / FCD ref model once
    model = load_ref_model()
    test_fc = get_predictions(model, can_test)
    mu_real, sigma_real = np.mean(test_fc, axis=0), np.cov(test_fc, rowvar=False)
    
    # SNN
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

        # Novelty + FCD using ChemNet features
        if len(unique_list) > 0:
            can_gen = [w for w in canonical_smiles(unique_list) if w is not None]
            # can_gen = [w for w in canonical_smiles(valid_smiles) if w is not None]

            # Novelty wrt training set
            can_novel = set(can_gen) - set(can_train)
            novelty = len(can_novel) / len(can_gen) if len(can_gen) > 0 else 0.0

            # FCD between real(test) and generated
            gen_fc = get_predictions(model, can_gen)
            mu_gen, sigma_gen = np.mean(gen_fc, axis=0), np.cov(gen_fc, rowvar=False)

            # Use the same Frechet formula as CatFlow
            def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
                mu_diff = mu1 - mu2
                sigma1_eps = sigma1 + np.eye(sigma1.shape[0]) * eps
                sigma2_eps = sigma2 + np.eye(sigma2.shape[0]) * eps
                covmean, _ = sqrtm(sigma1_eps.dot(sigma2_eps), disp=False)
                if np.iscomplexobj(covmean):
                    covmean = covmean.real
                trace_term = np.trace(sigma1_eps + sigma2_eps - 2 * covmean)
                return (mu_diff.dot(mu_diff) + trace_term).real
                
            # SNN (nearest-neighbor similarity via Tanimoto on Morgan fingerprints)
            ref_valid_mols = [Chem.MolFromSmiles(s) for s in can_gen if s is not None]
            ref_valid_mols = [m for m in ref_valid_mols if m is not None]
            try:
                snn = compute_snn_tanimoto(
                    gen_mols= ref_valid_mols, # valid_mols,
                    ref_mols=ref_test_mols, # or train+test if you prefer
                    radius=2,
                    nBits=2048,
                )
            except Exception:
                snn = 0.0

            # Scaffold similarity (histogram distance of Bemis–Murcko scaffolds)
            try:
                scaffold_l1 = compute_scaffold_l1_distance(
                    gen_mols=ref_valid_mols, # valid_mols,
                    ref_mols=ref_test_mols,
                    include_none=True,
                )
            except Exception:
                scaffold_l1 = 1e8

            # KL-divergence for various physicochemical descriptors
            try:
                per_physchem_kl, mean_physchem_kl = physchem_kl_test_vs_gen(
                    test_mols=test_mols,
                    gen_mols=ref_valid_mols,
                    n_bins=32,
                    clip_quantiles=(0.01, 0.99),
                )
            except Exception:
                mean_physchem_kl = 0.0
            
        else:
            novelty = 0.0
            snn = 0.0
            scaffold_l1 = 1e8
            mean_physchem_kl = 0.0
            
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
            snn=snn,
            scaffold_l1=scaffold_l1,
            physchem_kl_mean=mean_physchem_kl,
        )

    return all_valid_mols, all_metrics