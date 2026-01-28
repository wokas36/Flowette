from typing import List, Optional, Tuple, Dict
import random
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import rdchem
import re

from fm_graphex.config import (
    DEVICE,
    BOND_TYPES,
    NUM_ATOM_TYPES,
    NUM_BOND_TYPES,
    BOND_ORDERS,
    TYPE_TO_Z,
    TYPE_TO_SYM,
)
from fm_graphex.config import get_spec
from fm_graphex.models.encoder import GINEncoder, SimpleGraphDecoder
from fm_graphex.models.velocity_field import VelocityFieldGNNTransformer
from fm_graphex.flow_matching.dense_graph import build_dense_type_graph, normalize_adjacency, relu
from fm_graphex.data.graphex_noise import sample_graphex_noise_graph_batch
from fm_graphex.data.rdkit_conversions import sanitize_mol_safe, kekulize_and_remove_hs, kekulize_preserve_hs


_PT = rdchem.GetPeriodicTable()


# Sampling from graphex noise graphs + FM field
@torch.no_grad()
def integrate_flow(
    vnet: VelocityFieldGNNTransformer,
    A0: torch.Tensor,
    X0: torch.Tensor,
    E0: torch.Tensor,
    n_steps: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Integrate the learned flow from t=0 to t=1 using simple Euler steps.
    A0, X0, E0 are initial adjacency, atom-type one-hot, and bond-type one-hot tensors.
    Fixed integration:
      - uses standard clamp for adjacency (nonnegative)
    """
    vnet.eval()

    A_t = A0.clone().to(DEVICE)
    X_t = X0.clone().to(DEVICE)
    E_t = E0.clone().to(DEVICE)

    dt = 1.0 / float(n_steps)

    for step in range(n_steps):
        t_scalar = step / float(n_steps)
        t = torch.tensor(t_scalar, dtype=torch.float32, device=DEVICE)

        v_A, v_X, v_E = vnet(A_t, X_t, E_t, t)

        A_t = A_t + dt * v_A
        X_t = X_t + dt * v_X
        E_t = E_t + dt * v_E

    # Keep adjacency non-negative
    A_t = relu(A_t)
    A_t = 0.5 * (A_t + A_t.T)
    
    return A_t, X_t, E_t


def _approx_max_valence(z: int,
                        task: str, 
                        formal_charge: int = 0, 
                        is_aromatic: bool = False) -> float:
    try:
        vals = _PT.GetValenceList(int(z))
        base = max(vals) if vals else 4
    except Exception:
        base = 4

    if task == "guacamol":
        if is_aromatic:
            base = min(base, 3)
        if formal_charge > 0:
            base = min(base + 1, 6)
        if formal_charge < 0:
            base = min(base, 4)

    elif task == "moses":
        # allow hypervalent S/P
        if z in (16, 15): # S, P
            base = max(base, 6)
        if formal_charge != 0:
            base = max(base, 4)

    return float(base)


def project_valence_prune(
    A_soft: torch.Tensor,
    X_logits: torch.Tensor,
    E_logits: torch.Tensor,
    base_threshold: float = 0.2,
) -> torch.Tensor:
    """
    Valence-aware projection that starts from an unconstrained graph
    and only prunes edges when valence is exceeded.

    Args
    ----
    A_soft        : (N, N) soft adjacency
    X_logits      : (N, num_atom_types) node logits
    E_logits      : (N, N, num_bond_types) edge logits
    base_threshold: keep edges with A_ij >= base_threshold initially

    Returns
    -------
    bonds : (N, N) long tensor
        bonds[i,j] in {0..NUM_BOND_TYPES-1} for bonds,
        or NUM_BOND_TYPES for "no bond".
    """
    device = X_logits.device
    N = X_logits.size(0)

    # Atom identity from first NUM_ATOM_TYPES dims
    atom_type_idx = X_logits[:, :NUM_ATOM_TYPES].argmax(dim=-1) # (N,)
    
    task = get_spec().name
    
    if task not in ("qm9", "zinc"):
        charge_norm = X_logits[:, NUM_ATOM_TYPES + 0]
        arom_flag   = X_logits[:, NUM_ATOM_TYPES + 1]

    # map type idx -> atomic number
    atom_z = torch.tensor([TYPE_TO_Z[int(i)] for i in atom_type_idx.cpu().tolist()], device=device)
    
    if task not in ("qm9", "zinc"):
        formal_charge = torch.round(torch.clamp(charge_norm, -1.0, 1.0) * 3.0).to(torch.int32)
        is_aromatic = (arom_flag > 0.5)

    max_valence = torch.zeros((N,), device=device, dtype=torch.float32)
    for i in range(N):
        z = int(atom_z[i].item())
        if task not in ("qm9", "zinc"):
            fc = int(formal_charge[i].item())
            arom = bool(is_aromatic[i].item())
            max_valence[i] = _approx_max_valence(z, task, fc, arom)
        else:
            max_valence[i] = _approx_max_valence(z, task)

    # Symmetrize adjacency
    A_sym = 0.5 * (A_soft + A_soft.T)
    bond_probs = torch.softmax(E_logits, dim=-1) # (N,N,edge_dim)

    bonds = torch.full((N, N), NUM_BOND_TYPES, dtype=torch.long, device=device)
    scores = torch.zeros((N, N), dtype=A_soft.dtype, device=device)

    for i in range(N):
        for j in range(i + 1, N):
            a_ij = float(A_sym[i, j].item())
            if a_ij < base_threshold:
                continue
            probs_ij = bond_probs[i, j, :NUM_BOND_TYPES]
            
            # dataset-aware aromatic bias
            if task == "guacamol":
                if is_aromatic[i] and is_aromatic[j]:
                    aromatic_idx = BOND_TYPES.index(rdchem.BondType.AROMATIC)
                    probs_ij = probs_ij.clone()
                    probs_ij[aromatic_idx] += 0.2
            
            b_idx = int(probs_ij.argmax().item())
            p_bond = float(probs_ij[b_idx].item())
            score = a_ij * p_bond
            bonds[i, j] = b_idx
            bonds[j, i] = b_idx
            scores[i, j] = score
            scores[j, i] = score

    # used valence
    used = torch.zeros((N,), device=device, dtype=torch.float32)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            b = int(bonds[i, j].item())
            if b >= NUM_BOND_TYPES:
                continue
            used[i] += float(BOND_ORDERS[b].item())

    # prune excess
    for i in range(N):
        while used[i] - max_valence[i] > 1e-6:
            incident = []
            for j in range(N):
                if i == j:
                    continue
                b = int(bonds[i, j].item())
                if b >= NUM_BOND_TYPES:
                    continue
                incident.append((float(scores[i, j].item()), j))
            if not incident:
                break
            incident.sort(key=lambda x: x[0]) # weakest first
            _, j = incident[0]
            b = int(bonds[i, j].item())
            order = float(BOND_ORDERS[b].item())
            bonds[i, j] = NUM_BOND_TYPES
            bonds[j, i] = NUM_BOND_TYPES
            used[i] -= order
            used[j] -= order

    bonds[torch.arange(N), torch.arange(N)] = NUM_BOND_TYPES
    return bonds

def project_valence_prune_topk(
    A_soft: torch.Tensor, # (N, N)
    X_logits: torch.Tensor, # (N, num_atom_types + extras)
    E_logits: torch.Tensor, # (N, N, num_bond_types)
    max_extra_bonds: int = 1, # allow slight flexibility
) -> torch.Tensor:
    """
    Top-k valence-aware projection for continuous FM graphs.
    Returns:
        bonds: (N, N) long tensor in {0..NUM_BOND_TYPES-1} or NUM_BOND_TYPES=no bond
    """

    device = A_soft.device
    N = A_soft.size(0)
    task = get_spec().name

    # atom identity
    atom_type_idx = X_logits[:, :NUM_ATOM_TYPES].argmax(dim=-1)
    atom_z = torch.tensor(
        [TYPE_TO_Z[int(i)] for i in atom_type_idx.cpu().tolist()],
        device=device,
    )

    # extra features (Guacamol / MOSES)
    if task not in ("qm9", "zinc"):
        charge_norm = X_logits[:, NUM_ATOM_TYPES + 0]
        formal_charge = torch.round(torch.clamp(charge_norm, -1.0, 1.0)).int()
    else:
        formal_charge = torch.zeros(N, device=device, dtype=torch.int)

    # max valence per atom
    max_valence = torch.zeros(N, device=device)
    for i in range(N):
        z = int(atom_z[i])
        fc = int(formal_charge[i])
        max_valence[i] = _approx_max_valence(z, task, fc)

    # symmetrized adjacency score
    A = torch.clamp(0.5 * (A_soft + A_soft.T), 0.0, 1.0)

    # initialize empty bond matrix
    bonds = torch.full(
        (N, N), NUM_BOND_TYPES, dtype=torch.long, device=device
    )

    # TOP-K EDGE SELECTION PER NODE
    edge_scores = A.clone()
    edge_scores.fill_diagonal_(0.0)

    selected_edges = set()

    for i in range(N):
        k = int(max_valence[i].item()) + max_extra_bonds
        if k <= 0:
            continue

        scores_i = edge_scores[i]
        vals, idxs = torch.topk(scores_i, k=min(k, N - 1))

        for v, j in zip(vals, idxs):
            if v <= 0:
                continue
            a, b = min(i, j.item()), max(i, j.item())
            selected_edges.add((a, b))

    # ASSIGN BOND TYPES
    bond_probs = torch.softmax(E_logits, dim=-1)

    for (i, j) in selected_edges:
        probs = bond_probs[i, j, :NUM_BOND_TYPES]
        b_idx = int(probs.argmax().item())
        bonds[i, j] = b_idx
        bonds[j, i] = b_idx

    # STEP 3: HARD VALENCE CHECK (FINAL SAFETY)
    used = torch.zeros(N, device=device)

    for i in range(N):
        for j in range(N):
            b = bonds[i, j]
            if b < NUM_BOND_TYPES:
                used[i] += BOND_ORDERS[b]

    for i in range(N):
        while used[i] > max_valence[i] + 1e-6:
            incident = []
            for j in range(N):
                b = bonds[i, j]
                if b < NUM_BOND_TYPES:
                    score = A[i, j].item()
                    incident.append((score, j))

            if not incident:
                break

            # remove weakest bond
            incident.sort(key=lambda x: x[0])
            _, j = incident[0]
            order = BOND_ORDERS[bonds[i, j]]
            bonds[i, j] = NUM_BOND_TYPES
            bonds[j, i] = NUM_BOND_TYPES
            used[i] -= order
            used[j] -= order

    bonds.fill_diagonal_(NUM_BOND_TYPES)
    return bonds

ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 5, 16: 6, 17: 1, 35: 1, 53: 1}

def _check_valency_from_exception(mol: Chem.Mol):
    """
    Try a light property-only sanitize. If there is a valence error,
    parse RDKit's error string to get [atom_idx, valence].
    """
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        s = str(e)
        p = s.find("#")
        if p == -1:
            return False, None
        atomid_valence = list(map(int, re.findall(r"\d+", s[p:])))
        return False, atomid_valence # e.g. [idx, val]

    
def _defog_charge_repair_once(mol: Chem.RWMol) -> bool:
    """
    One step of Defog-style charge repair:
    if valence is exactly +1 over neutral for N/O/S, assign +1 formal charge.
    """
    ok, atomid_valence = _check_valency_from_exception(mol)
    if ok or atomid_valence is None or len(atomid_valence) != 2:
        return False

    idx, v = atomid_valence
    atom = mol.GetAtomWithIdx(int(idx))
    an = atom.GetAtomicNum()
    base = ATOM_VALENCY.get(an, None)
    # if base is not None and (v - base) == 1:
    if an in (7, 8, 16) and (v - base) == 1:
        atom.SetFormalCharge(atom.GetFormalCharge() + 1)
        return True
    return False


def defog_charge_repair(mol: Chem.Mol, max_iter: int = 8) -> Chem.Mol:
    """
    Run the Defog-style repair a few times until no more fixes are possible.
    """
    rw = Chem.RWMol(mol)
    for _ in range(max_iter):
        changed = _defog_charge_repair_once(rw)
        if not changed:
            break
    return rw.GetMol()

def sanitize_mol_relaxed(mol: Chem.Mol) -> Chem.Mol:
    """
    Relaxed sanitization: enough to compute properties / SMILES in more cases,
    but avoids hard failures. If it still fails, returns None.
    """
    if mol is None:
        return None

    mol = Chem.Mol(mol)
    try:
        # Partial sanitize: skip SANITIZE_PROPERTIES? (we already cached)
        # Keep it minimal: find rings + kekulization can be tricky for invalid aromatics.
        Chem.SanitizeMol(
            mol,
            sanitizeOps=(Chem.SanitizeFlags.SANITIZE_PROPERTIES),
        )
        return mol
    except Exception:
        # fallback: at least try cache update only
        try:
            mol.UpdatePropertyCache(strict=False)
            return mol
        except Exception:
            return None

@torch.no_grad()
def dense_to_mol(
    A_soft: torch.Tensor,
    X_soft: torch.Tensor,
    E_soft: torch.Tensor,
    bond_threshold: float = 0.2,
) -> Optional[Chem.Mol]:
    """
    Convert final continuous A, X, E into an RDKit molecule by
    valence-aware discretization with pruning.
    """
    device = X_soft.device
    N = X_soft.size(0)

    # atom identity + extras
    atom_idx = X_soft[:, :NUM_ATOM_TYPES].argmax(dim=-1).cpu().numpy()
    
    task = get_spec().name
    
    # extra channels only for guacamol/moses
    if task not in ("qm9", "zinc"):
        charge_norm = X_soft[:, NUM_ATOM_TYPES + 0].detach().cpu()
        arom_flag   = X_soft[:, NUM_ATOM_TYPES + 1].detach().cpu()

    mol = rdchem.RWMol()
    for i, idx in enumerate(atom_idx):
        sym = TYPE_TO_SYM.get(int(idx), "C")
        a = rdchem.Atom(sym)

        if task not in ("qm9", "zinc"):
            fc = int(torch.round(torch.clamp(charge_norm[i], -1.0, 1.0)).item())  # -> {-1,0,+1}
            if fc != 0:
                a.SetFormalCharge(fc)

            # aromatic flag (RDKit will validate later)
            if float(arom_flag[i].item()) > 0.5:
                a.SetIsAromatic(True)

        mol.AddAtom(a)

    # clamp A for stable scoring
    A_use = torch.clamp(A_soft, 0.0, 1.0)
    
    if task=="qm9" or task=="zinc":
        bonds = project_valence_prune(
            A_soft=A_use,
            X_logits=X_soft,
            E_logits=E_soft,
            base_threshold=bond_threshold,
        )
    else:
        bonds = project_valence_prune_topk(
            A_soft=A_use,
            X_logits=X_soft,
            E_logits=E_soft,
        )

    for i in range(N):
        for j in range(i + 1, N):
            b_idx = int(bonds[i, j].item())
            if b_idx >= NUM_BOND_TYPES:
                continue
            bt = BOND_TYPES[min(b_idx, len(BOND_TYPES) - 1)]
            try:
                mol.AddBond(i, j, bt)
            except Exception:
                continue

    # aromatic consistency
    if task not in ("qm9", "zinc"): # , "moses"
        for bond in mol.GetBonds():
            if bond.GetBondType() == rdchem.BondType.AROMATIC:
                bond.GetBeginAtom().SetIsAromatic(True)
                bond.GetEndAtom().SetIsAromatic(True)

        for atom in mol.GetAtoms():
            if atom.GetIsAromatic():
                has_arom_bond = any(b.GetBondType() == rdchem.BondType.AROMATIC for b in atom.GetBonds())
                if not has_arom_bond:
                    atom.SetIsAromatic(False)
                    
    rmol = mol.GetMol()
    
    if task == "guacamol" or task == "moses":
        rmol = sanitize_mol_relaxed(rmol)
        rmol = defog_charge_repair(rmol)
        rmol = sanitize_mol_relaxed(rmol)
    else:
        rmol = sanitize_mol_safe(rmol)
    
    return rmol


def generate_molecules_from_graphex(
    vnet: VelocityFieldGNNTransformer,
    graphon: np.ndarray,
    atom_prob: np.ndarray,
    atom_z: np.ndarray,
    bond_prob: np.ndarray,
    cycle_counts: List[dict],
    node_counts: List,
    num_samples: int = 100,
    n_steps: int = 50,
    bond_threshold: float = 0.5,
    gen_batch_size: int = 16,
    fixed_nodes: int = 30,
    task: str = "qm9",
) -> List[Optional[Chem.Mol]]:
    max_atoms = graphon.shape[0]
    mols: List[Optional[Chem.Mol]] = []
    remaining = num_samples
    start_index = 0 ####

    while remaining > 0:
        cur_bs = min(gen_batch_size, remaining)
        end_index = start_index + cur_bs
        n_nodes_list = node_counts[start_index:end_index]
        n_cycle_list = cycle_counts[start_index:end_index]

        noise_graphs, _ = sample_graphex_noise_graph_batch(
            graphon=graphon,
            atom_prob=atom_prob,
            atom_z=atom_z,
            bond_prob=bond_prob,
            max_atoms=max_atoms,
            cycle_counts=n_cycle_list,
            n_nodes_list=n_nodes_list,
            return_batch=False,
        )

        for g0 in noise_graphs:
            N = min(g0.num_nodes, max_atoms)
            if N == 0:
                mols.append(None)
                remaining -= 1
                continue

            A0, X0, E0 = build_dense_type_graph(g0, num_nodes=N, edge_feat_dim=NUM_BOND_TYPES)

            A1, X1, E1 = integrate_flow(vnet=vnet, A0=A0, X0=X0, E0=E0, n_steps=n_steps)
            mol_raw = dense_to_mol(A1, X1, E1, bond_threshold=bond_threshold)
            if task == "qm9" or task == "zinc":
                mol = kekulize_and_remove_hs(mol_raw)
            else:
                mol = kekulize_preserve_hs(mol_raw)
            mols.append(mol)
            remaining -= 1
            if remaining <= 0:
                break
                
        start_index = start_index + cur_bs
    return mols