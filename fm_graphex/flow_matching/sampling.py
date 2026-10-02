from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple
import itertools

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import rdchem

import fm_graphex.config as cfg
from fm_graphex.data.graphex_noise import sample_graphex_noise_graph_batch
from fm_graphex.data.rdkit_conversions import (
    kekulize_and_remove_hs,
    kekulize_preserve_hs,
    sanitize_mol_safe,
)
from fm_graphex.flow_matching.dense_graph import build_dense_type_graph
from fm_graphex.models.velocity_field import VelocityFieldGNNTransformer


@torch.no_grad()
def integrate_flow(
    vnet: VelocityFieldGNNTransformer,
    A0: torch.Tensor,
    X0: torch.Tensor,
    E0: torch.Tensor,
    n_steps: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    vnet.eval()
    A_t = A0.clone().to(cfg.DEVICE)
    X_t = X0.clone().to(cfg.DEVICE)
    E_t = E0.clone().to(cfg.DEVICE)
    dt = 1.0 / float(n_steps)

    for step in range(n_steps):
        t = torch.tensor(step / float(n_steps), dtype=torch.float32, device=cfg.DEVICE)
        v_A, v_X, v_E = vnet(A_t, X_t, E_t, t)
        A_t = A_t + dt * v_A
        X_t = X_t + dt * v_X
        E_t = E_t + dt * v_E

    A_t = 0.5 * (A_t + A_t.T)
    A_t = F.softplus(A_t, beta=1.0, threshold=20.0)
    
    E_t = 0.5 * (E_t + E_t.transpose(0, 1))
    
    return A_t, X_t, E_t


def formal_charge_indices_from_X(X: torch.Tensor, task: str) -> Optional[torch.Tensor]:
    if not cfg.use_formal_charge(task):
        return None
    q0 = cfg.NUM_ATOM_TYPES
    qdim = len(cfg.FORMAL_CHARGES)
    if X.size(-1) < q0 + qdim:
        return torch.full(
            (X.size(0),), cfg.NEUTRAL_CHARGE_IDX, dtype=torch.long, device=X.device
        )
    return X[:, q0:q0 + qdim].argmax(dim=-1)


def atom_valence_limits(
    atom_idx: torch.Tensor,
    charge_idx: Optional[torch.Tensor] = None,
    task: str = "qm9",
) -> torch.Tensor:
    max_table = cfg.MAX_VALENCE_PER_TYPE.to(atom_idx.device)
    vals = max_table[atom_idx.clamp(max=len(max_table) - 1)].float().clone()
    if not cfg.use_formal_charge(task) or charge_idx is None:
        return vals

    for i, (ai, qi) in enumerate(zip(atom_idx.tolist(), charge_idx.tolist())):
        sym = cfg.ATOM_TYPES[min(int(ai), len(cfg.ATOM_TYPES) - 1)]
        charge = cfg.formal_charge_from_index(int(qi))
        if sym == "N" and charge == 1:
            vals[i] = max(float(vals[i]), 4.0)
        elif sym == "O" and charge == -1:
            vals[i] = max(float(vals[i]), 1.0)
        elif sym in {"S", "Se"}:
            vals[i] = max(float(vals[i]), 6.0)
        elif sym == "P":
            vals[i] = max(float(vals[i]), 5.0)
    return vals


def _task_edge_multiplier(task: str) -> float:
    return {
        "qm9": 1.05,
        "zinc": 1.20,
        "moses": 1.35,
        "guacamol": 1.40,
    }.get(str(task).lower(), 1.15)


def expected_num_edges(n: int, task: str) -> int:
    if n <= 1:
        return 0
    max_edges = n * (n - 1) // 2
    m = int(round(_task_edge_multiplier(task) * n))
    return max(n - 1, min(m, max_edges))


def threshold_adjacency(
    A_soft: torch.Tensor,
    n_nodes: int,
    expected_edges: int,
) -> torch.Tensor:
    if n_nodes <= 1:
        return torch.zeros(max(n_nodes, 0), max(n_nodes, 0), device=A_soft.device)

    A = A_soft[:n_nodes, :n_nodes]
    A = torch.sigmoid(0.5 * (A + A.T))
    A = A.clone()
    A.fill_diagonal_(0.0)

    max_edges = n_nodes * (n_nodes - 1) // 2
    k = max(n_nodes - 1, min(int(expected_edges), max_edges))
    iu, ju = torch.triu_indices(n_nodes, n_nodes, offset=1, device=A.device)

    scores = A[iu, ju]
    k = max(n_nodes - 1, min(int(expected_edges), scores.numel()))
    tau = torch.topk(scores, k=k, largest=True).values[-1]
    selected = scores >= tau

    A_bin = torch.zeros(n_nodes, n_nodes, dtype=A.dtype, device=A.device)
    A_bin[iu[selected], ju[selected]] = 1.0
    A_bin[ju[selected], iu[selected]] = 1.0
    return A_bin


def symmetrize_bond_logits(E_soft: torch.Tensor, task: str) -> torch.Tensor:
    E = 0.5 * (E_soft + E_soft.transpose(0, 1)).clone()
    # Preserve the archive's successful QM9/ZINC behavior. Aromatic bond logits
    # remain active for GuacaMol/MOSES, where aromatic chemistry is essential.
    if task in {"qm9", "zinc"} and E.size(-1) >= 4:
        aromatic_idx = (
            cfg.BOND_TYPES.index(rdchem.BondType.AROMATIC)
            if rdchem.BondType.AROMATIC in cfg.BOND_TYPES else 3
        )
        E[..., aromatic_idx] = -1e9
    return E


def largest_connected_component(A_bin: torch.Tensor) -> Set[int]:
    n = A_bin.size(0)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if A_bin[i, j] > 0:
                graph.add_edge(i, j)
    comps = list(nx.connected_components(graph))
    return set(max(comps, key=len)) if comps else set(range(n))


def _make_rdkit_atom(symbol: str, formal_charge: int = 0) -> rdchem.Atom:
    atom = rdchem.Atom(symbol)
    if formal_charge:
        atom.SetFormalCharge(int(formal_charge))
    return atom


def _atom_charge(
    i: int,
    charge_idx: Optional[torch.Tensor],
    task: str,
) -> int:
    if not cfg.use_formal_charge(task) or charge_idx is None:
        return 0
    return cfg.formal_charge_from_index(int(charge_idx[i].detach().cpu()))


def _build_typed_molecule(
    A_bin: torch.Tensor,
    atom_idx: torch.Tensor,
    bond_idx: torch.Tensor,
    charge_idx: Optional[torch.Tensor],
    active: Set[int],
    task: str,
) -> Optional[Chem.Mol]:
    rw = rdchem.RWMol()
    idxmap: Dict[int, int] = {}
    for i in range(A_bin.size(0)):
        if i not in active:
            continue
        a = int(atom_idx[i].detach().cpu())
        sym = cfg.ATOM_TYPES[min(a, len(cfg.ATOM_TYPES) - 1)]
        idxmap[i] = rw.AddAtom(_make_rdkit_atom(sym, _atom_charge(i, charge_idx, task)))

    for i in range(A_bin.size(0)):
        for j in range(i + 1, A_bin.size(0)):
            if A_bin[i, j] <= 0 or i not in idxmap or j not in idxmap:
                continue
            bt_idx = min(int(bond_idx[i, j].detach().cpu()), cfg.NUM_BOND_TYPES - 1)
            bt = cfg.BOND_TYPES[bt_idx]
            try:
                rw.AddBond(idxmap[i], idxmap[j], bt)
                if bt == rdchem.BondType.AROMATIC:
                    ai, aj = rw.GetAtomWithIdx(idxmap[i]), rw.GetAtomWithIdx(idxmap[j])
                    ai.SetIsAromatic(True); aj.SetIsAromatic(True)
                    bond = rw.GetBondBetweenAtoms(idxmap[i], idxmap[j])
                    if bond is not None:
                        bond.SetIsAromatic(True)
            except Exception:
                continue

    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
        return out if out.GetNumAtoms() > 0 else None
    except Exception:
        return None


def fallback_single_bond_decode(
    A_bin: torch.Tensor,
    atom_idx: torch.Tensor,
    charge_idx: Optional[torch.Tensor],
    active: Set[int],
    task: str,
) -> Optional[Chem.Mol]:
    rw = rdchem.RWMol()
    idxmap: Dict[int, int] = {}
    for i in range(A_bin.size(0)):
        if i not in active:
            continue
        a = int(atom_idx[i].detach().cpu())
        sym = cfg.ATOM_TYPES[min(a, len(cfg.ATOM_TYPES) - 1)]
        idxmap[i] = rw.AddAtom(_make_rdkit_atom(sym, _atom_charge(i, charge_idx, task)))

    for i in range(A_bin.size(0)):
        for j in range(i + 1, A_bin.size(0)):
            if A_bin[i, j] > 0 and i in idxmap and j in idxmap:
                try:
                    rw.AddBond(idxmap[i], idxmap[j], rdchem.BondType.SINGLE)
                except Exception:
                    pass
    try:
        out = rw.GetMol()
        Chem.SanitizeMol(out)
        return out if out.GetNumAtoms() > 0 else None
    except Exception:
        return None


def prune_by_valence(
    A_bin: torch.Tensor,
    atom_idx: torch.Tensor,
    bond_idx: torch.Tensor,
    n: float,
    charge_idx: Optional[torch.Tensor] = None,
    task: str = "qm9",
    keep_lcc: bool = True,
) -> torch.Tensor:
    A = A_bin.clone()
    n = A.size(0)
    max_val = atom_valence_limits(atom_idx, charge_idx, task).to(A.device)
    bond_orders = cfg.BOND_ORDERS.to(A.device)

    changed = True
    while changed:
        changed = False
        val = torch.zeros(n, device=A.device)
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j] <= 0:
                    continue
                bt = min(int(bond_idx[i, j].detach().cpu()), len(bond_orders) - 1)
                order = float(bond_orders[bt].detach().cpu())
                val[i] += order
                val[j] += order
                edges.append((i, j, order))

        for i in range(n):
            if val[i] <= max_val[i] + 1e-6:
                continue
            incident = [(u, v, order) for u, v, order in edges if u == i or v == i]
            if not incident:
                continue
            u, v, _ = sorted(incident, key=lambda item: item[2], reverse=True)[0]
            A[u, v] = A[v, u] = 0.0
            changed = True
            break

    active = largest_connected_component(A) if keep_lcc else set(range(n))
    mol = _build_typed_molecule(A, atom_idx, bond_idx, charge_idx, active, task)
    if mol is not None:
        return mol
    return fallback_single_bond_decode(A, atom_idx, charge_idx, active, task)


@torch.no_grad()
def dense_to_mol(
    A_soft: torch.Tensor,
    X_soft: torch.Tensor,
    E_soft: torch.Tensor,
    expected_edges: Optional[int] = None,
    keep_lcc: bool = False,
) -> Optional[Chem.Mol]:
    n = X_soft.size(0)
    task = cfg.get_spec().name
    if n == 0:
        return None
    if X_soft.size(1) < cfg.node_feat_dim():
        raise ValueError(
            f"X has {X_soft.size(1)} channels; {task} requires {cfg.node_feat_dim()}. "
            "Do not load pre-charge checkpoints for GuacaMol/MOSES."
        )

    expected_edges = expected_num_edges(n, task) if expected_edges is None else int(expected_edges)
    E = symmetrize_bond_logits(E_soft, task)
    A_bin = threshold_adjacency(A_soft, n, expected_edges)
    atom_idx = X_soft[:, :cfg.NUM_ATOM_TYPES].argmax(dim=-1)
    charge_idx = formal_charge_indices_from_X(X_soft, task)
    bond_idx = E.argmax(dim=-1)
    A_bin = prune_by_valence(A_bin, atom_idx, bond_idx, n, charge_idx, task)

    return A_bin


@torch.no_grad()
def generate_molecules_from_graphex(
    vnet: VelocityFieldGNNTransformer,
    graphon: np.ndarray,
    atom_prob: np.ndarray,
    atom_z: np.ndarray,
    bond_prob: np.ndarray,
    cycle_counts: List[dict],
    node_counts: List[int],
    num_samples: int = 100,
    n_steps: int = 50,
    bond_threshold: float = 0.5,
    gen_batch_size: int = 16,
    fixed_nodes: int = 30,
    task: str = "qm9",
    formal_charge_prob: Optional[np.ndarray] = None,
) -> List[Optional[Chem.Mol]]:
    active_task = cfg.get_spec().name
    if str(task).lower() != active_task:
        raise ValueError(
            f"Generation task={task!r} does not match active config={active_task!r}. "
            "Call cfg.set_dataset(task) before constructing/loading the model."
        )
    if not node_counts:
        node_counts = [fixed_nodes]
    if not cycle_counts:
        cycle_counts = [{}]

    max_atoms = int(graphon.shape[0])
    mols: List[Optional[Chem.Mol]] = []
    node_iter = itertools.cycle(node_counts)
    cycle_iter = itertools.cycle(cycle_counts)

    while len(mols) < num_samples:
        cur_bs = min(gen_batch_size, num_samples - len(mols))
        n_batch = [max(1, min(int(next(node_iter)), max_atoms)) for _ in range(cur_bs)]
        c_batch = [next(cycle_iter) for _ in range(cur_bs)]

        noise_graphs, _ = sample_graphex_noise_graph_batch(
            graphon=graphon,
            atom_prob=atom_prob,
            atom_z=atom_z,
            bond_prob=bond_prob,
            formal_charge_prob=formal_charge_prob,
            max_atoms=max_atoms,
            cycle_counts=c_batch,
            n_nodes_list=n_batch,
            return_batch=False,
        )

        for g0 in noise_graphs:
            n = int(g0.num_nodes)
            if n <= 0:
                mols.append(None)
                continue
            A0, X0, E0 = build_dense_type_graph(
                g0, num_nodes=n, edge_feat_dim=cfg.NUM_BOND_TYPES
            )
            A1, X1, E1 = integrate_flow(vnet, A0, X0, E0, n_steps=n_steps)
            
            raw = dense_to_mol(A1, X1, E1, expected_edges=expected_num_edges(n, task))

            if task in {"qm9", "zinc"}:
                mols.append(kekulize_and_remove_hs(raw))
            else:
                mols.append(kekulize_preserve_hs(raw))
            if len(mols) >= num_samples:
                break

    return mols