from __future__ import annotations

from typing import List, Optional
import ast
import os

import numpy as np
from torch_geometric.data import Batch, Data

from fm_graphex.data.graphex_sampling import generate_graph_from_graphex
from fm_graphex.features.molecule_like_features import igraph_to_pyg


def load_cycle_counts(path: str) -> List[dict]:
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Using fallback cycle counts.")
        return [{5: 1, 6: 1}, {6: 2}, {5: 2}]
    with open(path, "r", encoding="utf-8") as f:
        return ast.literal_eval(f.read())


def load_node_counts(path: str) -> List[int]:
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Using fallback node counts.")
        return [5, 20, 25, 29, 38]
    with open(path, "r", encoding="utf-8") as f:
        return ast.literal_eval(f.read())


def sample_graphex_noise_graph_single(
    graphon: np.ndarray,
    atom_prob: np.ndarray,
    atom_z: np.ndarray,
    bond_prob: np.ndarray,
    max_atoms: int,
    cycle_counts: dict,
    n_nodes: Optional[int] = None,
    formal_charge_prob: Optional[np.ndarray] = None,
) -> Data:
    if n_nodes is None:
        n_nodes = int(np.random.randint(5, max_atoms + 1))
    n_nodes = max(1, min(int(n_nodes), int(max_atoms)))

    gr = generate_graph_from_graphex(
        graphon,
        n=n_nodes,
        sparsity_factor=None,
        epsilon=1e-3,
        ring_proportion=0.1,
        ring_sizes=cycle_counts,
        embedproportion=0.1,
    )
    if gr.vcount() != n_nodes:
        raise RuntimeError(
            f"Graphex source has {gr.vcount()} nodes but {n_nodes} were requested."
        )

    return igraph_to_pyg(
        gr,
        atom_prob=atom_prob,
        atom_z=atom_z,
        bond_prob=bond_prob,
        formal_charge_prob=formal_charge_prob,
    )


def sample_graphex_noise_graph_batch(
    graphon: np.ndarray,
    atom_prob: np.ndarray,
    atom_z: np.ndarray,
    bond_prob: np.ndarray,
    max_atoms: int,
    cycle_counts: List[dict],
    n_nodes_list: List[int],
    return_batch: bool = False,
    formal_charge_prob: Optional[np.ndarray] = None,
):
    if len(cycle_counts) < len(n_nodes_list):
        if not cycle_counts:
            cycle_counts = [{} for _ in n_nodes_list]
        else:
            cycle_counts = list(cycle_counts) + [cycle_counts[-1]] * (len(n_nodes_list) - len(cycle_counts))

    noise_graphs = []
    for i, n in enumerate(n_nodes_list):
        data = sample_graphex_noise_graph_single(
            graphon=graphon,
            atom_prob=atom_prob,
            atom_z=atom_z,
            bond_prob=bond_prob,
            max_atoms=max_atoms,
            cycle_counts=cycle_counts[i],
            n_nodes=min(int(n), int(max_atoms)),
            formal_charge_prob=formal_charge_prob,
        )
        noise_graphs.append(data)

    batch_noise = Batch.from_data_list(noise_graphs) if return_batch else None
    return noise_graphs, batch_noise
