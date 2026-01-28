from typing import Optional, List, Tuple
import os, ast, random
import numpy as np
from torch_geometric.data import Data, Batch

from fm_graphex.features.molecule_like_features import igraph_to_pyg
from fm_graphex.data.graphex_sampling import generate_graph_from_graphex

def load_cycle_counts(path: str) -> List[dict]:
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Using simple fallback ring counts.")
        return [{5: 1, 6: 1}, {6: 2}, {5: 2}]
    with open(path, "r") as f:
        content = f.read()
    cycle_counts = ast.literal_eval(content)
    return cycle_counts

def load_node_counts(path: str) -> List:
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Using simple fallback ring counts.")
        return [5, 20, 25, 29, 38]
    with open(path, "r") as f:
        content = f.read()
    return ast.literal_eval(content)

def sample_graphex_noise_graph_single(
    graphon: np.ndarray,
    atom_prob: np.ndarray,
    atom_z: np.ndarray,
    bond_prob: np.ndarray,
    max_atoms: int,
    cycle_counts: dict, # List[dict],
    n_nodes: Optional[int] = None,
) -> Data:
    if n_nodes is None:
        n_nodes = np.random.randint(5, max_atoms + 1)

    random_rings = cycle_counts # random.choice(cycle_counts)

    gr = generate_graph_from_graphex(
        graphon,
        n=n_nodes,
        sparsity_factor=None,
        epsilon=1e-3,
        ring_proportion=0.1,
        ring_sizes=random_rings,
        embedproportion=0.1,
    )

    z0_graph = igraph_to_pyg(gr, atom_prob, atom_z, bond_prob)
    return z0_graph

def sample_graphex_noise_graph_batch(
    graphon: np.ndarray,
    atom_prob: np.ndarray,
    atom_z: np.ndarray,
    bond_prob: np.ndarray,
    max_atoms: int,
    cycle_counts: List[dict],
    n_nodes_list: List[int],
    return_batch: bool = False,
):
    noise_graphs = []
    for i, n in enumerate(n_nodes_list):
        n_clipped = min(n, max_atoms)
        data = sample_graphex_noise_graph_single(
            graphon=graphon,
            atom_prob=atom_prob,
            atom_z=atom_z,
            bond_prob=bond_prob,
            max_atoms=max_atoms,
            cycle_counts=cycle_counts[i],
            n_nodes=n_clipped,
        )
        noise_graphs.append(data)

    batch_noise = Batch.from_data_list(noise_graphs) if return_batch else None
    return noise_graphs, batch_noise