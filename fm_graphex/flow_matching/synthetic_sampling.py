from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

from fm_graphex.flow_matching.dense_graph import build_dense_type_graph
from fm_graphex.flow_matching.sampling import integrate_flow, threshold_adjacency
from fm_graphex.models.velocity_field import VelocityFieldGNNTransformer
from fm_graphex.data.synthetic import nx_to_pyg_graph, pyg_to_nx_undirected_simple, undirected_edge_count
from fm_graphex.data.synthetic_graphex_noise import (
    SBMMixture,
    sample_graphette_noise_batch,
    sample_sbm_noise_batch_for_sizes,
)


def build_size_conditioned_edge_counts(dataset: Sequence[Data]) -> Dict[int, List[int]]:
    by_n: Dict[int, List[int]] = {}
    for data in dataset:
        n = int(data.num_nodes)
        if n > 0:
            by_n.setdefault(n, []).append(undirected_edge_count(data))
    if not by_n:
        raise ValueError("Cannot estimate edge budgets from an empty dataset.")
    return by_n


def _edge_density_from_count(n_nodes: int, num_edges: int) -> float:
    n = int(n_nodes)
    max_edges = n * (n - 1) // 2
    return float(num_edges / max_edges) if max_edges > 0 else 0.0


def expected_num_edges_synthetic(
    n_nodes: int,
    edge_counts_by_n: Dict[int, List[int]],
    fallback_density: float,
    mode: str = "sample",
    require_connected: bool = True,
    task: str = "ego",
) -> int:
    """Size-conditioned expected edge budget used before shared thresholding."""
    n = int(n_nodes)
    if n <= 1:
        return 0
    mode = str(mode).lower()
    if mode not in {"sample", "mean", "median"}:
        raise ValueError("edge budget mode must be sample, mean, or median")

    max_edges = n * (n - 1) // 2
    values = edge_counts_by_n.get(n, [])
    if values:
        clean = [int(v) for v in values]
        if mode == "sample":
            k = int(random.choice(clean))
        elif mode == "mean":
            k = int(round(float(np.mean(clean))))
        else:
            k = int(round(float(np.median(clean))))
    elif str(task).lower() == "tree" and require_connected:
        k = n - 1
    else:
        keys = sorted(int(k0) for k0 in edge_counts_by_n if int(k0) > 1)
        if keys:
            nearest_n = min(keys, key=lambda z: (abs(z - n), z))
            densities = [
                _edge_density_from_count(nearest_n, int(m))
                for m in edge_counts_by_n.get(nearest_n, [])
            ]
            density = float(np.mean(densities)) if densities else float(fallback_density)
        else:
            density = float(fallback_density)
        k = int(round(float(np.clip(density, 0.0, 1.0)) * max_edges))

    if require_connected:
        k = max(n - 1, k)
    return max(0, min(int(k), max_edges))


def dense_to_pyg_graph(A_bin: torch.Tensor) -> Data:
    A = A_bin.detach().cpu().float().clone()
    n = int(A.size(0))
    A = (0.5 * (A + A.T) > 0).float()
    A.fill_diagonal_(0.0)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    iu, ju = torch.triu_indices(n, n, offset=1)
    mask = A[iu, ju] > 0
    G.add_edges_from((int(u), int(v)) for u, v in zip(iu[mask], ju[mask]))
    return nx_to_pyg_graph(G)


def keep_largest_connected_component(A_bin: torch.Tensor) -> torch.Tensor:
    """Molecule-style LCC retention used by the current SBM notebook."""
    A = A_bin.detach().cpu()
    n = int(A.size(0))
    G = nx.Graph()
    G.add_nodes_from(range(n))
    iu, ju = torch.triu_indices(n, n, offset=1)
    mask = A[iu, ju] > 0
    G.add_edges_from((int(u), int(v)) for u, v in zip(iu[mask], ju[mask]))
    if G.number_of_nodes() == 0:
        return A_bin[:0, :0]
    comps = list(nx.connected_components(G))
    if len(comps) <= 1:
        return A_bin
    largest = sorted(max(comps, key=len))
    idx = torch.tensor(largest, dtype=torch.long, device=A_bin.device)
    return A_bin.index_select(0, idx).index_select(1, idx)


def _sample_generation_sources(
    task: str,
    graphon: np.ndarray,
    n_nodes_list: Sequence[int],
    ring_sizes_pool: Optional[List[Dict[int, int]]],
    remove_ring: bool,
    sbm_mix: Optional[SBMMixture],
) -> List[Data]:
    task = str(task).lower()
    if task == "sbm":
        if sbm_mix is None:
            raise ValueError("SBM generation requires sbm_mix.")
        return sample_sbm_noise_batch_for_sizes(sbm_mix, n_nodes_list)
    return sample_graphette_noise_batch(
        graphon=graphon,
        n_nodes_list=n_nodes_list,
        ring_sizes_pool=ring_sizes_pool,
        remove_ring=remove_ring,
        task=task,
    )


@torch.no_grad()
def generate_graphs_from_graphex_fm(
    vnet: VelocityFieldGNNTransformer,
    graphon: np.ndarray,
    node_counts: List[int],
    ring_sizes_pool: Optional[List[Dict[int, int]]],
    edge_counts_by_n: Dict[int, List[int]],
    fallback_density: float,
    num_samples: int = 200,
    n_steps: int = 200,
    edge_budget_mode: str = "sample",
    gen_batch_size: int = 32,
    require_connected: bool = True,
    remove_ring: bool = False,
    task: str = "ego",
    sbm_mix: Optional[SBMMixture] = None,
) -> List[Data]:
    """Generate Ego-small, Tree, or SBM graphs with one shared decoder."""
    if not node_counts:
        raise ValueError("node_counts must contain at least one graph size.")
    if num_samples <= 0:
        return []

    task = str(task).lower()
    max_nodes = max(int(n) for n in node_counts)
    generated: List[Data] = []

    while len(generated) < int(num_samples):
        cur_bs = min(int(gen_batch_size), int(num_samples) - len(generated))
        n_nodes_list = [
            max(1, min(int(random.choice(node_counts)), max_nodes))
            for _ in range(cur_bs)
        ]
        noise_graphs = _sample_generation_sources(
            task=task,
            graphon=graphon,
            n_nodes_list=n_nodes_list,
            ring_sizes_pool=ring_sizes_pool,
            remove_ring=remove_ring,
            sbm_mix=sbm_mix,
        )

        for g0 in noise_graphs:
            N = int(g0.num_nodes)
            if N <= 1:
                generated.append(g0)
                continue

            # Reuse the common molecular dense conversion; synthetic features are
            # simply one node channel and one edge channel.
            A0, X0, E0 = build_dense_type_graph(g0, num_nodes=N, edge_feat_dim=1)
            A1, X1, E1 = integrate_flow(vnet, A0, X0, E0, n_steps=n_steps)

            k = expected_num_edges_synthetic(
                n_nodes=N,
                edge_counts_by_n=edge_counts_by_n,
                fallback_density=fallback_density,
                mode=edge_budget_mode,
                require_connected=require_connected,
                task=task,
            )

            A_bin = threshold_adjacency(A1, n_nodes=N, expected_edges=k)

            if task == "sbm" and require_connected:
                A_bin = keep_largest_connected_component(A_bin)

            g = dense_to_pyg_graph(A_bin)
            if task in {"ego", "tree"} and int(g.num_nodes) != N:
                raise RuntimeError(f"Generated graph changed node count: requested {N}, got {g.num_nodes}.")
            if task == "sbm" and require_connected and g.num_nodes > 1:
                if not nx.is_connected(pyg_to_nx_undirected_simple(g)):
                    raise RuntimeError("SBM LCC post-processing did not produce a connected graph.")

            generated.append(g)
            if len(generated) >= int(num_samples):
                break

    return generated