from __future__ import annotations

import os
import pickle as pkl
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
from networkx import to_numpy_array
from torch_geometric.data import Data, download_url
import torch_geometric.utils

import fm_graphex.config as cfg


DATA_URLS: Dict[str, str] = {
    "sbm": "https://raw.githubusercontent.com/AndreasBergmeister/graph-generation/main/data/sbm.pkl",
    "tree": "https://raw.githubusercontent.com/AndreasBergmeister/graph-generation/main/data/tree.pkl",
    "ego": "https://raw.githubusercontent.com/tufts-ml/graph-generation-EDGE/main/graphs/Ego.pkl",
}


def nx_to_pyg_graph(G: nx.Graph, add_edge_attr: bool = True) -> Data:
    """Convert an undirected NetworkX graph to the synthetic PyG format."""
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    G = nx.convert_node_labels_to_integers(G)

    n = G.number_of_nodes()
    x = torch.ones((n, 1), dtype=torch.float32)

    edges = []
    for u, v in G.edges():
        edges.append([u, v])
        edges.append([v, u])

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = (
            torch.ones((edge_index.size(1), 1), dtype=torch.float32)
            if add_edge_attr
            else None
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float32) if add_edge_attr else None

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def pyg_to_nx_undirected_simple(data: Data) -> nx.Graph:
    n = int(data.num_nodes)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    if data.edge_index is not None and data.edge_index.numel() > 0:
        ei = data.edge_index.detach().cpu()
        for e in range(ei.size(1)):
            u = int(ei[0, e])
            v = int(ei[1, e])
            if u != v:
                G.add_edge(u, v)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def _adj_to_pyg(adj: torch.Tensor) -> Data:
    adj = adj.float().clone()
    adj.fill_diagonal_(0.0)
    edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
    n = int(adj.size(0))
    x = torch.ones((n, 1), dtype=torch.float32)
    edge_attr = torch.ones((edge_index.size(1), 1), dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def _download_if_needed(url: str, raw_dir: str) -> str:
    os.makedirs(raw_dir, exist_ok=True)
    filename = os.path.basename(url)
    expected = os.path.join(raw_dir, filename)
    if os.path.exists(expected):
        return expected
    return download_url(url, raw_dir)


def load_spectre_style_dataset(
    name: str,
    root: str = "./spectre_data",
    seed: int = 1234,
) -> Tuple[List[Data], List[Data], List[Data], int]:
    """Load the Ego-small, Tree, or SBM datasets used by the notebooks."""
    name = str(name).lower()
    if name not in DATA_URLS:
        raise ValueError(f"Unknown synthetic dataset {name!r}; use ego, tree, or sbm.")

    raw_dir = os.path.join(root, "raw")
    path = _download_if_needed(DATA_URLS[name], raw_dir)

    if name in {"sbm", "tree"}:
        with open(path, "rb") as f:
            dataset = pkl.load(f)

        def convert(graphs: Sequence[nx.Graph]) -> List[Data]:
            out: List[Data] = []
            for G in graphs:
                adj = torch.tensor(to_numpy_array(G), dtype=torch.float32)
                adj.fill_diagonal_(0.0)
                out.append(_adj_to_pyg(adj))
            return out

        train = convert(dataset["train"])
        val = convert(dataset["val"])
        test = convert(dataset["test"])

    else:  # ego
        with open(path, "rb") as f:
            networks = pkl.load(f)

        adjs = []
        for G in networks:
            adj = torch.tensor(to_numpy_array(G), dtype=torch.float32)
            adj.fill_diagonal_(0.0)
            adjs.append(adj)

        # Preserve the split logic used by the supplied Ego notebook.
        generator = torch.Generator().manual_seed(seed)
        num_graphs = len(adjs)
        train_len = int(round(num_graphs * 0.8))
        val_len = int(round(num_graphs * 0.2))
        indices = torch.randperm(num_graphs, generator=generator)
        train_set = set(indices[:train_len].tolist())
        val_set = set(indices[:val_len].tolist())
        test_set = set(indices[train_len:].tolist())

        train, val, test = [], [], []
        for i, adj in enumerate(adjs):
            if i in train_set:
                train.append(_adj_to_pyg(adj))
            if i in val_set:
                val.append(_adj_to_pyg(adj))
            if i in test_set:
                test.append(_adj_to_pyg(adj))

    all_graphs = train + val + test
    max_nodes = max((int(d.num_nodes) for d in all_graphs), default=0)
    return train, val, test, max_nodes


def estimate_edge_density(dataset: Sequence[Data]) -> float:
    num_edges = 0
    num_possible = 0
    for d in dataset:
        n = int(d.num_nodes)
        if n <= 1:
            continue
        directed = int(d.edge_index.size(1)) if d.edge_index is not None else 0
        num_edges += directed // 2
        num_possible += n * (n - 1) // 2
    return float(num_edges / max(1, num_possible))


def node_count_list(dataset: Sequence[Data]) -> List[int]:
    return [int(d.num_nodes) for d in dataset if int(d.num_nodes) > 0]


def pyg_to_dense_adj(data: Data, n_nodes: Optional[int] = None) -> torch.Tensor:
    n = int(data.num_nodes)
    N = n if n_nodes is None else min(int(n_nodes), n)
    A = torch.zeros((N, N), dtype=torch.float32, device=cfg.DEVICE)
    if data.edge_index is not None and data.edge_index.numel() > 0:
        ei = data.edge_index.to(cfg.DEVICE)
        keep = (ei[0] < N) & (ei[1] < N) & (ei[0] != ei[1])
        A[ei[0, keep], ei[1, keep]] = 1.0
    A = torch.maximum(A, A.T)
    A.fill_diagonal_(0.0)
    return A


def undirected_edge_count(data: Data) -> int:
    if data.edge_index is None or data.edge_index.numel() == 0:
        return 0
    ei = data.edge_index.detach().cpu()
    pairs = set()
    for e in range(ei.size(1)):
        u, v = int(ei[0, e]), int(ei[1, e])
        if u != v:
            pairs.add((min(u, v), max(u, v)))
    return len(pairs)