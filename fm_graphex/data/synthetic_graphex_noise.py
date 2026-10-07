from __future__ import annotations

import ast
import os
import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import igraph as ig
import networkx as nx
import numpy as np
from networkx.algorithms.community import greedy_modularity_communities
from torch_geometric.data import Data

from fm_graphex.data.synthetic import nx_to_pyg_graph, pyg_to_nx_undirected_simple


_EGO_GRAPHON_CACHE: Dict[Tuple[int, float], np.ndarray] = {}


def sample_graph_from_graphon(graphon: np.ndarray, n: int) -> np.ndarray:
    m = int(graphon.shape[0])
    n = int(n)
    u = np.random.uniform(0.0, 1.0, n)
    idx = np.clip((u * m).astype(int), 0, m - 1)
    A = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.rand() < float(graphon[idx[i], idx[j]]):
                A[i, j] = A[j, i] = 1.0
    return A


def sample_graphon_from_sparsified_graphon(
    graphon: np.ndarray,
    n: int,
    sparsity_factor: Optional[float],
    epsilon: float = 1e-3,
) -> np.ndarray:
    meanW = float(np.mean(graphon))
    if sparsity_factor is None:
        sparsity_factor = 1.0 / (max(meanW, 1e-6) * max(int(n), 1)) + epsilon
    W = np.clip(graphon * float(sparsity_factor), 0.0, 1.0)
    return sample_graph_from_graphon(W, int(n))


def create_exp_matrix(nrow: int, scalar: float) -> np.ndarray:
    idx = np.arange(1, int(nrow) + 1, dtype=np.float64)
    return np.exp(-(idx[:, None] + idx[None, :]) / float(scalar))


def get_ego_graphon(resolution: int = 1000, scalar: float = 30.0) -> np.ndarray:
    key = (int(resolution), float(scalar))
    if key not in _EGO_GRAPHON_CACHE:
        _EGO_GRAPHON_CACHE[key] = create_exp_matrix(*key).astype(np.float32)
    return _EGO_GRAPHON_CACHE[key]


def star_f3(xvals: np.ndarray, t: float) -> np.ndarray:
    return t * np.exp(-1.0 * (xvals + 1.0))


def sample_star_function(star_f, num_samples: int, t: float) -> np.ndarray:
    xvals = np.linspace(0, 1, int(num_samples))
    return np.random.poisson(lam=star_f(xvals, t))


def add_star_edges_budgeted(gr: ig.Graph, star_f, t: float, max_nodes: int) -> ig.Graph:
    result = gr.copy()
    max_nodes = int(max_nodes)
    initial_nodes = result.vcount()
    if initial_nodes <= 0 or initial_nodes >= max_nodes:
        return result
    samples = sample_star_function(star_f, initial_nodes, t)
    for i in np.where(samples > 0)[0]:
        remaining = max_nodes - result.vcount()
        if remaining <= 0:
            break
        count = min(int(samples[i]), remaining)
        if count <= 0:
            continue
        cur = result.vcount()
        result.add_vertices(count)
        result.add_edges([(int(i), v) for v in range(cur, cur + count)])
    return result


def add_disjoint_ring_components(gr: ig.Graph, ring_size_counts: Dict[int, int], max_nodes: int) -> ig.Graph:
    result = gr.copy()
    for ring_size, count in ring_size_counts.items():
        for _ in range(max(0, int(count))):
            remaining = int(max_nodes) - result.vcount()
            if remaining <= 0:
                return result
            r = min(int(ring_size), remaining)
            if r < 3:
                continue
            ring = ig.Graph.Ring(r)
            cur = result.vcount()
            result = result.disjoint_union(ring)
            if cur > 0:
                result.add_edge(
                    int(np.random.randint(0, cur)),
                    int(np.random.randint(cur, cur + r)),
                )
    return result


def add_path_based_ring_closures(gr: ig.Graph, ring_size_dict: Dict[int, int], max_nodes: int) -> ig.Graph:
    result = gr.copy()
    max_nodes = int(max_nodes)
    for ring_size, count in ring_size_dict.items():
        ring_size = int(ring_size)
        for _ in range(max(0, int(count))):
            if result.vcount() >= max_nodes:
                return result
            n_vertices = result.vcount()
            if n_vertices == 0:
                r = min(ring_size, max_nodes)
                if r < 3:
                    return result
                result.add_vertices(r)
                result.add_edges([(i, (i + 1) % r) for i in range(r)])
                continue

            existing_path = None
            path_limit = max(2, ring_size - 1)
            for _attempt in range(80):
                start = int(np.random.randint(0, n_vertices))
                ends = [i for i in range(n_vertices) if i != start]
                if not ends:
                    break
                sampled = np.random.choice(ends, size=min(10, len(ends)), replace=False)
                for end in sampled:
                    paths = result.get_shortest_paths(start, to=int(end), mode="all")
                    if paths and 2 <= len(paths[0]) <= path_limit:
                        existing_path = paths[0]
                        break
                if existing_path is not None:
                    break

            if existing_path is None:
                edges = result.get_edgelist()
                if edges:
                    existing_path = list(edges[np.random.randint(len(edges))])
                elif n_vertices >= 2:
                    existing_path = np.random.choice(n_vertices, 2, replace=False).tolist()
                else:
                    existing_path = [0]

            add = min(max(0, ring_size - len(existing_path)), max_nodes - result.vcount())
            edges_to_add = []
            if add > 0:
                cur = result.vcount()
                result.add_vertices(add)
                new_nodes = list(range(cur, cur + add))
                edges_to_add.append((existing_path[0], new_nodes[0]))
                edges_to_add.extend((new_nodes[j], new_nodes[j + 1]) for j in range(add - 1))
                edges_to_add.append((new_nodes[-1], existing_path[-1]))
            elif len(existing_path) >= 2 and not result.are_adjacent(existing_path[0], existing_path[-1]):
                edges_to_add.append((existing_path[0], existing_path[-1]))
            if edges_to_add:
                result.add_edges(edges_to_add)
    return result


def remove_cycles(gr: ig.Graph) -> ig.Graph:
    return gr.as_undirected().spanning_tree()


def ensure_exact_connected_node_count(gr: ig.Graph, n: int) -> ig.Graph:
    """Restore the requested source size without deleting the learned motifs."""
    n = int(n)
    if n <= 0:
        return ig.Graph()
    result = gr.copy()
    result.simplify(multiple=True, loops=True)
    if result.vcount() == 0:
        result.add_vertices(n)
        if n > 1:
            result.add_edges([(i - 1, i) for i in range(1, n)])
        return result
    if result.vcount() > n:
        order = result.bfs(0)[0][:n] if result.ecount() else list(range(n))
        result = result.subgraph(order)
    while result.vcount() < n:
        old_n = result.vcount()
        result.add_vertex()
        if old_n > 0:
            result.add_edge(int(np.random.randint(0, old_n)), old_n)
    result.simplify(multiple=True, loops=True)
    if result.vcount() > 1 and not result.is_connected():
        comps = result.connected_components()
        reps = [int(next(iter(c))) for c in comps]
        result.add_edges([(a, b) for a, b in zip(reps[:-1], reps[1:]) if not result.are_adjacent(a, b)])
    if result.vcount() != n:
        raise RuntimeError(f"Exact source-size repair failed: requested={n}, got={result.vcount()}.")
    return result


def igraph_to_pyg_simple(gr: ig.Graph) -> Data:
    G = nx.Graph()
    G.add_nodes_from(range(gr.vcount()))
    G.add_edges_from(gr.get_edgelist())
    return nx_to_pyg_graph(G)


def generate_graphette_source(
    graphon: np.ndarray,
    n: int,
    ring_sizes: Optional[Dict[int, int]] = None,
    remove_ring: bool = False,
    task: str = "ego",
) -> ig.Graph:
    """Task-specific Ego/Tree graphette source with exact requested size."""
    task = str(task).lower()
    n = int(n)
    if n <= 0:
        return ig.Graph()

    if task == "ego":
        W = get_ego_graphon(resolution=1000, scalar=30.0)
        A = sample_graph_from_graphon(W, n=n)
        gr = ig.Graph.Adjacency((A > 0).tolist(), mode="undirected")
        gr.simplify(multiple=True, loops=True)
        if gr.vcount() > 0 and gr.ecount() > 0:
            gr = gr.subgraph(max(gr.connected_components(), key=len))
        elif gr.vcount() > 0:
            gr = gr.subgraph([0])
        if ring_sizes:
            gr = add_disjoint_ring_components(gr, ring_sizes, max_nodes=n)
            gr = add_path_based_ring_closures(gr, ring_sizes, max_nodes=n)
        gr = add_star_edges_budgeted(gr, star_f3, t=int(np.sqrt(n)), max_nodes=n)
        if remove_ring:
            raise ValueError("Ego-small uses remove_ring=False.")
        gr = ensure_exact_connected_node_count(gr, n)
        if n > 1 and not gr.is_connected():
            raise RuntimeError("Ego source is disconnected after exact-size repair.")
        return gr

    if task == "tree":
        A = sample_graphon_from_sparsified_graphon(graphon, n=n, sparsity_factor=None)
        gr = ig.Graph.Adjacency((A > 0).tolist(), mode="undirected")
        if gr.vcount() > 0 and gr.ecount() > 0:
            gr = gr.subgraph(max(gr.connected_components(), key=len))
        elif gr.vcount() > 0:
            gr = gr.subgraph([0])
        if remove_ring:
            gr = remove_cycles(gr)
        gr = ensure_exact_connected_node_count(gr, n)
        if remove_ring and n > 0:
            if not gr.is_connected() or gr.ecount() != max(0, n - 1):
                raise RuntimeError(f"Tree source invariant failed: n={n}, m={gr.ecount()}.")
        return gr

    raise ValueError(f"generate_graphette_source supports ego/tree, got {task!r}.")


def sample_graphette_noise_batch(
    graphon: np.ndarray,
    n_nodes_list: Sequence[int],
    ring_sizes_pool: Optional[List[Dict[int, int]]] = None,
    remove_ring: bool = False,
    task: str = "ego",
) -> List[Data]:
    out: List[Data] = []
    for requested_n in n_nodes_list:
        rings = random.choice(ring_sizes_pool) if ring_sizes_pool else None
        gr = generate_graphette_source(
            graphon=graphon,
            n=int(requested_n),
            ring_sizes=rings,
            remove_ring=remove_ring,
            task=task,
        )
        if gr.vcount() != int(requested_n):
            raise RuntimeError(f"Source has {gr.vcount()} nodes; requested {requested_n}.")
        out.append(igraph_to_pyg_simple(gr))
    return out


def _ego_cycle_count_dict(data: Data, min_cycle: int = 3, max_cycle: int = 8) -> Dict[int, int]:
    G = pyg_to_nx_undirected_simple(data)
    counts: Dict[int, int] = {}
    try:
        basis = nx.cycle_basis(G)
    except Exception:
        basis = []
    for cyc in basis:
        k = len(cyc)
        if min_cycle <= k <= max_cycle:
            counts[k] = counts.get(k, 0) + 1
    return counts


def load_ego_ring_pool(
    path: str = "cycle_count_ego.txt",
    train_dataset: Optional[Sequence[Data]] = None,
    fallback_max_graphs: int = 256,
) -> List[Dict[int, int]]:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            pool = ast.literal_eval(f.read())
        if not isinstance(pool, list):
            raise ValueError(f"{path} must contain a list of dictionaries.")
        return [
            {int(k): int(v) for k, v in item.items() if int(k) >= 3 and int(v) > 0}
            for item in pool if isinstance(item, dict)
        ]
    if train_dataset is None:
        return []
    pool = [_ego_cycle_count_dict(g) for g in train_dataset[:fallback_max_graphs]]
    return [d for d in pool if d]


@dataclass
class SBMParams:
    k: int
    pi: np.ndarray
    P: np.ndarray


@dataclass
class SBMMixture:
    k_min: int
    k_max: int
    pk: Dict[int, float]
    params: Dict[int, SBMParams]


def estimate_k_greedy_modularity(G: nx.Graph, k_min: int = 2, k_max: int = 5) -> int:
    if G.number_of_nodes() < 4 or G.number_of_edges() == 0:
        return k_min
    return int(max(k_min, min(k_max, len(list(greedy_modularity_communities(G))))))


def estimate_sbm_params_for_fixed_k(train_pyg: Sequence[Data], k: int, seed: int = 0) -> SBMParams:
    P_num = np.zeros((k, k), dtype=np.float64)
    P_den = np.zeros((k, k), dtype=np.float64)
    pi_acc = np.zeros((k,), dtype=np.float64)
    used = 0

    for d in train_pyg:
        G = pyg_to_nx_undirected_simple(d)
        if G.number_of_nodes() < 2:
            continue
        comms = sorted(greedy_modularity_communities(G), key=len, reverse=True)
        if not comms:
            comms = [set(G.nodes())]
        while len(comms) < k:
            largest = list(comms[0])
            if len(largest) <= 1:
                break
            half = max(1, len(largest) // 2)
            comms[0] = set(largest[:half])
            comms.append(set(largest[half:]))
            comms = sorted(comms, key=len, reverse=True)
        if len(comms) > k:
            for extra in comms[k:]:
                comms[0] = set(comms[0]).union(extra)
            comms = comms[:k]
        while len(comms) < k:
            comms.append(set())

        label = {u: bi for bi, S in enumerate(comms) for u in S}
        sizes = np.zeros((k,), dtype=np.int64)
        for u in G.nodes():
            sizes[label.get(u, 0)] += 1
        if sizes.sum() == 0:
            continue
        used += 1
        pi_acc += sizes / sizes.sum()
        for a in range(k):
            for b in range(k):
                P_den[a, b] += sizes[a] * (sizes[a] - 1) / 2.0 if a == b else sizes[a] * sizes[b]
        for u, v in G.edges():
            a, b = label.get(u, 0), label.get(v, 0)
            P_num[a, b] += 1.0
            P_num[b, a] += 1.0

    if used == 0:
        pi = np.ones((k,), dtype=np.float32) / k
        P = np.full((k, k), 0.02, dtype=np.float32)
        np.fill_diagonal(P, 0.2)
        return SBMParams(k, pi, P)

    pi = pi_acc / max(1e-8, pi_acc.sum())
    P = np.clip(0.5 * ((P_num / (P_den + 1e-8)) + (P_num / (P_den + 1e-8)).T), 0.0, 1.0)
    return SBMParams(k, pi.astype(np.float32), P.astype(np.float32))


def estimate_sbm_mixture_from_train(
    train_pyg: Sequence[Data], k_min: int = 2, k_max: int = 5, seed: int = 0
) -> SBMMixture:
    ks = [
        estimate_k_greedy_modularity(pyg_to_nx_undirected_simple(d), k_min, k_max)
        for d in train_pyg
    ]
    cnt = Counter(ks)
    total = sum(cnt.values())
    pk = {k: cnt.get(k, 0) / max(1, total) for k in range(k_min, k_max + 1)}
    params = {k: estimate_sbm_params_for_fixed_k(train_pyg, k, seed) for k in range(k_min, k_max + 1)}
    return SBMMixture(k_min, k_max, pk, params)


def _connect_sbm_components_preserve_nodes(G: nx.Graph, sbm: SBMParams, rng: np.random.RandomState) -> nx.Graph:
    G = nx.Graph(G).copy()
    G.remove_edges_from(nx.selfloop_edges(G))
    while G.number_of_nodes() > 1 and not nx.is_connected(G):
        comps = [set(c) for c in nx.connected_components(G)]
        best_score, best_pairs = -1.0, []
        for a in range(len(comps)):
            for b in range(a + 1, len(comps)):
                for u in comps[a]:
                    bu = min(max(int(G.nodes[u].get("block", 0)), 0), sbm.k - 1)
                    for v in comps[b]:
                        bv = min(max(int(G.nodes[v].get("block", 0)), 0), sbm.k - 1)
                        score = float(sbm.P[bu, bv])
                        if score > best_score + 1e-15:
                            best_score, best_pairs = score, [(int(u), int(v))]
                        elif abs(score - best_score) <= 1e-15:
                            best_pairs.append((int(u), int(v)))
        if not best_pairs:
            G.add_edge(next(iter(comps[0])), next(iter(comps[1])))
        else:
            G.add_edge(*best_pairs[int(rng.randint(len(best_pairs)))])
    return G


def sample_sbm_noise_graph_for_k(sbm: SBMParams, n: int, seed: Optional[int] = None) -> nx.Graph:
    n = int(n)
    if n <= 0:
        return nx.Graph()
    rng = np.random.RandomState(seed if seed is not None else np.random.randint(0, 10**9))
    sizes = rng.multinomial(n, sbm.pi)
    if n >= sbm.k:
        for _ in range(20):
            zeros = np.where(sizes == 0)[0]
            donors = np.where(sizes > 1)[0]
            if len(zeros) == 0 or len(donors) == 0:
                break
            sizes[int(rng.choice(donors))] -= 1
            sizes[int(rng.choice(zeros))] += 1
    G = nx.stochastic_block_model(list(map(int, sizes)), np.asarray(sbm.P), seed=int(rng.randint(0, 10**9)))
    G = _connect_sbm_components_preserve_nodes(nx.Graph(G), sbm, rng)
    if G.number_of_nodes() != n:
        raise RuntimeError(f"SBM source-size invariant failed: {G.number_of_nodes()} != {n}.")
    return G


def sample_sbm_noise_batch_match_targets(sbm_mix: SBMMixture, graphs_tgt: Sequence[Data]) -> List[Data]:
    out = []
    for g in graphs_tgt:
        n = int(g.num_nodes)
        k_hat = estimate_k_greedy_modularity(pyg_to_nx_undirected_simple(g), sbm_mix.k_min, sbm_mix.k_max)
        out.append(nx_to_pyg_graph(sample_sbm_noise_graph_for_k(sbm_mix.params[k_hat], n=n)))
    return out


def sample_k_from_pk(pk: Dict[int, float]) -> int:
    ks = sorted(pk)
    ps = np.asarray([pk[k] for k in ks], dtype=np.float64)
    ps = ps / ps.sum() if ps.sum() > 0 else np.ones_like(ps) / len(ps)
    return int(np.random.choice(ks, p=ps))


def sample_sbm_noise_batch_for_sizes(sbm_mix: SBMMixture, n_nodes_list: Sequence[int]) -> List[Data]:
    out = []
    for n in n_nodes_list:
        k = sample_k_from_pk(sbm_mix.pk)
        out.append(nx_to_pyg_graph(sample_sbm_noise_graph_for_k(sbm_mix.params[k], int(n))))
    return out