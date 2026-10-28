from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from torch_geometric.data import Data

try:
    from eval_helper import degree_stats as _degree_stats
    from eval_helper import clustering_stats as _clustering_stats
    from eval_helper import orbit_stats_all as _orbit_stats_all
    USING_EXTERNAL_EVAL_HELPER = True
except ImportError:
    USING_EXTERNAL_EVAL_HELPER = False
    _degree_stats = _clustering_stats = _orbit_stats_all = None


@dataclass
class ValidityConfig:
    require_connected: bool = True
    min_nodes: int = 2
    disallow_self_loops: bool = True


@dataclass
class EvalConfig:
    degree_max: Optional[int] = None
    clustering_bins: int = 10
    validity: ValidityConfig = field(default_factory=ValidityConfig)
    wl_iters: int = 3



def pyg_to_nx_undirected_simple(data: Data) -> nx.Graph:
    G = nx.Graph()
    n = int(data.num_nodes) if data.num_nodes is not None else 0
    G.add_nodes_from(range(n))
    if data.edge_index is None or data.edge_index.numel() == 0:
        return G
    ei = data.edge_index.detach().cpu().numpy()
    for e in range(ei.shape[1]):
        u = int(ei[0, e]); v = int(ei[1, e])
        if u == v:
            continue
        if 0 <= u < n and 0 <= v < n:
            a, b = (u, v) if u < v else (v, u)
            G.add_edge(a, b)
    return G

def is_valid_graph_nx(G: nx.Graph, cfg: ValidityConfig) -> bool:
    if G.number_of_nodes() < cfg.min_nodes:
        return False
    if cfg.disallow_self_loops and any(u == v for u, v in G.edges()):
        return False
    if cfg.require_connected and G.number_of_nodes() > 0 and not nx.is_connected(G):
        return False
    return True


def wl_hash_graph(G: nx.Graph, iterations: int = 3) -> str:
    return nx.weisfeiler_lehman_graph_hash(G, iterations=iterations)


def compute_vun(
    train_graphs: Sequence[nx.Graph],
    gen_graphs: Sequence[nx.Graph],
    validity_cfg: ValidityConfig,
    wl_iters: int = 3,
) -> Dict[str, float]:
    train_hashes = {
        wl_hash_graph(G, wl_iters)
        for G in train_graphs
        if is_valid_graph_nx(G, validity_cfg)
    }
    valid_graphs = [G for G in gen_graphs if is_valid_graph_nx(G, validity_cfg)]
    validity = len(valid_graphs) / max(1, len(gen_graphs))
    if not valid_graphs:
        return {"validity": validity, "uniqueness": 0.0, "novelty": 0.0, "vun": 0.0}
    hashes = [wl_hash_graph(G, wl_iters) for G in valid_graphs]
    unique_hashes = set(hashes)
    uniqueness = len(unique_hashes) / len(valid_graphs)
    novelty = sum(h not in train_hashes for h in unique_hashes) / max(1, len(unique_hashes))
    return {
        "validity": float(validity),
        "uniqueness": float(uniqueness),
        "novelty": float(novelty),
        "vun": float(validity * uniqueness * novelty),
    }


def _motif_vector(G: nx.Graph) -> np.ndarray:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if n == 0:
        return np.zeros(6, dtype=np.float64)
    deg = np.asarray([d for _, d in G.degree()], dtype=np.int64)
    triangles = sum(nx.triangles(G).values()) // 3
    wedges = max(0, int(np.sum(deg * (deg - 1) // 2) - 3 * triangles))
    nodes = list(G.nodes())
    neigh = {u: set(G.neighbors(u)) for u in nodes}
    four_cycle_sum = 0
    for ai, u in enumerate(nodes):
        for v in nodes[ai + 1:]:
            c = len(neigh[u].intersection(neigh[v]))
            if c >= 2:
                four_cycle_sum += c * (c - 1) // 2
    four_cycles = four_cycle_sum / 2.0
    four_cliques = sum(1 for c in nx.enumerate_all_cliques(G) if len(c) == 4)
    path3 = sum(max(0, G.degree(u)-1) * max(0, G.degree(v)-1) for u, v in G.edges()) / 2.0
    return np.asarray([m, wedges, triangles, four_cycles, four_cliques, path3], dtype=np.float64)


def _degree_vector(G: nx.Graph, max_degree: int) -> np.ndarray:
    h = np.zeros(max_degree + 1, dtype=np.float64)
    for _, d in G.degree():
        h[min(int(d), max_degree)] += 1.0
    return h / max(h.sum(), 1.0)


def _clustering_vector(G: nx.Graph, bins: int = 20) -> np.ndarray:
    vals = list(nx.clustering(G).values())
    h, _ = np.histogram(vals, bins=bins, range=(0.0, 1.0))
    h = h.astype(np.float64)
    return h / max(h.sum(), 1.0)


def _rbf_mmd(v1: Sequence[np.ndarray], v2: Sequence[np.ndarray]) -> float:
    if not v1 or not v2:
        return 0.0
    X = np.asarray(v1, dtype=np.float64)
    Y = np.asarray(v2, dtype=np.float64)
    Z = np.concatenate([X, Y], axis=0)
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    positive = d2[d2 > 0]
    sigma2 = float(np.median(positive)) if positive.size else 1.0
    sigma2 = max(sigma2, 1e-8)
    Kxx = np.exp(-((X[:, None, :] - X[None, :, :]) ** 2).sum(-1) / (2 * sigma2))
    Kyy = np.exp(-((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1) / (2 * sigma2))
    Kxy = np.exp(-((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1) / (2 * sigma2))
    return float(Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean())


def _fallback_stats(ref: Sequence[nx.Graph], pred: Sequence[nx.Graph], kind: str) -> float:
    if kind == "degree":
        max_degree = max([d for G in list(ref)+list(pred) for _, d in G.degree()] or [0])
        return _rbf_mmd([_degree_vector(G, max_degree) for G in ref], [_degree_vector(G, max_degree) for G in pred])
    if kind == "clustering":
        return _rbf_mmd([_clustering_vector(G) for G in ref], [_clustering_vector(G) for G in pred])
    return _rbf_mmd([_motif_vector(G) for G in ref], [_motif_vector(G) for G in pred])


def _metric(ref: Sequence[nx.Graph], pred: Sequence[nx.Graph], kind: str) -> float:
    if USING_EXTERNAL_EVAL_HELPER:
        if kind == "degree":
            return float(_degree_stats(ref, pred, compute_emd=False))
        if kind == "clustering":
            return float(_clustering_stats(ref, pred, compute_emd=False))
        return float(_orbit_stats_all(ref, pred, compute_emd=False))
    return _fallback_stats(ref, pred, kind)


def evaluate_deg_clus_orb_vun(
    train_pyg: Sequence[Data],
    test_pyg: Sequence[Data],
    gen_pyg: Sequence[Data],
    cfg: EvalConfig,
    dataset: str,
) -> Dict[str, float]:
    train_nx = [pyg_to_nx_undirected_simple(d) for d in train_pyg]
    test_nx = [pyg_to_nx_undirected_simple(d) for d in test_pyg]
    gen_nx = [pyg_to_nx_undirected_simple(d) for d in gen_pyg]

    train_valid = [G for G in train_nx if is_valid_graph_nx(G, cfg.validity)]
    test_valid = [G for G in test_nx if is_valid_graph_nx(G, cfg.validity)]
    gen_valid = [G for G in gen_nx if is_valid_graph_nx(G, cfg.validity)]

    deg_mmd = _metric(test_valid, gen_valid, "degree")
    clus_mmd = _metric(test_valid, gen_valid, "clustering")
    orb_mmd = _metric(test_valid, gen_valid, "orbit")
    deg_ref = _metric(test_valid, train_valid, "degree")
    clus_ref = _metric(test_valid, train_valid, "clustering")
    orb_ref = _metric(test_valid, train_valid, "orbit")

    # print("GEN deg mmd:", deg_mmd, "----- TEST deg ref:", deg_ref)
    # print("GEN clus mmd:", clus_mmd, "----- TEST clus ref:", clus_ref)
    # print("GEN orb mmd:", orb_mmd, "----- TEST orb ref:", orb_ref)

    ratios = []
    dataset = str(dataset).lower()
    if dataset in {"sbm", "ego"}:
        for num, den in [(deg_mmd, deg_ref), (clus_mmd, clus_ref), (orb_mmd, orb_ref)]:
            if den > 1e-8:
                ratios.append(num / den)
    elif dataset == "tree" and deg_ref > 1e-8:
        ratios.append(deg_mmd / deg_ref)
    average_ratio = float(sum(ratios) / len(ratios)) if ratios else -1.0

    vun = compute_vun(train_nx, gen_nx, cfg.validity, cfg.wl_iters)
    return {
        "Clus": clus_mmd,
        "Orb": orb_mmd,
        "Validity": vun["validity"],
        "Uniqueness": vun["uniqueness"],
        "Novelty": vun["novelty"],
        "VUN": vun["vun"],
        "n_gen": len(gen_pyg),
        "n_gen_valid": len(gen_valid),
        "n_test_valid": len(test_valid),
        "eval_helper": "external" if USING_EXTERNAL_EVAL_HELPER else "internal_fallback",
    }


def get_layout(G: nx.Graph, layout: str = "spring", seed: int = 0):
    """Choose a layout (spring/kamada/circular/spectral/planar if possible)."""
    if G.number_of_nodes() == 0:
        return {}

    layout = layout.lower()
    if layout == "spring":
        return nx.spring_layout(G, seed=seed, k=None)
    if layout == "kamada":
        return nx.kamada_kawai_layout(G)
    if layout == "circular":
        return nx.circular_layout(G)
    if layout == "spectral":
        return nx.spectral_layout(G)

    # planar (only works if graph is planar & networkx has planar_layout)
    if layout == "planar":
        try:
            is_planar, emb = nx.check_planarity(G)
            if is_planar:
                return nx.planar_layout(G)
        except Exception:
            pass
        # fallback
        return nx.spring_layout(G, seed=seed)

    # fallback
    return nx.spring_layout(G, seed=seed)


def draw_graph(
    G: nx.Graph,
    ax=None,
    title: str = "",
    layout: str = "spring",
    seed: int = 0,
    node_size: int = 80,
    with_labels: bool = False,
):
    """Draw a single graph on an axis."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_title(title)
    ax.axis("off")

    pos = get_layout(G, layout=layout, seed=seed)

    # color nodes by degree (nice quick visual cue)
    deg = dict(G.degree())
    node_color = [deg[i] for i in G.nodes()] if G.number_of_nodes() > 0 else []

    nx.draw_networkx(
        G,
        pos=pos,
        ax=ax,
        node_size=node_size,
        node_color=node_color,
        cmap=plt.cm.viridis,
        with_labels=with_labels,
        linewidths=0.3,
        edgecolors="black",
        width=0.7,
    )
    return ax


def visualize_graph_samples(
    gen_graphs_pyg: List,
    out_dir: str = "images",
    run_name: str = "gen_graphs",
    max_show: int = 36,
    cols: int = 6,
    layout: str = "spring",
    seed: int = 0,
    node_size: int = 60,
    with_labels: bool = False,
) -> str:
    """
    Visualize up to max_show generated graphs in a grid, save as PNG, and show inline.
    Returns the saved path.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Convert to NX (if not already)
    nx_graphs = []
    for g in gen_graphs_pyg[:max_show]:
        if isinstance(g, nx.Graph):
            nx_graphs.append(g)
        else:
            nx_graphs.append(pyg_to_nx_undirected_simple(g))

    n = len(nx_graphs)
    if n == 0:
        print("[Viz] No graphs to visualize.")
        return ""

    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.axis("off")
        if i < n:
            G = nx_graphs[i]
            title = f"#{i} | n={G.number_of_nodes()} m={G.number_of_edges()}"
            draw_graph(
                G,
                ax=ax,
                title=title,
                layout=layout,
                seed=seed + i,
                node_size=node_size,
                with_labels=with_labels,
            )

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{run_name}_grid.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"[Viz] Saved: {out_path}")
    return out_path


def save_individual_graph_images(
    gen_graphs_pyg: List,
    out_dir: str = "images/graphs",
    layout: str = "spring",
    seed: int = 0,
    node_size: int = 80,
    with_labels: bool = False,
    max_save: int = 50,
) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, g in enumerate(gen_graphs_pyg[:max_save]):
        G = g if isinstance(g, nx.Graph) else pyg_to_nx_undirected_simple(g)

        fig, ax = plt.subplots(figsize=(4, 4))
        title = f"#{i} | n={G.number_of_nodes()} m={G.number_of_edges()}"
        draw_graph(G, ax=ax, title=title, layout=layout, seed=seed + i,
                   node_size=node_size, with_labels=with_labels)
        plt.tight_layout()

        path = os.path.join(out_dir, f"graph_{i:04d}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    print(f"[Viz] Saved {len(paths)} individual graphs to: {out_dir}")
    return paths