from typing import Tuple, List, Dict
import numpy as np
import torch
import ot
from torch_geometric.data import Data

from fm_graphex.models.encoder import GINEncoder
from fm_graphex.config import DEVICE
import pdb


def pairwise_sq_dist_matrix(X: np.ndarray) -> np.ndarray:
    """
    Squared Euclidean distance matrix for a single point set.

    X : (n, d)
    returns M : (n, n) with M[i,j] = ||X[i] - X[j]||^2
    """
    X_sq = np.sum(X**2, axis=1, keepdims=True) # (n, 1)
    cross = X @ X.T # (n, n)
    M = X_sq + X_sq.T - 2 * cross # broadcasting
    M = np.maximum(M, 0.0).astype(np.float32) # numerical safety
    return M


def feature_distance_matrix(X0: np.ndarray, X1: np.ndarray) -> np.ndarray:
    """
    Squared Euclidean distance between two point sets X0, X1.

    X0 : (n0, d)
    X1 : (n1, d)
    returns M : (n0, n1) with M[i,j] = ||X0[i] - X1[j]||^2
    """
    X0_sq = np.sum(X0**2, axis=1, keepdims=True) # (n0, 1)
    X1_sq = np.sum(X1**2, axis=1, keepdims=True).T # (1, n1)
    cross = X0 @ X1.T # (n0, n1)
    M = X0_sq + X1_sq - 2 * cross
    M = np.maximum(M, 0.0).astype(np.float32)
    return M


# FGW distance between two PyG graphs
@torch.no_grad()
def fgw_distance_and_coupling_pyg(
    encoder: GINEncoder,
    data_src: Data,
    data_tgt: Data,
    alpha: float = 0.5,
) -> Tuple[float, np.ndarray]:
    """
    Fused Gromov–Wasserstein distance between two PyG graphs using:

      - encoder(node features, edge_index, edge_attr) as latent node embeddings
      - original node features as "appearance" features

    Returns
    -------
    dist : float (FGW distance).
    T    : (n_src, n_tgt) optimal coupling matrix.
    """
    # Move to device
    data_src = data_src.to(DEVICE)
    data_tgt = data_tgt.to(DEVICE)

    # Latent embeddings via encoder
    z_src = encoder(data_src.x, data_src.edge_index, data_src.edge_attr)
    z_tgt = encoder(data_tgt.x, data_tgt.edge_index, data_tgt.edge_attr)

    # Use original node features for appearance
    x_src = data_src.x
    x_tgt = data_tgt.x

    Z0 = z_src.detach().cpu().numpy()
    Z1 = z_tgt.detach().cpu().numpy()
    X0 = x_src.detach().cpu().numpy()
    X1 = x_tgt.detach().cpu().numpy()

    # Intra-graph distances (latent)
    C0 = pairwise_sq_dist_matrix(Z0)
    C1 = pairwise_sq_dist_matrix(Z1)

    # Cross-graph feature cost (appearance)
    M = feature_distance_matrix(X0, X1)

    n0 = X0.shape[0]
    n1 = X1.shape[0]
    p = np.ones((n0,), dtype=np.float32) / n0
    q = np.ones((n1,), dtype=np.float32) / n1

    T, log = ot.gromov.fused_gromov_wasserstein(
        M,
        C0,
        C1,
        p=p,
        q=q,
        alpha=alpha,
        loss_fun="square_loss",
        log=True,
    )
    dist = float(log["fgw_dist"])
    return dist, T


# Batched FGW distance matrix
def batch_fgw_distance_matrix(
    encoder: GINEncoder,
    noise_graphs: List[Data],
    target_graphs: List[Data],
    alpha: float = 0.5,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], np.ndarray]]:
    """
    Compute the FGW distance matrix between:
      - noise_graphs: list of graphex noise graphs
      - target_graphs: list of QM9/ZINC graphs

    Returns
    -------
    D      : (Bn, Bt) matrix with FGW distances
    T_dict : dict[(i,j) -> coupling matrix T_ij]
    """
    Bn = len(noise_graphs)
    Bt = len(target_graphs)
    D = np.zeros((Bn, Bt), dtype=np.float32)
    T_dict: Dict[Tuple[int, int], np.ndarray] = {}

    for i, g_src in enumerate(noise_graphs):
        for j, g_tgt in enumerate(target_graphs):
            dist, T = fgw_distance_and_coupling_pyg(
                encoder=encoder,
                data_src=g_src,
                data_tgt=g_tgt,
                alpha=alpha,
            )
            D[i, j] = dist
            T_dict[(i, j)] = T
    return D, T_dict