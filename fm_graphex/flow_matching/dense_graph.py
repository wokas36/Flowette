from typing import Tuple
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from fm_graphex.config import DEVICE

def normalize_adjacency(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Row-normalized adjacency: D^{-1} A
    """
    row_sum = A.sum(dim=-1, keepdim=True)
    row_sum = row_sum + eps
    return A / row_sum


def relu(x):
    """
    Sign passes through one side of the fold.
    Velocity field remains locally directional.
    FM behaves like a reflection, not a projection.
    Folded ReLU preserves signed velocity information.
    """
    return F.relu(x) + F.relu(-x)


def leaky_relu(x):
    """
    Leaky fold ReLU controlled softness & better stability for long integrations steps.
    """
    return F.leaky_relu(x, 0.01) + F.leaky_relu(-x, 0.01)


def normalised_softplus(x):
    """
    Apply a normalized Softplus activation to enforce non-negativity while preserving relative edge strengths in 
    soft adjacency matrices.
    Renormalizing by the maximum value to keep all entries in [0, 1].
    """
    x = F.softplus(x)
    x = x / (x.max() + 1e-6)
    return x


def build_dense_type_graph(data: Data, num_nodes: int, edge_feat_dim: int):
    num_nodes = min(num_nodes, int(data.num_nodes))

    x = data.x[:num_nodes].to(DEVICE)
    X = x.float()

    A = torch.zeros(num_nodes, num_nodes, device=DEVICE)
    E = torch.zeros(num_nodes, num_nodes, edge_feat_dim, device=DEVICE)

    edge_index = data.edge_index.to(DEVICE)
    edge_attr = data.edge_attr.to(DEVICE)

    for e in range(edge_index.size(1)):
        i = int(edge_index[0, e]); j = int(edge_index[1, e])
        if i >= num_nodes or j >= num_nodes:
            continue
        A[i, j] = 1.0
        A[j, i] = 1.0
        E[i, j, :edge_feat_dim] = edge_attr[e, :edge_feat_dim]
        E[j, i, :edge_feat_dim] = edge_attr[e, :edge_feat_dim]

    return A, X, E