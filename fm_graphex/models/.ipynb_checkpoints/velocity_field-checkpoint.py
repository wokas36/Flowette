from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class VelocityLayer(nn.Module):
    """
    A simple edge-aware self-attention-like layer over nodes:

      - Node states h (N, H)
      - Edge states g (N, N, H)
      - Adjacency A (N, N) used as a bias in attention scores.

    Produces updated (h', g').
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)

        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self, node_h: torch.Tensor, edge_h: torch.Tensor, A_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        node_h : (N, H)
        edge_h : (N, N, H)
        A_t    : (N, N)
        """
        N, H = node_h.size()
        q = self.q_proj(node_h) # (N, H)
        k = self.k_proj(node_h) # (N, H)

        # attention scores: (N, N)
        scores = (q @ k.T) / (H**0.5)
        scores = scores + A_t # inject topology

        attn = F.softmax(scores, dim=-1) # over j

        # build messages using both node and edge states
        node_i = node_h.unsqueeze(1).expand(N, N, H)
        node_j = node_h.unsqueeze(0).expand(N, N, H)
        edge_input = torch.cat([node_i, node_j, edge_h], dim=-1) # (N, N, 3H)

        m_ij = self.edge_mlp(edge_input) # (N, N, H)

        # aggregate messages per node
        weighted = (attn.unsqueeze(-1) * m_ij).sum(dim=1) # (N, H)
        delta_node = self.node_mlp(weighted)

        new_node_h = node_h + delta_node
        new_edge_h = edge_h + self.edge_update(m_ij)

        return new_node_h, new_edge_h


class VelocityFieldGNNTransformer(nn.Module):
    """
    Transformer-like GNN that predicts velocity fields on:
      - adjacency A_t    : (N, N)
      - node types X_t   : (N, NUM_ATOM_TYPES)
      - bond types E_t   : (N, N, NUM_BOND_TYPES)

    forward(A_t, X_t, E_t, t) -> (A_vel, X_vel, E_vel)
    """

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden_dim = hidden_dim

        # Input projection
        self.node_in = nn.Linear(node_feat_dim, hidden_dim)
        # edge_feat_dim + 1 (we append A_t as a channel)
        self.edge_in = nn.Linear(edge_feat_dim + 1, hidden_dim)

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.layers = nn.ModuleList(
            [VelocityLayer(hidden_dim=hidden_dim) for _ in range(num_layers)]
        )

        # Output projections to velocities
        self.node_out = nn.Linear(hidden_dim, node_feat_dim)
        self.edge_out_E = nn.Linear(hidden_dim, edge_feat_dim)
        self.edge_out_A = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        A_t: torch.Tensor,
        X_t: torch.Tensor,
        E_t: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        A_t : (N, N)
        X_t : (N, node_feat_dim)
        E_t : (N, N, edge_feat_dim)
        t   : scalar tensor or shape (1,)
        """
        device = X_t.device
        N = X_t.size(0)

        if t.dim() == 0:
            t = t.view(1, 1)
        elif t.dim() == 1:
            t = t.view(-1, 1)
        t = t.to(device=device, dtype=X_t.dtype)
        # use same time embedding for all nodes/edges
        t_emb = self.time_mlp(t).view(1, -1) # (1, H)
        
        node_h = self.node_in(X_t) + t_emb # (N, H)
        A_channel = A_t.unsqueeze(-1) # (N, N, 1)
        edge_input = torch.cat([E_t, A_channel], dim=-1)
        edge_h = self.edge_in(edge_input)
        edge_h = edge_h + t_emb.view(1, 1, -1) # (N, N, H)

        for layer in self.layers:
            node_h, edge_h = layer(node_h, edge_h, A_t)

        v_X = self.node_out(node_h) # (N, node_feat_dim)
        v_E = self.edge_out_E(edge_h) # (N, N, edge_feat_dim)
        v_A = self.edge_out_A(edge_h).squeeze(-1) # (N, N)

        # Symmetrize adjacency- and edge-related velocities (undirected graphs)
        v_A = 0.5 * (v_A + v_A.T)
        v_E = 0.5 * (v_E + v_E.transpose(0, 1))

        return v_A, v_X, v_E