from typing import Sequence, List, Tuple, Optional, Dict
import numpy as np
import os
import copy
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from fm_graphex.config import (
    DEVICE,
    NUM_ATOM_TYPES,
    NUM_BOND_TYPES,
)
from fm_graphex.config import node_feat_dim
from fm_graphex.models.encoder import GINEncoder
from fm_graphex.models.velocity_field import VelocityFieldGNNTransformer
from fm_graphex.data.graphex_noise import sample_graphex_noise_graph_batch
from fm_graphex.ot.fgw import batch_fgw_distance_matrix
from fm_graphex.flow_matching.dense_graph import build_dense_type_graph, normalize_adjacency, relu

import pdb


def _upper_mask(N: int, device: torch.device):
    return torch.triu(torch.ones(N, N, dtype=torch.bool, device=device), diagonal=1)


def soft_valence_penalty(
    A: torch.Tensor,
    X_logits: torch.Tensor,
    E_logits: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    from fm_graphex.config import MAX_VALENCE_PER_TYPE, BOND_ORDERS, get_spec

    device = X_logits.device
    N = X_logits.size(0)
    if N == 0:
        return torch.zeros((), device=device)

    A_w = torch.clamp(A, 0.0, 1.0)
    A_w = 0.5 * (A_w + A_w.T)

    num_atom_types = MAX_VALENCE_PER_TYPE.shape[0]
    atom_logits = X_logits[:, :num_atom_types]
    atom_idx = atom_logits.argmax(dim=-1)
    max_val = MAX_VALENCE_PER_TYPE.to(device)[atom_idx]

    task = get_spec().name
    if task != "qm9" and task != "zinc":
        charge_norm = X_logits[:, num_atom_types + 0]
        arom_flag   = X_logits[:, num_atom_types + 1]

        formal_charge = torch.round(torch.clamp(charge_norm, -1.0, 1.0) * 3.0)
        max_val = max_val + formal_charge.float()

        is_arom = arom_flag > 0.5
        max_val = torch.where(is_arom, torch.minimum(max_val, torch.tensor(3.0, device=device)), max_val)

    p_b = F.softmax(E_logits, dim=-1)
    bond_orders = BOND_ORDERS.to(device).view(1, 1, -1)
    exp_order = (p_b * bond_orders).sum(dim=-1)

    exp_val = (A_w * exp_order).sum(dim=-1)
    excess = torch.relu(exp_val - max_val)

    return weight * excess.mean()


def atom_hist_loss(
    X_hat_logits: torch.Tensor, # (N, A)
    X_true: torch.Tensor, # (N, A) one-hot
    weight: float = 1.0
) -> torch.Tensor:
    if X_hat_logits.numel() == 0:
        return torch.zeros((), device=X_hat_logits.device)
    p_hat = F.softmax(X_hat_logits, dim=-1).mean(dim=0)
    p_true = X_true.mean(dim=0)
    return weight * F.mse_loss(p_hat, p_true)


def bond_hist_loss(
    A_hat: torch.Tensor, # (N,N)
    E_hat_logits: torch.Tensor, # (N,N,B)
    E_true: torch.Tensor, # (N,N,B) one-hot (dense)
    weight: float = 1.0
) -> torch.Tensor:
    device = E_hat_logits.device
    N = E_hat_logits.size(0)
    if N <= 1:
        return torch.zeros((), device=device)

    mask = _upper_mask(N, device)
    A_w = torch.clamp(0.5 * (A_hat + A_hat.T), 0.0, 1.0)  # (N,N)
    p_b = F.softmax(E_hat_logits, dim=-1) # (N,N,B)

    # weight edges by A_w for a soft "edge existence"
    w = A_w[mask].unsqueeze(-1)  # (M,1)
    p_hat = (p_b[mask] * w).sum(dim=0) / (w.sum() + 1e-8) # (B,)

    # true bond marginal over edges where E_true has a bond
    # we treat any nonzero one-hot as a bond entry; use A_true if you want stricter.
    p_true = E_true[mask].mean(dim=0)  # (B,)
    return weight * F.mse_loss(p_hat, p_true)

def fm_batch_loss(
    vnet: VelocityFieldGNNTransformer,
    encoder: GINEncoder,
    graphs_tgt: List[Data],
    cycle_counts: List[dict],
    graphon: np.ndarray,
    atom_prob: np.ndarray,
    atom_z: np.ndarray,
    bond_prob: np.ndarray,
    max_atoms: int,
    lambda_x: float,
    lambda_e: float,
    alpha_fgw: float = 0.5,
    beta_end: float = 1.0, # endpoint consistency strength
    beta_valence: float = 0.5, # soft valence penalty
    beta_atom_hist: float = 0.2, # atom marginal match
    beta_bond_hist: float = 0.2, # bond marginal match
    beta_deg: float = 0.1, # degree stats match
    beta_cycle: float = 0.05, # trace(A^k) ring proxy
) -> Optional[torch.Tensor]:
    """
    Returns a scalar loss tensor (on DEVICE). If no valid pairs, returns None.
    
    FM loss:
      - SIGNED target_A
      - endpoint consistency loss (x1_hat = x_t + (1-t)*v)
      - chemistry regularizers
    """
    if len(graphs_tgt) == 0:
        return None

    # sample matched noise graphs with same node counts
    n_nodes_list = [g.num_nodes for g in graphs_tgt]
    noise_graphs, _ = sample_graphex_noise_graph_batch(
        graphon=graphon,
        atom_prob=atom_prob,
        atom_z=atom_z,
        bond_prob=bond_prob,
        max_atoms=max_atoms,
        cycle_counts=cycle_counts,
        n_nodes_list=n_nodes_list,
        return_batch=False,
    )

    # FGW align noise <-> target
    D, _ = batch_fgw_distance_matrix(
        encoder=encoder,
        noise_graphs=noise_graphs,
        target_graphs=graphs_tgt,
        alpha=alpha_fgw,
    )
    row_ind, col_ind = linear_sum_assignment(D)
    matched_noise = [noise_graphs[i] for i in row_ind]
    matched_tgt   = [graphs_tgt[j] for j in col_ind]

    batch_loss = 0.0
    pc = 0

    for g0, g1 in zip(matched_noise, matched_tgt):
        n0, n1 = g0.num_nodes, g1.num_nodes
        if n0 == 0 or n1 == 0:
            continue
        N = min(n0, n1, max_atoms)
        if N <= 1:
            continue

        # build raw adjacency A in {0,1}
        A0, X0, E0 = build_dense_type_graph(g0, num_nodes=N, edge_feat_dim=NUM_BOND_TYPES)
        A1, X1, E1 = build_dense_type_graph(g1, num_nodes=N, edge_feat_dim=NUM_BOND_TYPES)

        # time sample
        t = torch.rand(1, device=DEVICE) # (1,)
        t_scalar = t.view(1, 1) # broadcast helper

        # linear interpolant
        A_t = (1.0 - t) * A0 + t * A1
        X_t = (1.0 - t) * X0 + t * X1
        E_t = (1.0 - t) * E0 + t * E1

        # SIGNED targets
        target_A = (A1 - A0)
        target_X = (X1 - X0)
        target_E = (E1 - E0)

        vA, vX, vE = vnet(A_t, X_t, E_t, t)

        # velocity matching
        loss_vel = (
            F.mse_loss(vA, target_A)
            + lambda_x * F.mse_loss(vX, target_X)
            + lambda_e * F.mse_loss(vE, target_E)
        )

        # endpoint consistency (rectified flow identity)
        A1_hat = A_t + (1.0 - t) * vA
        X1_hat = X_t + (1.0 - t) * vX
        E1_hat = E_t + (1.0 - t) * vE

        loss_end = (
            F.mse_loss(A1_hat, A1)
            + lambda_x * F.mse_loss(X1_hat, X1)
            + lambda_e * F.mse_loss(E1_hat, E1)
        )

        # chemistry/stat regularizers (apply on endpoint prediction)
        loss_val = soft_valence_penalty(A1_hat, X1_hat, E1_hat, weight=beta_valence)
        loss_ah  = atom_hist_loss(X1_hat, X1, weight=beta_atom_hist)
        loss_bh  = bond_hist_loss(A1_hat, E1_hat, E1, weight=beta_bond_hist)

        loss = loss_vel + beta_end * loss_end + loss_val + loss_ah # loss for QM9, ZINC, Guacamol
        # loss = loss_vel + beta_end * loss_end + loss_ah # Guacamol

        batch_loss = batch_loss + loss
        pc += 1

    if pc == 0:
        return None
    return batch_loss / pc


# FM training: adds val loop + best model selection
def train_flow_matching_graphex(
    encoder: GINEncoder,
    train_dataset: Sequence[Data],
    val_dataset: Optional[Sequence[Data]],
    cycle_counts: List[dict],
    graphon: np.ndarray,
    atom_prob: np.ndarray,
    atom_z: np.ndarray,
    bond_prob: np.ndarray,
    fm_hidden_dim: int = 128,
    fm_layers: int = 3,
    batch_size: int = 16,
    num_epochs: int = 5,
    lr: float = 1e-3,
    lambda_x: float = 1.0,
    lambda_e: float = 1.0,
    alpha_fgw: float = 0.5,
    grad_clip: Optional[float] = 1.0,
    ckpt_path: Optional[str] = None,
    beta_end: float = 1.0,
    beta_valence: float = 0.5,
    beta_atom_hist: float = 0.2,
    beta_bond_hist: float = 0.2,
    beta_deg: float = 0.1,
    beta_cycle: float = 0.05,
):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    vnet = VelocityFieldGNNTransformer(
        node_feat_dim=node_feat_dim(), # NUM_ATOM_TYPES + EXTRA_NODE_FEATS,
        edge_feat_dim=NUM_BOND_TYPES,
        hidden_dim=fm_hidden_dim,
        num_layers=fm_layers,
    ).to(DEVICE)

    opt = torch.optim.AdamW(vnet.parameters(), lr=lr)

    # encoder is only for FGW distances; keep frozen + eval
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    max_atoms = int(graphon.shape[0])

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, num_epochs + 1):
        # Train
        vnet.train()
        train_loss_sum = 0.0
        train_used = 0

        for batch in train_loader:
            graphs_tgt = batch.to_data_list()
            loss = fm_batch_loss(
                vnet=vnet,
                encoder=encoder,
                graphs_tgt=graphs_tgt,
                cycle_counts=cycle_counts,
                graphon=graphon,
                atom_prob=atom_prob,
                atom_z=atom_z,
                bond_prob=bond_prob,
                max_atoms=max_atoms,
                lambda_x=lambda_x,
                lambda_e=lambda_e,
                alpha_fgw=alpha_fgw,
                beta_end=beta_end,
                beta_valence=beta_valence,
                beta_atom_hist=beta_atom_hist,
                beta_bond_hist=beta_bond_hist,
                beta_deg=beta_deg,
                beta_cycle=beta_cycle,
            )
            if loss is None:
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(vnet.parameters(), grad_clip)
            opt.step()

            train_loss_sum += float(loss.item())
            train_used += 1

        train_loss = train_loss_sum / max(1, train_used)

        # Validate
        if val_loader is not None:
            vnet.eval()
            val_loss_sum = 0.0
            val_used = 0
            with torch.no_grad():
                for batch in val_loader:
                    graphs_tgt = batch.to_data_list()
                    loss = fm_batch_loss(
                        vnet=vnet,
                        encoder=encoder,
                        graphs_tgt=graphs_tgt,
                        cycle_counts=cycle_counts,
                        graphon=graphon,
                        atom_prob=atom_prob,
                        atom_z=atom_z,
                        bond_prob=bond_prob,
                        max_atoms=max_atoms,
                        lambda_x=lambda_x,
                        lambda_e=lambda_e,
                        alpha_fgw=alpha_fgw,
                        beta_end=beta_end,
                        beta_valence=beta_valence,
                        beta_atom_hist=beta_atom_hist,
                        beta_bond_hist=beta_bond_hist,
                        beta_deg=beta_deg,
                        beta_cycle=beta_cycle,
                    )
                    if loss is None:
                        continue
                    val_loss_sum += float(loss.item())
                    val_used += 1

            val_loss = val_loss_sum / max(1, val_used)
            print(f"[FM] Epoch {epoch}/{num_epochs} | train={train_loss:.4f} | val={val_loss:.4f}")

            # Best model by validation
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = copy.deepcopy(vnet.state_dict())
                if ckpt_path is not None:
                    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                    torch.save(best_state, ckpt_path)
        else:
            # No val set provided → best by train loss (fallback)
            print(f"[FM] Epoch {epoch}/{num_epochs} | train={train_loss:.4f} | val=N/A")
            if train_loss < best_loss:
                best_loss = train_loss
                best_state = copy.deepcopy(vnet.state_dict())
                if ckpt_path is not None:
                    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                    torch.save(best_state, ckpt_path)

    # restore best weights
    if best_state is not None:
        vnet.load_state_dict(best_state)

    print(f"[FM] Best loss = {best_loss:.4f} (used {'val' if val_loader is not None else 'train'} criterion)")
    return vnet