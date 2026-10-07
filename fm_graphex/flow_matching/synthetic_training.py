from __future__ import annotations

import copy
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import fm_graphex.config as cfg
from fm_graphex.flow_matching.dense_graph import build_dense_type_graph
from fm_graphex.flow_matching.training import _equal_size_assignment, soft_align_target_with_fgw
from fm_graphex.models.encoder import DenseGINEncoder
from fm_graphex.models.velocity_field import VelocityFieldGNNTransformer
from fm_graphex.ot.fgw import batch_fgw_distance_matrix
from fm_graphex.data.synthetic import pyg_to_dense_adj
from fm_graphex.data.synthetic_graphex_noise import (
    SBMMixture,
    sample_graphette_noise_batch,
    sample_sbm_noise_batch_match_targets,
)



def pretrain_encoder_reconstruct_adj(
    encoder: DenseGINEncoder,
    train_dataset: Sequence[Data],
    num_epochs: int = 3,
    lr: float = 1e-3,
    max_graphs: int = 2000,
) -> DenseGINEncoder:
    """Unsupervised adjacency-reconstruction pretraining used by FGW."""
    encoder = encoder.to(cfg.DEVICE)
    opt = torch.optim.AdamW(encoder.parameters(), lr=lr)
    subset = list(train_dataset[:max_graphs])

    for ep in range(1, int(num_epochs) + 1):
        encoder.train()
        losses = []
        for d in subset:
            n = int(d.num_nodes)
            if n < 2:
                continue
            A = pyg_to_dense_adj(d, n_nodes=n)
            X = d.x[:n].to(cfg.DEVICE).float()
            z = encoder.forward_dense(A, X)
            logits = z @ z.T
            target = (A > 0.5).float()
            mask = ~torch.eye(n, dtype=torch.bool, device=cfg.DEVICE)
            loss = F.binary_cross_entropy_with_logits(logits[mask], target[mask])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        mean_loss = float(np.mean(losses)) if losses else 0.0
        print(f"[EncPretrain] epoch {ep}/{num_epochs} | loss={mean_loss:.4f}")

    encoder.eval()
    return encoder


def _sample_training_sources(
    task: str,
    graphs_tgt: Sequence[Data],
    graphon: np.ndarray,
    ring_sizes_pool: Optional[List[Dict[int, int]]],
    remove_ring: bool,
    sbm_mix: Optional[SBMMixture],
) -> List[Data]:
    task = str(task).lower()
    if task == "sbm":
        if sbm_mix is None:
            raise ValueError("SBM training requires sbm_mix.")
        return sample_sbm_noise_batch_match_targets(sbm_mix, graphs_tgt)

    sizes = [int(g.num_nodes) for g in graphs_tgt]
    return sample_graphette_noise_batch(
        graphon=graphon,
        n_nodes_list=sizes,
        ring_sizes_pool=ring_sizes_pool,
        remove_ring=remove_ring,
        task=task,
    )


def fm_batch_loss(
    vnet: VelocityFieldGNNTransformer,
    encoder: DenseGINEncoder,
    graphs_tgt: List[Data],
    graphon: np.ndarray,
    ring_sizes_pool: Optional[List[Dict[int, int]]],
    max_nodes: int,
    lambda_x: float,
    lambda_e: float,
    alpha_fgw: float = 0.5,
    beta_end: float = 1.0,
    remove_ring: bool = False,
    task: str = "ego",
    sbm_mix: Optional[SBMMixture] = None,
) -> Optional[torch.Tensor]:
    """Synthetic Flowette objective: velocity MSE + endpoint consistency."""
    if not graphs_tgt:
        return None

    noise_graphs = _sample_training_sources(
        task=task,
        graphs_tgt=graphs_tgt,
        graphon=graphon,
        ring_sizes_pool=ring_sizes_pool,
        remove_ring=remove_ring,
        sbm_mix=sbm_mix,
    )

    target_sizes = [int(g.num_nodes) for g in graphs_tgt]
    source_sizes = [int(g.num_nodes) for g in noise_graphs]
    if sorted(source_sizes) != sorted(target_sizes):
        raise RuntimeError(
            f"Source/target node-count invariant failed: source={source_sizes}, target={target_sizes}."
        )

    D, T_dict = batch_fgw_distance_matrix(
        encoder=encoder,
        noise_graphs=noise_graphs,
        target_graphs=graphs_tgt,
        alpha=alpha_fgw,
    )
    rows, cols = _equal_size_assignment(D, noise_graphs, graphs_tgt)

    total = None
    used = 0
    for i, j in zip(rows, cols):
        i, j = int(i), int(j)
        g0, g1 = noise_graphs[i], graphs_tgt[j]
        n0, n1 = int(g0.num_nodes), int(g1.num_nodes)
        if n0 <= 1 or n1 <= 1:
            continue
        if n0 != n1:
            raise RuntimeError(f"Equal-size pairing invariant violated: {n0} vs {n1}.")
        if n0 > int(max_nodes):
            raise RuntimeError(f"Graph has {n0} nodes but max_nodes={max_nodes}.")

        A0, X0, E0 = build_dense_type_graph(g0, num_nodes=n0, edge_feat_dim=1)
        A1, X1, E1 = build_dense_type_graph(g1, num_nodes=n1, edge_feat_dim=1)

        # Shared soft FGW transport from the molecular data.
        A1, X1, E1, _ = soft_align_target_with_fgw(A1, X1, E1, T_dict[(i, j)])

        t = torch.rand(1, device=cfg.DEVICE)
        A_t = (1.0 - t) * A0 + t * A1
        X_t = (1.0 - t) * X0 + t * X1
        E_t = (1.0 - t) * E0 + t * E1

        target_A, target_X, target_E = A1 - A0, X1 - X0, E1 - E0
        vA, vX, vE = vnet(A_t, X_t, E_t, t)
        loss_vel = (
            F.mse_loss(vA, target_A)
            + float(lambda_x) * F.mse_loss(vX, target_X)
            + float(lambda_e) * F.mse_loss(vE, target_E)
        )

        A_hat = A_t + (1.0 - t) * vA
        X_hat = X_t + (1.0 - t) * vX
        E_hat = E_t + (1.0 - t) * vE
        loss_end = (
            F.mse_loss(A_hat, A1)
            + float(lambda_x) * F.mse_loss(X_hat, X1)
            + float(lambda_e) * F.mse_loss(E_hat, E1)
        )
        loss = loss_vel + float(beta_end) * loss_end
        total = loss if total is None else total + loss
        used += 1

    return None if used == 0 else total / used


def train_flow_matching_synthetic(
    encoder: DenseGINEncoder,
    train_dataset: Sequence[Data],
    val_dataset: Optional[Sequence[Data]],
    graphon: np.ndarray,
    ring_sizes_pool: Optional[List[Dict[int, int]]],
    fm_hidden_dim: int = 128,
    fm_layers: int = 3,
    batch_size: int = 32,
    num_epochs: int = 10,
    lr: float = 1e-3,
    lambda_x: float = 0.5,
    lambda_e: float = 0.5,
    alpha_fgw: float = 0.5,
    grad_clip: Optional[float] = 1.0,
    ckpt_path: Optional[str] = None,
    beta_end: float = 1.0,
    remove_ring: bool = False,
    task: str = "ego",
    sbm_mix: Optional[SBMMixture] = None,
) -> VelocityFieldGNNTransformer:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None

    node_feat_dim = int(train_dataset[0].x.size(1))
    vnet = VelocityFieldGNNTransformer(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=1,
        hidden_dim=fm_hidden_dim,
        num_layers=fm_layers,
    ).to(cfg.DEVICE)
    opt = torch.optim.AdamW(vnet.parameters(), lr=lr)

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    max_nodes = max(int(g.num_nodes) for g in train_dataset)
    best_loss, best_state = float("inf"), None

    for epoch in range(1, int(num_epochs) + 1):
        vnet.train()
        train_sum = 0.0
        train_used = 0
        for batch in train_loader:
            loss = fm_batch_loss(
                vnet=vnet,
                encoder=encoder,
                graphs_tgt=batch.to_data_list(),
                graphon=graphon,
                ring_sizes_pool=ring_sizes_pool,
                max_nodes=max_nodes,
                lambda_x=lambda_x,
                lambda_e=lambda_e,
                alpha_fgw=alpha_fgw,
                beta_end=beta_end,
                remove_ring=remove_ring,
                task=task,
                sbm_mix=sbm_mix,
            )
            if loss is None:
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(vnet.parameters(), grad_clip)
            opt.step()
            train_sum += float(loss.item())
            train_used += 1
        train_loss = train_sum / max(train_used, 1)

        criterion = train_loss
        val_text = "N/A"
        if val_loader is not None:
            vnet.eval()
            val_sum, val_used = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    loss = fm_batch_loss(
                        vnet=vnet,
                        encoder=encoder,
                        graphs_tgt=batch.to_data_list(),
                        graphon=graphon,
                        ring_sizes_pool=ring_sizes_pool,
                        max_nodes=max_nodes,
                        lambda_x=lambda_x,
                        lambda_e=lambda_e,
                        alpha_fgw=alpha_fgw,
                        beta_end=beta_end,
                        remove_ring=remove_ring,
                        task=task,
                        sbm_mix=sbm_mix,
                    )
                    if loss is not None:
                        val_sum += float(loss.item())
                        val_used += 1
            criterion = val_sum / max(val_used, 1)
            val_text = f"{criterion:.4f}"

        print(f"[FM][{task}] epoch={epoch}/{num_epochs} train={train_loss:.4f} val={val_text}")
        if criterion < best_loss:
            best_loss = criterion
            best_state = copy.deepcopy(vnet.state_dict())
            if ckpt_path:
                directory = os.path.dirname(ckpt_path)
                if directory:
                    os.makedirs(directory, exist_ok=True)
                torch.save(best_state, ckpt_path)

    if best_state is not None:
        vnet.load_state_dict(best_state)
    print(f"[FM][{task}] best={best_loss:.4f}")
    return vnet