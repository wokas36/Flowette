from __future__ import annotations

from typing import Dict, List, Literal, Optional, Sequence, Tuple
import copy
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import fm_graphex.config as cfg
from fm_graphex.data.graphex_noise import sample_graphex_noise_graph_batch
from fm_graphex.flow_matching.dense_graph import build_dense_type_graph
from fm_graphex.models.encoder import GINEncoder
from fm_graphex.models.velocity_field import VelocityFieldGNNTransformer
from fm_graphex.ot.fgw import batch_fgw_distance_matrix


PairingStrategy = Literal["online", "fixed_pool", "periodic"]


class PairingProtocolStats:
    def __init__(self):
        self.fgw_hungarian_time_s = 0.0
        self.model_time_s = 0.0
        self.num_pairing_calls = 0

    def as_dict(self) -> Dict[str, float]:
        total = self.fgw_hungarian_time_s + self.model_time_s
        return {
            "fgw_hungarian_time_s": self.fgw_hungarian_time_s,
            "model_time_s": self.model_time_s,
            "num_pairing_calls": self.num_pairing_calls,
            "pairing_overhead_pct": 100.0 * self.fgw_hungarian_time_s / total if total else 0.0,
        }


def _equal_size_assignment(
    D: np.ndarray,
    noise_graphs: Sequence[Data],
    target_graphs: Sequence[Data],
) -> Tuple[np.ndarray, np.ndarray]:
    """Hungarian assignment constrained to equal node counts.

    The source node-count multiset is generated directly from the targets, so a
    complete equal-size assignment must exist. Without this mask, Hungarian can
    cross-match different sizes and the old loss code silently skips those
    pairs, wasting much of a diverse GuacaMol/MOSES minibatch.
    """
    cost = np.asarray(D, dtype=np.float64).copy()
    for i, g0 in enumerate(noise_graphs):
        for j, g1 in enumerate(target_graphs):
            if int(g0.num_nodes) != int(g1.num_nodes):
                cost[i, j] = np.inf
    rows, cols = linear_sum_assignment(cost)
    if not np.isfinite(cost[rows, cols]).all():
        src_sizes = [int(g.num_nodes) for g in noise_graphs]
        tgt_sizes = [int(g.num_nodes) for g in target_graphs]
        raise RuntimeError(
            "No complete equal-size FGW assignment exists. "
            f"source sizes={src_sizes}, target sizes={tgt_sizes}."
        )
    return rows, cols


class PairingCache:
    def __init__(self):
        self._cache: Dict[int, Tuple[Data, np.ndarray]] = {}
        self.last_refresh_epoch = -1

    def get(self, target_idx: int) -> Optional[Tuple[Data, np.ndarray]]:
        return self._cache.get(target_idx)

    def refresh(
        self,
        encoder: GINEncoder,
        train_dataset: Sequence[Data],
        cycle_counts: List[dict],
        graphon: np.ndarray,
        atom_prob: np.ndarray,
        atom_z: np.ndarray,
        bond_prob: np.ndarray,
        formal_charge_prob: Optional[np.ndarray],
        max_atoms: int,
        batch_size: int,
        alpha_fgw: float,
        epoch: int,
        stats: Optional[PairingProtocolStats] = None,
    ) -> None:
        self._cache.clear()
        n_total = len(train_dataset)
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            idxs = list(range(start, end))
            targets = [train_dataset[i] for i in idxs]
            sizes = [int(g.num_nodes) for g in targets]
            cycles = [cycle_counts[i] if i < len(cycle_counts) else {} for i in idxs]

            t0 = time.perf_counter()
            noise, _ = sample_graphex_noise_graph_batch(
                graphon=graphon,
                atom_prob=atom_prob,
                atom_z=atom_z,
                bond_prob=bond_prob,
                formal_charge_prob=formal_charge_prob,
                max_atoms=max_atoms,
                cycle_counts=cycles,
                n_nodes_list=sizes,
                return_batch=False,
            )
            D, T_dict = batch_fgw_distance_matrix(encoder, noise, targets, alpha=alpha_fgw)
            rows, cols = _equal_size_assignment(D, noise, targets)
            if stats is not None:
                stats.fgw_hungarian_time_s += time.perf_counter() - t0
                stats.num_pairing_calls += 1

            for r, c in zip(rows, cols):
                r, c = int(r), int(c)
                self._cache[idxs[c]] = (noise[r], T_dict[(r, c)])
        self.last_refresh_epoch = epoch


def _zero_like(X: torch.Tensor) -> torch.Tensor:
    return torch.zeros((), device=X.device, dtype=X.dtype)


def _formal_charge_indices(X: torch.Tensor) -> Optional[torch.Tensor]:
    if not cfg.use_formal_charge():
        return None
    q0 = cfg.NUM_ATOM_TYPES
    qdim = len(cfg.FORMAL_CHARGES)
    if X.size(-1) < q0 + qdim:
        return torch.full((X.size(0),), cfg.NEUTRAL_CHARGE_IDX, device=X.device, dtype=torch.long)
    return X[:, q0:q0 + qdim].argmax(dim=-1)


def _atom_valence_limits(atom_idx: torch.Tensor, charge_idx: Optional[torch.Tensor]) -> torch.Tensor:
    table = cfg.MAX_VALENCE_PER_TYPE.to(atom_idx.device)
    vals = table[atom_idx.clamp(max=len(table) - 1)].float().clone()
    if charge_idx is None:
        return vals
    for i, (ai, qi) in enumerate(zip(atom_idx.tolist(), charge_idx.tolist())):
        sym = cfg.ATOM_TYPES[min(int(ai), len(cfg.ATOM_TYPES) - 1)]
        q = cfg.formal_charge_from_index(int(qi))
        if sym == "N" and q == 1:
            vals[i] = max(float(vals[i]), 4.0)
        elif sym in {"S", "Se"}:
            vals[i] = max(float(vals[i]), 6.0)
        elif sym == "P":
            vals[i] = max(float(vals[i]), 5.0)
    return vals


def soft_valence_penalty(
    A: torch.Tensor,
    X_logits: torch.Tensor,
    E_logits: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    if A.numel() == 0 or weight == 0:
        return _zero_like(X_logits)
    A_w = torch.clamp(0.5 * (A + A.T), 0.0, 1.0)
    atom_idx = X_logits[:, :cfg.NUM_ATOM_TYPES].argmax(dim=-1)
    max_val = _atom_valence_limits(atom_idx, _formal_charge_indices(X_logits))
    p_bond = F.softmax(E_logits[..., :cfg.NUM_BOND_TYPES], dim=-1)
    orders = cfg.BOND_ORDERS.to(E_logits.device).view(1, 1, -1)
    exp_order = (p_bond * orders).sum(dim=-1)
    exp_valence = (A_w * exp_order).sum(dim=-1)
    return weight * torch.relu(exp_valence - max_val).mean()


def atom_hist_loss(X_hat: torch.Tensor, X_true: torch.Tensor, weight: float) -> torch.Tensor:
    if weight == 0 or X_hat.numel() == 0:
        return _zero_like(X_hat)
    pred = F.softmax(X_hat[:, :cfg.NUM_ATOM_TYPES], dim=-1).mean(dim=0)
    target = X_true[:, :cfg.NUM_ATOM_TYPES].mean(dim=0)
    target = target / target.sum().clamp_min(1e-8)
    return weight * F.mse_loss(pred, target)


def soft_align_target_with_fgw(
    A1: torch.Tensor,
    X1: torch.Tensor,
    E1: torch.Tensor,
    T_ij: np.ndarray,
    eps: float = 1e-12,
    remove_self_loops: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    T = torch.as_tensor(T_ij, dtype=X1.dtype, device=X1.device)
    if T.ndim != 2:
        raise ValueError(f"FGW coupling must be 2D, got {T.shape}.")
    n_src, n_tgt = T.shape
    if X1.shape[0] != n_tgt or A1.shape != (n_tgt, n_tgt) or E1.shape[:2] != (n_tgt, n_tgt):
        raise ValueError("Target tensors do not match the FGW coupling dimensions.")
    if not torch.isfinite(T).all():
        raise ValueError("FGW coupling contains non-finite values.")

    R = T / T.sum(dim=1, keepdim=True).clamp_min(eps)
    X_soft = R @ X1
    A_soft = R @ A1 @ R.T
    E_soft = torch.einsum("aj,jkc,bk->abc", R, E1, R)
    A_soft = 0.5 * (A_soft + A_soft.T)
    E_soft = 0.5 * (E_soft + E_soft.transpose(0, 1))
    if remove_self_loops:
        diag = torch.arange(n_src, device=A_soft.device)
        A_soft[diag, diag] = 0.0
        E_soft[diag, diag, :] = 0.0
    return A_soft, X_soft, E_soft, R


def fm_batch_loss(
    vnet: VelocityFieldGNNTransformer,
    encoder: GINEncoder,
    graphs_tgt: List[Data],
    cycle_counts: List[dict],
    graphon: np.ndarray,
    atom_prob: np.ndarray,
    atom_z: np.ndarray,
    bond_prob: np.ndarray,
    formal_charge_prob: Optional[np.ndarray],
    max_atoms: int,
    lambda_x: float,
    lambda_e: float,
    alpha_fgw: float = 0.5,
    beta_end: float = 1.0,
    beta_valence: float = 0.5,
    beta_atom_hist: float = 0.2,
    precomputed_pairs: Optional[List[Tuple[Data, Data, np.ndarray]]] = None,
    stats: Optional[PairingProtocolStats] = None,
) -> Optional[torch.Tensor]:
    """Compute the same Flowette objective for every dataset.

    The objective is

        L = L_velocity + beta_end * L_endpoint
            + beta_valence * L_valence
            + beta_atom_hist * L_atom_hist.

    For GuacaMol and MOSES, the formal-charge one-hot channels remain part
    of X, so they are supervised by the node velocity and endpoint losses.
    """
    if not graphs_tgt:
        return None

    if precomputed_pairs is not None:
        if len(precomputed_pairs) != len(graphs_tgt):
            raise ValueError("precomputed_pairs must align 1:1 with graphs_tgt.")
        matched_pairs = precomputed_pairs
    else:
        sizes = [int(g.num_nodes) for g in graphs_tgt]
        t0 = time.perf_counter()
        noise, _ = sample_graphex_noise_graph_batch(
            graphon=graphon,
            atom_prob=atom_prob,
            atom_z=atom_z,
            bond_prob=bond_prob,
            formal_charge_prob=formal_charge_prob,
            max_atoms=max_atoms,
            cycle_counts=cycle_counts,
            n_nodes_list=sizes,
            return_batch=False,
        )
        D, T_dict = batch_fgw_distance_matrix(encoder, noise, graphs_tgt, alpha=alpha_fgw)
        rows, cols = _equal_size_assignment(D, noise, graphs_tgt)
        matched_pairs = [
            (noise[int(i)], graphs_tgt[int(j)], T_dict[(int(i), int(j))])
            for i, j in zip(rows, cols)
        ]
        if stats is not None:
            stats.fgw_hungarian_time_s += time.perf_counter() - t0
            stats.num_pairing_calls += 1

    t_model = time.perf_counter()
    total = None
    used = 0
    for g0, g1, T_ij in matched_pairs:
        n0, n1 = int(g0.num_nodes), int(g1.num_nodes)
        if n0 <= 1 or n1 <= 1 or n0 > max_atoms or n1 > max_atoms:
            continue
        if n0 != n1:
            raise RuntimeError(f"Equal-size pairing invariant violated: {n0} vs {n1}.")

        A0, X0, E0 = build_dense_type_graph(g0, n0, cfg.NUM_BOND_TYPES)
        A1, X1, E1 = build_dense_type_graph(g1, n1, cfg.NUM_BOND_TYPES)
        if X0.size(1) != cfg.node_feat_dim() or X1.size(1) != cfg.node_feat_dim():
            raise RuntimeError(
                f"Node feature mismatch: source={X0.size(1)}, target={X1.size(1)}, "
                f"expected={cfg.node_feat_dim()} for {cfg.get_spec().name}."
            )

        A1, X1, E1, _ = soft_align_target_with_fgw(A1, X1, E1, T_ij)
        t = torch.rand(1, device=cfg.DEVICE)
        A_t = (1 - t) * A0 + t * A1
        X_t = (1 - t) * X0 + t * X1
        E_t = (1 - t) * E0 + t * E1
        target_A, target_X, target_E = A1 - A0, X1 - X0, E1 - E0

        vA, vX, vE = vnet(A_t, X_t, E_t, t)
        loss_vel = (
            F.mse_loss(vA, target_A)
            + lambda_x * F.mse_loss(vX, target_X)
            + lambda_e * F.mse_loss(vE, target_E)
        )
        A_hat = A_t + (1 - t) * vA
        X_hat = X_t + (1 - t) * vX
        E_hat = E_t + (1 - t) * vE
        loss_end = (
            F.mse_loss(A_hat, A1)
            + lambda_x * F.mse_loss(X_hat, X1)
            + lambda_e * F.mse_loss(E_hat, E1)
        )

        # Identical objective for QM9, ZINC, GuacaMol and MOSES.
        loss_valence = soft_valence_penalty(
            A_hat, X_hat, E_hat, weight=beta_valence
        )
        loss_atom_hist = atom_hist_loss(
            X_hat, X1, weight=beta_atom_hist
        )
        loss = (
            loss_vel
            + beta_end * loss_end
            + loss_valence
            + loss_atom_hist
        )

        total = loss if total is None else total + loss
        used += 1

    if stats is not None:
        stats.model_time_s += time.perf_counter() - t_model
    return None if used == 0 else total / used


def train_flow_matching_graphex(
    encoder: GINEncoder,
    train_dataset: Sequence[Data],
    val_dataset: Optional[Sequence[Data]],
    cycle_counts: List[dict],
    graphon: np.ndarray,
    atom_prob: np.ndarray,
    atom_z: np.ndarray,
    bond_prob: np.ndarray,
    formal_charge_prob: Optional[np.ndarray] = None,
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
    pairing_strategy: PairingStrategy = "online",
    pairing_refresh_every: int = 1,
    verbose_pairing_stats: bool = True,
) -> VelocityFieldGNNTransformer:
    if pairing_strategy not in {"online", "fixed_pool", "periodic"}:
        raise ValueError(f"Unknown pairing_strategy={pairing_strategy!r}.")
    use_cache = pairing_strategy in {"fixed_pool", "periodic"}

    indexed_train: List[Data] = []
    for i, graph in enumerate(train_dataset):
        item = copy.copy(graph)
        item.cache_idx = torch.tensor([i], dtype=torch.long)
        indexed_train.append(item)

    train_loader = DataLoader(indexed_train, batch_size=batch_size, shuffle=True)
    val_loader = (
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        if val_dataset is not None and len(val_dataset) else None
    )
    cache = PairingCache() if use_cache else None

    vnet = VelocityFieldGNNTransformer(
        node_feat_dim=cfg.node_feat_dim(),
        edge_feat_dim=cfg.NUM_BOND_TYPES,
        hidden_dim=fm_hidden_dim,
        num_layers=fm_layers,
    ).to(cfg.DEVICE)
    opt = torch.optim.AdamW(vnet.parameters(), lr=lr)

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    max_atoms = int(graphon.shape[0])
    best_loss = float("inf")
    best_state = None

    print(
        "[FM objective] shared across all datasets: "
        "velocity + endpoint + charge-aware valence + atom histogram "
        f"(beta_end={beta_end}, beta_valence={beta_valence}, "
        f"beta_atom_hist={beta_atom_hist})"
    )

    for epoch in range(1, num_epochs + 1):
        stats = PairingProtocolStats()
        if use_cache:
            refresh = (
                pairing_strategy == "fixed_pool" and cache.last_refresh_epoch < 0
            ) or (
                pairing_strategy == "periodic"
                and (epoch - 1) % max(1, pairing_refresh_every) == 0
            )
            if refresh:
                cache.refresh(
                    encoder=encoder,
                    train_dataset=indexed_train,
                    cycle_counts=cycle_counts,
                    graphon=graphon,
                    atom_prob=atom_prob,
                    atom_z=atom_z,
                    bond_prob=bond_prob,
                    formal_charge_prob=formal_charge_prob,
                    max_atoms=max_atoms,
                    batch_size=batch_size,
                    alpha_fgw=alpha_fgw,
                    epoch=epoch,
                    stats=stats,
                )

        vnet.train()
        train_sum = 0.0; train_used = 0
        for batch in train_loader:
            targets = batch.to_data_list()
            idxs = [int(g.cache_idx.item()) for g in targets]
            precomputed = None
            if use_cache:
                cached = [cache.get(i) for i in idxs]
                if all(item is not None for item in cached):
                    precomputed = [
                        (noise_g, target_g, coupling)
                        for target_g, (noise_g, coupling) in zip(targets, cached)
                    ]
            cycles = [cycle_counts[i] if i < len(cycle_counts) else {} for i in idxs]
            loss = fm_batch_loss(
                vnet=vnet,
                encoder=encoder,
                graphs_tgt=targets,
                cycle_counts=cycles,
                graphon=graphon,
                atom_prob=atom_prob,
                atom_z=atom_z,
                bond_prob=bond_prob,
                formal_charge_prob=formal_charge_prob,
                max_atoms=max_atoms,
                lambda_x=lambda_x,
                lambda_e=lambda_e,
                alpha_fgw=alpha_fgw,
                beta_end=beta_end,
                beta_valence=beta_valence,
                beta_atom_hist=beta_atom_hist,
                precomputed_pairs=precomputed,
                stats=stats,
            )
            if loss is None:
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(vnet.parameters(), grad_clip)
            opt.step()
            train_sum += float(loss.item()); train_used += 1
        train_loss = train_sum / max(train_used, 1)

        criterion = train_loss
        val_text = "N/A"
        if val_loader is not None:
            vnet.eval()
            val_sum = 0.0; val_used = 0
            with torch.no_grad():
                offset = 0
                for batch in val_loader:
                    targets = batch.to_data_list()
                    cycles = [cycle_counts[(offset + i) % max(len(cycle_counts), 1)] if cycle_counts else {}
                              for i in range(len(targets))]
                    offset += len(targets)
                    loss = fm_batch_loss(
                        vnet=vnet,
                        encoder=encoder,
                        graphs_tgt=targets,
                        cycle_counts=cycles,
                        graphon=graphon,
                        atom_prob=atom_prob,
                        atom_z=atom_z,
                        bond_prob=bond_prob,
                        formal_charge_prob=formal_charge_prob,
                        max_atoms=max_atoms,
                        lambda_x=lambda_x,
                        lambda_e=lambda_e,
                        alpha_fgw=alpha_fgw,
                        beta_end=beta_end,
                        beta_valence=beta_valence,
                        beta_atom_hist=beta_atom_hist,
                    )
                    if loss is not None:
                        val_sum += float(loss.item()); val_used += 1
            criterion = val_sum / max(val_used, 1)
            val_text = f"{criterion:.4f}"

        if verbose_pairing_stats:
            s = stats.as_dict()
            print(
                f"[FM][{pairing_strategy}] epoch={epoch}/{num_epochs} "
                f"train={train_loss:.4f} val={val_text} | "
                f"FGW={s['fgw_hungarian_time_s']:.2f}s model={s['model_time_s']:.2f}s "
                f"overhead={s['pairing_overhead_pct']:.1f}% calls={s['num_pairing_calls']}"
            )
        else:
            print(f"[FM] epoch={epoch}/{num_epochs} train={train_loss:.4f} val={val_text}")

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
    print(
        f"[FM] best={best_loss:.4f} task={cfg.get_spec().name} "
        f"node_dim={cfg.node_feat_dim()} feature_version={cfg.FEATURE_VERSION}"
    )
    return vnet