from __future__ import annotations

from typing import Optional, Sequence, Tuple
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import rdchem
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv

import fm_graphex.config as cfg


class GINEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        out_dim: int = 64,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.edge_dim = edge_dim

        dim = in_dim
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=edge_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            dim = hidden_dim
        self.proj = nn.Linear(dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = F.relu(bn(conv(h, edge_index, edge_attr)))
        return self.proj(h)


class SimpleGraphDecoder(nn.Module):
    """Auxiliary encoder pretraining decoder.

    ``node_logits`` contains atom logits followed, for GuacaMol/MOSES, by
    three formal-charge logits. Training applies separate categorical losses
    to the two groups, so atom and charge classes never compete in one softmax.
    """

    def __init__(
        self,
        node_dim: int,
        num_atom_types: int,
        num_bond_types: int,
        formal_charge_dim: int = 0,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.num_atom_types = num_atom_types
        self.num_bond_types = num_bond_types
        self.formal_charge_dim = formal_charge_dim
        self.node_output_dim = num_atom_types + formal_charge_dim

        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, self.node_output_dim),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, num_bond_types + 1),
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n = z.size(0)
        node_logits = self.node_mlp(z)
        zi = z.unsqueeze(1).expand(n, n, self.node_dim)
        zj = z.unsqueeze(0).expand(n, n, self.node_dim)
        edge_logits = self.edge_mlp(torch.cat([zi, zj], dim=-1))
        return node_logits, edge_logits

    def decode_molecule(self, z: torch.Tensor, bond_threshold: float = 0.5) -> Optional[Chem.Mol]:
        node_logits, edge_logits = self.forward(z)
        atom_idx = node_logits[:, :self.num_atom_types].argmax(dim=-1)
        charge_idx = None
        if self.formal_charge_dim:
            q0 = self.num_atom_types
            charge_idx = node_logits[:, q0:q0 + self.formal_charge_dim].argmax(dim=-1)

        edge_probs = F.softmax(edge_logits, dim=-1)
        rw = rdchem.RWMol()
        for i, aidx in enumerate(atom_idx.detach().cpu().tolist()):
            atom = rdchem.Atom(cfg.TYPE_TO_SYM.get(int(aidx), "C"))
            if charge_idx is not None:
                q = cfg.formal_charge_from_index(int(charge_idx[i].item()))
                if q:
                    atom.SetFormalCharge(q)
            rw.AddAtom(atom)

        n = z.size(0)
        for i in range(n):
            for j in range(i + 1, n):
                probs = edge_probs[i, j]
                bidx = int(probs[:-1].argmax().item())
                if float(probs[bidx]) <= bond_threshold or float(probs[bidx]) <= float(probs[-1]):
                    continue
                bt = cfg.BOND_TYPES[min(bidx, cfg.NUM_BOND_TYPES - 1)]
                try:
                    rw.AddBond(i, j, bt)
                    if bt == rdchem.BondType.AROMATIC:
                        rw.GetAtomWithIdx(i).SetIsAromatic(True)
                        rw.GetAtomWithIdx(j).SetIsAromatic(True)
                except Exception:
                    continue
        try:
            out = rw.GetMol()
            Chem.SanitizeMol(out)
            return out
        except Exception:
            return None


def atom_targets_from_data(data: Data) -> torch.Tensor:
    x = getattr(data, "x", None)
    if x is not None and x.dim() == 2 and x.size(1) >= cfg.NUM_ATOM_TYPES:
        return x[:, :cfg.NUM_ATOM_TYPES].argmax(dim=-1).long()
    if hasattr(data, "z") and data.z is not None:
        vals = [cfg.Z_TO_TYPE.get(int(z), cfg.Z_TO_TYPE.get(6, 0)) for z in data.z.detach().cpu().tolist()]
        return torch.tensor(vals, dtype=torch.long, device=data.z.device)
    raise ValueError("Cannot build atom targets: need data.x or data.z.")


def charge_targets_from_data(data: Data) -> Optional[torch.Tensor]:
    if not cfg.use_formal_charge():
        return None
    x = getattr(data, "x", None)
    q0 = cfg.NUM_ATOM_TYPES
    qdim = len(cfg.FORMAL_CHARGES)
    if x is not None and x.dim() == 2 and x.size(1) >= q0 + qdim:
        return x[:, q0:q0 + qdim].argmax(dim=-1).long()
    charges = getattr(data, "formal_charge", None)
    if charges is not None:
        vals = [cfg.formal_charge_to_index(int(q)) for q in charges.detach().cpu().tolist()]
        return torch.tensor(vals, dtype=torch.long, device=charges.device)
    return torch.full(
        (int(data.num_nodes),),
        cfg.NEUTRAL_CHARGE_IDX,
        dtype=torch.long,
        device=data.x.device,
    )


def build_edge_targets(n: int, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
    target = torch.full(
        (n, n),
        fill_value=cfg.NUM_BOND_TYPES,
        dtype=torch.long,
        device=edge_index.device,
    )
    seen = set()
    for e in range(edge_index.size(1)):
        i, j = int(edge_index[0, e]), int(edge_index[1, e])
        if i == j:
            continue
        a, b = sorted((i, j))
        if (a, b) in seen:
            continue
        seen.add((a, b))
        bt = int(edge_attr[e, :cfg.NUM_BOND_TYPES].argmax().item())
        target[a, b] = min(bt, cfg.NUM_BOND_TYPES - 1)
    return target


def _load_dataset(task: str, root: str, max_num_graphs: Optional[int]):
    # Import only the selected loader. Besides reducing startup cost, this
    # prevents a QM9/ZINC run from requiring Hugging Face datasets solely
    # because the GuacaMol loader exists in the same project.
    if task == "qm9":
        from fm_graphex.data.qm9 import load_qm9_heavy_dataset
        return load_qm9_heavy_dataset(root=root, max_num_graphs=max_num_graphs)
    if task == "zinc":
        from fm_graphex.data.zinc import load_zinc_dataset
        return load_zinc_dataset(root=root, max_num_graphs=max_num_graphs)
    if task == "guacamol":
        from fm_graphex.data.guacamol import load_guacamol_dataset
        return load_guacamol_dataset(root=root, max_nodes=88, max_num_graphs=max_num_graphs)
    if task == "moses":
        from fm_graphex.data.moses import load_moses_dataset
        return load_moses_dataset(root=root, max_nodes=27, max_num_graphs=max_num_graphs)
    raise ValueError(task)


def _checkpoint_suffix(task: str) -> str:
    return f"_{cfg.FEATURE_VERSION}" if cfg.use_formal_charge(task) else ""


def train_autoencoder(
    task: str,
    root: str,
    node_dim: int = 64,
    hidden_dim: int = 128,
    batch_size: int = 32,
    num_epochs: int = 5,
    lr: float = 1e-3,
    max_num_graphs: Optional[int] = None,
    ckpt_dir: str = "./checkpoints_ae",
    load_if_exists: bool = True,
    save_after_train: bool = True,
) -> Tuple[GINEncoder, SimpleGraphDecoder, Sequence[Data], Sequence[Data], Sequence[Data], int]:
    os.makedirs(ckpt_dir, exist_ok=True)
    train_dataset, val_dataset, test_dataset, max_atoms = _load_dataset(task, root, max_num_graphs)
    if not train_dataset:
        raise RuntimeError(f"No training molecules were loaded for {task} from {root}.")

    in_dim = int(train_dataset[0].x.shape[1])
    edge_dim = int(train_dataset[0].edge_attr.shape[1])
    expected_in = cfg.node_feat_dim()
    if in_dim != expected_in:
        raise RuntimeError(
            f"Dataset x dimension is {in_dim}, but active {task} configuration expects {expected_in}. "
            "Restart the kernel after updating the code and delete stale processed/cache objects."
        )

    encoder = GINEncoder(in_dim, edge_dim, hidden_dim, 3, node_dim).to(cfg.DEVICE)
    decoder = SimpleGraphDecoder(
        node_dim=node_dim,
        num_atom_types=cfg.NUM_ATOM_TYPES,
        num_bond_types=cfg.NUM_BOND_TYPES,
        formal_charge_dim=cfg.formal_charge_dim(task),
    ).to(cfg.DEVICE)

    suffix = _checkpoint_suffix(task)
    enc_path = os.path.join(ckpt_dir, f"encoder_{task}_nd{node_dim}_hd{hidden_dim}{suffix}.pt")
    dec_path = os.path.join(ckpt_dir, f"decoder_{task}_nd{node_dim}_hd{hidden_dim}{suffix}.pt")

    if load_if_exists and os.path.exists(enc_path) and os.path.exists(dec_path):
        try:
            encoder.load_state_dict(torch.load(enc_path, map_location=cfg.DEVICE, weights_only=True))
            decoder.load_state_dict(torch.load(dec_path, map_location=cfg.DEVICE, weights_only=True))
            print(f"[AE] Loaded {enc_path} and {dec_path}")
            encoder.eval(); decoder.eval()
            return encoder, decoder, train_dataset, val_dataset, test_dataset, max_atoms
        except (RuntimeError, ValueError) as exc:
            print(f"[AE] Checkpoint incompatible with {cfg.FEATURE_VERSION}; retraining. Details: {exc}")

    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    print(f"[AE] Training {task} encoder with node feature dim {in_dim}.")

    for epoch in range(1, num_epochs + 1):
        encoder.train(); decoder.train()
        total = 0.0; used = 0
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            batch_loss = torch.zeros((), device=cfg.DEVICE)
            count = 0
            for data in batch.to_data_list():
                data = data.to(cfg.DEVICE)
                n = int(data.num_nodes)
                if n <= 0:
                    continue
                z = encoder(data.x, data.edge_index, data.edge_attr)
                node_logits, edge_logits = decoder(z)

                atom_target = atom_targets_from_data(data)
                loss = F.cross_entropy(node_logits[:, :cfg.NUM_ATOM_TYPES], atom_target)
                charge_target = charge_targets_from_data(data)
                if charge_target is not None:
                    q0 = cfg.NUM_ATOM_TYPES
                    loss = loss + F.cross_entropy(
                        node_logits[:, q0:q0 + len(cfg.FORMAL_CHARGES)],
                        charge_target,
                    )

                edge_target = build_edge_targets(n, data.edge_index, data.edge_attr)
                mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=cfg.DEVICE), diagonal=1)
                loss = loss + F.cross_entropy(edge_logits[mask], edge_target[mask])
                batch_loss = batch_loss + loss
                count += 1

            if count:
                batch_loss = batch_loss / count
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(decoder.parameters()), 1.0
                )
                opt.step()
                total += float(batch_loss.item()); used += 1
        print(f"[AE] Epoch {epoch}/{num_epochs} | loss={total / max(used, 1):.4f}")

    if save_after_train:
        torch.save(encoder.state_dict(), enc_path)
        torch.save(decoder.state_dict(), dec_path)
        print(f"[AE] Saved {enc_path}")
        print(f"[AE] Saved {dec_path}")

    return encoder, decoder, train_dataset, val_dataset, test_dataset, max_atoms