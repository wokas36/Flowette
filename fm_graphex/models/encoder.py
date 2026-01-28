from typing import Sequence, Tuple, Optional, List
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv
from rdkit import Chem

from fm_graphex.config import (
    DEVICE,
    NUM_ATOM_TYPES,
    NUM_BOND_TYPES,
    Z_TO_TYPE,
)
from fm_graphex.data.qm9 import load_qm9_heavy_dataset
from fm_graphex.data.zinc import load_zinc_dataset
from fm_graphex.data.guacamol import load_guacamol_dataset
from fm_graphex.data.moses import load_moses_dataset


# GNN encoder
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
            self.convs.append(GINEConv(mlp, edge_dim=self.edge_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            dim = hidden_dim

        self.proj = nn.Linear(dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index, edge_attr)
            h = bn(h)
            h = F.relu(h)
        z = self.proj(h)
        return z


# GNN decoder
class SimpleGraphDecoder(nn.Module):
    def __init__(self, node_dim: int, num_atom_types: int, num_bond_types: int):
        super().__init__()
        self.node_dim = node_dim
        self.num_atom_types = num_atom_types
        self.num_bond_types = num_bond_types

        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, num_atom_types),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, num_bond_types + 1), # +1 = no bond
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        N = z.size(0)
        node_logits = self.node_mlp(z)

        zi = z.unsqueeze(1).expand(N, N, self.node_dim)
        zj = z.unsqueeze(0).expand(N, N, self.node_dim)
        pair = torch.cat([zi, zj], dim=-1)
        edge_logits = self.edge_mlp(pair)
        return node_logits, edge_logits

    def decode_molecule(
        self, z: torch.Tensor, bond_threshold: float = 0.5
    ) -> Optional[Chem.Mol]:
        node_logits, edge_logits = self.forward(z)
        atom_probs = F.softmax(node_logits, dim=-1)
        edge_probs = F.softmax(edge_logits, dim=-1)

        N = z.size(0)
        atom_types = atom_probs.argmax(dim=-1).cpu().numpy()

        mol = rdchem.RWMol()
        for aidx in atom_types:
            sym = TYPE_TO_SYM.get(int(aidx), "C")
            mol.AddAtom(rdchem.Atom(sym))

        for i in range(N):
            for j in range(i + 1, N):
                probs_ij = edge_probs[i, j]
                bond_type_idx = probs_ij[:-1].argmax().item()
                bond_prob = probs_ij[bond_type_idx].item()
                no_bond_prob = probs_ij[-1].item()
                if bond_prob > bond_threshold and bond_prob > no_bond_prob:
                    bt = BOND_TYPES[min(bond_type_idx, len(BOND_TYPES) - 1)]
                    try:
                        mol.AddBond(int(i), int(j), bt)
                    except Exception:
                        continue

        rmol = mol.GetMol()
        try:
            Chem.SanitizeMol(rmol)
        except Exception:
            return None
        return rmol
    

# Encoder training helpers
def atom_targets_from_data(data: Data) -> torch.Tensor:
    """
    Robust atom targets:
      - if data.x is one-hot of size NUM_ATOM_TYPES -> argmax
      - else if data.z exists -> map via Z_TO_TYPE
    """
    if hasattr(data, "x") and data.x is not None and data.x.dim() == 2 and data.x.size(1) == NUM_ATOM_TYPES:
        return data.x.argmax(dim=-1).long()
    if hasattr(data, "z") and data.z is not None:
        z_np = data.z.detach().cpu().tolist()
        out = []
        for z in z_np:
            z = int(z)
            out.append(Z_TO_TYPE.get(z, Z_TO_TYPE.get(6, 0)))
        return torch.tensor(out, dtype=torch.long, device=data.z.device)
    raise ValueError("Cannot build atom targets: need data.x one-hot or data.z")


def build_edge_targets(
    n: int,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
) -> torch.Tensor:
    edge_target = torch.full(
        (n, n),
        fill_value=NUM_BOND_TYPES,
        dtype=torch.long,
        device=edge_index.device,
    )
    E = edge_index.size(1)
    seen = set()
    for e in range(E):
        i = int(edge_index[0, e])
        j = int(edge_index[1, e])
        if i == j:
            continue
        a, b = sorted((i, j))
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        bt_idx = int(edge_attr[e, :NUM_BOND_TYPES].argmax().item())
        bt_idx = min(bt_idx, NUM_BOND_TYPES - 1)
        edge_target[a, b] = bt_idx
    return edge_target


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
) -> Tuple[GINEncoder, SimpleGraphDecoder, Sequence[Data], Sequence[Data], Sequence[Data]]:
    """
    Train (or load) the GIN encoder + simple decoder on heavy-atom QM9.

    If 'load_if_exists' is True and checkpoint files are found, it will
    load the pre-trained models and skip training.

    Returns:
        encoder, decoder, train_dataset, val_dataset, test_dataset
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Load dataset according to task
    if task == "qm9":
        train_dataset, val_dataset, test_dataset, max_atoms = load_qm9_heavy_dataset(root=root, max_num_graphs=max_num_graphs)
    elif task == "zinc":
        train_dataset, val_dataset, test_dataset, max_atoms = load_zinc_dataset(root=root, max_num_graphs=max_num_graphs)
    elif task == "guacamol":
        train_dataset, val_dataset, test_dataset, max_atoms = load_guacamol_dataset(root=root, max_nodes=88, 
                                                                                    max_num_graphs=max_num_graphs)
    elif task == "moses":
        train_dataset, val_dataset, test_dataset, max_atoms = load_moses_dataset(root=root, max_nodes=27, 
                                                                                 max_num_graphs=max_num_graphs)
    else:
        raise ValueError(task)

    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    in_dim  = train_dataset[0].x.shape[1]
    edge_dim = train_dataset[0].edge_attr.shape[1]

    encoder = GINEncoder(
        in_dim=in_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        num_layers=3,
        out_dim=node_dim,
    ).to(DEVICE)

    decoder = SimpleGraphDecoder(
        node_dim=node_dim,
        num_atom_types=NUM_ATOM_TYPES,
        num_bond_types=NUM_BOND_TYPES,
    ).to(DEVICE)

    # Try to LOAD existing checkpoints 
    enc_ckpt_path = os.path.join(ckpt_dir, f"encoder_{task}_nd{node_dim}_hd{hidden_dim}.pt")
    dec_ckpt_path = os.path.join(ckpt_dir, f"decoder_{task}_nd{node_dim}_hd{hidden_dim}.pt")

    if load_if_exists and os.path.exists(enc_ckpt_path) and os.path.exists(dec_ckpt_path):
        print(f"[AE] Found existing checkpoints in {ckpt_dir}, loading them...")
        encoder.load_state_dict(torch.load(enc_ckpt_path, map_location=DEVICE, weights_only=True))
        decoder.load_state_dict(torch.load(dec_ckpt_path, map_location=DEVICE, weights_only=True))
        encoder.eval()
        decoder.eval()
        return encoder, decoder, train_dataset, val_dataset, test_dataset, max_atoms

    # Train from scratch if no checkpoint is found
    print("[AE] No existing checkpoint found (or loading disabled). Training from scratch...")

    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=lr,
    )

    for epoch in range(1, num_epochs + 1):
        encoder.train()
        decoder.train()
        total_loss = 0.0

        for batch in loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()

            graphs = batch.to_data_list()
            batch_loss = 0.0
            count = 0

            for data in graphs:
                data = data.to(DEVICE)
                x, edge_index, edge_attr = (
                    data.x,
                    data.edge_index,
                    data.edge_attr,
                )
                n = x.size(0)
                if n == 0:
                    continue
                
                # forward
                z_latent = encoder(x, edge_index, edge_attr)
                node_logits, edge_logits = decoder(z_latent)

                # targets
                atom_target = atom_targets_from_data(data)
                edge_target = build_edge_targets(n, edge_index, edge_attr)

                node_loss = F.cross_entropy(node_logits, atom_target)

                mask = torch.triu(
                    torch.ones(n, n, dtype=torch.bool, device=DEVICE), 1
                )
                edge_pred = edge_logits[mask]
                edge_gt   = edge_target[mask]
                edge_loss = F.cross_entropy(edge_pred, edge_gt)

                batch_loss += node_loss + edge_loss
                count += 1

            if count > 0:
                batch_loss = batch_loss / count
                batch_loss.backward()
                opt.step()
                total_loss += batch_loss.item()

        print(f"[AE] Epoch {epoch}/{num_epochs} | Loss: {total_loss / len(loader):.4f}")

    # Save checkpoints for future runs
    if save_after_train:
        torch.save(encoder.state_dict(), enc_ckpt_path)
        torch.save(decoder.state_dict(), dec_ckpt_path)
        print(f"[AE] Saved encoder to {enc_ckpt_path}")
        print(f"[AE] Saved decoder to {dec_ckpt_path}")

    return encoder, decoder, train_dataset, val_dataset, test_dataset, max_atoms